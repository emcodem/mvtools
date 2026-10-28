#define NOMINMAX
#define NOGDI
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "AvstpWrapper.h"
#include "GpuBackend.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

extern "C" IMAGE_DOS_HEADER __ImageBase;

namespace
{
  struct ProfTable
  {
    std::mutex m;
    std::map<std::string, std::pair<double, long>> t;
    ~ProfTable()
    {
      if (t.empty())
        return;
      std::fprintf(stderr, "mvtools gpu profile (wall ms summed over all threads, calls, average):\n");
      for (auto &e : t)
        std::fprintf(stderr, "  %-44s %10.1f ms %7ld calls %8.3f ms\n", e.first.c_str(), e.second.first, e.second.second,
          e.second.first / e.second.second);
    }
  };
  ProfTable &prof_table()
  {
    static ProfTable p;
    return p;
  }
}

bool gpu_prof_enabled()
{
  static const bool on = (std::getenv("MVTOOLS_GPU_PROFILE") != nullptr);
  return on;
}

void gpu_prof_add(const char *name, double ms)
{
  ProfTable &p = prof_table();
  std::lock_guard<std::mutex> lk(p.m);
  auto &e = p.t[name];
  e.first += ms;
  e.second++;
}

GpuBackend &GpuBackend::instance()
{
  static GpuBackend inst; // thread-safe initialisation (C++11)
  return inst;
}

bool GpuBackend::log_enabled()
{
  static const bool on = (std::getenv("MVTOOLS_GPU_LOG") != nullptr);
  return on;
}

bool GpuBackend::force_fallback()
{
  static const bool on = (std::getenv("MVTOOLS_GPU_FORCE_FALLBACK") != nullptr);
  return on;
}

int GpuBackend::slice_threads(bool mt_flag)
{
  if (!mt_flag)
    return 1;
  // MVTOOLS_GPU_SLICES overrides the slice count (more slices = shorter serial chains on the GPU, but a different
  // vector field than the CPU with the default thread count; the CPU result depends on its thread count as well)
  if (const char *e = std::getenv("MVTOOLS_GPU_SLICES"))
  {
    const int v = std::atoi(e);
    if (v >= 1)
      return v;
  }
  const int n = AvstpWrapper::use_instance().get_nbr_threads();
  return n > 1 ? n : 24;
}

unsigned long long GpuBackend::frame_key(const void *clip, int n)
{
  unsigned long long k = (unsigned long long)(uintptr_t)clip * 0x9E3779B97F4A7C15ull;
  k ^= ((unsigned long long)(n + 1) * 0xC2B2AE3D27D4EB4Full) + (k << 6) + (k >> 2);
  return k | 1ull; // never 0 (0 = uncached)
}

void GpuBackend::add_user()
{
  InterlockedIncrement(&_users);
}

void GpuBackend::remove_user()
{
  if (InterlockedDecrement(&_users) == 0 && cache_clear != nullptr)
    cache_clear();
}

GpuBackend::GpuBackend()
  : analyse_host(nullptr)
  , recalc_host(nullptr)
  , flow_host(nullptr)
  , cache_clear(nullptr)
  , _usable(false)
  , _users(0)
{
  if (std::getenv("MVTOOLS_GPU_DISABLE") != nullptr)
  {
    if (log_enabled())
      std::fprintf(stderr, "mvtools gpu: disabled by MVTOOLS_GPU_DISABLE\n");
    return;
  }

  wchar_t path[MAX_PATH] = {};
  if (GetModuleFileNameW(reinterpret_cast<HMODULE>(&__ImageBase), path, MAX_PATH) == 0)
  {
    if (log_enabled())
      std::fprintf(stderr, "mvtools gpu: cannot determine the plugin folder\n");
    return;
  }
  std::wstring dir(path);
  const size_t slash = dir.find_last_of(L"\\/");
  if (slash == std::wstring::npos)
  {
    if (log_enabled())
      std::fprintf(stderr, "mvtools gpu: plugin path has no folder part\n");
    return;
  }
  dir.resize(slash + 1);

  HMODULE lib = LoadLibraryW((dir + L"mvtools_cuda.dll").c_str());
  if (lib == nullptr)
  {
    if (log_enabled())
      std::fprintf(stderr, "mvtools gpu: mvtools_cuda.dll not found next to mvtools2.dll, using the CPU\n");
    return;
  }
  auto version = reinterpret_cast<int (*)()>(GetProcAddress(lib, "mvcuda_api_version"));
  auto count = reinterpret_cast<int (*)()>(GetProcAddress(lib, "mvcuda_device_count"));
  analyse_host = reinterpret_cast<decltype(analyse_host)>(GetProcAddress(lib, "mvcuda_analyse_host"));
  recalc_host = reinterpret_cast<decltype(recalc_host)>(GetProcAddress(lib, "mvcuda_recalc_host"));
  flow_host = reinterpret_cast<decltype(flow_host)>(GetProcAddress(lib, "mvcuda_flow_host"));
  cache_clear = reinterpret_cast<decltype(cache_clear)>(GetProcAddress(lib, "mvcuda_cache_clear"));
  if (version == nullptr || count == nullptr || analyse_host == nullptr || recalc_host == nullptr || flow_host == nullptr || cache_clear == nullptr
    || version() != MVCUDA_API_VERSION || count() <= 0)
  {
    if (log_enabled())
      std::fprintf(stderr, "mvtools gpu: library unusable (missing exports, API version or no device), using the CPU\n");
    analyse_host = nullptr;
    recalc_host = nullptr;
    return;
  }
  _usable = true;
  if (log_enabled())
    std::fprintf(stderr, "mvtools gpu: backend loaded\n");
}
