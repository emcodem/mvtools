#define NOMINMAX
#define NOGDI
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "AvstpWrapper.h"
#include "GpuBackend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

bool GpuBackend::read_token(const void *p, int row_bytes, MvCudaTokenHeader &h)
{
  if (p == nullptr || row_bytes < (int)sizeof(h))
    return false;
  std::memcpy(&h, p, sizeof(h));
  return h.magic == MVCUDA_TOKEN_MAGIC && h.layout == MVCUDA_TOKEN_LAYOUT;
}

bool GpuBackend::download_token(const MvCudaTokenHeader &h, void *const dst[3], const int dst_pitch[3], void *env)
{
  if (!_usable)
    return false;
  int rc = frame_download(h.token, dst, dst_pitch);
  if (rc == MVCUDA_TOKEN_GONE && h.producer != 0)
  {
    MvCudaTokenHeader again;
    if (token_regenerate(h.producer, h.frame, env, &again) == 0)
      rc = frame_download(again.token, dst, dst_pitch);
  }
  if (log_enabled())
    std::fprintf(stderr, "mvtools gpu: token frame %d downloaded, rc=%d\n", h.frame, rc);
  return rc == 0;
}

bool GpuBackend::lite_token(const PVideoFrame &f, const VideoInfo &super_vi, MvCudaTokenHeader &h)
{
  if (!super_vi.IsPlanar())
    return false;
  int64_t packed;
  std::memcpy(&packed, &super_vi.num_audio_samples, 8);
  const unsigned char hpad = (unsigned char)((packed >> 16) & 0xff), vpad = (unsigned char)((packed >> 24) & 0xff);
  const int pixelsize = super_vi.ComponentSize();
  const int pic_row = (super_vi.width - 2 * hpad) * pixelsize;
  return read_token(f->GetReadPtr(PLANAR_Y) + (size_t)vpad * f->GetPitch(PLANAR_Y) + (size_t)hpad * pixelsize, pic_row, h);
}

bool GpuBackend::frame_token(const PVideoFrame &f, const VideoInfo &vi, MvCudaTokenHeader &h)
{
  return vi.IsPlanar() && read_token(f->GetReadPtr(PLANAR_Y), f->GetRowSize(PLANAR_Y), h);
}

PVideoFrame GpuBackend::resolve_frame(const PVideoFrame &f, const VideoInfo &vi, IScriptEnvironment *env)
{
  MvCudaTokenHeader h;
  if (!frame_token(f, vi, h))
    return f;
  PVideoFrame pic = env->NewVideoFrame(vi);
  const int planes[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
  void *d[3] = { 0, 0, 0 };
  int dp[3] = { 0, 0, 0 };
  for (int p = 0; p < (vi.IsY() ? 1 : 3); p++)
  {
    d[p] = pic->GetWritePtr(planes[p]);
    dp[p] = pic->GetPitch(planes[p]);
  }
  if (!download_token(h, d, dp, env))
    env->ThrowError("mvtools: GPU frame %d is no longer available on the device", h.frame);
  return pic;
}

void GpuBackend::write_token(PVideoFrame &dst, const MvCudaTokenHeader &h)
{
  std::memcpy(dst->GetWritePtr(PLANAR_Y), &h, sizeof(h));
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
  , analyse_token(nullptr)
  , frame_download(nullptr)
  , token_regenerate(nullptr)
  , token_pin(nullptr)
  , token_unpin(nullptr)
  , producer_register(nullptr)
  , producer_unregister(nullptr)
  , compensate(nullptr)
  , degrain(nullptr)
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
  analyse_token = reinterpret_cast<decltype(analyse_token)>(GetProcAddress(lib, "mvcuda_analyse_token"));
  frame_download = reinterpret_cast<decltype(frame_download)>(GetProcAddress(lib, "mvcuda_frame_download"));
  token_regenerate = reinterpret_cast<decltype(token_regenerate)>(GetProcAddress(lib, "mvcuda_token_regenerate"));
  token_pin = reinterpret_cast<decltype(token_pin)>(GetProcAddress(lib, "mvcuda_token_pin"));
  token_unpin = reinterpret_cast<decltype(token_unpin)>(GetProcAddress(lib, "mvcuda_token_unpin"));
  producer_register = reinterpret_cast<decltype(producer_register)>(GetProcAddress(lib, "mvcuda_producer_register"));
  producer_unregister = reinterpret_cast<decltype(producer_unregister)>(GetProcAddress(lib, "mvcuda_producer_unregister"));
  compensate = reinterpret_cast<decltype(compensate)>(GetProcAddress(lib, "mvcuda_compensate"));
  degrain = reinterpret_cast<decltype(degrain)>(GetProcAddress(lib, "mvcuda_degrain"));
  if (version == nullptr || count == nullptr || analyse_host == nullptr || recalc_host == nullptr || flow_host == nullptr || cache_clear == nullptr
    || analyse_token == nullptr || frame_download == nullptr || token_regenerate == nullptr || token_pin == nullptr || token_unpin == nullptr || producer_register == nullptr || producer_unregister == nullptr || compensate == nullptr || degrain == nullptr
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
