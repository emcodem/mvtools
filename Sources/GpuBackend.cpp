#define NOMINMAX
#define NOGDI
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "AvstpWrapper.h"
#include "GpuBackend.h"
#include "MVPlaneSet.h"
#include "SuperParams64Bits.h"

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
  BYTE *y = dst->GetWritePtr(PLANAR_Y);
  const int pitch = dst->GetPitch(PLANAR_Y), row = dst->GetRowSize(PLANAR_Y), height = dst->GetHeight(PLANAR_Y);
  std::memcpy(y, &h, sizeof(h));
  for (int r = 8; r < height; r += 8)
    std::memset(y + (size_t)r * pitch, 0, row);
}

namespace
{
  bool make_format(const VideoInfo &vi, int width, int height, bool chroma, MvCudaFrameFormat &f)
  {
    if (!vi.IsPlanar() || vi.IsRGB() || vi.IsYUY2() || (vi.ComponentSize() != 1 && vi.ComponentSize() != 2))
      return false;
    f = MvCudaFrameFormat();
    f.width = width;
    f.height = height;
    f.planes = chroma && !vi.IsY() ? 3 : 1;
    f.bytes_per_sample = vi.ComponentSize();
    f.bits_per_sample = vi.BitsPerComponent();
    f.x_ratio_uv = f.planes == 3 ? 1 << vi.GetPlaneWidthSubsampling(PLANAR_U) : 1;
    f.y_ratio_uv = f.planes == 3 ? 1 << vi.GetPlaneHeightSubsampling(PLANAR_U) : 1;
    return f.x_ratio_uv <= 2 && f.y_ratio_uv <= 2 && width > 0 && height > 0;
  }
}

bool GpuBackend::super_input(const PVideoFrame &f, const VideoInfo &super_vi, const void *clip, int n, Input &in)
{
  SuperParams64Bits params;
  std::memcpy(&params, &super_vi.num_audio_samples, 8);
  in = Input();
  // the picture's chroma planes exist in the super frame only when MSuper processed them
  if (!make_format(super_vi, super_vi.width - 2 * params.nHPad, params.nHeight, (params.nModeYUV & UPLANE) != 0, in.fmt))
    return false;
  const int planes_l[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
  for (int p = 0; p < in.fmt.planes; p++)
  {
    const int xr = p ? in.fmt.x_ratio_uv : 1, yr = p ? in.fmt.y_ratio_uv : 1;
    in.pitch[p] = f->GetPitch(planes_l[p]);
    in.planes[p] = f->GetReadPtr(planes_l[p]) + (size_t)(params.nVPad / yr) * in.pitch[p] + (size_t)(params.nHPad / xr) * in.fmt.bytes_per_sample;
  }
  if (read_token(in.planes[0], in.fmt.width * in.fmt.bytes_per_sample, in.h))
    return true; // a device frame already (lite super of a token clip)
  in.host = true;
  in.key = frame_key(clip, n);
  return true;
}

bool GpuBackend::frame_input(const PVideoFrame &f, const VideoInfo &vi, const void *clip, int n, Input &in)
{
  in = Input();
  if (!make_format(vi, vi.width, vi.height, true, in.fmt))
    return false;
  if (frame_token(f, vi, in.h))
    return true;
  const int planes_l[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
  for (int p = 0; p < in.fmt.planes; p++)
  {
    in.pitch[p] = f->GetPitch(planes_l[p]);
    in.planes[p] = f->GetReadPtr(planes_l[p]);
  }
  in.host = true;
  in.key = frame_key(clip, n);
  return true;
}

bool GpuBackend::download_frame(const MvCudaTokenHeader &h, PVideoFrame &dst, const VideoInfo &vi)
{
  const int planes_l[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
  void *d[3] = { 0, 0, 0 };
  int dp[3] = { 0, 0, 0 };
  for (int p = 0; p < (vi.IsY() ? 1 : 3); p++)
  {
    d[p] = dst->GetWritePtr(planes_l[p]);
    dp[p] = dst->GetPitch(planes_l[p]);
  }
  return frame_download(h.token, d, dp) == 0;
}

void GpuBackend::remove_user()
{
  if (InterlockedDecrement(&_users) != 0)
    return;
  if (log_enabled() && frame_stats != nullptr)
  {
    MvCudaFrameStats s;
    frame_stats(&s);
    std::fprintf(stderr, "mvtools gpu: device frames %llu (%.1f of %.1f MB), uploads %llu (shared %llu), downloads %llu, evictions %llu, regenerations %llu, lookups of evicted tokens %llu\n",
      s.frames, s.bytes / 1048576.0, s.budget / 1048576.0, s.uploads, s.upload_hits, s.downloads, s.evictions, s.regenerations, s.lookups_gone);
  }
  if (cache_clear != nullptr)
    cache_clear();
}

GpuBackend::GpuBackend()
  : cache_clear(nullptr)
  , analyse_token(nullptr)
  , recalc_token(nullptr)
  , flow_token(nullptr)
  , flow_token_host(nullptr)
  , frame_upload_keyed(nullptr)
  , frame_download(nullptr)
  , frame_stats(nullptr)
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
  cache_clear = reinterpret_cast<decltype(cache_clear)>(GetProcAddress(lib, "mvcuda_cache_clear"));
  analyse_token = reinterpret_cast<decltype(analyse_token)>(GetProcAddress(lib, "mvcuda_analyse_token"));
  recalc_token = reinterpret_cast<decltype(recalc_token)>(GetProcAddress(lib, "mvcuda_recalc_token"));
  flow_token = reinterpret_cast<decltype(flow_token)>(GetProcAddress(lib, "mvcuda_flow_token"));
  flow_token_host = reinterpret_cast<decltype(flow_token_host)>(GetProcAddress(lib, "mvcuda_flow_token_host"));
  frame_upload_keyed = reinterpret_cast<decltype(frame_upload_keyed)>(GetProcAddress(lib, "mvcuda_frame_upload_keyed"));
  frame_download = reinterpret_cast<decltype(frame_download)>(GetProcAddress(lib, "mvcuda_frame_download"));
  frame_stats = reinterpret_cast<decltype(frame_stats)>(GetProcAddress(lib, "mvcuda_frame_stats"));
  token_regenerate = reinterpret_cast<decltype(token_regenerate)>(GetProcAddress(lib, "mvcuda_token_regenerate"));
  token_pin = reinterpret_cast<decltype(token_pin)>(GetProcAddress(lib, "mvcuda_token_pin"));
  token_unpin = reinterpret_cast<decltype(token_unpin)>(GetProcAddress(lib, "mvcuda_token_unpin"));
  producer_register = reinterpret_cast<decltype(producer_register)>(GetProcAddress(lib, "mvcuda_producer_register"));
  producer_unregister = reinterpret_cast<decltype(producer_unregister)>(GetProcAddress(lib, "mvcuda_producer_unregister"));
  compensate = reinterpret_cast<decltype(compensate)>(GetProcAddress(lib, "mvcuda_compensate"));
  degrain = reinterpret_cast<decltype(degrain)>(GetProcAddress(lib, "mvcuda_degrain"));
  if (version == nullptr || count == nullptr || cache_clear == nullptr || recalc_token == nullptr || flow_token == nullptr
    || frame_upload_keyed == nullptr || flow_token_host == nullptr || analyse_token == nullptr || frame_download == nullptr || token_regenerate == nullptr || token_pin == nullptr || token_unpin == nullptr || producer_register == nullptr || producer_unregister == nullptr || compensate == nullptr || degrain == nullptr
    || version() != MVCUDA_API_VERSION || count() <= 0)
  {
    if (log_enabled())
      std::fprintf(stderr, "mvtools gpu: library unusable (missing exports, API version or no device), using the CPU\n");
    return;
  }
  _usable = true;
  if (log_enabled())
    std::fprintf(stderr, "mvtools gpu: backend loaded\n");
}
