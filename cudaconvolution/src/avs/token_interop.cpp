#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS  // std::getenv, as in the rest of the plugin
#include <windows.h>
#include <psapi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <mutex>
#include <vector>

#include "avs/token_interop.h"

namespace cudaconvolution::token {

namespace {

constexpr const wchar_t* kLibName = L"mvtools_cuda.dll";

bool logging() {
  static const bool on = std::getenv("CUDACONVOLUTION_LOG") != nullptr;
  return on;
}

std::filesystem::path module_dir() {
  HMODULE self = nullptr;
  if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCWSTR>(&module_dir), &self))
    return {};
  wchar_t buf[MAX_PATH * 4];
  const DWORD n = GetModuleFileNameW(self, buf, static_cast<DWORD>(std::size(buf)));
  if (n == 0 || n >= std::size(buf)) return {};
  return std::filesystem::path(buf).parent_path();
}

template <class F>
F symbol(HMODULE lib, const char* name) {
  return reinterpret_cast<F>(GetProcAddress(lib, name));
}

// the library the GPU plugins of this process use: an instance already loaded (pinned: never unloaded under us), else
// the copy in this plugin's folder (loaded without error dialog boxes: an unattended encode must not hang on one)
HMODULE find_library(std::string& why) {
  HMODULE lib = nullptr;
  if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN, kLibName, &lib) && lib) return lib;
  const std::filesystem::path dir = module_dir();
  if (dir.empty()) {
    why = "the plugin folder could not be determined";
    return nullptr;
  }
  DWORD old_mode = 0;
  SetThreadErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX, &old_mode);
  lib = LoadLibraryExW((dir / kLibName).c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  SetThreadErrorMode(old_mode, nullptr);
  if (!lib) why = "no GPU plugin has loaded mvtools_cuda.dll and there is none in CudaConvolution's folder";
  return lib;
}

// number of distinct loaded modules named mvtools_cuda.dll
int library_instances() {
  HMODULE mods[1024];
  DWORD bytes = 0;
  if (!K32EnumProcessModules(GetCurrentProcess(), mods, sizeof(mods), &bytes)) return 1;  // cannot tell: assume one
  const DWORD count = std::min<DWORD>(bytes / sizeof(HMODULE), static_cast<DWORD>(std::size(mods)));
  int found = 0;
  wchar_t name[MAX_PATH];
  for (DWORD i = 0; i < count; ++i)
    if (K32GetModuleBaseNameW(GetCurrentProcess(), mods[i], name, MAX_PATH) && _wcsicmp(name, kLibName) == 0) ++found;
  return found;
}

}  // namespace

Registry& Registry::get() {
  static Registry r;
  if (!r.available()) {
    std::lock_guard<std::mutex> lock(r.m_);
    if (!r.available() && r.not_found_) r.init();
  }
  return r;
}

std::string Registry::why() const {
  std::lock_guard<std::mutex> lock(m_);
  return why_;
}

void Registry::init() {
  if (std::getenv("CUDACONVOLUTION_NO_TOKENS")) {
    why_ = "disabled by the environment variable CUDACONVOLUTION_NO_TOKENS";
    not_found_ = false;
    return;
  }
  std::string why;
  HMODULE lib = find_library(why);
  if (!lib) {
    why_ = why;
    return;
  }
  not_found_ = false;  // found: decided once, below
  auto version = symbol<int (*)()>(lib, "mvcuda_api_version");
  auto count = symbol<int (*)()>(lib, "mvcuda_device_count");
  auto token_abi = symbol<int (*)()>(lib, "mvcuda_token_abi");
  download_ = symbol<decltype(download_)>(lib, "mvcuda_frame_download");
  upload_ = symbol<decltype(upload_)>(lib, "mvcuda_frame_upload");
  producer_register_ = symbol<decltype(producer_register_)>(lib, "mvcuda_producer_register");
  producer_unregister_ = symbol<decltype(producer_unregister_)>(lib, "mvcuda_producer_unregister");
  regenerate_ = symbol<decltype(regenerate_)>(lib, "mvcuda_token_regenerate");
  pin_ = symbol<decltype(pin_)>(lib, "mvcuda_token_pin");
  unpin_ = symbol<decltype(unpin_)>(lib, "mvcuda_token_unpin");
  if (!version || !count || !download_ || !upload_ || !producer_register_ || !producer_unregister_ || !regenerate_ ||
      !pin_ || !unpin_) {
    why_ = "mvtools_cuda.dll does not export the device-frame functions (too old?)";
    return;
  }
  api_ = version();
  // never call through function signatures this file does not know
  if (token_abi) {
    if (const int abi = token_abi(); abi != kTokenAbi) {
      why_ = "mvtools_cuda.dll has token API " + std::to_string(abi) + ", this CudaConvolution knows " +
             std::to_string(kTokenAbi) + ": update CudaConvolution";
      return;
    }
    // part of token ABI 1 (optional all the same: without them, the copy path)
    download_checked_ = symbol<decltype(download_checked_)>(lib, "mvcuda_frame_download_checked");
    run_ = symbol<decltype(run_)>(lib, "mvcuda_frame_run");
  } else if (api_ < kMinApiVersion || api_ > kMaxApiVersion) {
    why_ = "mvtools_cuda.dll API version " + std::to_string(api_) + " is not one this CudaConvolution knows (" +
           std::to_string(kMinApiVersion) + ".." + std::to_string(kMaxApiVersion) + "): update CudaConvolution";
    return;
  }
  if (count() <= 0) {
    why_ = "mvtools_cuda.dll found no usable NVIDIA GPU";
    return;
  }
  ok_.store(true, std::memory_order_release);
  if (logging()) std::fprintf(stderr, "CudaConvolution: token interop available (mvtools_cuda API %d)\n", api_);
}

void Registry::require_single_instance(IScriptEnvironment* env) const {
  // re-checked only when the number of loaded modules changes (a plugin loaded later could bring a second copy)
  static std::mutex m;
  static DWORD checked_modules = 0;
  static int instances = 1;
  HMODULE probe[1];
  DWORD bytes = 0;
  K32EnumProcessModules(GetCurrentProcess(), probe, sizeof(probe), &bytes);
  {
    std::lock_guard<std::mutex> lock(m);
    if (bytes != checked_modules) {
      instances = library_instances();
      checked_modules = bytes;
      if (logging()) std::fprintf(stderr, "CudaConvolution: %d loaded copies of mvtools_cuda.dll\n", instances);
    }
    if (instances <= 1) return;
  }
  env->ThrowError("CudaConvolution: %d different copies of mvtools_cuda.dll are loaded in this process; GPU token frames of "
                  "one cannot be read through another. Load all GPU plugins (and CudaConvolution.dll) from one folder.",
                  instances);
}

int Registry::download(unsigned long long token, const FrameFormat& expect, void* const dst[3],
                       const int pitch[3]) const {
  return download_checked_ ? download_checked_(token, &expect, dst, pitch) : download_(token, dst, pitch);
}
int Registry::run(const unsigned long long* in_tokens, int n_in, const FrameFormat* out_fmt,
                  unsigned long long producer, int frame, RunFn fn, void* user, Header* out) const {
  return run_ ? run_(in_tokens, n_in, out_fmt, producer, frame, fn, user, out) : -4;
}
int Registry::upload(const FrameFormat& f, const void* const src[3], const int pitch[3], unsigned long long producer,
                     int frame, Header& out) const {
  return upload_(&f, src, pitch, producer, frame, &out);
}
unsigned long long Registry::producer_register(RegenFn fn, void* user) const { return producer_register_(fn, user); }
void Registry::producer_unregister(unsigned long long producer) const { producer_unregister_(producer); }
int Registry::regenerate(unsigned long long producer, int frame, void* env, Header* out) const {
  return regenerate_(producer, frame, env, out);
}
int Registry::pin(unsigned long long token) const { return pin_(token); }
void Registry::unpin(unsigned long long token) const { unpin_(token); }

bool make_format(const VideoInfo& vi, FrameFormat& f) {
  std::memset(&f, 0, sizeof(f));
  if (!vi.IsPlanar() || vi.IsRGB() || vi.IsYUVA() || vi.width <= 0 || vi.height <= 0) return false;
  f.width = vi.width;
  f.height = vi.height;
  f.planes = vi.IsY() ? 1 : 3;
  f.bytes_per_sample = vi.ComponentSize();
  f.bits_per_sample = vi.BitsPerComponent();
  if (f.bytes_per_sample != 1 && f.bytes_per_sample != 2) return false;
  f.x_ratio_uv = f.y_ratio_uv = 1;
  if (f.planes == 3) {
    f.x_ratio_uv = 1 << vi.GetPlaneWidthSubsampling(PLANAR_U);
    f.y_ratio_uv = 1 << vi.GetPlaneHeightSubsampling(PLANAR_U);
    if (f.x_ratio_uv > 2 || f.y_ratio_uv > 2) return false;
  }
  return static_cast<size_t>(vi.width) * f.bytes_per_sample >= sizeof(Header);
}

bool same_format(const FrameFormat& a, const FrameFormat& b) {
  return a.width == b.width && a.height == b.height && a.planes == b.planes &&
         a.bytes_per_sample == b.bytes_per_sample && a.bits_per_sample == b.bits_per_sample &&
         (a.planes == 1 || (a.x_ratio_uv == b.x_ratio_uv && a.y_ratio_uv == b.y_ratio_uv));
}

Kind classify(const PVideoFrame& f, const VideoInfo& vi, Header& h) {
  // token frames are planar Y / YUV with row 0 at least as wide as the header (the library's own rule)
  if (!vi.IsPlanar() || vi.IsRGB() || f->GetRowSize(PLANAR_Y) < static_cast<int>(sizeof(h))) return Kind::Picture;
  std::memcpy(&h, f->GetReadPtr(PLANAR_Y), sizeof(h));
  if (h.magic != kMagic) return Kind::Picture;
  return h.layout == kLayout ? Kind::Token : Kind::UnknownLayout;
}

void write_token(PVideoFrame& dst, const Header& h) {
  BYTE* y = dst->GetWritePtr(PLANAR_Y);
  const int pitch = dst->GetPitch(PLANAR_Y), row = dst->GetRowSize(PLANAR_Y), height = dst->GetHeight(PLANAR_Y);
  std::memcpy(y, &h, sizeof(h));
  for (int r = 8; r < height; r += 8) std::memset(y + static_cast<size_t>(r) * pitch, 0, row);
}

PVideoFrame materialize(const PVideoFrame& f, const VideoInfo& vi, int n, IScriptEnvironment* env) {
  Header h;
  const Kind kind = classify(f, vi, h);
  if (kind == Kind::Picture) return f;
  if (kind == Kind::UnknownLayout)
    env->ThrowError("CudaConvolution: frame %d is a GPU token frame of layout %d, which this CudaConvolution cannot read (it knows "
                    "layout %d). Update CudaConvolution, or put CudaDownload() in front of it.",
                    n, h.layout, kLayout);
  const Registry& reg = Registry::get();
  if (!reg.available())
    env->ThrowError("CudaConvolution: frame %d is a GPU token frame, but CudaConvolution cannot read it: %s. Put CudaDownload() in "
                    "front of CudaConvolution, or load CudaConvolution.dll from the folder of the GPU plugins.",
                    n, reg.why().c_str());
  reg.require_single_instance(env);
  FrameFormat want;
  if (!make_format(vi, want) || !same_format(h.format, want))
    env->ThrowError("CudaConvolution: frame %d is a GPU token frame of another format than the clip announces.", n);

  PVideoFrame dst = env->NewVideoFrameP(vi, const_cast<PVideoFrame*>(&f));
  void* planes[3] = {};
  int pitches[3] = {};
  static constexpr int kPlanes[3] = {PLANAR_Y, PLANAR_U, PLANAR_V};
  for (int p = 0; p < want.planes; ++p) {
    planes[p] = dst->GetWritePtr(kPlanes[p]);
    pitches[p] = dst->GetPitch(kPlanes[p]);
  }
  int rc = reg.download(h.token, want, planes, pitches);
  if (rc == kTokenGone) {
    // evicted: its producer makes the frame again; pinned while we copy it so that it cannot be evicted once more
    Header fresh;
    int rr = -100;
    if (h.producer && (rr = reg.regenerate(h.producer, h.frame, env, &fresh)) == 0) {
      if (!same_format(fresh.format, want)) {
        rc = -4;
      } else if (reg.pin(fresh.token) == 0) {
        rc = reg.download(fresh.token, want, planes, pitches);
        reg.unpin(fresh.token);
      } else {
        rc = kTokenGone;
      }
    }
    if (logging())
      std::fprintf(stderr, "CudaConvolution: token of frame %d (producer %llu) evicted, regenerate=%d, download=%d\n", n,
                   h.producer, rr, rc);
  }
  if (rc != 0)
    env->ThrowError("CudaConvolution: frame %d: the GPU token frame could not be read back (code %d%s). Raise "
                    "MVCUDA_CACHE_MB or put CudaDownload() in front of CudaConvolution.",
                    n, rc, rc == kTokenGone ? ", evicted from the registry and not made again" : "");
  return dst;  // NewVideoFrameP above already carried the frame properties over
}

}  // namespace cudaconvolution::token
