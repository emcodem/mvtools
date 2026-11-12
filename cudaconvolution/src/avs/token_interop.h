#pragma once
// Optional interop with the companion GPU plugins' device-frame registry ("token frames").
//
// A token frame is an ordinary AviSynth frame whose luma row 0 holds a small header (magic, token, producer, frame
// number, format); the picture lives in device memory, owned by the registry in mvtools_cuda.dll. Nothing links
// against that DLL: CudaConvolution uses the instance the GPU plugins already loaded into the process (the registry the
// tokens belong to), or loads it from its own folder when none is loaded yet. Without it CudaConvolution behaves as before,
// except that a token frame it cannot read is an error, never picture data. The structs mirror the library's public
// header (layout 1); the library is accepted only with a token API this file knows (see Registry).
#include <atomic>
#include <cstddef>
#include <mutex>
#include <string>

#include "avisynth/avisynth.h"

namespace cudaconvolution::token {

constexpr unsigned long long kMagic = 0x4B5441445543564DULL;  // "MVCUDATK" as a little-endian 64-bit integer
constexpr int kLayout = 1;
constexpr int kTokenGone = -6;  // the token is not (any longer) in the registry
// Library API versions whose token functions have the signatures below (unchanged from 13, when they were added, to
// 30). A library that exports mvcuda_token_abi() is accepted when that returns kTokenAbi, whatever its API version.
constexpr int kMinApiVersion = 13, kMaxApiVersion = 30;
constexpr int kTokenAbi = 1;

struct FrameFormat {
  int width, height;           // luma size in samples
  int planes;                  // 1 (Y only) or 3 (Y, U, V)
  int bytes_per_sample;        // 1 or 2
  int bits_per_sample;         // 8..16
  int x_ratio_uv, y_ratio_uv;  // chroma subsampling (1 or 2)
};

struct Header {
  unsigned long long magic;
  int layout;
  int device;
  unsigned long long token;
  unsigned long long producer;  // regeneration source (0: none)
  int frame;
  int reserved;
  FrameFormat format;
};
static_assert(sizeof(FrameFormat) == 28, "token interop: FrameFormat layout");
static_assert(sizeof(Header) == 72, "token interop: Header layout");

// producer: fn(user, frame, env, out) makes the frame again and returns 0
using RegenFn = int (*)(void* user, int frame, void* env, Header* out);
// zero-copy work on device frames (mvcuda_frame_run, token ABI 1): runs on the library's submitter thread with a
// stream ordered after the inputs; in_planes / in_pitch: [i * 3 + plane]; in_fmt: the frames' real formats
using RunFn = int (*)(void* user, void* stream, int n_in, const void* const* in_planes, const int* in_pitch,
                      const FrameFormat* in_fmt, void* const* out_planes, const int* out_pitch);
constexpr int kRunMaxInputs = 8;

// The registry library, looked up on first use (thread-safe). available() is false without a usable DLL; why() says why.
class Registry {
 public:
  // While no library is found, every call looks again (a GPU plugin may load it later, e.g. after CudaConvolution_Version()).
  static Registry& get();
  bool available() const { return ok_.load(std::memory_order_acquire); }
  std::string why() const;  // only meaningful while !available()
  int api_version() const { return api_; }

  // Two different mvtools_cuda.dll files in one process are two registries whose token and producer numbers collide:
  // reading a token in the wrong one returns an unrelated frame. Throws (env->ThrowError) when more than one is loaded.
  void require_single_instance(IScriptEnvironment* env) const;

  // copies a token's picture into dst, sized for `expect` (checked by the library when it can: API 31+)
  int download(unsigned long long token, const FrameFormat& expect, void* const dst[3], const int pitch[3]) const;
  int upload(const FrameFormat& f, const void* const src[3], const int pitch[3], unsigned long long producer, int frame,
             Header& out) const;
  unsigned long long producer_register(RegenFn fn, void* user) const;
  void producer_unregister(unsigned long long producer) const;
  int regenerate(unsigned long long producer, int frame, void* env, Header* out) const;
  int pin(unsigned long long token) const;
  void unpin(unsigned long long token) const;
  // zero-copy (only libraries with token ABI 1 and mvcuda_frame_run): out_fmt nullptr = no output frame
  bool can_run() const { return run_ != nullptr; }
  int run(const unsigned long long* in_tokens, int n_in, const FrameFormat* out_fmt, unsigned long long producer,
          int frame, RunFn fn, void* user, Header* out) const;

 private:
  Registry() = default;
  void init();
  mutable std::mutex m_;  // guards init() and why_
  std::atomic<bool> ok_{false};
  bool not_found_ = true;  // no library found (yet): look again on the next get()
  std::string why_ = "the GPU library was not loaded";
  int api_ = 0;
  int (*download_)(unsigned long long, void* const[3], const int[3]) = nullptr;
  int (*upload_)(const FrameFormat*, const void* const[3], const int[3], unsigned long long, int, Header*) = nullptr;
  unsigned long long (*producer_register_)(RegenFn, void*) = nullptr;
  void (*producer_unregister_)(unsigned long long) = nullptr;
  int (*regenerate_)(unsigned long long, int, void*, Header*) = nullptr;
  int (*pin_)(unsigned long long) = nullptr;
  void (*unpin_)(unsigned long long) = nullptr;
  int (*download_checked_)(unsigned long long, const FrameFormat*, void* const[3], const int[3]) = nullptr;
  int (*run_)(const unsigned long long*, int, const FrameFormat*, unsigned long long, int, RunFn, void*,
              Header*) = nullptr;
};

// the device-frame format of a clip's frames; false when the registry cannot hold it (RGB, alpha, 4:1:1, ...)
bool make_format(const VideoInfo& vi, FrameFormat& f);
bool same_format(const FrameFormat& a, const FrameFormat& b);

enum class Kind {
  Picture,        // an ordinary frame
  Token,          // a token frame of the layout this file knows (h filled)
  UnknownLayout,  // a token frame of another layout (newer library): must not be read as picture, cannot be read
};
Kind classify(const PVideoFrame& f, const VideoInfo& vi, Header& h);
// writes the header into dst and blackens every 8th luma row, so a filter that does not understand tokens shows stripes
void write_token(PVideoFrame& dst, const Header& h);

// A normal frame with the pixels of token frame `f` (downloaded; made again by its producer when evicted); ordinary
// frames are returned as they are. Throws via env->ThrowError if a token frame cannot be read - detection does not
// depend on the library being available.
PVideoFrame materialize(const PVideoFrame& f, const VideoInfo& vi, int n, IScriptEnvironment* env);

}  // namespace cudaconvolution::token
