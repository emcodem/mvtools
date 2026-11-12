// AviSynth+ plugin entry and filter class.
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "avisynth/avisynth.h"
#include "avs/token_interop.h"
#include "core/engine.h"
#include "core/format.h"
#include "core/selftest.h"
#include "core/preset.h"
#include "core/presets.h"
#ifdef CUDACONVOLUTION_HAVE_CUDA
#include "cuda/device.h"
#include "cuda/engine_cuda.h"
#endif

const AVS_Linkage* AVS_linkage = nullptr;

namespace {

using cudaconvolution::Preset;

constexpr const char* kDefaultPreset = "balanced";

// preset: nullptr -> built-in default; a built-in name; otherwise a path to a preset JSON file
nlohmann::json load_preset_arg(const char* preset) {
  if (!preset) return cudaconvolution::load_preset(kDefaultPreset);
  const std::string arg = preset;
  for (const std::string& name : cudaconvolution::preset_names())
    if (name == arg) return cudaconvolution::load_preset(name);
  std::ifstream f(arg);
  if (!f) {
    std::string known;
    for (const std::string& n : cudaconvolution::preset_names()) known += (known.empty() ? "" : ", ") + n;
    throw std::runtime_error("preset '" + arg + "' is neither a readable preset file nor a built-in preset (" + known + ")");
  }
  try {
    nlohmann::json j = nlohmann::json::parse(f);
    if (!j.contains("name")) j["name"] = arg;
    return j;
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error("preset file '" + arg + "': " + e.what());
  }
}

// AviSynth name of the clip's pixel type, for messages
std::string format_name(const VideoInfo& vi) {
  if (vi.IsYUY2()) return "YUY2";
  if (vi.IsRGB24()) return "RGB24";
  if (vi.IsRGB32()) return "RGB32";
  if (vi.IsRGB48()) return "RGB48";
  if (vi.IsRGB64()) return "RGB64";
  const int bits = vi.BitsPerComponent();
  const std::string depth = bits == 32 ? "S" : std::to_string(bits);
  if (vi.IsPlanarRGBA()) return "RGBAP" + depth;
  if (vi.IsPlanarRGB()) return "RGBP" + depth;
  if (vi.IsY()) return "Y" + depth;
  const char* sub = vi.Is420() ? "420" : vi.Is422() ? "422" : vi.Is444() ? "444" : vi.IsYV411() ? "411" : "";
  return std::string(vi.IsYUVA() ? "YUVA" : "YUV") + sub + "P" + depth;
}

// Supported input formats; throws a message naming the format and how to convert otherwise.
cudaconvolution::SampleFormat sample_format(const VideoInfo& vi) {
  const int bits = vi.BitsPerComponent();
  cudaconvolution::SampleFormat f;
  f.bits = bits;
  const bool depth_ok = bits == 8 || bits == 10 || bits == 12 || bits == 16;
  if (vi.IsPlanarRGB() && depth_ok) f.layout = cudaconvolution::Layout::RGB;
  else if (vi.IsYUV() && vi.IsPlanar() && !vi.IsYUVA() && vi.Is422() && depth_ok) f.layout = cudaconvolution::Layout::YUV422;
  else if (vi.IsYUV() && vi.IsPlanar() && !vi.IsYUVA() && vi.Is444() && depth_ok) f.layout = cudaconvolution::Layout::YUV444;
  else
    throw std::runtime_error("input format " + format_name(vi) +
                             " is not supported. Supported: YUV422P8/10/12/16, YUV444P8/10/12/16, RGBP8/10/12/16 "
                             "(planar, no alpha). Convert upstream, e.g. ConvertBits(10), ConvertToYUV422(), "
                             "ConvertToPlanarRGB(), RemoveAlphaPlane(), or drop convert=false.");
  return f;
}

bool supported(const VideoInfo& vi) {
  try {
    sample_format(vi);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

// Pass-through in front of the automatic conversion: GPU token frames become pictures (a conversion filter would
// read their header as picture data); ordinary frames pass unchanged. Checks every frame (clips may mix both).
class TokenRead : public GenericVideoFilter {
 public:
  explicit TokenRead(PClip c) : GenericVideoFilter(c) {}
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override {
    return cudaconvolution::token::materialize(child->GetFrame(n, env), vi, n, env);
  }
  int __stdcall SetCacheHints(int hints, int) override { return hints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0; }
};

// convert=true: brings an unsupported input to the nearest supported format with the AviSynth+ built-ins, keeping
// YUV as YUV and RGB as RGB (no matrix is guessed): packed RGB -> planar, alpha removed, other YUV layouts
// (4:2:0, 4:1:1, Y, YUY2) -> 4:2:2, other bit depths (14, 32 float) -> 16.
PClip to_supported(PClip c, IScriptEnvironment* env) {
  auto call = [&](const char* fn, int bits = 0) {
    try {
      const AVSValue args[2] = {c, bits};
      c = env->Invoke(fn, AVSValue(args, bits ? 2 : 1)).AsClip();
    } catch (const AvisynthError& e) {
      env->ThrowError("CudaConvolution: automatic %s() failed: %s", fn, e.msg);
    } catch (const IScriptEnvironment::NotFound&) {
      env->ThrowError("CudaConvolution: automatic conversion needs %s(), which this AviSynth does not have.", fn);
    }
  };
  if (supported(c->GetVideoInfo())) return c;
  c = new TokenRead(c);
  if (c->GetVideoInfo().IsRGB() && !c->GetVideoInfo().IsPlanar()) call("ConvertToPlanarRGB");
  if (c->GetVideoInfo().IsPlanarRGBA() || c->GetVideoInfo().IsYUVA()) call("RemoveAlphaPlane");
  const VideoInfo& vi = c->GetVideoInfo();
  if (vi.IsYUV() && (!vi.IsPlanar() || !(vi.Is422() || vi.Is444()))) call("ConvertToYUV422");
  const int bits = c->GetVideoInfo().BitsPerComponent();
  if (bits != 8 && bits != 10 && bits != 12 && bits != 16) call("ConvertBits", 16);
  return c;
}

// 8-bit plane -> packed 10-bit uint16_t buffer (the engines' sample type)
cudaconvolution::ConstPlane widen(const PVideoFrame& f, int pl, std::vector<uint16_t>& buf) {
  const int w = f->GetRowSize(pl), h = f->GetHeight(pl), pitch = f->GetPitch(pl);
  buf.resize(static_cast<size_t>(w) * h);
  const uint8_t* s = f->GetReadPtr(pl);
  uint16_t* d = buf.data();
  for (int y = 0; y < h; ++y, s += pitch, d += w)
    for (int x = 0; x < w; ++x) d[x] = cudaconvolution::from_8bit(s[x]);
  return {buf.data(), w, w, h};
}

class CudaConvolution : public GenericVideoFilter {
 public:
  // device < 0: CPU engine; output size = input size / (p.factor_h, p.factor_v);
  // clip_min / clip_max: output limits in the input's bit depth, < 0 = not given (storage range)
  CudaConvolution(PClip clip, Preset p, int clip_min, int clip_max, int device, int threads, bool token_out,
           IScriptEnvironment* env)
      : GenericVideoFilter(clip), token_out_(token_out), threads_(threads) {
    try {
      format_ = sample_format(vi);
      p_ = cudaconvolution::adapt_preset(std::move(p), {format_.layout, eight_bit() ? 10 : format_.bits});
    } catch (const std::exception& e) {
      env->ThrowError("CudaConvolution: %s", e.what());
    }
    const int max_code = (1 << format_.bits) - 1;
    for (auto [name, v] : {std::pair{"clip_min", clip_min}, {"clip_max", clip_max}})
      if (v > max_code)
        env->ThrowError("CudaConvolution: %s must be within 0..%d for %d-bit input, got %d.", name, max_code, format_.bits, v);
    // 8 bit: the limits apply after rounding to 8 bit; the 10-bit arithmetic keeps its storage range
    int& lo = eight_bit() ? lo8_ : p_.output_lo;
    int& hi = eight_bit() ? hi8_ : p_.output_hi;
    if (clip_min >= 0) lo = clip_min;
    if (clip_max >= 0) hi = clip_max;
    if (lo > hi) env->ThrowError("CudaConvolution: clip_min (%d) is above clip_max (%d).", lo, hi);
    if (vi.IsFieldBased())
      env->ThrowError("CudaConvolution: input is field-based (after SeparateFields?); progressive frames are needed.");
    if (p_.interlaced && vi.num_frames < 2)
      env->ThrowError("CudaConvolution: interlaced output needs at least 2 input frames, got %d.", vi.num_frames);
    const int out_w = vi.width / p_.factor_h, out_h = vi.height / p_.factor_v;
    if (format_.layout == cudaconvolution::Layout::YUV422 && ((vi.width / 2) % p_.factor_h || out_w % 2))
      env->ThrowError("CudaConvolution: 4:2:2 needs an even output width whose chroma is an integer part of the input chroma: "
                      "%dx%d -> %dx%d (chroma %d -> %d) does not fit. Choose another width or crop the input.",
                      vi.width, vi.height, out_w, out_h, vi.width / 2, out_w / 2);
    if (p_.interlaced && out_h % 2)
      env->ThrowError("CudaConvolution: interlaced output needs an even height, got %d. Choose another height or crop the input.",
                      out_h);

    vi.width = out_w;
    vi.height = out_h;
    vi.Clear(VideoInfo::IT_BFF | VideoInfo::IT_TFF);
    if (p_.interlaced) {
      vi.num_frames /= 2;  // a trailing single progressive frame is dropped
      vi.MulDivFPS(1, 2);
      vi.Set(p_.tff ? VideoInfo::IT_TFF : VideoInfo::IT_BFF);
    }
#ifdef CUDACONVOLUTION_HAVE_CUDA
    if (device >= 0) {
      try {
        gpu_ = std::make_unique<cudaconvolution::CudaEngine>(device, p_);
      } catch (const std::exception& e) {
        env->ThrowError("CudaConvolution: %s", e.what());
      }
    }
#else
    (void)device;
#endif
    if (token_out_) {
      namespace tk = cudaconvolution::token;
      const tk::Registry& reg = tk::Registry::get();
      if (device < 0) env->ThrowError("CudaConvolution: token_out=true needs the GPU engine (not cpu=true).");
      if (!reg.available())
        env->ThrowError("CudaConvolution: token_out=true needs the GPU plugins' mvtools_cuda.dll (load CudaConvolution.dll from their "
                        "folder): %s.",
                        reg.why().c_str());
      if (!tk::make_format(vi, out_format_))
        env->ThrowError("CudaConvolution: token_out=true: the output format %s cannot be a GPU token frame (planar Y/YUV, 8..16 "
                        "bit, 4:2:0/4:2:2/4:4:4 only).",
                        format_name(vi).c_str());
      producer_ = reg.producer_register(&CudaConvolution::regenerate, this);
    }
  }

  ~CudaConvolution() override {
    if (producer_) cudaconvolution::token::Registry::get().producer_unregister(producer_);
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override {
    n = std::clamp(n, 0, vi.num_frames - 1);
    if (token_out_) {
      cudaconvolution::token::Header h;
      PVideoFrame dst = make_token(n, env, h);
      cudaconvolution::token::write_token(dst, h);
      return dst;
    }
    PVideoFrame a, b, dst;
    fetch(n, env, a, b);
    if (zero_copy(n, a, b, env, nullptr, dst)) return dst;
    return compute(n, a, b, env);
  }

  int __stdcall SetCacheHints(int hints, int) override { return hints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0; }

 private:
  // the registry asks for a frame it evicted: make it again
  static int regenerate(void* user, int n, void* env, cudaconvolution::token::Header* out) {
    CudaConvolution* self = static_cast<CudaConvolution*>(user);
    try {
      self->make_token(std::clamp(n, 0, self->vi.num_frames - 1), static_cast<IScriptEnvironment*>(env), *out);
      return 0;
    } catch (...) {
      return -2;
    }
  }

  // token_out: frame n as a device frame of this producer (h: its header). Zero-copy when the inputs are device frames
  // the engine can read in place; else computed on the host path and uploaded. The returned frame carries the frame
  // properties (and, on the host path, the pixels); the caller writes the header into it.
  PVideoFrame make_token(int n, IScriptEnvironment* env, cudaconvolution::token::Header& h) {
    cudaconvolution::token::Registry::get().require_single_instance(env);
    PVideoFrame a, b, dst;
    fetch(n, env, a, b);
    if (zero_copy(n, a, b, env, &h, dst)) return dst;
    dst = compute(n, a, b, env);
    if (upload(dst, n, h) != 0)
      env->ThrowError("CudaConvolution: frame %d: the result could not be uploaded to the GPU registry (token_out=true).", n);
    return dst;
  }

  // the input frames of output frame n (as the child returns them: maybe token frames)
  void fetch(int n, IScriptEnvironment* env, PVideoFrame& a, PVideoFrame& b) {
    const int na = p_.interlaced ? 2 * n : n;
    a = child->GetFrame(na, env);
    b = p_.interlaced ? child->GetFrame(na + 1, env) : a;
  }

  void set_props(PVideoFrame& dst, IScriptEnvironment* env) const {
    AVSMap* props = env->getFramePropsRW(dst);
    env->propSetInt(props, "_FieldBased", p_.interlaced ? (p_.tff ? 2 : 1) : 0, PROPAPPENDMODE_REPLACE);
    env->propDeleteKey(props, "_DurationNum");  // the source frame duration no longer applies
    env->propDeleteKey(props, "_DurationDen");
  }

#ifdef CUDACONVOLUTION_HAVE_CUDA
  struct RunCtx {
    cudaconvolution::CudaEngine* engine;
    cudaconvolution::CudaEngine::DeviceJob* job;
    cudaconvolution::token::FrameFormat in_fmt, out_fmt;
    bool token_out;
  };

  // runs on the registry's submitter thread (mvcuda_frame_run): check the real formats, then only queue GPU work
  static int run_on_device(void* user, void* stream, int n_in, const void* const* in_planes, const int* in_pitch,
                           const cudaconvolution::token::FrameFormat* in_fmt, void* const* out_planes, const int* out_pitch) {
    const RunCtx& c = *static_cast<const RunCtx*>(user);
    auto plane_bytes = [](const cudaconvolution::token::FrameFormat& f, int p) {
      return (p == 0 ? f.width : f.width / f.x_ratio_uv) * f.bytes_per_sample;
    };
    for (int i = 0; i < n_in; ++i) {
      if (!cudaconvolution::token::same_format(in_fmt[i], c.in_fmt)) return -4;
      for (int p = 0; p < 3; ++p)
        if (!in_planes[i * 3 + p] || in_pitch[i * 3 + p] != plane_bytes(c.in_fmt, p)) return -4;
    }
    cudaconvolution::CudaEngine::DevicePlanes first{}, second{};
    const int s = n_in > 1 ? 3 : 0;
    for (int p = 0; p < 3; ++p) {
      first.plane[p] = static_cast<const uint16_t*>(in_planes[p]);
      second.plane[p] = static_cast<const uint16_t*>(in_planes[s + p]);
    }
    uint16_t* out[3] = {};
    if (c.token_out)
      for (int p = 0; p < 3; ++p) {
        if (!out_planes[p] || out_pitch[p] != plane_bytes(c.out_fmt, p)) return -4;
        out[p] = static_cast<uint16_t*>(out_planes[p]);
      }
    return c.engine->launch_device(*c.job, first, second, c.token_out ? out : nullptr, stream);
  }
#endif

  // Zero-copy: when every input frame is a device frame on the engine's GPU (10..16 bit), the engine reads it in place
  // through the registry (mvcuda_frame_run). token: nullptr = output as pixels in dst; else the output stays a device
  // frame of this producer and *token receives its header. false: not possible here (the caller takes the copy path,
  // which also makes evicted inputs again and reports unreadable ones).
  bool zero_copy(int n, const PVideoFrame& a, const PVideoFrame& b, IScriptEnvironment* env, cudaconvolution::token::Header* token,
                 PVideoFrame& dst) {
#ifdef CUDACONVOLUTION_HAVE_CUDA
    namespace tk = cudaconvolution::token;
    if (!gpu_ || eight_bit() || format_.layout == cudaconvolution::Layout::RGB) return false;
    const VideoInfo& cvi = child->GetVideoInfo();
    tk::Header ha, hb;
    if (tk::classify(a, cvi, ha) != tk::Kind::Token) return false;
    hb = ha;
    if (p_.interlaced && tk::classify(b, cvi, hb) != tk::Kind::Token) return false;
    const tk::Registry& reg = tk::Registry::get();
    tk::FrameFormat want;
    if (!reg.available() || !reg.can_run() || !tk::make_format(cvi, want) || !tk::same_format(ha.format, want) ||
        !tk::same_format(hb.format, want) || ha.device != gpu_->device() || hb.device != gpu_->device())
      return false;
    reg.require_single_instance(env);

    dst = env->NewVideoFrameP(vi, const_cast<PVideoFrame*>(&a));
    static constexpr int kYuv[3] = {PLANAR_Y, PLANAR_U, PLANAR_V};
    cudaconvolution::Frame shape;
    for (int i = 0; i < 3; ++i) {
      const int pl = kYuv[i];
      shape.plane[i] = {token ? nullptr : reinterpret_cast<uint16_t*>(dst->GetWritePtr(pl)), dst->GetPitch(pl) / 2,
                        dst->GetRowSize(pl) / 2, dst->GetHeight(pl)};
    }
    cudaconvolution::CudaEngine::DeviceJobPtr job;
    try {
      job = gpu_->prepare_device(shape, token == nullptr);
    } catch (const std::exception& e) {
      env->ThrowError("CudaConvolution: frame %d: %s", n, e.what());
    }
    RunCtx ctx{gpu_.get(), job.get(), want, out_format_, token != nullptr};
    const unsigned long long tokens[2] = {ha.token, hb.token};
    const int rc = reg.run(tokens, p_.interlaced ? 2 : 1, token ? &out_format_ : nullptr, producer_, n,
                           &CudaConvolution::run_on_device, &ctx, token);
    if (rc == tk::kTokenGone) return false;  // an input was evicted: the copy path has it made again
    if (rc != 0) env->ThrowError("CudaConvolution: frame %d: the zero-copy GPU call failed (code %d).", n, rc);
    if (!token) gpu_->finish_device(*job, shape);
    set_props(dst, env);
    return true;
#else
    (void)n, (void)a, (void)b, (void)env, (void)token, (void)dst;
    return false;
#endif
  }

  // uploads a normal result frame as frame n of this filter; 0 on success
  int upload(const PVideoFrame& f, int n, cudaconvolution::token::Header& h) const {
    static constexpr int kPl[3] = {PLANAR_Y, PLANAR_U, PLANAR_V};
    const void* planes[3] = {};
    int pitches[3] = {};
    for (int p = 0; p < out_format_.planes; ++p) {
      planes[p] = f->GetReadPtr(kPl[p]);
      pitches[p] = f->GetPitch(kPl[p]);
    }
    return cudaconvolution::token::Registry::get().upload(out_format_, planes, pitches, producer_, n, h);
  }

  // output frame n as normal pixels from its input frames; token frames are read back first (see token_interop.h)
  PVideoFrame compute(int n, const PVideoFrame& a_in, const PVideoFrame& b_in, IScriptEnvironment* env) {
    const VideoInfo& cvi = child->GetVideoInfo();
    const int na = p_.interlaced ? 2 * n : n;
    PVideoFrame a = cudaconvolution::token::materialize(a_in, cvi, na, env);
    PVideoFrame b = a;
    if (p_.interlaced) b = cudaconvolution::token::materialize(b_in, cvi, na + 1, env);
    PVideoFrame dst = env->NewVideoFrameP(vi, &a);

    static constexpr int kYuv[3] = {PLANAR_Y, PLANAR_U, PLANAR_V}, kRgb[3] = {PLANAR_G, PLANAR_B, PLANAR_R};
    const int* kPlanes = format_.layout == cudaconvolution::Layout::RGB ? kRgb : kYuv;
    const bool narrow = eight_bit();  // the engines work on uint16_t samples: 8 bit goes through packed 10-bit buffers
    thread_local std::vector<uint16_t> wide[3][3];  // [a, b, dst][plane]
    cudaconvolution::ConstFrame fa, fb;
    cudaconvolution::Frame fd;
    for (int i = 0; i < 3; ++i) {
      const int pl = kPlanes[i];
      if (narrow) {
        fa.plane[i] = widen(a, pl, wide[0][i]);
        fb.plane[i] = p_.interlaced ? widen(b, pl, wide[1][i]) : fa.plane[i];
        const int w = dst->GetRowSize(pl), h = dst->GetHeight(pl);
        wide[2][i].resize(static_cast<size_t>(w) * h);
        fd.plane[i] = {wide[2][i].data(), w, w, h};
        continue;
      }
      fa.plane[i] = {reinterpret_cast<const uint16_t*>(a->GetReadPtr(pl)), a->GetPitch(pl) / 2, a->GetRowSize(pl) / 2,
                     a->GetHeight(pl)};
      fb.plane[i] = {reinterpret_cast<const uint16_t*>(b->GetReadPtr(pl)), b->GetPitch(pl) / 2, b->GetRowSize(pl) / 2,
                     b->GetHeight(pl)};
      fd.plane[i] = {reinterpret_cast<uint16_t*>(dst->GetWritePtr(pl)), dst->GetPitch(pl) / 2,
                     dst->GetRowSize(pl) / 2, dst->GetHeight(pl)};
    }
#ifdef CUDACONVOLUTION_HAVE_CUDA
    if (gpu_) {
      try {
        gpu_->process_frame(fa, fb, fd);
      } catch (const std::exception& e) {
        env->ThrowError("CudaConvolution: frame %d: %s", n, e.what());
      }
    } else
#endif
    {
      thread_local std::vector<float> scratch;
      cudaconvolution::process_frame(fa, fb, fd, p_, scratch, cudaconvolution::CpuPath::Auto, threads_);
    }
    if (narrow)
      for (int i = 0; i < 3; ++i) {
        const int pl = kPlanes[i], pitch = dst->GetPitch(pl);
        const uint8_t lo = static_cast<uint8_t>(lo8_), hi = static_cast<uint8_t>(hi8_);
        uint8_t* d = dst->GetWritePtr(pl);
        const cudaconvolution::Plane& s = fd.plane[i];
        for (int y = 0; y < s.height; ++y, d += pitch) {
          const uint16_t* row = s.data + y * s.stride;
          for (int x = 0; x < s.width; ++x) d[x] = std::clamp(cudaconvolution::to_8bit(row[x]), lo, hi);
        }
      }
    set_props(dst, env);
    return dst;
  }

  bool eight_bit() const { return format_.bits == 8; }

  bool token_out_;                          // result frames are GPU token frames (needs mvtools_cuda.dll)
  cudaconvolution::token::FrameFormat out_format_{};
  unsigned long long producer_ = 0;         // registry producer id (token_out)
  int threads_;  // CPU engine worker threads per frame
  cudaconvolution::SampleFormat format_;
  Preset p_;
  int lo8_ = 0, hi8_ = 255;  // 8-bit input: output limits after rounding to 8 bit
#ifdef CUDACONVOLUTION_HAVE_CUDA
  std::unique_ptr<cudaconvolution::CudaEngine> gpu_;
#endif
};

// Integer scale factor for one axis (input n_in -> output n_out); throws a message with valid alternatives.
int axis_factor(int n_in, int n_out, const char* what) {
  const std::string axis = std::string(what) + " " + std::to_string(n_in) + " -> " + std::to_string(n_out);
  if (n_out <= 0) throw std::runtime_error(std::string(what) + " must be > 0");
  if (n_out > n_in)
    throw std::runtime_error(axis + ": upscaling is not supported; scale up before CudaConvolution (e.g. Spline36Resize)");
  if (n_in % n_out == 0) {
    if (n_in / n_out > cudaconvolution::kMaxFactor)
      throw std::runtime_error(axis + ": factor " + std::to_string(n_in / n_out) + " is above the maximum " +
                               std::to_string(cudaconvolution::kMaxFactor) + "; scale down before CudaConvolution");
    return n_in / n_out;
  }
  std::string outs, crops;  // valid output sizes for the neighbouring factors, or input crops keeping n_out
  for (int f = std::max(1, n_in / n_out); f <= std::min(cudaconvolution::kMaxFactor, n_in / n_out + 1); ++f) {
    if (n_in % f == 0)
      outs += (outs.empty() ? "" : " or ") + std::string(what) + "=" + std::to_string(n_in / f) + " (factor " + std::to_string(f) + ")";
    if (n_out * f <= n_in)
      crops += (crops.empty() ? "" : " or ") + std::to_string(n_out * f) + " (factor " + std::to_string(f) + ")";
  }
  char ratio[32];
  std::snprintf(ratio, sizeof(ratio), "%.3f", static_cast<double>(n_in) / n_out);
  std::string hint = outs.empty() ? "" : "; use " + outs;
  if (!crops.empty()) hint += std::string(outs.empty() ? "; " : ", or ") + "crop the input " + what + " to " + crops;
  throw std::runtime_error(axis + " is not an integer ratio (" + ratio + ")" + hint);
}

AVSValue __cdecl create_cudaconvolution(AVSValue args, void*, IScriptEnvironment* env) {
  // args: c, width, height, preset, field_order, clip_min, clip_max, device, cpu, threads, convert, token_out
  PClip clip = args[0].AsClip();
  if (args[10].AsBool(true)) clip = to_supported(clip, env);
  const int width = args[1].AsInt(0), height = args[2].AsInt(0);  // 0: not given
  const char* preset = args[3].AsString(nullptr);
  const char* field_order = args[4].AsString(nullptr);
  const int clip_min = args[5].AsInt(-1), clip_max = args[6].AsInt(-1);  // -1: no clipping
  const int device = args[7].AsInt(0);
  const bool cpu = args[8].AsBool(false);
  // CPU engine threads per frame: default all hardware threads (with Prefetch(N) at the end, threads=1 avoids
  // oversubscription); no effect on the GPU engine
  int threads = args[9].AsInt(0);
  if (threads < 0) env->ThrowError("CudaConvolution: threads must be >= 0 (0 = all hardware threads), got %d.", threads);
  if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
  if ((args[5].Defined() && clip_min < 0) || (args[6].Defined() && clip_max < 0))
    env->ThrowError("CudaConvolution: clip_min / clip_max must not be negative.");

  // output size: default = input size (pure P -> i); if only one is given, the other axis uses the same factor
  const VideoInfo& in = clip->GetVideoInfo();
  Preset p;
  try {
    int fh = 1, fv = 1;
    if (args[1].Defined()) fh = axis_factor(in.width, width, "width");
    if (args[2].Defined()) fv = axis_factor(in.height, height, "height");
    if (args[1].Defined() && !args[2].Defined()) {
      if (in.height % fh) throw std::runtime_error("height " + std::to_string(in.height) + " is not divisible by the width "
                                                   "factor " + std::to_string(fh) + "; give height= explicitly");
      fv = fh;
    } else if (args[2].Defined() && !args[1].Defined()) {
      if (in.width % fv) throw std::runtime_error("width " + std::to_string(in.width) + " is not divisible by the height "
                                                  "factor " + std::to_string(fv) + "; give width= explicitly");
      fh = fv;
    }
    p = cudaconvolution::parse_preset(load_preset_arg(preset), fh, fv);
  } catch (const std::exception& e) {
    env->ThrowError("CudaConvolution: %s", e.what());
  }
  if (field_order) {
    const std::string fo = field_order;
    if (fo != "tff" && fo != "bff") env->ThrowError("CudaConvolution: field_order must be \"tff\" or \"bff\", got \"%s\"", field_order);
    p.tff = fo == "tff";
  }

  if (!cpu) {
#ifdef CUDACONVOLUTION_HAVE_CUDA
    const cudaconvolution::CudaStatus& st = cudaconvolution::cuda_status(device);
    if (!st.ok) env->ThrowError("CudaConvolution: GPU engine not available: %s", st.error.c_str());
#else
    (void)device;
    env->ThrowError("CudaConvolution: this build has no GPU engine (built without CUDA). Use CudaConvolution(..., cpu=true).");
#endif
  }
  return new CudaConvolution(clip, std::move(p), clip_min, clip_max, cpu ? -1 : device, threads, args[11].AsBool(false), env);
}

AVSValue __cdecl cudaconvolution_version(AVSValue, void*, IScriptEnvironment* env) {
  std::string v = "CudaConvolution " CUDACONVOLUTION_VERSION;
#ifdef CUDACONVOLUTION_HAVE_CUDA
  const cudaconvolution::CudaStatus& st = cudaconvolution::cuda_status(0);
  v += st.ok ? " - GPU: " + st.info : " - GPU not available: " + st.error;
#else
  v += " - CPU only build";
#endif
  const cudaconvolution::token::Registry& reg = cudaconvolution::token::Registry::get();
  v += reg.available() ? " - GPU token frames: available (mvtools_cuda API " + std::to_string(reg.api_version()) + ")"
                       : " - GPU token frames: not available (" + reg.why() + ")";
  return AVSValue(env->SaveString(v.c_str()));
}

// Runs the embedded test vectors through every engine; one line per engine, e.g. "GPU: 18/18 OK (<device>)".
AVSValue __cdecl cudaconvolution_selftest(AVSValue args, void*, IScriptEnvironment* env) {
  const int device = args[0].AsInt(0);
  std::string report = "CudaConvolution " CUDACONVOLUTION_VERSION " self-test";
  bool all_ok = true;
  auto run = [&](const char* label, const cudaconvolution::EngineFn& fn) {
    int ok = 0, total = 0;
    std::string failed;
    for (const cudaconvolution::SelftestCase& c : cudaconvolution::selftest_cases()) {
      ++total;
      std::string md5;
      try {
        md5 = cudaconvolution::selftest_md5(c, fn);
      } catch (const std::exception& e) {
        md5 = std::string("error: ") + e.what();
      }
      if (md5 == c.md5) ++ok;
      else failed += (failed.empty() ? "" : ", ") + c.label();
    }
    all_ok &= ok == total;
    report += std::string("\n") + label + ": " + std::to_string(ok) + "/" + std::to_string(total) +
              (ok == total ? " OK" : " FAILED: " + failed);
  };
  auto cpu = [](cudaconvolution::CpuPath path) {
    return [path](const cudaconvolution::ConstFrame& a, const cudaconvolution::ConstFrame& b, const cudaconvolution::Frame& o,
                  const Preset& p) {
      std::vector<float> scratch;
      cudaconvolution::process_frame(a, b, o, p, scratch, path);
    };
  };
  run("CPU scalar", cpu(cudaconvolution::CpuPath::Scalar));
  if (cudaconvolution::cpu_has_avx2_fma()) run("CPU AVX2", cpu(cudaconvolution::CpuPath::Avx2));
  else report += "\nCPU AVX2: not available on this CPU (scalar path is used)";
#ifdef CUDACONVOLUTION_HAVE_CUDA
  const cudaconvolution::CudaStatus& st = cudaconvolution::cuda_status(device);
  if (st.ok) {
    run(("GPU " + st.info).c_str(), [device](const cudaconvolution::ConstFrame& a, const cudaconvolution::ConstFrame& b,
                                             const cudaconvolution::Frame& o, const Preset& p) {
      cudaconvolution::CudaEngine(device, p).process_frame(a, b, o);
    });
  } else {
    all_ok = false;
    report += "\nGPU: not available: " + st.error;
  }
#else
  (void)device;
  report += "\nGPU: CPU only build";
#endif
  report += all_ok ? "\nRESULT: PASS" : "\nRESULT: FAIL";
  return AVSValue(env->SaveString(report.c_str()));
}

AVSValue __cdecl too_old_avisynth(AVSValue, void*, IScriptEnvironment* env) {
  env->ThrowError("CudaConvolution: this AviSynth is too old - AviSynth+ 3.7 or newer (plugin interface version 8) is required. "
                  "Install the current AviSynth+ from https://github.com/AviSynth/AviSynthPlus/releases");
  return AVSValue();
}

}  // namespace

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env,
                                                                            const AVS_Linkage* const vectors) {
  AVS_linkage = vectors;
  try {
    env->CheckVersion(8);  // frame properties, AviSynth+ 3.7+
  } catch (const AvisynthError&) {
    // still register the names, so a script gets this message instead of "there is no function named CudaConvolution"
    env->AddFunction("CudaConvolution", ".*", too_old_avisynth, nullptr);
    env->AddFunction("CudaConvolution_Version", ".*", too_old_avisynth, nullptr);
    env->AddFunction("CudaConvolution_SelfTest", ".*", too_old_avisynth, nullptr);
    return "CudaConvolution: needs AviSynth+ 3.7 or newer";
  }
  env->AddFunction("CudaConvolution", "c[width]i[height]i[preset]s[field_order]s[clip_min]i[clip_max]i[device]i[cpu]b[threads]i[convert]b[token_out]b", create_cudaconvolution, nullptr);
  env->AddFunction("CudaConvolution_Version", "", cudaconvolution_version, nullptr);
  env->AddFunction("CudaConvolution_SelfTest", "[device]i", cudaconvolution_selftest, nullptr);
  return "CudaConvolution: progressive UHD/HD -> interlaced 1080i resampler";
}
