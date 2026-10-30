// cudafilters.dll: AviSynth+ filters that run on the GPU through mvtools_cuda.dll (GPU_PIPELINE_PLAN.md). This file:
// the library loader, the token-frame helpers every GPU filter of the plugin uses, CudaUpload and CudaDownload
// (milestone D), and the plugin entry point.
//
// Token frames: a GPU filter keeps its result in device memory and returns an ordinary frame of the declared format
// whose first bytes (row 0 of the Y plane) hold a MvCudaTokenHeader. Every 8th luma row of a token frame is black, so
// a filter that does not understand tokens shows obvious stripes instead of a plausible picture. CudaDownload turns
// tokens back into normal frames; a normal frame passes through both filters unchanged.

#include "CudaCommon.h"

extern "C" IMAGE_DOS_HEADER __ImageBase;

namespace cf
{
  CudaLib &CudaLib::get()
  {
    static CudaLib lib;
    return lib;
  }

  CudaLib::CudaLib()
  {
    if (std::getenv("MVTOOLS_GPU_DISABLE"))
      return;
    wchar_t path[MAX_PATH] = {};
    if (GetModuleFileNameW(reinterpret_cast<HMODULE>(&__ImageBase), path, MAX_PATH) == 0)
      return;
    std::wstring dir(path);
    const size_t slash = dir.find_last_of(L"\\/");
    if (slash == std::wstring::npos)
      return;
    dir.resize(slash + 1);
    HMODULE lib = LoadLibraryW((dir + L"mvtools_cuda.dll").c_str());
    if (!lib)
      return;
    auto version = reinterpret_cast<int (*)()>(GetProcAddress(lib, "mvcuda_api_version"));
    auto count = reinterpret_cast<int (*)()>(GetProcAddress(lib, "mvcuda_device_count"));
#define CF_LOAD(member, name) member = reinterpret_cast<decltype(member)>(GetProcAddress(lib, name))
    CF_LOAD(upload, "mvcuda_frame_upload");
    CF_LOAD(download, "mvcuda_frame_download");
    CF_LOAD(producer_register, "mvcuda_producer_register");
    CF_LOAD(producer_unregister, "mvcuda_producer_unregister");
    CF_LOAD(regenerate, "mvcuda_token_regenerate");
    CF_LOAD(stats, "mvcuda_frame_stats");
    CF_LOAD(resample_program_create, "mvcuda_resample_program_create");
    CF_LOAD(resample_program_free, "mvcuda_resample_program_free");
    CF_LOAD(resample_v, "mvcuda_resample_v");
    CF_LOAD(resample_2d, "mvcuda_resample_2d");
    CF_LOAD(temporal_soften, "mvcuda_temporal_soften");
    CF_LOAD(merge, "mvcuda_merge");
    CF_LOAD(lut2_create, "mvcuda_lut2_create");
    CF_LOAD(lut2_create_bits, "mvcuda_lut2_create_bits");
    CF_LOAD(lut2_free, "mvcuda_lut2_free");
    CF_LOAD(kobs, "mvcuda_kobs");
    CF_LOAD(removegrain, "mvcuda_removegrain");
    CF_LOAD(pin, "mvcuda_token_pin");
    CF_LOAD(unpin, "mvcuda_token_unpin");
    CF_LOAD(nnedi3_create, "mvcuda_nnedi3_create");
    CF_LOAD(nnedi3_free, "mvcuda_nnedi3_free");
    CF_LOAD(nnedi3, "mvcuda_nnedi3");
    CF_LOAD(mt, "mvcuda_mt");
    CF_LOAD(qtgmc_sharp, "mvcuda_qtgmc_sharp");
    CF_LOAD(qtgmc_backblend, "mvcuda_qtgmc_backblend");
    CF_LOAD(temporal_binomial, "mvcuda_temporal_binomial");
#undef CF_LOAD
    ok = version && count && upload && download && producer_register && producer_unregister && regenerate && stats
      && resample_program_create && resample_program_free && resample_v && resample_2d && temporal_soften && merge && lut2_create && lut2_create_bits && lut2_free && kobs && removegrain && pin && unpin && nnedi3_create && nnedi3_free && nnedi3 && mt && qtgmc_sharp && qtgmc_backblend && temporal_binomial && version() == MVCUDA_API_VERSION && count() > 0;
  }

  bool make_format(const VideoInfo &vi, MvCudaFrameFormat &f)
  {
    std::memset(&f, 0, sizeof(f));
    if (!vi.IsPlanar() || vi.IsRGB() || vi.IsYUVA() || vi.width <= 0 || vi.height <= 0)
      return false;
    f.width = vi.width;
    f.height = vi.height;
    f.planes = vi.IsY() ? 1 : 3;
    f.bytes_per_sample = vi.ComponentSize();
    f.bits_per_sample = vi.BitsPerComponent();
    if (f.bytes_per_sample != 1 && f.bytes_per_sample != 2)
      return false;
    f.x_ratio_uv = f.y_ratio_uv = 1;
    if (f.planes == 3)
    {
      f.x_ratio_uv = 1 << vi.GetPlaneWidthSubsampling(PLANAR_U);
      f.y_ratio_uv = 1 << vi.GetPlaneHeightSubsampling(PLANAR_U);
      if (f.x_ratio_uv > 2 || f.y_ratio_uv > 2)
        return false;
    }
    return (size_t)vi.width * f.bytes_per_sample >= sizeof(MvCudaTokenHeader);
  }

  bool read_token(const PVideoFrame &f, const VideoInfo &vi, MvCudaTokenHeader &h)
  {
    if (!vi.IsPlanar() || vi.IsRGB() || f->GetRowSize(PLANAR_Y) < (int)sizeof(h))
      return false;
    std::memcpy(&h, f->GetReadPtr(PLANAR_Y), sizeof(h));
    return h.magic == MVCUDA_TOKEN_MAGIC && h.layout == MVCUDA_TOKEN_LAYOUT;
  }

  void write_token(PVideoFrame &dst, const MvCudaTokenHeader &h)
  {
    BYTE *y = dst->GetWritePtr(PLANAR_Y);
    const int pitch = dst->GetPitch(PLANAR_Y), row = dst->GetRowSize(PLANAR_Y), height = dst->GetHeight(PLANAR_Y);
    std::memcpy(y, &h, sizeof(h));
    for (int r = 8; r < height; r += 8)
      std::memset(y + (size_t)r * pitch, 0, row);
  }

  bool same_format(const MvCudaFrameFormat &a, const MvCudaFrameFormat &b)
  {
    return a.width == b.width && a.height == b.height && a.planes == b.planes && a.bytes_per_sample == b.bytes_per_sample
      && a.bits_per_sample == b.bits_per_sample && (a.planes == 1 || (a.x_ratio_uv == b.x_ratio_uv && a.y_ratio_uv == b.y_ratio_uv));
  }

  void print_stats(const char *who)
  {
    CudaLib &lib = CudaLib::get();
    if (!lib.ok || !std::getenv("CUDAFILTERS_STATS"))
      return;
    MvCudaFrameStats s;
    lib.stats(&s);
    std::fprintf(stderr, "%s: device frames %llu (%.1f of %.1f MB), uploads %llu, downloads %llu, evictions %llu, regenerations %llu, lookups of evicted tokens %llu, redirected %llu\n",
      who, s.frames, s.bytes / 1048576.0, s.budget / 1048576.0, s.uploads, s.downloads, s.evictions, s.regenerations, s.lookups_gone, s.redirects);
  }

  int input_token(const PVideoFrame &src, const VideoInfo &vi, const MvCudaFrameFormat &f, unsigned long long &token)
  {
    MvCudaTokenHeader h;
    if (read_token(src, vi, h))
    {
      if (!same_format(h.format, f))
        return -4;
      token = h.token;
      return 0;
    }
    const void *planes[3] = {};
    int pitches[3] = {};
    for (int p = 0; p < f.planes; p++)
    {
      planes[p] = src->GetReadPtr(g_planes[p]);
      pitches[p] = src->GetPitch(g_planes[p]);
    }
    const int rc = CudaLib::get().upload(&f, planes, pitches, 0, 0, &h);
    if (rc == 0)
      token = h.token;
    return rc;
  }
}

using namespace cf;

namespace
{
  // ---- CudaUpload(clip): device frame + token for every frame (normal frames if the GPU is unavailable) --------------
  class CudaUpload : public GenericVideoFilter
  {
    MvCudaFrameFormat _fmt;
    bool _ok;
    unsigned long long _producer;

    // uploads src as frame n; 0 on success
    int upload(const PVideoFrame &src, int n, MvCudaTokenHeader &h)
    {
      const void *planes[3] = {};
      int pitches[3] = {};
      for (int p = 0; p < _fmt.planes; p++)
      {
        planes[p] = src->GetReadPtr(g_planes[p]);
        pitches[p] = src->GetPitch(g_planes[p]);
      }
      return CudaLib::get().upload(&_fmt, planes, pitches, _producer, n, &h);
    }

    static int regen(void *user, int frame, void *env, MvCudaTokenHeader *out)
    {
      CudaUpload *self = static_cast<CudaUpload *>(user);
      PVideoFrame src = self->child->GetFrame(frame, static_cast<IScriptEnvironment *>(env));
      return self->upload(src, frame, *out);
    }

  public:
    CudaUpload(PClip c, IScriptEnvironment *env)
      : GenericVideoFilter(c), _ok(false), _producer(0)
    {
      CudaLib &lib = CudaLib::get();
      _ok = lib.ok && make_format(vi, _fmt);
      if (_ok)
        _producer = lib.producer_register(&CudaUpload::regen, this);
      if (log_enabled())
        std::fprintf(stderr, "CudaUpload: %s\n", _ok ? "device frames" : "GPU or format not usable, frames stay on the CPU");
    }

    ~CudaUpload()
    {
      if (_producer)
        CudaLib::get().producer_unregister(_producer);
      print_stats("CudaUpload");
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env) override
    {
      PVideoFrame src = child->GetFrame(n, env);
      MvCudaTokenHeader h;
      if (!_ok || read_token(src, vi, h))
        return src; // GPU not usable, or already a token
      if (upload(src, n, h) != 0)
        return src; // the frame stays on the CPU; consumers accept normal frames
      PVideoFrame dst = env->NewVideoFrameP(vi, &src);
      write_token(dst, h);
      return dst;
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) override
    {
      return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }
  };

  // ---- CudaDownload(clip): normal frames from tokens (regenerated when evicted); normal frames pass through ----------
  class CudaDownload : public GenericVideoFilter
  {
    MvCudaFrameFormat _fmt;
    bool _fmt_ok;

  public:
    CudaDownload(PClip c, IScriptEnvironment *env)
      : GenericVideoFilter(c)
    {
      _fmt_ok = make_format(vi, _fmt);
    }

    ~CudaDownload()
    {
      print_stats("CudaDownload");
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env) override
    {
      PVideoFrame src = child->GetFrame(n, env);
      MvCudaTokenHeader h;
      if (!read_token(src, vi, h))
        return src;
      CudaLib &lib = CudaLib::get();
      if (!lib.ok || !_fmt_ok || !same_format(h.format, _fmt))
        env->ThrowError("CudaDownload: frame %d is a GPU token of another format, or the GPU library is not loaded", n);
      PVideoFrame dst = env->NewVideoFrameP(vi, &src);
      void *planes[3] = {};
      int pitches[3] = {};
      for (int p = 0; p < _fmt.planes; p++)
      {
        planes[p] = dst->GetWritePtr(g_planes[p]);
        pitches[p] = dst->GetPitch(g_planes[p]);
      }
      int rc = lib.download(h.token, planes, pitches);
      if (rc == MVCUDA_TOKEN_GONE && h.producer)
      {
        // evicted from the device frame registry: the producing filter makes it again
        MvCudaTokenHeader h2;
        rc = lib.regenerate(h.producer, h.frame, env, &h2);
        if (rc == 0)
          rc = lib.download(h2.token, planes, pitches);
      }
      if (rc != 0)
        env->ThrowError("CudaDownload: frame %d could not be read from the GPU (code %d)", n, rc);
      return dst;
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) override
    {
      return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }
  };

  AVSValue __cdecl Create_CudaUpload(AVSValue args, void *, IScriptEnvironment *env)
  {
    return new CudaUpload(args[0].AsClip(), env);
  }

  AVSValue __cdecl Create_CudaDownload(AVSValue args, void *, IScriptEnvironment *env)
  {
    return new CudaDownload(args[0].AsClip(), env);
  }
}

PClip cf::cpu_input(PClip c, IScriptEnvironment *env)
{
  return new CudaDownload(c, env);
}

AVSValue __cdecl Create_CudaBob(AVSValue args, void *, IScriptEnvironment *env); // CudaBob.cpp
AVSValue __cdecl Create_CudaTemporalSoften(AVSValue args, void *, IScriptEnvironment *env); // CudaTemporal.cpp
AVSValue __cdecl Create_CudaMerge(AVSValue args, void *, IScriptEnvironment *env); // CudaMerge.cpp
AVSValue __cdecl Create_CudaMergeLuma(AVSValue args, void *, IScriptEnvironment *env);
AVSValue __cdecl Create_CudaMergeChroma(AVSValue args, void *, IScriptEnvironment *env);
AVSValue __cdecl Create_CudaKobs(AVSValue args, void *, IScriptEnvironment *env); // CudaKobs.cpp
AVSValue __cdecl Create_CudaRemoveGrain(AVSValue args, void *, IScriptEnvironment *env);
AVSValue __cdecl Create_CudaGaussResize(AVSValue args, void *, IScriptEnvironment *env); // CudaResize.cpp
AVSValue __cdecl Create_CudaBilinearResize(AVSValue args, void *, IScriptEnvironment *env);
AVSValue __cdecl Create_CudaNnedi3(AVSValue args, void *, IScriptEnvironment *env); // CudaNnedi3.cpp
AVSValue __cdecl Create_CudaLogic(AVSValue args, void *, IScriptEnvironment *env); // CudaMt.cpp
AVSValue __cdecl Create_CudaClamp(AVSValue args, void *, IScriptEnvironment *env);
AVSValue __cdecl Create_CudaExpand(AVSValue args, void *, IScriptEnvironment *env);
AVSValue __cdecl Create_CudaInpand(AVSValue args, void *, IScriptEnvironment *env);
AVSValue __cdecl Create_CudaLutxy(AVSValue args, void *, IScriptEnvironment *env);
AVSValue __cdecl Create_CudaQtgmcSharpLimit(AVSValue args, void *, IScriptEnvironment *env); // CudaQtgmc.cpp
AVSValue __cdecl Create_CudaQtgmcBinomial0(AVSValue args, void *, IScriptEnvironment *env);
AVSValue __cdecl Create_CudaQtgmcBackBlend(AVSValue args, void *, IScriptEnvironment *env);
AVSValue __cdecl Create_CudaQtgmcAddNoise(AVSValue args, void *, IScriptEnvironment *env);

const AVS_Linkage *AVS_linkage = 0;

extern "C" __declspec(dllexport) const char *__stdcall AvisynthPluginInit3(IScriptEnvironment *env, const AVS_Linkage *const vectors)
{
  AVS_linkage = vectors;
  // frame properties, NewVideoFrameP and GetEnvProperty are interface V8 (AviSynth+ 3.6); on an older core their vtable slots
  // do not exist and a call would run some unrelated function, so refuse to load
  try { env->CheckVersion(8); }
  catch (const AvisynthError &) { env->ThrowError("cudafilters: needs AviSynth+ 3.6 or newer (interface V8); update AviSynth+, 3.7.5 is the tested version"); }
  env->AddFunction("CudaUpload", "c", Create_CudaUpload, 0);
  env->AddFunction("CudaDownload", "c", Create_CudaDownload, 0);
  env->AddFunction("CudaBob", "c[b]f[c]f[height]i", Create_CudaBob, 0);
  env->AddFunction("CudaTemporalSoften", "ciii[scenechange]i[mode]i", Create_CudaTemporalSoften, 0);
  env->AddFunction("CudaMerge", "cc[weight]f", Create_CudaMerge, 0);
  env->AddFunction("CudaMergeLuma", "cc[weight]f", Create_CudaMergeLuma, 0);
  env->AddFunction("CudaMergeChroma", "cc[weight]f", Create_CudaMergeChroma, 0);
  env->AddFunction("CudaKeepOnlyBobShimmerFixes", "cc[rep]i[chroma]b", Create_CudaKobs, 0);
  env->AddFunction("CudaRemoveGrain", "c[mode]i[modeU]i[modeV]i", Create_CudaRemoveGrain, 0);
  env->AddFunction("CudaGaussResize", "cii[src_left]f[src_top]f[src_width]f[src_height]f[p]f[b]f[s]f[force]i[keep_center]b[placement]s", Create_CudaGaussResize, 0);
  env->AddFunction("CudaBilinearResize", "cii[src_left]f[src_top]f[src_width]f[src_height]f[force]i[keep_center]b[placement]s", Create_CudaBilinearResize, 0);
  env->AddFunction("CudaNNEDI3", "c[field]i[dh]b[Y]b[U]b[V]b[nsize]i[nns]i[qual]i[etype]i[pscrn]i[threads]i[opt]i[fapprox]i", Create_CudaNnedi3, 0);
  env->AddFunction("CudaLogic", "cc[mode]s[Y]i[U]i[V]i", Create_CudaLogic, 0);
  env->AddFunction("CudaClamp", "ccc[overshoot]i[undershoot]i[Y]i[U]i[V]i", Create_CudaClamp, 0);
  env->AddFunction("CudaExpand", "c[mode]s[Y]i[U]i[V]i", Create_CudaExpand, 0);
  env->AddFunction("CudaInpand", "c[mode]s[Y]i[U]i[V]i", Create_CudaInpand, 0);
  env->AddFunction("CudaLutxy", "cc[expr]s[use_expr]i[Y]i[U]i[V]i", Create_CudaLutxy, 0);
  env->AddFunction("CudaQtgmcSharpLimit", "ccccsii[degrain2]c[fComp3]c[bComp3]c[tr1]i[smode]i[precise]b[nref]i", Create_CudaQtgmcSharpLimit, 0);
  env->AddFunction("CudaQtgmcBackBlend", "ccff[edi]c[fComp1]c[bComp1]c[fComp3]c[bComp3]c[nref]i[sovs]i", Create_CudaQtgmcBackBlend, 0);
  env->AddFunction("CudaQtgmcAddNoise", "ccs[use_expr]i[clamp_float]b[lutUV]i[addUV]i", Create_CudaQtgmcAddNoise, 0);
  env->AddFunction("CudaQtgmcBinomial0", "c[luma_threshold]i[chroma_threshold]i[scenechange]i[chroma]b", Create_CudaQtgmcBinomial0, 0);
  return "cudafilters: GPU filters for AviSynth+ (token frames through mvtools_cuda.dll)";
}
