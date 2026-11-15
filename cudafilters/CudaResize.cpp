// The AviSynth+ 3.7.5 core resizers on the GPU, bit for bit, with the same arguments as the CPU filter plus the Cuda
// prefix: CudaPointResize, CudaBilinearResize, CudaBicubicResize, CudaLanczosResize, CudaLanczos4Resize,
// CudaBlackmanResize, CudaSpline16Resize, CudaSpline36Resize, CudaSpline64Resize, CudaGaussResize, CudaSincResize,
// CudaSinPowerResize, CudaSincLin2Resize, CudaUserDefined2Resize (QTGMC blurs the search clip with
// GaussResize(w, h, 0, 0, w + eps, h + eps, p=2); SrchClipPP 1 uses BilinearResize to half size and back). The passes,
// their order and the programs follow FilteredResize::CreateResize / CreateResizeH / CreateResizeV and the
// FilteredResizeH / V constructors of AviSynth+ 3.7.5 (resample.cpp, GPL v2 or later); the kernels are in Resample.cpp,
// the arithmetic in mvcuda_resample.cu. The CPU filter with the same arguments is built too (VideoInfo, exact fallback).

#include "CudaCommon.h"
#include "Resample.h"

#include <cctype>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace cf;

namespace
{
  // ChromaLocation_e (convert_helper.h)
  enum { CL_UNUSED = -1, CL_LEFT = 0, CL_CENTER = 1, CL_TOP_LEFT = 2, CL_TOP = 3, CL_BOTTOM_LEFT = 4, CL_BOTTOM = 5, CL_DV = 6 };

  // getChromaLocation names (convert_helper.cpp); false for unknown names
  bool chroma_location_by_name(const char *name, int &out)
  {
    static const struct { const char *n; int v; } table[] = { { "left", CL_LEFT }, { "center", CL_CENTER }, { "top_left", CL_TOP_LEFT },
      { "top", CL_TOP }, { "bottom_left", CL_BOTTOM_LEFT }, { "bottom", CL_BOTTOM }, { "dv", CL_DV }, { "mpeg1", CL_CENTER },
      { "mpeg2", CL_LEFT }, { "jpeg", CL_CENTER } };
    for (const auto &e : table)
      if (_stricmp(name, e.n) == 0)
      {
        out = e.v;
        return true;
      }
    return false;
  }

  // ChromaPlacement_compute_cplace for plane U, not interlaced (resample.cpp)
  void chroma_center(int cplace, int ss_h, int ss_v, double &cp_h, double &cp_v)
  {
    cp_h = 0.5;
    cp_v = 0.5;
    if (ss_h > 0 && (cplace == CL_LEFT || cplace == CL_DV || cplace == CL_TOP_LEFT || cplace == CL_BOTTOM_LEFT))
      cp_h = 0.5 / (1 << ss_h);
    if (ss_v == 1)
    {
      if (cplace == CL_LEFT)
        cp_v = 0.5;
      else if (cplace == CL_DV || cplace == CL_TOP_LEFT || cplace == CL_TOP)
        cp_v = 0.25;
      else if (cplace == CL_BOTTOM_LEFT || cplace == CL_BOTTOM)
        cp_v = 0.75;
    }
  }

  // One resizer as AviSynth+ 3.7.5 registers it (resample.cpp: the function table and FilteredResize::Create_*):
  // the argument signature, the index of src_left (the source window follows it), of force, keep_center and placement,
  // and the kernel made from the arguments with the same defaults and the same AsFloat / AsDblDef / AsInt reads.
  // BlackmanResize reads keep_center from the force index and placement from the keep_center index (a value of another
  // type reads as the default, so it always works as keep_center=true, placement "auto"); kept as it is, because the
  // result is compared with that filter.
  struct ResizeKind
  {
    const char *cpu_name, *gpu_name, *params;
    int window, force, keep_center, placement;
    std::unique_ptr<ResampleKernel> (*kernel)(const AVSValue &args);
  };

#define CF_WINDOW "[src_left]f[src_top]f[src_width]f[src_height]f"
#define CF_TAIL "[force]i[keep_center]b[placement]s"
  const ResizeKind kResizeKinds[] = {
    { "PointResize", "CudaPointResize", "cii" CF_WINDOW CF_TAIL, 3, 7, 8, 9,
      [](const AVSValue &) -> std::unique_ptr<ResampleKernel> { return std::make_unique<PointKernel>(); } },
    { "BilinearResize", "CudaBilinearResize", "cii" CF_WINDOW CF_TAIL, 3, 7, 8, 9,
      [](const AVSValue &) -> std::unique_ptr<ResampleKernel> { return std::make_unique<TriangleKernel>(); } },
    { "BicubicResize", "CudaBicubicResize", "cii[b]f[c]f" CF_WINDOW CF_TAIL, 5, 9, 10, 11,
      [](const AVSValue &a) -> std::unique_ptr<ResampleKernel> {
        return std::make_unique<MitchellNetravaliKernel>(a[3].AsDblDef(1. / 3.), a[4].AsDblDef(1. / 3.)); } },
    { "LanczosResize", "CudaLanczosResize", "cii" CF_WINDOW "[taps]i" CF_TAIL, 3, 8, 9, 10,
      [](const AVSValue &a) -> std::unique_ptr<ResampleKernel> { return std::make_unique<LanczosKernel>(a[7].AsInt(3)); } },
    { "Lanczos4Resize", "CudaLanczos4Resize", "cii" CF_WINDOW CF_TAIL, 3, 7, 8, 9,
      [](const AVSValue &) -> std::unique_ptr<ResampleKernel> { return std::make_unique<LanczosKernel>(4); } },
    { "BlackmanResize", "CudaBlackmanResize", "cii" CF_WINDOW "[taps]i" CF_TAIL, 3, 8, 8, 9,
      [](const AVSValue &a) -> std::unique_ptr<ResampleKernel> { return std::make_unique<BlackmanKernel>(a[7].AsInt(4)); } },
    { "Spline16Resize", "CudaSpline16Resize", "cii" CF_WINDOW CF_TAIL, 3, 7, 8, 9,
      [](const AVSValue &) -> std::unique_ptr<ResampleKernel> { return std::make_unique<Spline16Kernel>(); } },
    { "Spline36Resize", "CudaSpline36Resize", "cii" CF_WINDOW CF_TAIL, 3, 7, 8, 9,
      [](const AVSValue &) -> std::unique_ptr<ResampleKernel> { return std::make_unique<Spline36Kernel>(); } },
    { "Spline64Resize", "CudaSpline64Resize", "cii" CF_WINDOW CF_TAIL, 3, 7, 8, 9,
      [](const AVSValue &) -> std::unique_ptr<ResampleKernel> { return std::make_unique<Spline64Kernel>(); } },
    { "GaussResize", "CudaGaussResize", "cii" CF_WINDOW "[p]f[b]f[s]f" CF_TAIL, 3, 10, 11, 12,
      [](const AVSValue &a) -> std::unique_ptr<ResampleKernel> {
        return std::make_unique<GaussianKernel>(a[7].AsFloat(30.0f), a[8].AsFloat(2.0f), a[9].AsFloat(4.0f)); } },
    { "SincResize", "CudaSincResize", "cii" CF_WINDOW "[taps]i" CF_TAIL, 3, 8, 9, 10,
      [](const AVSValue &a) -> std::unique_ptr<ResampleKernel> { return std::make_unique<SincKernel>(a[7].AsInt(4)); } },
    { "SinPowerResize", "CudaSinPowerResize", "cii" CF_WINDOW "[p]f" CF_TAIL, 3, 8, 9, 10,
      [](const AVSValue &a) -> std::unique_ptr<ResampleKernel> { return std::make_unique<SinPowerKernel>(a[7].AsFloat(2.5f)); } },
    { "SincLin2Resize", "CudaSincLin2Resize", "cii" CF_WINDOW "[taps]i" CF_TAIL, 3, 8, 9, 10,
      [](const AVSValue &a) -> std::unique_ptr<ResampleKernel> { return std::make_unique<SincLin2Kernel>(a[7].AsInt(15)); } },
    { "UserDefined2Resize", "CudaUserDefined2Resize", "cii[b]f[c]f[s]f" CF_WINDOW CF_TAIL, 6, 10, 11, 12,
      [](const AVSValue &a) -> std::unique_ptr<ResampleKernel> {
        return std::make_unique<UserDefined2Kernel>(a[3].AsFloat(121.0f), a[4].AsFloat(19.0f), a[5].AsFloat(2.3f)); } },
  };
#undef CF_WINDOW
#undef CF_TAIL

  // the argument names of a signature ("cii[src_left]f..."): null for the unnamed ones
  std::vector<std::string> param_names(const char *params)
  {
    std::vector<std::string> names;
    for (const char *s = params; *s;)
    {
      if (*s == '[')
      {
        const char *e = std::strchr(s, ']');
        names.emplace_back(s + 1, e);
        s = e + 2; // the type letter after the name
      }
      else
      {
        names.emplace_back();
        s++;
      }
    }
    return names;
  }

  class CudaResize : public GenericVideoFilter
  {
    PClip _cpu;
    VideoInfo _src_vi;
    MvCudaFrameFormat _src_fmt = {}, _out_fmt = {};
    bool _ok = false, _h_first = true;
    unsigned long long _ph[3] = {}, _pv[3] = {}, _producer = 0;

    int make(int n, IScriptEnvironment *env, MvCudaTokenHeader &h)
    {
      PVideoFrame frames[1] = { child->GetFrame(n, env) };
      return with_inputs(frames, 1, _src_vi, _src_fmt, env, [&](const unsigned long long *tok) {
        return CudaLib::get().resample_2d(_ph, _pv, _h_first ? 1 : 0, tok[0], &_out_fmt, _producer, n, &h);
      });
    }

    static int regen(void *user, int frame, void *env, MvCudaTokenHeader *out)
    {
      CudaResize *self = static_cast<CudaResize *>(user);
      if (self->make(frame, static_cast<IScriptEnvironment *>(env), *out) == 0)
        return 0;
      return cpu_regen(self->_cpu, frame, self->vi, self->_out_fmt, self->_producer, env, out); // as GetFrame
    }

  public:
    CudaResize(const AVSValue &args, const ResizeKind &kind, IScriptEnvironment *env)
      : GenericVideoFilter(args[0].AsClip()), _src_vi(args[0].AsClip()->GetVideoInfo())
    {
      // the CPU filter reads pixels: token inputs go through CudaDownload (cpu_input); undefined optional arguments left out
      const std::vector<std::string> names = param_names(kind.params);
      const int nmax = (int)names.size();
      std::vector<AVSValue> cargs(nmax);
      std::vector<const char *> cnames(nmax);
      int na = 0;
      for (int i = 0; i < nmax && i < args.ArraySize(); i++)
        if (i < 3 || args[i].Defined())
        {
          cargs[na] = i == 0 ? AVSValue(cpu_input(args[0].AsClip(), env)) : args[i];
          cnames[na++] = names[i].empty() ? nullptr : names[i].c_str();
        }
      _cpu = env->Invoke(kind.cpu_name, AVSValue(cargs.data(), na), cnames.data()).AsClip();
      vi = _cpu->GetVideoInfo();

      CudaLib &lib = CudaLib::get();
      const char *const fmt_why = "only planar Y / YUV 8 to 16 bit, 4:2:0 / 4:2:2 / 4:4:0 / 4:4:4, at least 68 bytes wide";
      _ok = lib.ok && make_format(_src_vi, _src_fmt) && make_format(vi, _out_fmt);
      if (!_ok)
      {
        gpu_decide(kind.gpu_name, kind.cpu_name, false, fmt_why, env);
        return;
      }
      // ---- FilteredResize::CreateResize
      const int w0 = kind.window;
      const int target_w = args[1].AsInt(), target_h = args[2].AsInt();
      const double sub_left = args[w0].AsFloat(0), sub_top = args[w0 + 1].AsFloat(0);
      double sub_w = args[w0 + 2].AsDblDef(_src_vi.width), sub_h = args[w0 + 3].AsDblDef(_src_vi.height);
      if (sub_w <= 0.0)
        sub_w = _src_vi.width - sub_left + sub_w;
      if (sub_h <= 0.0)
        sub_h = _src_vi.height - sub_top + sub_h;
      const std::unique_ptr<ResampleKernel> kernel = kind.kernel(args);
      const int force = args[kind.force].AsInt(0);
      const bool keep_center = args[kind.keep_center].AsBool(true);
      // a placement index that holds another type (BlackmanResize) counts as not given
      const char *placement = args[kind.placement].IsString() ? args[kind.placement].AsString() : "auto";
      int cplace = CL_UNUSED;
      if (_src_vi.Is420() || _src_vi.Is422() || _src_vi.IsYV411())
      {
        PVideoFrame f0 = child->GetFrame(0, env);
        const AVSMap *props = env->getFramePropsRO(f0);
        if (env->propNumElements(props, "_ChromaLocation") > 0)
          cplace = (int)env->propGetInt(props, "_ChromaLocation", 0, nullptr); // V8; propGetIntSaturated is V11 and breaks AviSynth+ 3.7.3
      }
      int named;
      if (!(placement[0] == 0 || _stricmp(placement, "auto") == 0) && chroma_location_by_name(placement, named))
        cplace = named;
      const bool force_h = force == 1 || force == 3, force_v = force == 2 || force == 3;
      _h_first = !(sub_h * target_w < sub_w * target_h);

      const int planes = _src_fmt.planes, bits = _src_fmt.bits_per_sample;
      const int sx = planes == 3 ? (_src_fmt.x_ratio_uv == 2 ? 1 : 0) : 0, sy = planes == 3 ? (_src_fmt.y_ratio_uv == 2 ? 1 : 0) : 0;
      double cp_h_chroma = 0.0, cp_v_chroma = 0.0;
      if (keep_center)
        chroma_center(cplace, sx, sy, cp_h_chroma, cp_v_chroma);
      const double cp_luma = keep_center ? 0.5 : 0.0;
      // the width / height each pass sees (the second pass works on the first one's output)
      const int h_src_w = _src_vi.width, v_src_h = _src_vi.height;
      // CreateResizeH / CreateResizeV skip a pass that is the identity
      const bool do_h = force_h || !(sub_left == 0 && sub_w == target_w && sub_w == h_src_w);
      const bool do_v = force_v || !(sub_top == 0 && sub_h == target_h && sub_h == v_src_h);
      for (int p = 0; p < planes && _ok; p++)
      {
        const int hx = p ? sx : 0, hy = p ? sy : 0, divx = 1 << hx, divy = 1 << hy;
        if (do_h)
        {
          const ResampleProgram prog = make_program(*kernel, h_src_w >> hx, sub_left / divx, sub_w / divx, target_w >> hx, bits,
            p ? cp_h_chroma : cp_luma, p ? cp_h_chroma : cp_luma);
          _ph[p] = lib.resample_program_create(prog.target_size, prog.taps, prog.offset.data(), prog.coeff.data());
          _ok = _ph[p] != 0;
        }
        if (do_v && _ok)
        {
          const ResampleProgram prog = make_program(*kernel, v_src_h >> hy, sub_top / divy, sub_h / divy, target_h >> hy, bits,
            p ? cp_v_chroma : cp_luma, p ? cp_v_chroma : cp_luma);
          _pv[p] = lib.resample_program_create(prog.target_size, prog.taps, prog.offset.data(), prog.coeff.data());
          _ok = _pv[p] != 0;
        }
      }
      _ok = gpu_decide(kind.gpu_name, kind.cpu_name, _ok, "a resampling program could not be made on the GPU", env);
      if (_ok)
        _producer = lib.producer_register(&CudaResize::regen, this);
    }

    ~CudaResize()
    {
      CudaLib &lib = CudaLib::get();
      if (_producer)
        lib.producer_unregister(_producer);
      for (int p = 0; p < 3; p++)
      {
        if (_ph[p])
          lib.resample_program_free(_ph[p]);
        if (_pv[p])
          lib.resample_program_free(_pv[p]);
      }
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env) override
    {
      if (!_ok)
        return _cpu->GetFrame(n, env);
      MvCudaTokenHeader h;
      if (make(n, env, h) != 0)
        return cpu_token(_cpu->GetFrame(n, env), vi, _out_fmt, _producer, n, env); // exact CPU result, uploaded
      PVideoFrame src = child->GetFrame(n, env);
      PVideoFrame dst = env->NewVideoFrameP(vi, &src);
      write_token(dst, h);
      return dst;
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) override
    {
      return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }
  };

  AVSValue __cdecl Create_CudaResize(AVSValue args, void *user_data, IScriptEnvironment *env)
  {
    return new CudaResize(args, *static_cast<const ResizeKind *>(user_data), env);
  }
}

// registers every Cuda*Resize (called from AvisynthPluginInit3)
void register_cuda_resizers(IScriptEnvironment *env)
{
  for (const ResizeKind &k : kResizeKinds)
    env->AddFunction(k.gpu_name, k.params, Create_CudaResize, const_cast<ResizeKind *>(&k));
}
