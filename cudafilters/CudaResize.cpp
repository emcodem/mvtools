// CudaGaussResize(clip, width, height, src_left, src_top, src_width, src_height, p, b, s, force, keep_center, placement) and
// CudaBilinearResize(clip, width, height, src_left, src_top, src_width, src_height, force, keep_center, placement):
// AviSynth+ GaussResize / BilinearResize on the GPU (GPU_PIPELINE_PLAN.md section 15, stage 5: QTGMC blurs the search clip with
// GaussResize(w, h, 0, 0, w + eps, h + eps, p=2); SrchClipPP 1 uses BilinearResize to half size and back), bit for bit. The passes, their order and the programs follow
// FilteredResize::CreateResize / CreateResizeH / CreateResizeV and the FilteredResizeH / V constructors of AviSynth+ 3.7.5
// (resample.cpp, GPL v2 or later); the arithmetic is in mvcuda_resample.cu. The CPU filter with the same arguments is
// built too (VideoInfo, exact fallback).

#include "CudaCommon.h"
#include "Resample.h"

#include <cctype>
#include <cstring>

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

  // kind 0: GaussResize, 1: BilinearResize (no p / b / s arguments)
  // kind 0: GaussResize, 1: BilinearResize (no p / b / s arguments)
  class CudaGaussResize : public GenericVideoFilter
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
      return static_cast<CudaGaussResize *>(user)->make(frame, static_cast<IScriptEnvironment *>(env), *out);
    }

  public:
    CudaGaussResize(const AVSValue &args, int kind, IScriptEnvironment *env)
      : GenericVideoFilter(args[0].AsClip()), _src_vi(args[0].AsClip()->GetVideoInfo())
    {
      const char *gnames[13] = { nullptr, nullptr, nullptr, "src_left", "src_top", "src_width", "src_height", "p", "b", "s", "force",
        "keep_center", "placement" };
      const char *bnames[10] = { nullptr, nullptr, nullptr, "src_left", "src_top", "src_width", "src_height", "force", "keep_center", "placement" };
      const int nmax = kind ? 10 : 13, ex = kind ? 7 : 10; // ex: index of force (then keep_center, placement)
      // the CPU filter reads pixels: token inputs go through CudaDownload (cpu_input); undefined optional arguments left out
      AVSValue cargs[13];
      const char *cnames[13];
      int na = 0;
      for (int i = 0; i < nmax && i < args.ArraySize(); i++)
        if (i < 3 || args[i].Defined())
        {
          cargs[na] = i == 0 ? AVSValue(cpu_input(args[0].AsClip(), env)) : args[i];
          cnames[na++] = kind ? bnames[i] : gnames[i];
        }
      _cpu = env->Invoke(kind ? "BilinearResize" : "GaussResize", AVSValue(cargs, na), cnames).AsClip();
      vi = _cpu->GetVideoInfo();

      CudaLib &lib = CudaLib::get();
      _ok = lib.ok && make_format(_src_vi, _src_fmt) && make_format(vi, _out_fmt);
      if (!_ok)
        return;
      // ---- FilteredResize::CreateResize
      const int target_w = args[1].AsInt(), target_h = args[2].AsInt();
      const double sub_left = args[3].AsFloat(0), sub_top = args[4].AsFloat(0);
      double sub_w = args[5].AsDblDef(_src_vi.width), sub_h = args[6].AsDblDef(_src_vi.height);
      if (sub_w <= 0.0)
        sub_w = _src_vi.width - sub_left + sub_w;
      if (sub_h <= 0.0)
        sub_h = _src_vi.height - sub_top + sub_h;
      const GaussianKernel gauss(kind ? 30.0 : args[7].AsFloat(30.0f), kind ? 2.0 : args[8].AsFloat(2.0f), kind ? 4.0 : args[9].AsFloat(4.0f));
      const TriangleKernel triangle;
      const ResampleKernel &kernel = kind ? (const ResampleKernel &)triangle : (const ResampleKernel &)gauss;
      const int force = args[ex].AsInt(0);
      const bool keep_center = args[ex + 1].AsBool(true);
      const char *placement = args[ex + 2].AsString("auto");
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
          const ResampleProgram prog = make_program(kernel, h_src_w >> hx, sub_left / divx, sub_w / divx, target_w >> hx, bits,
            p ? cp_h_chroma : cp_luma, p ? cp_h_chroma : cp_luma);
          _ph[p] = lib.resample_program_create(prog.target_size, prog.taps, prog.offset.data(), prog.coeff.data());
          _ok = _ph[p] != 0;
        }
        if (do_v && _ok)
        {
          const ResampleProgram prog = make_program(kernel, v_src_h >> hy, sub_top / divy, sub_h / divy, target_h >> hy, bits,
            p ? cp_v_chroma : cp_luma, p ? cp_v_chroma : cp_luma);
          _pv[p] = lib.resample_program_create(prog.target_size, prog.taps, prog.offset.data(), prog.coeff.data());
          _ok = _pv[p] != 0;
        }
      }
      if (_ok)
        _producer = lib.producer_register(&CudaGaussResize::regen, this);
      if (log_enabled())
        std::fprintf(stderr, "%s: %s\n", kind ? "CudaBilinearResize" : "CudaGaussResize", _ok ? "GPU" : "GPU or format not usable, using the CPU filter");
    }

    ~CudaGaussResize()
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
        return _cpu->GetFrame(n, env);
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
}

AVSValue __cdecl Create_CudaGaussResize(AVSValue args, void *, IScriptEnvironment *env)
{
  return new CudaGaussResize(args, 0, env);
}

AVSValue __cdecl Create_CudaBilinearResize(AVSValue args, void *, IScriptEnvironment *env)
{
  return new CudaGaussResize(args, 1, env);
}
