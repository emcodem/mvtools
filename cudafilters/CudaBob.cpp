// CudaBob(clip, b, c, height): AviSynth+ Bob on the GPU (GPU_PIPELINE_PLAN.md section 15, stage 1), bit for bit.
// Bob = SeparateFields, then FilteredResizeV of every field (Mitchell-Netravali kernel b, c; subrange_top +0.25 for top
// fields and -0.25 for bottom fields; Fieldwise picks by the field parity). The GPU reads the field rows straight from
// the frame (row step 2). Returns token frames (CudaDownload gives normal frames). The CPU Bob with the same arguments is
// built too: its VideoInfo and parity are used, and it produces the frame whenever the GPU cannot (exact fallback).

#include "CudaCommon.h"
#include "Resample.h"

using namespace cf;

namespace
{
  class CudaBob : public GenericVideoFilter
  {
    PClip _cpu;
    VideoInfo _src_vi;
    MvCudaFrameFormat _src_fmt = {}, _out_fmt = {};
    bool _ok = false;
    unsigned long long _prog[2][3] = {}; // [0]: bottom fields (subrange_top -0.25), [1]: top fields (+0.25)
    unsigned long long _producer = 0;

    // the token of output frame n; 0 on success
    int make(int n, IScriptEnvironment *env, MvCudaTokenHeader &h)
    {
      const bool top = child->GetParity(n >> 1) ^ ((n & 1) != 0); // SeparateFields::GetParity
      PVideoFrame src = child->GetFrame(n >> 1, env);
      unsigned long long tok = 0;
      int rc = input_token(src, _src_vi, _src_fmt, tok);
      if (rc)
        return rc;
      const int start = top ? 0 : 1; // the bottom field starts at row 1 (SeparateFields: Subframe offset one pitch)
      const int row_start[3] = { start, start, start }, row_step[3] = { 2, 2, 2 };
      return CudaLib::get().resample_v(_prog[top ? 1 : 0], tok, row_start, row_step, &_out_fmt, _producer, n, &h);
    }

    static int regen(void *user, int frame, void *env, MvCudaTokenHeader *out)
    {
      return static_cast<CudaBob *>(user)->make(frame, static_cast<IScriptEnvironment *>(env), *out);
    }

  public:
    CudaBob(PClip c, double b, double cc, int height, IScriptEnvironment *env)
      : GenericVideoFilter(c), _src_vi(c->GetVideoInfo())
    {
      AVSValue args[4] = { cpu_input(c, env), b, cc, height };
      const char *names[4] = { nullptr, "b", "c", "height" };
      _cpu = env->Invoke("Bob", AVSValue(args, 4), names).AsClip();
      vi = _cpu->GetVideoInfo();

      CudaLib &lib = CudaLib::get();
      _ok = lib.ok && !_src_vi.IsFieldBased() && make_format(_src_vi, _src_fmt) && make_format(vi, _out_fmt);
      if (_ok)
      {
        const MitchellNetravaliKernel kernel(b, cc);
        const int field_h = _src_vi.height / 2;
        const int bits = _src_vi.BitsPerComponent();
        const int shift = _src_fmt.planes == 3 ? (_src_fmt.y_ratio_uv == 2 ? 1 : 0) : 0;
        for (int t = 0; t < 2 && _ok; t++)
        {
          const double sub_top = t ? +0.25 : -0.25;
          for (int p = 0; p < _src_fmt.planes && _ok; p++)
          {
            const int sh = p ? shift : 0, div = 1 << sh;
            // GetCenterShiftForResizers: 0.5 for luma, and 0.5 for chroma with the default chroma placement
            const ResampleProgram prog = make_program(kernel, field_h >> sh, sub_top / div, (double)field_h / div, vi.height >> sh, bits, 0.5, 0.5);
            _prog[t][p] = lib.resample_program_create(prog.target_size, prog.taps, prog.offset.data(), prog.coeff.data());
            _ok = _prog[t][p] != 0;
          }
        }
      }
      if (_ok)
        _producer = lib.producer_register(&CudaBob::regen, this);
      if (log_enabled())
        std::fprintf(stderr, "CudaBob: %s\n", _ok ? "GPU" : "GPU or format not usable, using Bob on the CPU");
    }

    ~CudaBob()
    {
      CudaLib &lib = CudaLib::get();
      if (_producer)
        lib.producer_unregister(_producer);
      for (auto &t : _prog)
        for (unsigned long long p : t)
          if (p)
            lib.resample_program_free(p);
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env) override
    {
      if (!_ok)
        return _cpu->GetFrame(n, env);
      MvCudaTokenHeader h;
      if (make(n, env, h) != 0)
        return _cpu->GetFrame(n, env); // exact CPU result
      PVideoFrame src = child->GetFrame(n >> 1, env);
      PVideoFrame dst = env->NewVideoFrameP(vi, &src);
      write_token(dst, h);
      return dst;
    }

    bool __stdcall GetParity(int n) override
    {
      return _cpu->GetParity(n);
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) override
    {
      return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }
  };
}

AVSValue __cdecl Create_CudaBob(AVSValue args, void *, IScriptEnvironment *env)
{
  PClip c = args[0].AsClip();
  const double b = args[1].AsDblDef(1. / 3.), cc = args[2].AsDblDef(1. / 3.);
  const int height = args[3].AsInt(c->GetVideoInfo().height);
  return new CudaBob(c, b, cc, height, env);
}
