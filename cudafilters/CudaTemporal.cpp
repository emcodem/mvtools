// CudaTemporalSoften(clip, radius, luma_threshold, chroma_threshold, scenechange, mode): AviSynth+ TemporalSoften on the GPU
// (GPU_PIPELINE_PLAN.md section 15, stage 2), bit for bit; mode is ignored as by the CPU filter. Returns token frames. The
// CPU TemporalSoften with the same arguments is built too (VideoInfo, exact fallback).

#include "CudaCommon.h"

#include <algorithm>

using namespace cf;

namespace
{
  class CudaTemporalSoften : public GenericVideoFilter
  {
    PClip _cpu;
    MvCudaFrameFormat _fmt = {};
    bool _ok = false;
    int _radius, _luma, _chroma, _scenechange;
    unsigned long long _producer = 0;

    int make(int n, IScriptEnvironment *env, MvCudaTokenHeader &h)
    {
      PVideoFrame frames[2 * 7 + 1];
      const int count = 2 * _radius + 1;
      for (int k = 0; k < count; k++)
        frames[k] = child->GetFrame(std::clamp(n - _radius + k, 0, vi.num_frames - 1), env);
      return with_inputs(frames, count, vi, _fmt, env, [&](const unsigned long long *tok) {
        return CudaLib::get().temporal_soften(tok, _radius, _luma, _chroma, _scenechange, _producer, n, &h);
      });
    }

    static int regen(void *user, int frame, void *env, MvCudaTokenHeader *out)
    {
      return static_cast<CudaTemporalSoften *>(user)->make(frame, static_cast<IScriptEnvironment *>(env), *out);
    }

  public:
    CudaTemporalSoften(PClip c, int radius, int luma, int chroma, int scenechange, int mode, IScriptEnvironment *env)
      : GenericVideoFilter(c)
    {
      AVSValue args[6] = { cpu_input(c, env), radius, luma, chroma, scenechange, mode };
      const char *names[6] = { nullptr, nullptr, nullptr, nullptr, "scenechange", "mode" };
      _cpu = env->Invoke("TemporalSoften", AVSValue(args, 6), names).AsClip();
      vi = _cpu->GetVideoInfo();
      // as the TemporalSoften constructor
      _radius = std::min((unsigned)radius, 7u);
      _luma = (int)std::min((unsigned)luma, 255u);
      _chroma = (int)std::min((unsigned)chroma, 255u);
      _scenechange = scenechange >= 255 ? 0 : scenechange;
      _scenechange *= ((vi.width / 32) * 32) * vi.height;
      CudaLib &lib = CudaLib::get();
      _ok = lib.ok && make_format(vi, _fmt) && _radius > 0 && (_luma > 0 || _chroma > 0);
      if (_ok)
        _producer = lib.producer_register(&CudaTemporalSoften::regen, this);
      if (log_enabled())
        std::fprintf(stderr, "CudaTemporalSoften: %s\n", _ok ? "GPU" : "GPU, format or settings not usable, using TemporalSoften on the CPU");
    }

    ~CudaTemporalSoften()
    {
      if (_producer)
        CudaLib::get().producer_unregister(_producer);
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env) override
    {
      if (!_ok)
        return _cpu->GetFrame(n, env);
      MvCudaTokenHeader h;
      if (make(n, env, h) != 0)
        return _cpu->GetFrame(n, env); // exact CPU result
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

AVSValue __cdecl Create_CudaTemporalSoften(AVSValue args, void *, IScriptEnvironment *env)
{
  return new CudaTemporalSoften(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), args[3].AsInt(), args[4].AsInt(0), args[5].AsInt(1), env);
}
