// CudaMerge / CudaMergeLuma / CudaMergeChroma(clip, clip2, weight): AviSynth+ Merge, MergeLuma, MergeChroma on the GPU
// (GPU_PIPELINE_PLAN.md section 15), bit for bit with what AviSynth+ 3.7.5 runs on SSE2 / AVX2 CPUs. The per-plane
// decisions follow the GetFrame functions of merge.cpp; the arithmetic is in mvcuda_merge.cu. Returns token frames (or a
// source frame unchanged where the CPU filter returns it unchanged). The CPU filter is built too (VideoInfo, fallback).

#include "CudaCommon.h"

using namespace cf;

namespace
{
  enum Kind { MERGE_ALL, MERGE_LUMA, MERGE_CHROMA };

  class CudaMerge : public GenericVideoFilter
  {
    PClip _clip, _cpu;
    Kind _kind;
    float _weight;
    MvCudaFrameFormat _fmt = {};
    bool _ok = false;
    int _mode[3] = {}, _tail[3] = {}, _weight_i = 0, _invweight_i = 0;
    unsigned long long _producer = 0;

    int make(int n, IScriptEnvironment *env, MvCudaTokenHeader &h)
    {
      PVideoFrame frames[2] = { child->GetFrame(n, env), _clip->GetFrame(n, env) };
      return with_inputs(frames, 2, vi, _fmt, env, [&](const unsigned long long *tok) {
        return CudaLib::get().merge(tok[0], tok[1], _mode, _weight_i, _invweight_i, _tail, _producer, n, &h);
      });
    }

    static int regen(void *user, int frame, void *env, MvCudaTokenHeader *out)
    {
      return static_cast<CudaMerge *>(user)->make(frame, static_cast<IScriptEnvironment *>(env), *out);
    }

  public:
    CudaMerge(PClip c, PClip c2, float weight, Kind kind, IScriptEnvironment *env)
      : GenericVideoFilter(c), _clip(c2), _kind(kind)
    {
      AVSValue args[3] = { cpu_input(c, env), cpu_input(c2, env), weight };
      const char *names[3] = { nullptr, nullptr, "weight" };
      _cpu = env->Invoke(kind == MERGE_ALL ? "Merge" : kind == MERGE_LUMA ? "MergeLuma" : "MergeChroma", AVSValue(args, 3), names).AsClip();
      vi = _cpu->GetVideoInfo();
      _weight = weight < 0.0f ? 0.0f : weight > 1.0f ? 1.0f : weight;
      // merge_plane: average close to 0.5, else the weighted merge with the SIMD weights (getMergeFunc)
      const int merge_mode = (_weight > 0.4961f && _weight < 0.5039f) ? 2 : 1;
      _weight_i = (int)(_weight * 32767.0f + 0.5f);
      _invweight_i = 32767 - _weight_i;
      MvCudaFrameFormat f2;
      CudaLib &lib = CudaLib::get();
      _ok = lib.ok && make_format(vi, _fmt) && make_format(c2->GetVideoInfo(), f2) && same_format(_fmt, f2);
      const int planes = _fmt.planes;
      for (int p = 0; p < 3; p++)
        _mode[p] = 0;
      if (kind == MERGE_ALL)
        for (int p = 0; p < planes; p++)
          _mode[p] = merge_mode;
      else if (kind == MERGE_LUMA)
        _mode[0] = _weight > 0.9961f ? 3 : merge_mode;
      else
      {
        if (planes < 3)
          _ok = false;
        _mode[1] = _mode[2] = _weight < 0.9961f ? merge_mode : 3;
      }
      // the SIMD code covers row_size / 16 * 16 bytes of each row and leaves the rest to its scalar tail; MergeChroma
      // passes the aligned row size (no tail inside the picture)
      for (int p = 0; p < planes; p++)
      {
        const int w = p == 0 ? _fmt.width : _fmt.width / _fmt.x_ratio_uv;
        const int rowsize = w * _fmt.bytes_per_sample;
        _tail[p] = kind == MERGE_CHROMA ? w : (rowsize / 16 * 16) / _fmt.bytes_per_sample;
      }
      if (_ok)
        _producer = lib.producer_register(&CudaMerge::regen, this);
      if (log_enabled())
        std::fprintf(stderr, "CudaMerge: %s\n", _ok ? "GPU" : "GPU or formats not usable, using the CPU filter");
    }

    ~CudaMerge()
    {
      if (_producer)
        CudaLib::get().producer_unregister(_producer);
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env) override
    {
      if (!_ok)
        return _cpu->GetFrame(n, env);
      // frames the CPU filters return unchanged (also when they are tokens)
      if (_weight < 0.0039f)
        return child->GetFrame(n, env);
      if (_kind == MERGE_ALL && _weight > 0.9961f)
        return _clip->GetFrame(n, env);
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

AVSValue __cdecl Create_CudaMerge(AVSValue args, void *, IScriptEnvironment *env)
{
  return new CudaMerge(args[0].AsClip(), args[1].AsClip(), (float)args[2].AsFloat(0.5f), MERGE_ALL, env);
}

AVSValue __cdecl Create_CudaMergeLuma(AVSValue args, void *, IScriptEnvironment *env)
{
  return new CudaMerge(args[0].AsClip(), args[1].AsClip(), (float)args[2].AsFloat(1.0f), MERGE_LUMA, env);
}

AVSValue __cdecl Create_CudaMergeChroma(AVSValue args, void *, IScriptEnvironment *env)
{
  return new CudaMerge(args[0].AsClip(), args[1].AsClip(), (float)args[2].AsFloat(1.0f), MERGE_CHROMA, env);
}
