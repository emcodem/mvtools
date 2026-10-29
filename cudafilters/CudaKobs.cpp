// CudaKeepOnlyBobShimmerFixes(input, ref, rep, chroma): QTGMC_KeepOnlyBobShimmerFixes of QTGMC.avsi on the GPU (8 bit;
// GPU_PIPELINE_PLAN.md section 15, stages 4 and 16), and CudaRemoveGrain(clip, mode, modeU, modeV) for the modes the GPU
// has (12, 20 at 8 bit). Bit for bit with masktools2 / RgTools; the arithmetic is in mvcuda_kobs.cu.
//
// The function's two mt_lutxy expressions are not re-implemented: the plugin runs the installed mt_lutxy once on a
// 256 x 256 pattern of all (x, y) pairs and uses its output as the table, so the GPU applies exactly what the CPU filter
// computes. The CPU function / filter is built too (VideoInfo, exact fallback for unsupported settings and formats).

#include "CudaCommon.h"

using namespace cf;

namespace
{
  // one 256 x 256 Y8 frame whose samples are their column (by_row = false) or their row (by_row = true)
  class PatternClip : public IClip
  {
    VideoInfo _vi;
    bool _by_row;

  public:
    explicit PatternClip(bool by_row) : _by_row(by_row)
    {
      std::memset(&_vi, 0, sizeof(_vi));
      _vi.width = 256;
      _vi.height = 256;
      _vi.pixel_type = VideoInfo::CS_Y8;
      _vi.fps_numerator = 25;
      _vi.fps_denominator = 1;
      _vi.num_frames = 1;
    }
    PVideoFrame __stdcall GetFrame(int, IScriptEnvironment *env) override
    {
      PVideoFrame f = env->NewVideoFrame(_vi);
      BYTE *p = f->GetWritePtr(PLANAR_Y);
      const int pitch = f->GetPitch(PLANAR_Y);
      for (int y = 0; y < 256; y++)
        for (int x = 0; x < 256; x++)
          p[(size_t)y * pitch + x] = (BYTE)(_by_row ? y : x);
      return f;
    }
    bool __stdcall GetParity(int) override { return false; }
    void __stdcall GetAudio(void *, int64_t, int64_t, IScriptEnvironment *) override {}
    int __stdcall SetCacheHints(int, int) override { return 0; }
    const VideoInfo &__stdcall GetVideoInfo() override { return _vi; }
  };
}

PClip cf::pattern_clip(bool by_row)
{
  return new PatternClip(by_row);
}

unsigned long long cf::lut_from_frame(const PVideoFrame &f)
{
  const BYTE *p = f->GetReadPtr(PLANAR_Y);
  const int pitch = f->GetPitch(PLANAR_Y);
  unsigned char table[65536];
  for (int y = 0; y < 256; y++)
    for (int x = 0; x < 256; x++)
      table[x * 256 + y] = p[(size_t)y * pitch + x];
  return CudaLib::get().lut2_create(table);
}

// the table of mt_lutxy(x, y, expr, use_expr=...) as the installed masktools computes it: table[x * 256 + y]
unsigned long long cf::lut_from_cpu(const char *expr, int use_expr, IScriptEnvironment *env)
{
  PClip px = new PatternClip(false), py = new PatternClip(true);
  AVSValue args[4] = { px, py, expr, use_expr };
  const char *names[4] = { nullptr, nullptr, "expr", "use_expr" };
  PClip lut = env->Invoke("mt_lutxy", AVSValue(args, 4), names).AsClip();
  return lut_from_frame(lut->GetFrame(0, env));
}

namespace
{
  class CudaKobs : public GenericVideoFilter
  {
    PClip _ref, _cpu;
    MvCudaFrameFormat _fmt = {};
    bool _ok = false;
    int _rep, _mask;
    unsigned long long _lut1 = 0, _lut2 = 0, _producer = 0;

    int make(int n, IScriptEnvironment *env, MvCudaTokenHeader &h)
    {
      PVideoFrame frames[2] = { child->GetFrame(n, env), _ref->GetFrame(n, env) };
      return with_inputs(frames, 2, vi, _fmt, env, [&](const unsigned long long *tok) {
        return CudaLib::get().kobs(tok[0], tok[1], _rep, _mask, _lut1, _lut2, _producer, n, &h);
      });
    }

    static int regen(void *user, int frame, void *env, MvCudaTokenHeader *out)
    {
      return static_cast<CudaKobs *>(user)->make(frame, static_cast<IScriptEnvironment *>(env), *out);
    }

  public:
    CudaKobs(PClip input, PClip ref, int rep, bool chroma, IScriptEnvironment *env)
      : GenericVideoFilter(input), _ref(ref), _rep(rep)
    {
      AVSValue args[4] = { cpu_input(input, env), cpu_input(ref, env), rep, chroma };
      _cpu = env->Invoke("QTGMC_KeepOnlyBobShimmerFixes", AVSValue(args, 4)).AsClip();
      vi = _cpu->GetVideoInfo();
      const int ed = rep < 10 ? rep : rep / 10, od = rep < 10 ? 0 : rep % 10;
      MvCudaFrameFormat f2;
      CudaLib &lib = CudaLib::get();
      _ok = lib.ok && make_format(vi, _fmt) && make_format(ref->GetVideoInfo(), f2) && same_format(_fmt, f2) && _fmt.bits_per_sample == 8
        && !(ed == 2 || ed == 5 || od >= 3);
      _mask = 1 | ((chroma && _fmt.planes == 3) ? 6 : 0);
      if (_ok)
      {
        // the restore expressions of QTGMC_KeepOnlyBobShimmerFixes (QTGMC.avsi), with the use_expr values it passes
        _lut1 = lut_from_cpu("x 129 scalef < x y range_half < range_half y ? ?", 2, env);
        _lut2 = lut_from_cpu("x 127 scalef > x y range_half > range_half y ? ?", 1, env);
        _ok = _lut1 && _lut2;
      }
      if (_ok)
        _producer = lib.producer_register(&CudaKobs::regen, this);
      if (log_enabled())
        std::fprintf(stderr, "CudaKeepOnlyBobShimmerFixes: %s\n", _ok ? "GPU" : "settings or format not supported, using the CPU function");
    }

    ~CudaKobs()
    {
      CudaLib &lib = CudaLib::get();
      if (_producer)
        lib.producer_unregister(_producer);
      if (_lut1)
        lib.lut2_free(_lut1);
      if (_lut2)
        lib.lut2_free(_lut2);
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

  class CudaRemoveGrain : public GenericVideoFilter
  {
    PClip _cpu;
    MvCudaFrameFormat _fmt = {};
    bool _ok = false;
    int _mode[3];
    unsigned long long _producer = 0;

    int make(int n, IScriptEnvironment *env, MvCudaTokenHeader &h)
    {
      PVideoFrame frames[1] = { child->GetFrame(n, env) };
      return with_inputs(frames, 1, vi, _fmt, env, [&](const unsigned long long *tok) {
        return CudaLib::get().removegrain(tok[0], _mode, _producer, n, &h);
      });
    }

    static int regen(void *user, int frame, void *env, MvCudaTokenHeader *out)
    {
      return static_cast<CudaRemoveGrain *>(user)->make(frame, static_cast<IScriptEnvironment *>(env), *out);
    }

  public:
    CudaRemoveGrain(PClip c, int mode, int modeU, int modeV, IScriptEnvironment *env)
      : GenericVideoFilter(c)
    {
      AVSValue args[4] = { cpu_input(c, env), mode, modeU, modeV };
      const char *names[4] = { nullptr, "mode", "modeU", "modeV" };
      _cpu = env->Invoke("RemoveGrain", AVSValue(args, 4), names).AsClip();
      vi = _cpu->GetVideoInfo();
      _mode[0] = mode;
      _mode[1] = modeU;
      _mode[2] = modeV;
      CudaLib &lib = CudaLib::get();
      _ok = lib.ok && make_format(vi, _fmt) && _fmt.bits_per_sample == 8;
      for (int p = 0; p < _fmt.planes; p++)
        _ok = _ok && (_mode[p] == -1 || _mode[p] == 0 || _mode[p] == 12 || _mode[p] == 20
          || (_mode[p] == 11 && (env->GetCPUFlags() & CPUF_SSE4_1))); // RgTools' SSE4.1 mode 11 is mode 12
      if (_ok)
        _producer = lib.producer_register(&CudaRemoveGrain::regen, this);
    }

    ~CudaRemoveGrain()
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

AVSValue __cdecl Create_CudaKobs(AVSValue args, void *, IScriptEnvironment *env)
{
  return new CudaKobs(args[0].AsClip(), args[1].AsClip(), args[2].AsInt(1), args[3].AsBool(true), env);
}

AVSValue __cdecl Create_CudaRemoveGrain(AVSValue args, void *, IScriptEnvironment *env)
{
  // RgTools defaults: modeU = mode, modeV = modeU
  const int mode = args[1].AsInt(2);
  const int modeU = args[2].AsInt(mode);
  const int modeV = args[3].AsInt(modeU);
  return new CudaRemoveGrain(args[0].AsClip(), mode, modeU, modeV, env);
}
