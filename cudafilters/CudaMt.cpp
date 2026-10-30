// Single masktools2 operations on the GPU (8 to 16 bit, CudaLutxy up to 10 bit), bit for bit with masktools2 2.2.30 as
// it runs on SSE2+ CPUs; the arithmetic is mvcuda_mt (mvcuda_kobs.cu):
//   CudaLogic(a, b, mode, Y, U, V)                       mt_logic "min" / "max"
//   CudaClamp(c, bright, dark, overshoot, undershoot, Y, U, V)   mt_clamp
//   CudaExpand(c, mode, Y, U, V) / CudaInpand(...)       mt_expand / mt_inpand, mode "vertical"
//   CudaLutxy(x, y, expr, use_expr, Y, U, V)             mt_lutxy through a table made by the installed mt_lutxy
// Plane modes 3 (process) and 2 (copy the first clip) run on the GPU; anything else (or other modes / formats) uses the
// CPU filter, whose inputs are read through CudaDownload.

#include "CudaCommon.h"

using namespace cf;

namespace
{
  class CudaMtOp : public GenericVideoFilter
  {
    PClip _b, _c, _cpu;
    MvCudaFrameFormat _fmt = {};
    bool _ok = false;
    int _op[3] = { MVCUDA_MT_COPY_A, MVCUDA_MT_COPY_A, MVCUDA_MT_COPY_A };
    unsigned long long _lut = 0, _producer = 0;
    int _p1 = 0, _p2 = 0;

    int make(int n, IScriptEnvironment *env, MvCudaTokenHeader &h)
    {
      PVideoFrame frames[3] = { child->GetFrame(n, env), _b->GetFrame(n, env), _c->GetFrame(n, env) };
      return with_inputs(frames, 3, vi, _fmt, env, [&](const unsigned long long *tok) {
        return CudaLib::get().mt(_op, tok[0], tok[1], tok[2], _lut, _p1, _p2, _producer, n, &h);
      });
    }

    static int regen(void *user, int frame, void *env, MvCudaTokenHeader *out)
    {
      return static_cast<CudaMtOp *>(user)->make(frame, static_cast<IScriptEnvironment *>(env), *out);
    }

  public:
    // op: the operation of a processed plane; modes: masktools Y / U / V; cpu: the CPU filter (same arguments)
    CudaMtOp(PClip a, PClip b, PClip c, int op, const int modes[3], PClip cpu, unsigned long long lut, int p1, int p2, const char *who)
      : GenericVideoFilter(a), _b(b), _c(c), _cpu(cpu), _lut(lut), _p1(p1), _p2(p2)
    {
      vi = _cpu->GetVideoInfo();
      CudaLib &lib = CudaLib::get();
      _ok = lib.ok && make_format(vi, _fmt) && (op != MVCUDA_MT_LUT || lut != 0);
      for (int p = 0; p < _fmt.planes && _ok; p++)
      {
        if (modes[p] == 3)
          _op[p] = op;
        else if (modes[p] == 2)
          _op[p] = MVCUDA_MT_COPY_A;
        else
          _ok = false;
      }
      if (_ok)
        _producer = lib.producer_register(&CudaMtOp::regen, this);
      if (log_enabled())
        std::fprintf(stderr, "%s: %s\n", who, _ok ? "GPU" : "using the CPU filter");
    }

    ~CudaMtOp()
    {
      if (_producer)
        CudaLib::get().producer_unregister(_producer);
      if (_lut)
        CudaLib::get().lut2_free(_lut);
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env) override
    {
      if (!_ok)
        return _cpu->GetFrame(n, env);
      MvCudaTokenHeader h;
      if (make(n, env, h) != 0)
        return _cpu->GetFrame(n, env); // exact CPU result (its inputs go through CudaDownload)
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

  // masktools defaults: Y = 3, U = V = 1
  void plane_modes(const AVSValue &y, const AVSValue &u, const AVSValue &v, int m[3])
  {
    m[0] = y.AsInt(3);
    m[1] = u.AsInt(1);
    m[2] = v.AsInt(1);
  }

  // invokes the CPU filter with the clips behind CudaDownload and the given named arguments (undefined ones left out)
  PClip cpu_filter(const char *name, int nclips, const AVSValue &args, const char *const *names, int nargs, IScriptEnvironment *env)
  {
    AVSValue a[12];
    const char *n[12];
    int k = 0;
    for (int i = 0; i < nargs; i++)
    {
      if (i < nclips)
      {
        a[k] = cpu_input(args[i].AsClip(), env);
        n[k++] = nullptr;
      }
      else if (args[i].Defined())
      {
        a[k] = args[i];
        n[k++] = names[i];
      }
    }
    return env->Invoke(name, AVSValue(a, k), n).AsClip();
  }
}

AVSValue __cdecl Create_CudaLogic(AVSValue args, void *, IScriptEnvironment *env)
{
  // c, c, mode, Y, U, V
  const char *names[6] = { nullptr, nullptr, "mode", "Y", "U", "V" };
  PClip cpu = cpu_filter("mt_logic", 2, args, names, 6, env);
  const char *mode = args[2].AsString("and");
  int m[3];
  plane_modes(args[3], args[4], args[5], m);
  const int op = !_stricmp(mode, "min") ? MVCUDA_MT_MIN : !_stricmp(mode, "max") ? MVCUDA_MT_MAX : -1;
  if (op < 0)
    m[0] = m[1] = m[2] = 0; // other logic modes: the CPU filter
  return new CudaMtOp(args[0].AsClip(), args[1].AsClip(), args[1].AsClip(), op, m, cpu, 0, 0, 0, "CudaLogic");
}

AVSValue __cdecl Create_CudaClamp(AVSValue args, void *, IScriptEnvironment *env)
{
  // c, bright, dark, overshoot, undershoot, Y, U, V
  const char *names[8] = { nullptr, nullptr, nullptr, "overshoot", "undershoot", "Y", "U", "V" };
  PClip cpu = cpu_filter("mt_clamp", 3, args, names, 8, env);
  int m[3];
  plane_modes(args[5], args[6], args[7], m);
  const int over = args[3].AsInt(0), under = args[4].AsInt(0);
  if (over < 0 || under < 0 || over > 255 || under > 255)
    m[0] = m[1] = m[2] = 0;
  // overshoot / undershoot in sample units, as masktools scales them (paramscale "i8")
  const int bits = cpu->GetVideoInfo().BitsPerComponent();
  return new CudaMtOp(args[0].AsClip(), args[1].AsClip(), args[2].AsClip(), MVCUDA_MT_CLAMP, m, cpu, 0, scale_i8(over, bits),
    scale_i8(under, bits), "CudaClamp");
}

static AVSValue create_morph(AVSValue args, IScriptEnvironment *env, bool expand)
{
  // c, mode, Y, U, V
  const char *names[5] = { nullptr, "mode", "Y", "U", "V" };
  PClip cpu = cpu_filter(expand ? "mt_expand" : "mt_inpand", 1, args, names, 5, env);
  int m[3];
  plane_modes(args[2], args[3], args[4], m);
  if (_stricmp(args[1].AsString("square"), "vertical") != 0)
    m[0] = m[1] = m[2] = 0; // only the vertical mode on the GPU
  return new CudaMtOp(args[0].AsClip(), args[0].AsClip(), args[0].AsClip(), expand ? MVCUDA_MT_VEXPAND : MVCUDA_MT_VINPAND, m, cpu, 0, 0, 0,
    expand ? "CudaExpand" : "CudaInpand");
}

AVSValue __cdecl Create_CudaExpand(AVSValue args, void *, IScriptEnvironment *env)
{
  return create_morph(args, env, true);
}

AVSValue __cdecl Create_CudaInpand(AVSValue args, void *, IScriptEnvironment *env)
{
  return create_morph(args, env, false);
}

AVSValue __cdecl Create_CudaLutxy(AVSValue args, void *, IScriptEnvironment *env)
{
  // c, c, expr, use_expr, Y, U, V
  const char *names[7] = { nullptr, nullptr, "expr", "use_expr", "Y", "U", "V" };
  PClip cpu = cpu_filter("mt_lutxy", 2, args, names, 7, env);
  int m[3];
  plane_modes(args[4], args[5], args[6], m);
  const int bits = cpu->GetVideoInfo().BitsPerComponent();
  const unsigned long long lut = CudaLib::get().ok && bits <= 10 ? lut_from_cpu(args[2].AsString("x"), args[3].AsInt(0), env, bits) : 0;
  return new CudaMtOp(args[0].AsClip(), args[1].AsClip(), args[1].AsClip(), MVCUDA_MT_LUT, m, cpu, lut, 0, 0, "CudaLutxy");
}
