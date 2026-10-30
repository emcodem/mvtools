// QTGMC's output side (QTGMC.avsi "Create basic output" .. "Add back any extracted noise") as fused GPU steps; each filter's
// CPU fallback is exactly the chain of CPU filters it replaces (inputs through CudaDownload), which also runs for what the
// kernels do not cover (other modes; formats: SharpLimit with tables and AddNoise up to 10 bit, SharpLimit smode 0 up to
// 15 bit, BackBlend up to 16 bit, YUV / Y).
//
// CudaQtgmcSharpLimit(edi, degrain1, fComp1, bComp1, resharpExpr, rgBlur, Sovs, degrain2, fComp3, bComp3, tr1 = 1,
// smode = 2, precise = false, nref = 3): binomial1 .. sharpLimit1 in one step (mvcuda_qtgmc_sharp, mvcuda_kobs.cu):
//   binomial1 = TR1 0: edi;  1: degrain1.Merge(edi, 0.25);  2: degrain1.Merge(degrain2, 0.2).Merge(edi, 0.0625)
//   SMode 0: resharp = binomial1
//   SMode 1: resharp = binomial1.mt_lutxy(binomial1.RemoveGrain(rgBlur), resharpExpr, use_expr=1, U=3,V=3)
//   SMode 2: vresharp = Merge(binomial1.mt_expand(mode="vertical", U=3,V=3), binomial1.mt_inpand(mode="vertical", U=3,V=3))
//            Precise: vresharp = vresharp.mt_lutxy(binomial1, "x y < x 1 scalef + x y > x 1 scalef - x ? ?", use_expr=2, U=3,V=3)
//            resharp = binomial1.mt_lutxy(vresharp.RemoveGrain(rgBlur), resharpExpr, use_expr=1, U=3,V=3)
//   nref 3:  tMax = mt_logic(mt_logic(edi, fComp1, "max", U=3,V=3), bComp1, "max", U=3,V=3) (nref 5: then fComp3, bComp3;
//            tMin likewise with "min"); result = resharp.mt_clamp(tMax, tMin, Sovs, Sovs, U=3,V=3);  nref 0: resharp
// With smode = 0 and nref = 0 it is binomial1 itself (QTGMC's lossed1, the reference of the back blend).

#include "CudaCommon.h"
#include "Resample.h"

#include <algorithm>
#include <vector>

using namespace cf;

namespace
{
  PClip invoke(IScriptEnvironment *env, const char *name, std::initializer_list<AVSValue> args, std::initializer_list<const char *> names)
  {
    AVSValue a[12];
    const char *nm[12];
    int k = 0;
    auto ni = names.begin();
    for (const AVSValue &v : args)
    {
      a[k] = v;
      nm[k++] = *ni++;
    }
    return env->Invoke(name, AVSValue(a, k), nm).AsClip();
  }

  // QTGMC's temporal limit: resharp.mt_clamp(tMax, tMin, sovs, sovs, U=3,V=3) with tMax / tMin over refs (CPU clips), in
  // QTGMC.avsi's order (edi, fComp1, bComp1, fComp3, bComp3)
  PClip cpu_tclamp(PClip c, const PClip *refs, int nref, int sovs, IScriptEnvironment *env)
  {
    const AVSValue p3 = 3;
    PClip tmax = refs[0], tmin = refs[0];
    for (int j = 1; j < nref; j++)
    {
      tmax = invoke(env, "mt_logic", { tmax, refs[j], "max", p3, p3, p3 }, { nullptr, nullptr, "mode", "Y", "U", "V" });
      tmin = invoke(env, "mt_logic", { tmin, refs[j], "min", p3, p3, p3 }, { nullptr, nullptr, "mode", "Y", "U", "V" });
    }
    return invoke(env, "mt_clamp", { c, tmax, tmin, sovs, sovs, p3, p3, p3 }, { nullptr, nullptr, nullptr, "overshoot", "undershoot", "Y", "U", "V" });
  }

  // RemoveGrain modes the GPU computes: 12, and 11 where RgTools runs its SSE4.1 code (identical to 12 there)
  bool rg_blur_ok(int rg_blur, IScriptEnvironment *env)
  {
    return rg_blur == 12 || (rg_blur == 11 && (env->GetCPUFlags() & CPUF_SSE4_1));
  }

  // the GPU output frame of a token-producing filter, or the CPU result
  template <class Make>
  PVideoFrame token_or_cpu(const char *who, bool ok, const PClip &cpu, const PClip &child, const VideoInfo &vi, int n, IScriptEnvironment *env, Make make)
  {
    if (!ok)
      return cpu->GetFrame(n, env);
    MvCudaTokenHeader h;
    if (const int rc = make(h))
    {
      if (log_enabled())
        std::fprintf(stderr, "%s: frame %d rc=%d -> CPU filters\n", who, n, rc);
      return cpu->GetFrame(n, env); // exact CPU result (its inputs go through CudaDownload)
    }
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrameP(vi, &src);
    write_token(dst, h);
    return dst;
  }

  class CudaSharpLimit : public GenericVideoFilter
  {
  public:
    PClip _d1, _d2, _cpu;
    PClip _refs[5];
    MvCudaFrameFormat _fmt = {};
    bool _ok = false;
    MvCudaQtgmcSharp _q = {};
    unsigned long long _producer = 0;

    int make(int n, IScriptEnvironment *env, MvCudaTokenHeader &h)
    {
      // edi, degrain1, degrain2, refs
      PVideoFrame frames[8];
      int k = 0;
      frames[k++] = child->GetFrame(n, env);
      frames[k++] = _d1->GetFrame(n, env);
      frames[k++] = _d2->GetFrame(n, env);
      for (int j = 0; j < _q.nref; j++)
        frames[k++] = _refs[j]->GetFrame(n, env);
      return with_inputs(frames, k, vi, _fmt, env, [&](const unsigned long long *tok) {
        return CudaLib::get().qtgmc_sharp(&_q, tok[0], tok[1], tok[2], tok + 3, _producer, n, &h);
      });
    }

    static int regen(void *user, int frame, void *env, MvCudaTokenHeader *out)
    {
      return static_cast<CudaSharpLimit *>(user)->make(frame, static_cast<IScriptEnvironment *>(env), *out);
    }

    CudaSharpLimit(PClip edi, PClip d1, PClip d2, PClip fc1, PClip bc1, PClip fc3, PClip bc3, const char *expr, int rg_blur, int sovs,
      int tr1, int smode, bool precise, int nref, IScriptEnvironment *env)
      : GenericVideoFilter(edi), _d1(d1), _d2(d2)
    {
      if (tr1 < 0 || tr1 > 2 || smode < 0 || smode > 2 || !(nref == 0 || nref == 3 || nref == 5))
        env->ThrowError("CudaQtgmcSharpLimit: tr1 0..2, smode 0..2, nref 0 / 3 / 5");
      _refs[0] = edi;
      _refs[1] = fc1;
      _refs[2] = bc1;
      _refs[3] = fc3;
      _refs[4] = bc3;
      // the CPU chain, as QTGMC.avsi writes it
      const PClip e = cpu_input(edi, env);
      const AVSValue p3 = 3;
      PClip binomial = e;
      if (tr1 == 1)
        binomial = invoke(env, "Merge", { cpu_input(d1, env), e, 0.25 }, { nullptr, nullptr, "weight" });
      else if (tr1 == 2)
        binomial = invoke(env, "Merge", { invoke(env, "Merge", { cpu_input(d1, env), cpu_input(d2, env), 0.2 }, { nullptr, nullptr, "weight" }), e, 0.0625 },
          { nullptr, nullptr, "weight" });
      static const char *precise_expr = "x y < x 1 scalef + x y > x 1 scalef - x ? ?";
      PClip resharp = binomial;
      if (smode == 1)
        resharp = invoke(env, "mt_lutxy", { binomial, invoke(env, "RemoveGrain", { binomial, rg_blur }, { nullptr, "mode" }), expr, 1, p3, p3, p3 },
          { nullptr, nullptr, "expr", "use_expr", "Y", "U", "V" });
      else if (smode == 2)
      {
        const PClip vexp = invoke(env, "mt_expand", { binomial, "vertical", p3, p3, p3 }, { nullptr, "mode", "Y", "U", "V" });
        const PClip vinp = invoke(env, "mt_inpand", { binomial, "vertical", p3, p3, p3 }, { nullptr, "mode", "Y", "U", "V" });
        PClip vresharp = invoke(env, "Merge", { vexp, vinp }, { nullptr, nullptr });
        if (precise)
          vresharp = invoke(env, "mt_lutxy", { vresharp, binomial, precise_expr, 2, p3, p3, p3 }, { nullptr, nullptr, "expr", "use_expr", "Y", "U", "V" });
        const PClip rg = invoke(env, "RemoveGrain", { vresharp, rg_blur }, { nullptr, "mode" });
        resharp = invoke(env, "mt_lutxy", { binomial, rg, expr, 1, p3, p3, p3 }, { nullptr, nullptr, "expr", "use_expr", "Y", "U", "V" });
      }
      if (nref)
      {
        PClip cr[5];
        for (int j = 0; j < nref; j++)
          cr[j] = cpu_input(_refs[j], env);
        _cpu = cpu_tclamp(resharp, cr, nref, sovs, env);
      }
      else
        _cpu = resharp;
      vi = _cpu->GetVideoInfo();

      // Merge weights: the weighted mode of CudaMerge (below 16 bit: no scalar-tail difference)
      auto weight = [](float w, int &wi, int &inv) {
        wi = (int)(w * 32767.0f + 0.5f);
        inv = 32767 - wi;
      };
      _q.tr1 = tr1;
      weight(0.2f, _q.w1, _q.inv1);
      weight(tr1 == 2 ? 0.0625f : 0.25f, _q.w2, _q.inv2);
      _q.smode = smode;
      _q.nref = nref;
      CudaLib &lib = CudaLib::get();
      MvCudaFrameFormat f2;
      _ok = lib.ok && (smode == 0 || rg_blur_ok(rg_blur, env)) && sovs >= 0 && sovs <= 255 && make_format(vi, _fmt);
      const int bits = _fmt.bits_per_sample;
      _ok = _ok && (smode ? bits <= 10 : bits < 16); // tables up to 10 bit; 16-bit Merge rounds differently
      _q.sovs = _ok ? scale_i8(sovs, bits) : 0;   // mt_clamp's overshoot in sample units
      for (PClip c : { d1, d2, fc1, bc1, fc3, bc3 })
        _ok = _ok && make_format(c->GetVideoInfo(), f2) && same_format(_fmt, f2);
      if (_ok && smode)
        _ok = (_q.lut = lut_from_cpu(expr, 1, env, bits)) != 0;
      if (_ok && smode == 2 && precise)
        _ok = (_q.lut_precise = lut_from_cpu(precise_expr, 2, env, bits)) != 0;
      if (_ok)
        _producer = lib.producer_register(&CudaSharpLimit::regen, this);
      if (log_enabled())
        std::fprintf(stderr, "CudaQtgmcSharpLimit: %s\n", _ok ? "GPU" : "using the CPU filters");
    }

    ~CudaSharpLimit()
    {
      CudaLib &lib = CudaLib::get();
      if (_producer)
        lib.producer_unregister(_producer);
      if (_q.lut)
        lib.lut2_free(_q.lut);
      if (_q.lut_precise)
        lib.lut2_free(_q.lut_precise);
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env) override
    {
      return token_or_cpu("CudaQtgmcSharpLimit", _ok, _cpu, child, vi, n, env, [&](MvCudaTokenHeader &h) { return make(n, env, h); });
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) override
    {
      return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }
  };

  // CudaQtgmcBackBlend(thin, lossed, src_width, src_height, edi, fComp1, bComp1, fComp3, bComp3, nref = 0, sovs = 0): QTGMC's
  // back blend (Sbb), followed with nref > 0 by the temporal limit of CudaQtgmcSharpLimit:
  //   thin.mt_makediff(mt_makediff(thin, lossed, U=1,V=1).RemoveGrain(12, -1).GaussResize(w,h, 0,0, src_width,src_height, p=5), U=2,V=2)
  // (QTGMC passes src_width / src_height = w + epsilon / h + epsilon). Only the luma of the inner chain matters (the outer
  // mt_makediff copies thin's chroma), so the GPU resizes the luma alone (mvcuda_qtgmc_backblend, mvcuda_resample.cu).
  class CudaBackBlend : public GenericVideoFilter
  {
  public:
    PClip _lossed, _cpu;
    PClip _refs[5];
    int _nref = 0, _sovs = 0;
    MvCudaFrameFormat _fmt = {};
    bool _ok = false, _h_first = true;
    unsigned long long _ph = 0, _pv = 0, _producer = 0;

    int make(int n, IScriptEnvironment *env, MvCudaTokenHeader &h)
    {
      PVideoFrame frames[7];
      int k = 0;
      frames[k++] = child->GetFrame(n, env);
      frames[k++] = _lossed->GetFrame(n, env);
      for (int j = 0; j < _nref; j++)
        frames[k++] = _refs[j]->GetFrame(n, env);
      return with_inputs(frames, k, vi, _fmt, env, [&](const unsigned long long *tok) {
        return CudaLib::get().qtgmc_backblend(tok[0], tok[1], _ph, _pv, _h_first ? 1 : 0, tok + 2, _nref, _sovs, _producer, n, &h);
      });
    }

    static int regen(void *user, int frame, void *env, MvCudaTokenHeader *out)
    {
      return static_cast<CudaBackBlend *>(user)->make(frame, static_cast<IScriptEnvironment *>(env), *out);
    }

    CudaBackBlend(PClip thin, PClip lossed, double src_w, double src_h, const PClip refs[5], int nref, int sovs, IScriptEnvironment *env)
      : GenericVideoFilter(thin), _lossed(lossed), _nref(nref), _sovs(sovs)
    {
      if (!(nref == 0 || nref == 3 || nref == 5))
        env->ThrowError("CudaQtgmcBackBlend: nref 0 / 3 / 5");
      for (int j = 0; j < nref; j++)
        if (!refs[j])
          env->ThrowError("CudaQtgmcBackBlend: nref %d needs edi, fComp1, bComp1%s", nref, nref > 3 ? ", fComp3, bComp3" : "");
      for (int j = 0; j < 5; j++)
        _refs[j] = refs[j];
      // the CPU chain, as QTGMC.avsi writes it
      const PClip t = cpu_input(thin, env), l = cpu_input(lossed, env);
      const int w = vi.width, h = vi.height;
      const PClip d = invoke(env, "mt_makediff", { t, l, 1, 1 }, { nullptr, nullptr, "U", "V" });
      const PClip rg = invoke(env, "RemoveGrain", { d, 12, -1 }, { nullptr, "mode", "modeU" });
      const PClip g = invoke(env, "GaussResize", { rg, w, h, 0.0, 0.0, src_w, src_h, 5.0 },
        { nullptr, nullptr, nullptr, "src_left", "src_top", "src_width", "src_height", "p" });
      PClip bb = invoke(env, "mt_makediff", { t, g, 2, 2 }, { nullptr, nullptr, "U", "V" });
      if (nref)
      {
        PClip cr[5];
        for (int j = 0; j < nref; j++)
          cr[j] = cpu_input(refs[j], env);
        bb = cpu_tclamp(bb, cr, nref, sovs, env);
      }
      _cpu = bb;
      vi = _cpu->GetVideoInfo();

      CudaLib &lib = CudaLib::get();
      MvCudaFrameFormat f2;
      _ok = lib.ok && make_format(vi, _fmt) && sovs >= 0 && sovs <= 255 && make_format(lossed->GetVideoInfo(), f2) && same_format(_fmt, f2);
      const int bits = _fmt.bits_per_sample;
      _sovs = _ok ? scale_i8(sovs, bits) : 0; // mt_clamp's overshoot in sample units
      for (int j = 0; j < nref && _ok; j++)
        _ok = make_format(refs[j]->GetVideoInfo(), f2) && same_format(_fmt, f2);
      // the luma programs of GaussResize(w, h, 0, 0, src_w, src_h, p=5) (FilteredResize::CreateResize, see CudaResize.cpp):
      // both passes run (src_w != w), keep_center, the default b / s
      _ok = _ok && src_w != w && src_h != h && src_w > 0 && src_h > 0;
      if (_ok)
      {
        const GaussianKernel kernel(5.0, 2.0, 4.0);
        _h_first = !(src_h * w < src_w * h);
        const ResampleProgram ph = make_program(kernel, w, 0.0, src_w, w, bits, 0.5, 0.5);
        const ResampleProgram pv = make_program(kernel, h, 0.0, src_h, h, bits, 0.5, 0.5);
        _ph = lib.resample_program_create(ph.target_size, ph.taps, ph.offset.data(), ph.coeff.data());
        _pv = lib.resample_program_create(pv.target_size, pv.taps, pv.offset.data(), pv.coeff.data());
        _ok = _ph && _pv;
      }
      if (_ok)
        _producer = lib.producer_register(&CudaBackBlend::regen, this);
      if (log_enabled())
        std::fprintf(stderr, "CudaQtgmcBackBlend: %s\n", _ok ? "GPU" : "using the CPU filters");
    }

    ~CudaBackBlend()
    {
      CudaLib &lib = CudaLib::get();
      if (_producer)
        lib.producer_unregister(_producer);
      if (_ph)
        lib.resample_program_free(_ph);
      if (_pv)
        lib.resample_program_free(_pv);
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env) override
    {
      return token_or_cpu("CudaQtgmcBackBlend", _ok, _cpu, child, vi, n, env, [&](MvCudaTokenHeader &h) { return make(n, env, h); });
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) override
    {
      return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }
  };

  // CudaQtgmcAddNoise(c, noise, expr, use_expr = 0, clamp_float = false, lutUV = 1, addUV = 2): QTGMC's grain / noise restore
  //   c.mt_adddiff(noise.mt_lut(expr, use_expr=use_expr, clamp_float=clamp_float, U=lutUV,V=lutUV), U=addUV,V=addUV)
  // as one table lookup per sample: the plugin runs exactly that chain on the n x n pattern (x = c, y = noise; n = 256 at 8
  // bit, 1024 at 10 bit) and uses the result as the table (mvcuda_mt, MVCUDA_MT_LUT). Chroma on the GPU for addUV 2 (copy c) or lutUV = addUV = 3.
  class CudaAddNoise : public GenericVideoFilter
  {
  public:
    PClip _noise, _cpu;
    MvCudaFrameFormat _fmt = {};
    bool _ok = false;
    int _op[3] = { MVCUDA_MT_LUT, MVCUDA_MT_COPY_A, MVCUDA_MT_COPY_A };
    unsigned long long _lut = 0, _producer = 0;

    int make(int n, IScriptEnvironment *env, MvCudaTokenHeader &h)
    {
      PVideoFrame frames[2] = { child->GetFrame(n, env), _noise->GetFrame(n, env) };
      return with_inputs(frames, 2, vi, _fmt, env, [&](const unsigned long long *tok) {
        return CudaLib::get().mt(_op, tok[0], tok[1], tok[1], _lut, 0, 0, _producer, n, &h);
      });
    }

    static int regen(void *user, int frame, void *env, MvCudaTokenHeader *out)
    {
      return static_cast<CudaAddNoise *>(user)->make(frame, static_cast<IScriptEnvironment *>(env), *out);
    }

    CudaAddNoise(PClip c, PClip noise, const char *expr, int use_expr, bool clamp_float, int lut_uv, int add_uv, IScriptEnvironment *env)
      : GenericVideoFilter(c), _noise(noise)
    {
      auto chain = [&](PClip x, PClip y) {
        const PClip l = invoke(env, "mt_lut", { y, expr, use_expr, clamp_float, lut_uv, lut_uv },
          { nullptr, "expr", "use_expr", "clamp_float", "U", "V" });
        return invoke(env, "mt_adddiff", { x, l, add_uv, add_uv }, { nullptr, nullptr, "U", "V" });
      };
      _cpu = chain(cpu_input(c, env), cpu_input(noise, env));
      vi = _cpu->GetVideoInfo();
      CudaLib &lib = CudaLib::get();
      MvCudaFrameFormat f2;
      _ok = lib.ok && make_format(vi, _fmt) && _fmt.bits_per_sample <= 10 && make_format(noise->GetVideoInfo(), f2) && same_format(_fmt, f2)
        && (add_uv == 2 || (add_uv == 3 && lut_uv == 3));
      if (add_uv == 3)
        _op[1] = _op[2] = MVCUDA_MT_LUT;
      if (_ok)
        _ok = (_lut = lut_from_frame(chain(pattern_clip(false, _fmt.bits_per_sample), pattern_clip(true, _fmt.bits_per_sample))->GetFrame(0, env),
          _fmt.bits_per_sample)) != 0;
      if (_ok)
        _producer = lib.producer_register(&CudaAddNoise::regen, this);
      if (log_enabled())
        std::fprintf(stderr, "CudaQtgmcAddNoise: %s\n", _ok ? "GPU" : "using the CPU filters");
    }

    ~CudaAddNoise()
    {
      CudaLib &lib = CudaLib::get();
      if (_producer)
        lib.producer_unregister(_producer);
      if (_lut)
        lib.lut2_free(_lut);
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env) override
    {
      return token_or_cpu("CudaQtgmcAddNoise", _ok, _cpu, child, vi, n, env, [&](MvCudaTokenHeader &h) { return make(n, env, h); });
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) override
    {
      return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }
  };
}

AVSValue __cdecl Create_CudaQtgmcSharpLimit(AVSValue args, void *, IScriptEnvironment *env)
{
  // edi, degrain1, fComp1, bComp1, resharpExpr, rgBlur, Sovs, degrain2, fComp3, bComp3, tr1, smode, precise, nref
  const PClip edi = args[0].AsClip();
  auto opt = [&](int i) { return args[i].Defined() ? args[i].AsClip() : edi; };
  return new CudaSharpLimit(edi, args[1].AsClip(), opt(7), args[2].AsClip(), args[3].AsClip(), opt(8), opt(9), args[4].AsString(),
    args[5].AsInt(12), args[6].AsInt(0), args[10].AsInt(1), args[11].AsInt(2), args[12].AsBool(false), args[13].AsInt(3), env);
}

AVSValue __cdecl Create_CudaQtgmcBackBlend(AVSValue args, void *, IScriptEnvironment *env)
{
  // thin, lossed, src_width, src_height, edi, fComp1, bComp1, fComp3, bComp3, nref, sovs
  PClip refs[5];
  for (int j = 0; j < 5; j++)
    refs[j] = args[4 + j].Defined() ? args[4 + j].AsClip() : nullptr;
  return new CudaBackBlend(args[0].AsClip(), args[1].AsClip(), args[2].AsFloat(), args[3].AsFloat(), refs, args[9].AsInt(0), args[10].AsInt(0), env);
}

AVSValue __cdecl Create_CudaQtgmcAddNoise(AVSValue args, void *, IScriptEnvironment *env)
{
  // c, noise, expr, use_expr, clamp_float, lutUV, addUV
  return new CudaAddNoise(args[0].AsClip(), args[1].AsClip(), args[2].AsString(), args[3].AsInt(0), args[4].AsBool(false),
    args[5].AsInt(1), args[6].AsInt(2), env);
}

// CudaQtgmcBinomial0(b, luma_threshold = 255, chroma_threshold = 0, scenechange = 28, chroma = false): QTGMC's search-clip
// binomial for TR0 = 2, b.TemporalSoften(1, lt, ct, sc, 2).MergeLuma(b.TemporalSoften(2, lt, ct, sc, 2), 0.357).MergeLuma(b,
// 0.125) (chroma = true: Merge, QTGMC's ChromaMotion), in three kernels (mvcuda_temporal_binomial) instead of eight. The CPU
// fallback is that chain (input through CudaDownload).
namespace
{
  class CudaBinomial0 : public GenericVideoFilter
  {
    PClip _cpu;
    MvCudaFrameFormat _fmt = {};
    bool _ok = false;
    int _luma, _chroma, _scenechange, _w1, _inv1, _w2, _inv2, _tail, _merge_chroma, _tail_c;
    unsigned long long _producer = 0;

    int make(int n, IScriptEnvironment *env, MvCudaTokenHeader &h)
    {
      PVideoFrame frames[5];
      for (int k = 0; k < 5; k++)
        frames[k] = child->GetFrame(std::clamp(n - 2 + k, 0, vi.num_frames - 1), env);
      return with_inputs(frames, 5, vi, _fmt, env, [&](const unsigned long long *tok) {
        return CudaLib::get().temporal_binomial(tok, _luma, _chroma, _scenechange, _w1, _inv1, _w2, _inv2, _tail, _merge_chroma, _tail_c,
          _producer, n, &h);
      });
    }

    static int regen(void *user, int frame, void *env, MvCudaTokenHeader *out)
    {
      return static_cast<CudaBinomial0 *>(user)->make(frame, static_cast<IScriptEnvironment *>(env), *out);
    }

  public:
    CudaBinomial0(PClip b, int luma, int chroma, int scenechange, bool merge_chroma, IScriptEnvironment *env)
      : GenericVideoFilter(b), _merge_chroma(merge_chroma ? 1 : 0)
    {
      const PClip cb = cpu_input(b, env);
      const char *ts_names[6] = { nullptr, nullptr, nullptr, nullptr, "scenechange", "mode" };
      AVSValue ts1_args[6] = { cb, 1, luma, chroma, scenechange, 2 }, ts2_args[6] = { cb, 2, luma, chroma, scenechange, 2 };
      const PClip ts1 = env->Invoke("TemporalSoften", AVSValue(ts1_args, 6), ts_names).AsClip();
      const PClip ts2 = env->Invoke("TemporalSoften", AVSValue(ts2_args, 6), ts_names).AsClip();
      const char *m_names[3] = { nullptr, nullptr, "weight" };
      const char *merge = merge_chroma ? "Merge" : "MergeLuma";
      AVSValue m1_args[3] = { ts1, ts2, 0.357 };
      const PClip m1 = env->Invoke(merge, AVSValue(m1_args, 3), m_names).AsClip();
      AVSValue m2_args[3] = { m1, cb, 0.125 };
      _cpu = env->Invoke(merge, AVSValue(m2_args, 3), m_names).AsClip();
      vi = _cpu->GetVideoInfo();
      // as the TemporalSoften constructor and CudaMerge (weighted mode)
      _luma = (int)std::min((unsigned)luma, 255u);
      _chroma = (int)std::min((unsigned)chroma, 255u);
      _scenechange = scenechange >= 255 ? 0 : scenechange;
      _scenechange *= ((vi.width / 32) * 32) * vi.height;
      _w1 = (int)(0.357f * 32767.0f + 0.5f);
      _inv1 = 32767 - _w1;
      _w2 = (int)(0.125f * 32767.0f + 0.5f);
      _inv2 = 32767 - _w2;
      CudaLib &lib = CudaLib::get();
      _ok = lib.ok && make_format(vi, _fmt);
      _tail = _ok ? (_fmt.width * _fmt.bytes_per_sample / 16 * 16) / _fmt.bytes_per_sample : 0;
      const int wc = _ok && _fmt.planes == 3 ? _fmt.width / _fmt.x_ratio_uv : 0;
      _tail_c = (wc * _fmt.bytes_per_sample / 16 * 16) / (_fmt.bytes_per_sample ? _fmt.bytes_per_sample : 1);
      if (_ok)
        _producer = lib.producer_register(&CudaBinomial0::regen, this);
      if (log_enabled())
        std::fprintf(stderr, "CudaQtgmcBinomial0: %s\n", _ok ? "GPU" : "using the CPU filters");
    }

    ~CudaBinomial0()
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

AVSValue __cdecl Create_CudaQtgmcBinomial0(AVSValue args, void *, IScriptEnvironment *env)
{
  return new CudaBinomial0(args[0].AsClip(), args[1].AsInt(255), args[2].AsInt(0), args[3].AsInt(28), args[4].AsBool(false), env);
}