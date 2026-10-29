// CudaNNEDI3: plugins_JPSDR 4.1.0 nnedi3 on the GPU (mvtools_cuda.dll mvcuda_nnedi3), bit for bit for what the plugin runs
// on this machine when that is its opt 6 path (AVX2 + FMA3, no AVX-512): 8-bit planar input, dh = false, new prescreener
// (pscrn 2..4), int16 predictor with the fastest exp (fapprox with bits 2 and 8 set; default 15), qual 1 / 2, any nsize /
// nns / etype. Anything else, or a CPU whose rcpps / rsqrtps tables do not have the expected structure, uses the plugin.
//
// The network weights are read from the loaded plugins_JPSDR.dll (resource 101, as the plugin does) and prepared with
// the plugin's own code (constructor of nnedi3.cpp, same double arithmetic), in plain order instead of the asm order.

#include "CudaCommon.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <vector>

namespace
{
  using namespace cf;

  const int xdiaTable[7] = { 8, 16, 32, 48, 8, 16, 32 };
  const int ydiaTable[7] = { 6, 6, 6, 6, 4, 4, 4 };
  const int nnsTable[5] = { 16, 32, 64, 128, 256 };
  const int nnsTablePow2[5] = { 4, 5, 6, 7, 8 };

  int roundds(const double f) // nnedi3.cpp
  {
    if (f - floor(f) >= 0.5)
      return std::min((int)ceil(f), 32767);
    return std::max((int)floor(f), -32768);
  }

  unsigned fbits(float f)
  {
    unsigned u;
    std::memcpy(&u, &f, 4);
    return u;
  }
  float ffrom(unsigned u)
  {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  }

  // the CPU's rcpps / rsqrtps tables; ok = they depend only on the top 11 / 10 mantissa bits (and the exponent), which
  // is what the GPU lookup assumes. Measured once per process.
  struct HwTables
  {
    bool ok = false;
    unsigned rcp[2048], rsq[2048];
  };

  const HwTables &hw_tables()
  {
    static HwTables t;
    static std::once_flag once;
    std::call_once(once, [] {
      for (unsigned i = 0; i < 2048; i++)
        t.rcp[i] = fbits(_mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(ffrom(0x3F800000u | (i << 12))))));
      for (unsigned par = 0; par < 2; par++)
        for (unsigned i = 0; i < 1024; i++)
          t.rsq[par * 1024 + i] = fbits(_mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(ffrom((0x3F800000u + (par << 23)) | (i << 13))))));
      bool ok = true;
      for (unsigned m = 0; m < (1u << 23) && ok; m++)
      {
        if (fbits(_mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(ffrom(0x3F800000u | m))))) != t.rcp[m >> 12])
          ok = false;
        for (unsigned par = 0; par < 2 && ok; par++)
          if (fbits(_mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(ffrom((0x3F800000u + (par << 23)) | m))))) != t.rsq[par * 1024 + (m >> 13)])
            ok = false;
      }
      // exponent independence on a sample of the ranges used (|t| + 1, sums of exponentials, variances)
      unsigned seed = 12345;
      for (int i = 0; i < 200000 && ok; i++)
      {
        seed = seed * 1664525u + 1013904223u;
        const unsigned e = 20 + (seed >> 8) % 215, m = (seed * 2654435761u) & 0x7FFFFFu;
        const float x = ffrom((e << 23) | m);
        const int ex = (int)e - 127;
        if (fbits(_mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(x)))) != t.rcp[m >> 12] - ((unsigned)ex << 23))
          ok = false;
        const int par = ex & 1, k = (ex - par) >> 1;
        if (fbits(_mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x)))) != t.rsq[par * 1024 + (m >> 13)] - ((unsigned)k << 23))
          ok = false;
      }
      t.ok = ok;
    });
    return t;
  }

  class CudaNnedi3 : public GenericVideoFilter
  {
    PClip _cpu;
    MvCudaFrameFormat _fmt = {};
    bool _ok = false;
    int _field = 0;     // as the plugin normalises it: 0 / 1 single rate, 2 / 3 double rate (3 = top field first)
    int _mask = 0;
    int _vmin[3] = { 0, 0, 0 }, _vmax[3] = { 255, 255, 255 };
    void *_net = nullptr;
    unsigned long long _producer = 0;
    const char *_why = "";

    int make(int n, IScriptEnvironment *env, MvCudaTokenHeader &h)
    {
      int field_n = _field, x = n;
      if (_field > 1)
      {
        x = n >> 1;
        field_n = (n & 1) ? (_field == 3 ? 0 : 1) : (_field == 3 ? 1 : 0);
      }
      PVideoFrame frames[1] = { child->GetFrame(x, env) };
      return with_inputs(frames, 1, child->GetVideoInfo(), _fmt, env, [&](const unsigned long long *tok) {
        return CudaLib::get().nnedi3(_net, tok[0], field_n, _mask, _vmin, _vmax, _producer, n, &h);
      });
    }

    static int regen(void *user, int frame, void *env, MvCudaTokenHeader *out)
    {
      return static_cast<CudaNnedi3 *>(user)->make(frame, static_cast<IScriptEnvironment *>(env), *out);
    }

    // the plugin's weight preparation for the new prescreener and the int16 predictor, plain order
    bool prepare(int nsize, int nns, int qual, int etype, int pscrn, IScriptEnvironment *env)
    {
      HMODULE hmod = GetModuleHandleW(L"plugins_JPSDR.dll");
      if (!hmod)
      {
        _why = "plugins_JPSDR.dll not loaded";
        return false;
      }
      HRSRC hrsrc = FindResourceW(hmod, MAKEINTRESOURCEW(101), L"BINARY");
      HGLOBAL hglob = hrsrc ? LoadResource(hmod, hrsrc) : nullptr;
      const float *bdata = hglob ? (const float *)LockResource(hglob) : nullptr;
      const DWORD size = hrsrc ? SizeofResource(hmod, hrsrc) : 0;
      const int dims0 = 49 * 4 + 5 * 4 + 9 * 4, dims0new = 4 * 65 + 4 * 5;
      const int dims1 = (xdiaTable[nsize] * ydiaTable[nsize] + 1) << (nnsTablePow2[nns] + 1);
      int dims1tsize = 0, dims1offset = 0;
      for (int j = 0; j < 5; j++)
        for (int i = 0; i < 7; i++)
        {
          if (i == nsize && j == nns)
            dims1offset = dims1tsize;
          dims1tsize += (xdiaTable[i] * ydiaTable[i] + 1) << (nnsTablePow2[j] + 2);
        }
      if (!bdata || size != (DWORD)((dims0 + dims0new * 3 + dims1tsize * 2) * sizeof(float)))
      {
        _why = "weights resource not found";
        return false;
      }

      // prescreener (pscrn >= 2): offt = the source layout, mean removal and 1 / half scaling, int16
      std::vector<short> pre_w(4 * 64);
      std::vector<float> pre_f(28);
      {
        int offt[4 * 64];
        for (int j = 0; j < 4; j++)
          for (int k = 0; k < 64; k++)
            offt[j * 64 + k] = ((k >> 3) << 5) + j * 8 + (k & 7);
        const float *bdw = bdata + (dims0 + dims0new * (pscrn - 2));
        double mean[4] = { 0.0, 0.0, 0.0, 0.0 };
        for (int j = 0; j < 4; j++)
        {
          double cmean = 0.0;
          for (int k = 0; k < 64; k++)
            cmean += bdw[offt[j * 64 + k]];
          mean[j] = cmean / 64.0;
        }
        const int prescreener_bits = 8;
        const double half = (((int)1 << prescreener_bits) - 1) / 2.0;
        for (int j = 0; j < 4; j++)
        {
          double mval = 0.0;
          for (int k = 0; k < 64; k++)
            mval = std::max(mval, fabs((bdw[offt[j * 64 + k]] - mean[j]) / half));
          const double scale = 32767.0 / mval;
          for (int k = 0; k < 64; k++)
            pre_w[j * 64 + k] = (short)roundds(((bdw[offt[j * 64 + k]] - mean[j]) / half) * scale);
          pre_f[j] = (float)(mval / 32767.0);
        }
        std::memcpy(pre_f.data() + 4, bdw + 4 * 64, (dims0new - 4 * 64) * sizeof(float));
      }

      // predictor networks (int16): mean removal factored in, softmax global offset removed, scaled to int16
      const int nnst = nnsTable[nns], nnst2 = nnst << 1, asize = xdiaTable[nsize] * ydiaTable[nsize], boff = nnst2 * asize;
      std::vector<short> pw[2];
      std::vector<float> ps[2], pb[2];
      for (int i = 0; i < 2; i++)
      {
        const float *bdataT = bdata + (dims0 + dims0new * 3 + dims1tsize * etype + dims1offset + i * dims1);
        std::vector<double> mean(asize + 1 + nnst2, 0.0);
        int j_a = 0, j_d;
        for (int j = 0; j < nnst2; j++)
        {
          double cmean = 0.0;
          for (int k = 0; k < asize; k++)
            cmean += bdataT[j_a + k];
          mean[asize + 1 + j] = cmean / (double)asize;
          j_a += asize;
        }
        j_a = 0;
        j_d = asize + 1;
        for (int j = 0; j < nnst; j++)
        {
          for (int k = 0; k < asize; k++)
            mean[k] += bdataT[j_a + k] - mean[j_d];
          mean[asize] += bdataT[boff + j];
          j_a += asize;
          j_d++;
        }
        for (int j = 0; j < asize + 1; j++)
          mean[j] /= (double)(nnst);
        pw[i].assign((size_t)nnst2 * asize, 0);
        ps[i].assign(nnst2, 0.0f);
        pb[i].assign(nnst2, 0.0f);
        j_a = 0;
        j_d = asize + 1;
        for (int j = 0; j < nnst; j++) // softmax neurons
        {
          double mval = 0.0;
          for (int k = 0; k < asize; k++)
            mval = std::max(mval, fabs(bdataT[j_a + k] - mean[j_d] - mean[k]));
          const double scale = 32767.0 / mval;
          for (int k = 0; k < asize; k++)
            pw[i][j_a + k] = (short)roundds((bdataT[j_a + k] - mean[j_d] - mean[k]) * scale);
          ps[i][j] = (float)(mval / 32767.0);
          pb[i][j] = (float)(bdataT[boff + j] - mean[asize]);
          j_a += asize;
          j_d++;
        }
        for (int j = nnst; j < nnst2; j++) // elliott neurons
        {
          double mval = 0.0;
          for (int k = 0; k < asize; k++)
            mval = std::max(mval, fabs(bdataT[j_a + k] - mean[j_d]));
          const double scale = 32767.0 / mval;
          for (int k = 0; k < asize; k++)
            pw[i][j_a + k] = (short)roundds((bdataT[j_a + k] - mean[j_d]) * scale);
          ps[i][j] = (float)(mval / 32767.0);
          pb[i][j] = bdataT[boff + j];
          j_a += asize;
          j_d++;
        }
      }

      const HwTables &t = hw_tables();
      MvCudaNnediParams p = {};
      p.xdia = xdiaTable[nsize];
      p.ydia = ydiaTable[nsize];
      p.nns = nnst;
      p.qual = qual;
      p.pre_w = pre_w.data();
      p.pre_f = pre_f.data();
      for (int i = 0; i < 2; i++)
      {
        p.pred_w[i] = pw[i].data();
        p.pred_scale[i] = ps[i].data();
        p.pred_bias[i] = pb[i].data();
      }
      p.rcp_tab = t.rcp;
      p.rsq_tab = t.rsq;
      _net = CudaLib::get().nnedi3_create(&p);
      if (!_net)
        _why = "network upload failed";
      return _net != nullptr;
    }

  public:
    // args as nnedi3: c, field, dh, Y, U, V, nsize, nns, qual, etype, pscrn, threads, opt, fapprox
    CudaNnedi3(const AVSValue &args, IScriptEnvironment *env)
      : GenericVideoFilter(args[0].AsClip())
    {
      const char *names[14] = { nullptr, "field", "dh", "Y", "U", "V", "nsize", "nns", "qual", "etype", "pscrn", "threads", "opt", "fapprox" };
      AVSValue cargs[14];
      int na = 0;
      const char *cnames[14];
      cargs[na] = cpu_input(args[0].AsClip(), env);
      cnames[na++] = nullptr;
      for (int i = 1; i < 14; i++)
        if (args[i].Defined())
        {
          cargs[na] = args[i];
          cnames[na++] = names[i];
        }
      _cpu = env->Invoke("nnedi3", AVSValue(cargs, na), cnames).AsClip();
      vi = _cpu->GetVideoInfo();

      const VideoInfo &svi = child->GetVideoInfo();
      int field = args[1].AsInt(-1);
      const bool dh = args[2].AsBool(false);
      const bool Y = args[3].AsBool(true), U = args[4].AsBool(true), V = args[5].AsBool(true);
      const int nsize = args[6].AsInt(6), nns = args[7].AsInt(1), qual = args[8].AsInt(1), etype = args[9].AsInt(0);
      const int pscrn = args[10].AsInt(2), opt = args[12].AsInt(0), fapprox = args[13].AsInt(15);
      if (field == -2)
        field = child->GetParity(0) ? 3 : 2;
      else if (field == -1)
        field = child->GetParity(0) ? 1 : 0;
      _field = field;

      // the CPU path the plugin takes here (its constructor's auto detection)
      const int cpuf = env->GetCPUFlags();
      const bool avx512 = (cpuf & CPUF_AVX512F) && (cpuf & CPUF_AVX512DQ) && (cpuf & CPUF_AVX512BW);
      const int auto_opt = avx512 ? 8 : ((cpuf & CPUF_FMA4) && (cpuf & CPUF_AVX2)) ? 7 : ((cpuf & CPUF_FMA3) && (cpuf & CPUF_AVX2)) ? 6 : 0;
      const int eff_opt = opt == 0 ? auto_opt : opt;

      CudaLib &lib = CudaLib::get();
      if (!lib.ok)
        _why = "GPU library not usable";
      else if (!make_format(svi, _fmt) || _fmt.bits_per_sample != 8 || !svi.IsPlanar() || svi.IsRGB())
        _why = "only 8-bit planar YUV / Y";
      else if (dh || field < 0 || field > 3)
        _why = "dh not supported";
      else if (eff_opt != 6 && eff_opt != 7)
        _why = "the plugin does not take its AVX2 + FMA path on this CPU (AVX-512 or older CPU)";
      else if (pscrn < 2 || pscrn > 4 || !(fapprox & 2) || (fapprox & 12) < 8 || qual < 1 || qual > 2 || etype < 0 || etype > 1
        || nsize < 0 || nsize > 6 || nns < 0 || nns > 4)
        _why = "parameters not supported (pscrn 2..4, fapprox int16 predictor + fastest exp, qual 1 / 2)";
      else if (!hw_tables().ok)
        _why = "this CPU's rcpps / rsqrtps tables do not have the expected structure";
      else if (prepare(nsize, nns, qual, etype, pscrn, env))
      {
        _mask = (Y ? 1 : 0) | (_fmt.planes == 3 && U ? 2 : 0) | (_fmt.planes == 3 && V ? 4 : 0);
        _ok = true;
        _producer = lib.producer_register(&CudaNnedi3::regen, this);
      }
      if (log_enabled())
        std::fprintf(stderr, "CudaNNEDI3: %s%s\n", _ok ? "GPU" : "CPU nnedi3: ", _ok ? "" : _why);
    }

    ~CudaNnedi3()
    {
      if (_producer)
        CudaLib::get().producer_unregister(_producer);
      if (_net)
        CudaLib::get().nnedi3_free(_net);
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env) override
    {
      if (!_ok)
        return _cpu->GetFrame(n, env);
      MvCudaTokenHeader h;
      if (make(n, env, h) != 0)
        return _cpu->GetFrame(n, env); // exact CPU result (its input goes through CudaDownload)
      PVideoFrame src = child->GetFrame(_field > 1 ? n >> 1 : n, env);
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

AVSValue __cdecl Create_CudaNnedi3(AVSValue args, void *, IScriptEnvironment *env)
{
  return new CudaNnedi3(args, env);
}
