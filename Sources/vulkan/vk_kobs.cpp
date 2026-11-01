// mvtools_vulkan: MaskTools / RgTools building blocks on device frames, QTGMC_KeepOnlyBobShimmerFixes and QTGMC's fused
// sharpness limit, port of Sources/cuda/mvcuda_kobs.cu (bit for bit: see that file for the rules). Kernels:
// shaders/kobs.pix.comp, kobs_rg.pix.comp, mt_ops.pix.comp, qtg_sharp.pix.comp (helpers in kobs_common.glsl).

#include "vk_internal.h"
#include "vk_cf_log.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <unordered_map>

namespace
{
  // two lookup tables (luma, then chroma), each (1 << bits) x (1 << bits) entries of uint8_t (8 bit) or uint16_t
  struct DevLut
  {
    vkrt::DevBuf table;
    size_t one = 0;      // bytes of one table
    int bits = 8;
    uint64_t luma() const { return table.p; }
    uint64_t chroma() const { return table.p + one; }
  };
  std::mutex g_lut_mutex;
  std::unordered_map<unsigned long long, std::shared_ptr<DevLut>> g_luts;
  std::atomic<unsigned long long> g_next_lut{ 1 };

  std::shared_ptr<DevLut> lut_get(unsigned long long h)
  {
    std::lock_guard<std::mutex> lk(g_lut_mutex);
    auto it = g_luts.find(h);
    return it == g_luts.end() ? nullptr : it->second;
  }


  int pitch_of(const mvk::DevFrame &f, int p) { return (int)(f.pitch[p] / f.f.bytes_per_sample); }

  // 8-bit samples in bytes, 9..16-bit samples in 16-bit containers
  bool depth_ok(const mvk::DevFrame &f)
  {
    return (f.f.bytes_per_sample == 1 && f.f.bits_per_sample == 8)
      || (f.f.bytes_per_sample == 2 && f.f.bits_per_sample > 8 && f.f.bits_per_sample <= 16);
  }

  bool same_fmt(const mvk::DevFrame &a, const mvk::DevFrame &b)
  {
    return a.f.width == b.f.width && a.f.height == b.f.height && a.f.planes == b.f.planes && a.f.x_ratio_uv == b.f.x_ratio_uv
      && a.f.y_ratio_uv == b.f.y_ratio_uv && a.f.bytes_per_sample == b.f.bytes_per_sample && a.f.bits_per_sample == b.f.bits_per_sample;
  }

  int max_of(const mvk::DevFrame &f) { return (1 << f.f.bits_per_sample) - 1; }

  // one dispatch, submitted and waited for; then the frame is registered
  template <class Args>
  int run1(const char *k8, const char *k16, int bps, uint32_t gx, uint32_t gy, uint32_t gz, const Args &a,
    const std::shared_ptr<mvk::DevFrame> &dst, unsigned long long producer, int frame, MvCudaTokenHeader *out)
  {
    vkrt::Kernel *k = vkrt::kernel(bps == 2 ? k16 : k8);
    if (!k)
      return -2;
    vkrt::Cmd cmd;
    cmd.dispatch(k, gx, gy, gz, a);
    const int rc = cmd.submit_wait();
    if (rc)
      return rc;
    mvk::frame_put(dst, producer, frame, *out);
    return 0;
  }

  // ---- shader argument structs (declared identically in the shaders) ----
  const int KT_X = 32, KT_Y = 32; // shaders/kobs.pix.comp
  struct KobsArgs
  {
    uint64_t in_[3], rf[3], d[3];
    uint64_t lut1[2], lut2[2];
    int pin[3], prf[3], pd[3];
    int w[3], h[3], act_[3];
    int n1, n2, rg_first, od;
    int R, C;
    int maxv, half_;
  };

  struct RgArgs
  {
    uint64_t s[3], d[3];
    int sp[3], dp[3], w[3], h[3], mode[3], pad0;
  };

  struct MtArgs
  {
    uint64_t a[3], b[3], c[3], d[3];
    uint64_t lut[2];
    int pa[3], pb[3], pc[3], pd[3];
    int w[3], h[3], op[3];
    int p1, p2, maxv;
  };

  const int SL_X = 64, SL_Y = 16; // shaders/qtg_sharp.pix.comp
  const int SL_REFS = 5;
  struct SlArgs
  {
    uint64_t edi[3], d1[3], d2[3], ref[SL_REFS][3], d[3];
    uint64_t lut[2], lutp[2];
    int pe[3], p1[3], p2[3], pr[SL_REFS][3], pd[3];
    int w[3], h[3];
    int tr1, w1, inv1, w2, inv2;
    int smode, nref, sovs, maxv, pad0;
  };
}

extern "C" unsigned long long mvcuda_lut2_create_bits(const void *luma, const void *chroma, int bits)
{
  if (!luma || bits < 8 || bits > 10 || !vkrt::ready())
    return 0;
  auto l = std::make_shared<DevLut>();
  l->bits = bits;
  l->one = ((size_t)1 << (2 * bits)) * (bits == 8 ? 1 : 2);
  l->table = vkrt::DevBuf(2 * l->one);
  if (!l->table)
    return 0;
  vkrt::Cmd cmd;
  cmd.upload(l->table.p, luma, l->one);
  cmd.upload(l->table.p + l->one, chroma ? chroma : luma, l->one);
  if (cmd.submit_wait())
    return 0;
  const unsigned long long h = g_next_lut.fetch_add(1);
  std::lock_guard<std::mutex> lk(g_lut_mutex);
  g_luts[h] = l;
  return h;
}

extern "C" unsigned long long mvcuda_lut2_create(const unsigned char *table)
{
  return mvcuda_lut2_create_bits(table, table, 8);
}

extern "C" void mvcuda_lut2_free(unsigned long long lut)
{
  std::shared_ptr<DevLut> keep; // released outside the lock
  std::lock_guard<std::mutex> lk(g_lut_mutex);
  auto it = g_luts.find(lut);
  if (it != g_luts.end())
  {
    keep = it->second;
    g_luts.erase(it);
  }
}

static int kobs_impl(unsigned long long input, unsigned long long ref, int rep, int planes_mask, unsigned long long lut1,
  unsigned long long lut2, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  if (!out)
    return -4;
  // ed: erosion distance, od: over-dilation (QTGMC_KeepOnlyBobShimmerFixes)
  const int ed = rep < 10 ? rep : rep / 10, od = rep < 10 ? 0 : rep % 10;
  if (ed == 2 || ed == 5 || od >= 3)
    return -4;
  if (!vkrt::ready())
    return -1;
  auto fi = mvk::frame_get(input);
  auto fr = mvk::frame_get(ref);
  if (!fi || !fr)
    return MVCUDA_TOKEN_GONE;
  if (!depth_ok(*fi) || !same_fmt(*fi, *fr))
    return -4;
  auto l1 = lut_get(lut1), l2 = lut_get(lut2);
  if (!l1 || !l2 || l1->bits != fi->f.bits_per_sample || l2->bits != fi->f.bits_per_sample)
    return -4;
  auto dst = mvk::frame_alloc(fi->f);
  if (!dst)
    return -2;
  KobsArgs k = {};
  int max_w = 0, max_h = 0;
  for (int p = 0; p < fi->f.planes; p++)
  {
    k.in_[p] = fi->plane[p];
    k.rf[p] = fr->plane[p];
    k.d[p] = dst->plane[p];
    k.pin[p] = pitch_of(*fi, p);
    k.prf[p] = pitch_of(*fr, p);
    k.pd[p] = pitch_of(*dst, p);
    k.w[p] = fi->w[p];
    k.h[p] = fi->h[p];
    k.act_[p] = (planes_mask >> p) & 1;
    max_w = fi->w[p] > max_w ? fi->w[p] : max_w;
    max_h = fi->h[p] > max_h ? fi->h[p] : max_h;
  }
  k.lut1[0] = l1->luma();
  k.lut1[1] = l1->chroma();
  k.lut2[0] = l2->luma();
  k.lut2[1] = l2->chroma();
  k.n1 = 1 + (ed > 2 ? 1 : 0) + (ed > 5 ? 1 : 0);
  k.rg_first = (ed % 3 != 0) ? 1 : 0;
  k.n2 = 1 + (ed > 1 ? 1 : 0) + (ed > 4 ? 1 : 0);
  k.od = od;
  k.R = k.n1 + k.rg_first + k.n2 + k.od;
  k.C = k.rg_first + k.od;
  k.maxv = max_of(*fi);
  k.half_ = 1 << (fi->f.bits_per_sample - 1);
  const int rc = run1("kobs_8", "kobs_16", fi->f.bytes_per_sample, (max_w + KT_X - 1) / KT_X, (max_h + KT_Y - 1) / KT_Y,
    fi->f.planes, k, dst, producer, frame, out);
  return rc;
}

static int removegrain_impl(unsigned long long src, const int mode[3], unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  if (!mode || !out)
    return -4;
  if (!vkrt::ready())
    return -1;
  auto fs = mvk::frame_get(src);
  if (!fs)
    return MVCUDA_TOKEN_GONE;
  if (!depth_ok(*fs))
    return -4;
  for (int p = 0; p < fs->f.planes; p++)
    if (mode[p] != -1 && mode[p] != 0 && mode[p] != 11 && mode[p] != 12 && mode[p] != 20)
      return -4;
  auto dst = mvk::frame_alloc(fs->f);
  if (!dst)
    return -2;
  RgArgs g = {};
  int max_w = 0, max_h = 0;
  for (int p = 0; p < fs->f.planes; p++)
  {
    g.s[p] = fs->plane[p];
    g.d[p] = dst->plane[p];
    g.sp[p] = pitch_of(*fs, p);
    g.dp[p] = pitch_of(*dst, p);
    g.w[p] = fs->w[p];
    g.h[p] = fs->h[p];
    g.mode[p] = mode[p];
    max_w = fs->w[p] > max_w ? fs->w[p] : max_w;
    max_h = fs->h[p] > max_h ? fs->h[p] : max_h;
  }
  const int rc = run1("kobs_rg_8", "kobs_rg_16", fs->f.bytes_per_sample, (max_w + 127) / 128, (max_h + 7) / 8, fs->f.planes, g, dst, producer, // 8 = RG_ROWS of the shader
    frame, out);
  return rc;
}

static int mt_impl(const int op[3], unsigned long long a, unsigned long long b, unsigned long long c, unsigned long long lut,
  int p1, int p2, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  if (!op || !out)
    return -4;
  if (!vkrt::ready())
    return -1;
  auto fa = mvk::frame_get(a), fb = mvk::frame_get(b), fc = mvk::frame_get(c);
  if (!fa || !fb || !fc)
    return MVCUDA_TOKEN_GONE;
  if (!depth_ok(*fa) || !same_fmt(*fa, *fb) || !same_fmt(*fa, *fc))
    return -4;
  std::shared_ptr<DevLut> l;
  bool uses_lut = false;
  for (int p = 0; p < fa->f.planes; p++)
    uses_lut = uses_lut || op[p] == MVCUDA_MT_LUT;
  if (uses_lut && (!(l = lut_get(lut)) || l->bits != fa->f.bits_per_sample))
    return -4;
  auto dst = mvk::frame_alloc(fa->f);
  if (!dst)
    return -2;
  MtArgs m = {};
  int max_w = 0, max_h = 0;
  for (int p = 0; p < fa->f.planes; p++)
  {
    m.a[p] = fa->plane[p];
    m.b[p] = fb->plane[p];
    m.c[p] = fc->plane[p];
    m.d[p] = dst->plane[p];
    m.pa[p] = pitch_of(*fa, p);
    m.pb[p] = pitch_of(*fb, p);
    m.pc[p] = pitch_of(*fc, p);
    m.pd[p] = pitch_of(*dst, p);
    m.w[p] = fa->w[p];
    m.h[p] = fa->h[p];
    m.op[p] = op[p];
    max_w = fa->w[p] > max_w ? fa->w[p] : max_w;
    max_h = fa->h[p] > max_h ? fa->h[p] : max_h;
  }
  m.lut[0] = l ? l->luma() : 0;
  m.lut[1] = l ? l->chroma() : 0;
  m.p1 = p1;
  m.p2 = p2;
  m.maxv = max_of(*fa);
  const int rc = run1("mt_ops_8", "mt_ops_16", fa->f.bytes_per_sample, (max_w + 127) / 128, max_h, fa->f.planes, m, dst, producer,
    frame, out);
  return rc;
}

static int qtgmc_sharp_impl(const MvCudaQtgmcSharp *q, unsigned long long edi, unsigned long long deg1, unsigned long long deg2,
  const unsigned long long *refs, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  if (!q || !out || q->sovs < 0 || q->sovs > 65535 || q->tr1 < 0 || q->tr1 > 2 || q->smode < 0 || q->smode > 2 || q->nref < 0
    || q->nref > SL_REFS || (q->nref && !refs))
    return -4;
  if (!vkrt::ready())
    return -1;
  auto fe = mvk::frame_get(edi);
  std::shared_ptr<mvk::DevFrame> f1, f2, fr[SL_REFS];
  if (!fe || (q->tr1 > 0 && !(f1 = mvk::frame_get(deg1))) || (q->tr1 > 1 && !(f2 = mvk::frame_get(deg2))))
    return MVCUDA_TOKEN_GONE;
  for (int j = 0; j < q->nref; j++)
    if (!(fr[j] = mvk::frame_get(refs[j])))
      return MVCUDA_TOKEN_GONE;
  if (!depth_ok(*fe) || fe->f.bits_per_sample == 16 || (f1 && !same_fmt(*fe, *f1)) || (f2 && !same_fmt(*fe, *f2)))
    return -4;
  for (int j = 0; j < q->nref; j++)
    if (!same_fmt(*fe, *fr[j]))
      return -4;
  const int bits = fe->f.bits_per_sample;
  std::shared_ptr<DevLut> l, lp;
  if (q->smode && (!(l = lut_get(q->lut)) || l->bits != bits))
    return -4;
  if (q->smode == 2 && q->lut_precise && (!(lp = lut_get(q->lut_precise)) || lp->bits != bits))
    return -4;
  auto dst = mvk::frame_alloc(fe->f);
  if (!dst)
    return -2;
  SlArgs k = {};
  int max_w = 0, max_h = 0;
  for (int p = 0; p < fe->f.planes; p++)
  {
    k.edi[p] = fe->plane[p];
    k.pe[p] = pitch_of(*fe, p);
    k.d1[p] = f1 ? f1->plane[p] : fe->plane[p];
    k.p1[p] = f1 ? pitch_of(*f1, p) : k.pe[p];
    k.d2[p] = f2 ? f2->plane[p] : fe->plane[p];
    k.p2[p] = f2 ? pitch_of(*f2, p) : k.pe[p];
    for (int j = 0; j < q->nref; j++)
    {
      k.ref[j][p] = fr[j]->plane[p];
      k.pr[j][p] = pitch_of(*fr[j], p);
    }
    k.d[p] = dst->plane[p];
    k.pd[p] = pitch_of(*dst, p);
    k.w[p] = fe->w[p];
    k.h[p] = fe->h[p];
    max_w = fe->w[p] > max_w ? fe->w[p] : max_w;
    max_h = fe->h[p] > max_h ? fe->h[p] : max_h;
  }
  k.lut[0] = l ? l->luma() : 0;
  k.lut[1] = l ? l->chroma() : 0;
  k.lutp[0] = lp ? lp->luma() : 0;
  k.lutp[1] = lp ? lp->chroma() : 0;
  k.tr1 = q->tr1;
  k.w1 = q->w1;
  k.inv1 = q->inv1;
  k.w2 = q->w2;
  k.inv2 = q->inv2;
  k.smode = q->smode;
  k.nref = q->nref;
  k.sovs = q->sovs;
  k.maxv = max_of(*fe);
  const int rc = run1("qtg_sharp_8", "qtg_sharp_16", fe->f.bytes_per_sample, (max_w + SL_X - 1) / SL_X, (max_h + SL_Y - 1) / SL_Y,
    fe->f.planes, k, dst, producer, frame, out);
  return rc;
}

// ---- exported entry points (counted, see vk_cf_log.h) ----
extern "C" int mvcuda_kobs(unsigned long long input, unsigned long long ref, int rep, int planes_mask, unsigned long long lut1,
  unsigned long long lut2, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  static mvk::CallLog log("kobs");
  return log(kobs_impl(input, ref, rep, planes_mask, lut1, lut2, producer, frame, out));
}

extern "C" int mvcuda_removegrain(unsigned long long src, const int mode[3], unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  static mvk::CallLog log("removegrain");
  return log(removegrain_impl(src, mode, producer, frame, out));
}

extern "C" int mvcuda_mt(const int op[3], unsigned long long a, unsigned long long b, unsigned long long c, unsigned long long lut,
  int p1, int p2, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  static mvk::CallLog log("mt");
  return log(mt_impl(op, a, b, c, lut, p1, p2, producer, frame, out));
}

extern "C" int mvcuda_qtgmc_sharp(const MvCudaQtgmcSharp *q, unsigned long long edi, unsigned long long deg1, unsigned long long deg2,
  const unsigned long long *refs, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  static mvk::CallLog log("qtgmc_sharp");
  return log(qtgmc_sharp_impl(q, edi, deg1, deg2, refs, producer, frame, out));
}
