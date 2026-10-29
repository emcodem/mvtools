// 8-bit MaskTools / RgTools building blocks on device frames and QTGMC_KeepOnlyBobShimmerFixes (GPU_PIPELINE_PLAN.md
// section 15, stages 4, 5, 16), bit for bit with masktools2 2.2.30 and RgTools 1.2 as they run on SSE4.1 CPUs:
//   mt_makediff  clip(x - y + 128)               mt_adddiff  clip(x + y - 128)
//   mt_inpand / mt_expand mode="vertical" with the default threshold: min / max of the sample and its vertical neighbours,
//                the first and the last row use themselves as the missing neighbour
//   mt_logic min / max (default thresholds)      mt_lutxy    a 256 x 256 table made from the CPU filter by the plugin
//   RemoveGrain 12  rg_mode12_sse (cascade of rounding averages, one saturating -1), border rows / columns copied
//   RemoveGrain 20  rg_mode20_sse ((sum of 9 + 4) * 7282 >> 16), border rows / columns copied

#include "mvcuda_internal.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace
{
  struct DevLut
  {
    unsigned char *table = nullptr;
    ~DevLut()
    {
      mvc::pipe_flush(); // queued work may still read it (filter teardown only)
      mvc::dev_free(table, 65536);
    }
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

  struct Plane
  {
    unsigned char *p;
    int w, h;
    size_t pitch;
    __device__ unsigned char at(int x, int y) const { return p[(size_t)y * pitch + x]; }
  };

  __device__ __forceinline__ unsigned char clip255(int v) { return (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v); }
  __device__ __forceinline__ unsigned char avg(int a, int b) { return (unsigned char)((a + b + 1) >> 1); }

  struct RgPlanes
  {
    Plane s[3], d[3];
    int mode[3]; // 12 (11: RgTools' SSE4.1 mode 11 is mode 12), 20, otherwise copy
  };

  // RemoveGrain 12 / 20 (rg_mode12_sse / rg_mode20_sse), borders copied; blockIdx.z = plane
  __global__ void k_rg(RgPlanes g)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
    const Plane &s = g.s[blockIdx.z], &d = g.d[blockIdx.z];
    const int mode = g.mode[blockIdx.z] == 11 ? 12 : g.mode[blockIdx.z];
    if (x >= d.w || y >= d.h)
      return;
    unsigned char r;
    if ((mode != 12 && mode != 20) || x == 0 || y == 0 || x == s.w - 1 || y == s.h - 1)
      r = s.at(x, y);
    else
    {
      const int a1 = s.at(x - 1, y - 1), a2 = s.at(x, y - 1), a3 = s.at(x + 1, y - 1);
      const int a4 = s.at(x - 1, y), c = s.at(x, y), a5 = s.at(x + 1, y);
      const int a6 = s.at(x - 1, y + 1), a7 = s.at(x, y + 1), a8 = s.at(x + 1, y + 1);
      if (mode == 12)
      {
        const int a123 = avg(a2, avg(a1, a3));
        const int a678 = avg(a7, avg(a6, a8));
        const int a4c5 = avg(c, avg(a4, a5));
        int a123678 = avg(a123, a678);
        a123678 = a123678 > 0 ? a123678 - 1 : 0;
        r = avg(a4c5, a123678);
      }
      else
      {
        const unsigned sum = (unsigned)(a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + c + 4);
        r = (unsigned char)((sum * 7282u) >> 16); // _mm_mulhi_epu16(sum, (65536 + 4) / 9)
      }
    }
    d.p[(size_t)y * d.pitch + x] = r;
  }

  // ---- QTGMC_KeepOnlyBobShimmerFixes fused into one kernel (all planes): the same integer steps as the chain of masktools /
  // RemoveGrain filters in QTGMC.avsi, on a tile with a halo as large as the chain's reach (one row per vertical step, one row and
  // column per RemoveGrain 20 step); plane borders follow the same rules, evaluated in plane coordinates.
  const int KT_X = 64, KT_Y = 32;            // output tile
  const int KR_MAX = 9, KC_MAX = 3;          // largest reach: 3 + 1 + 3 vertical steps + 2 over-dilation, 1 + 2 rg steps
  const int KB_W = KT_X + 2 * KC_MAX, KB_H = KT_Y + 2 * KR_MAX;

  struct KobsPlanes
  {
    Plane in[3], rf[3], d[3];
    int active[3];
    const unsigned char *lut1, *lut2;
    int n1, n2, rg_first, od;                // first-direction vertical steps, second-direction steps, deflate step, over-dilation
    int R, C;                                // vertical / horizontal reach of one chain
  };

  // one step of a chain over the buffer region still valid (halo vr / vc around the tile): kind 0 vertical morph, 1 the
  // RemoveGrain 20 + min / max step; region coordinates are buffer rows / columns, plane coordinates gy0 + row, gx0 + col
  __device__ __forceinline__ void kobs_step(const unsigned char (*src)[KB_W], unsigned char (*dst)[KB_W], int kind, bool is_max,
    int vr, int vc, const KobsPlanes &k, int gx0, int gy0, int w, int h, int tid, int nt)
  {
    const int r0 = k.R - vr + 1, r1 = k.R + KT_Y + vr - 1;                       // rows produced (one less halo row)
    const int c0 = k.C - vc + (kind ? 1 : 0), c1 = k.C + KT_X + vc - (kind ? 1 : 0); // columns produced
    const int cols = c1 - c0, n = (r1 - r0) * cols;
    for (int i = tid; i < n; i += nt)
    {
      const int r = r0 + i / cols, c = c0 + i % cols;
      const int gy = gy0 + r, gx = gx0 + c;
      if (gy < 0 || gy >= h || gx < 0 || gx >= w)
        continue;
      const int v = src[r][c];
      int out;
      if (kind == 0)
      {
        const int up = gy > 0 ? src[r - 1][c] : v, dn = gy < h - 1 ? src[r + 1][c] : v;
        out = is_max ? max(v, max(up, dn)) : min(v, min(up, dn));
      }
      else
      {
        int rg = v; // QTGMC_deflate / inflate: mt_logic(c, c.RemoveGrain(20), "min" / "max"), RemoveGrain borders copied
        if (gx > 0 && gy > 0 && gx < w - 1 && gy < h - 1)
        {
          const unsigned sum = (unsigned)(src[r - 1][c - 1] + src[r - 1][c] + src[r - 1][c + 1] + src[r][c - 1] + v + src[r][c + 1]
            + src[r + 1][c - 1] + src[r + 1][c] + src[r + 1][c + 1] + 4);
          rg = (int)((sum * 7282u) >> 16);
        }
        out = is_max ? max(v, rg) : min(v, rg);
      }
      dst[r][c] = (unsigned char)out;
    }
  }

  __global__ void __launch_bounds__(256) k_kobs(KobsPlanes k)
  {
    __shared__ unsigned char diff[KB_H][KB_W], ba[KB_H][KB_W], bb[KB_H][KB_W], c1[KB_H][KB_W];
    const int p = blockIdx.z, tid = threadIdx.x, nt = blockDim.x;
    const Plane &in = k.in[p], &rf = k.rf[p], &d = k.d[p];
    const int w = d.w, h = d.h, tx0 = blockIdx.x * KT_X, ty0 = blockIdx.y * KT_Y;
    if (tx0 >= w || ty0 >= h)
      return;
    if (!k.active[p])
    {
      for (int i = tid; i < KT_X * KT_Y; i += nt)
      {
        const int x = tx0 + i % KT_X, y = ty0 + i / KT_X;
        if (x < w && y < h)
          d.p[(size_t)y * d.pitch + x] = in.at(x, y);
      }
      return;
    }
    const int gx0 = tx0 - k.C, gy0 = ty0 - k.R; // plane coordinates of buffer [0][0]
    const int rows = KT_Y + 2 * k.R, cols = KT_X + 2 * k.C;
    for (int i = tid; i < rows * cols; i += nt)
    {
      const int r = i / cols, c = i % cols, gy = gy0 + r, gx = gx0 + c;
      if (gy >= 0 && gy < h && gx >= 0 && gx < w)
        diff[r][c] = clip255((int)rf.at(gx, gy) - (int)in.at(gx, gy) + 128); // mt_makediff(Ref, Input)
    }
    __syncthreads();
    for (int chain = 0; chain < 2; chain++)
    {
      // chain 0: positive differences (inpand first) into c1; chain 1: negative (expand first), result left in cur
      const bool first_max = chain == 1;
      const unsigned char (*cur)[KB_W] = diff;
      int vr = k.R, vc = k.C;
      auto step = [&](int kind, bool is_max) {
        unsigned char (*out)[KB_W] = (cur == ba) ? bb : ba;
        kobs_step(cur, out, kind, is_max, vr, vc, k, gx0, gy0, w, h, tid, nt);
        __syncthreads();
        cur = out;
        vr--;
        if (kind)
          vc--;
      };
      for (int s = 0; s < k.n1; s++)
        step(0, first_max);
      if (k.rg_first)
        step(1, first_max);
      for (int s = 0; s < k.n2; s++)
        step(0, !first_max);
      for (int s = 0; s < k.od; s++)
        step(1, !first_max);
      if (chain == 0)
      {
        for (int i = tid; i < KT_X * KT_Y; i += nt)
        {
          const int r = k.R + i / KT_X, c = k.C + i % KT_X;
          c1[r][c] = cur[r][c];
        }
        __syncthreads();
      }
      else
      {
        // restore = diff.mt_lutxy(choke1, expr1).mt_lutxy(choke2, expr2); Input.mt_adddiff(restore)
        for (int i = tid; i < KT_X * KT_Y; i += nt)
        {
          const int r = k.R + i / KT_X, c = k.C + i % KT_X, x = tx0 + i % KT_X, y = ty0 + i / KT_X;
          if (x >= w || y >= h)
            continue;
          const int t1 = k.lut1[diff[r][c] * 256 + c1[r][c]];
          const int t2 = k.lut2[t1 * 256 + cur[r][c]];
          d.p[(size_t)y * d.pitch + x] = clip255((int)in.at(x, y) + t2 - 128);
        }
      }
    }
  }

  Plane plane_of(const mvc::DevFrame &f, int p)
  {
    return Plane{ (unsigned char *)f.plane[p], f.w[p], f.h[p], f.pitch[p] };
  }

  bool frame_8bit(const mvc::DevFrame &f)
  {
    return f.f.bytes_per_sample == 1 && f.f.bits_per_sample == 8;
  }
}

extern "C" unsigned long long mvcuda_lut2_create(const unsigned char *table)
{
  if (!table || !mvc::use_device())
    return 0;
  auto l = std::make_shared<DevLut>();
  l->table = (unsigned char *)mvc::dev_alloc(65536);
  if (!l->table || cudaMemcpy(l->table, table, 65536, cudaMemcpyHostToDevice) != cudaSuccess)
  {
    cudaGetLastError();
    return 0;
  }
  const unsigned long long h = g_next_lut.fetch_add(1);
  std::lock_guard<std::mutex> lk(g_lut_mutex);
  g_luts[h] = l;
  return h;
}

extern "C" void mvcuda_lut2_free(unsigned long long lut)
{
  std::lock_guard<std::mutex> lk(g_lut_mutex);
  g_luts.erase(lut);
}

extern "C" int mvcuda_kobs(unsigned long long input, unsigned long long ref, int rep, int planes_mask, unsigned long long lut1,
  unsigned long long lut2, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  if (!out)
    return -4;
  // ed: erosion distance, od: over-dilation (QTGMC_KeepOnlyBobShimmerFixes)
  const int ed = rep < 10 ? rep : rep / 10, od = rep < 10 ? 0 : rep % 10;
  if (ed == 2 || ed == 5 || od >= 3)
    return -4;
  if (!mvc::use_device())
    return -1;
  auto fi = mvc::frame_get(input);
  auto fr = mvc::frame_get(ref);
  if (!fi || !fr)
    return MVCUDA_TOKEN_GONE;
  if (!frame_8bit(*fi) || !frame_8bit(*fr) || fi->f.width != fr->f.width || fi->f.height != fr->f.height || fi->f.planes != fr->f.planes
    || fi->f.x_ratio_uv != fr->f.x_ratio_uv || fi->f.y_ratio_uv != fr->f.y_ratio_uv)
    return -4;
  auto l1 = lut_get(lut1), l2 = lut_get(lut2);
  if (!l1 || !l2)
    return -4;
  auto dst = mvc::frame_alloc(fi->f);
  if (!dst)
    return -2;
  // the whole function in one launch (k_kobs), all planes
  KobsPlanes k = {};
  int max_w = 0, max_h = 0;
  for (int p = 0; p < fi->f.planes; p++)
  {
    k.in[p] = plane_of(*fi, p);
    k.rf[p] = plane_of(*fr, p);
    k.d[p] = plane_of(*dst, p);
    k.active[p] = (planes_mask >> p) & 1;
    max_w = fi->w[p] > max_w ? fi->w[p] : max_w;
    max_h = fi->h[p] > max_h ? fi->h[p] : max_h;
  }
  k.lut1 = l1->table;
  k.lut2 = l2->table;
  k.n1 = 1 + (ed > 2 ? 1 : 0) + (ed > 5 ? 1 : 0);
  k.rg_first = (ed % 3 != 0) ? 1 : 0;
  k.n2 = 1 + (ed > 1 ? 1 : 0) + (ed > 4 ? 1 : 0);
  k.od = od;
  k.R = k.n1 + k.rg_first + k.n2 + k.od;
  k.C = k.rg_first + k.od;
  const dim3 grid((max_w + KT_X - 1) / KT_X, (max_h + KT_Y - 1) / KT_Y, fi->f.planes);
  mvc::pipe_submit([=](cudaStream_t s) { k_kobs<<<grid, 256, 0, s>>>(k); });  mvc::frame_put(dst, producer, frame, *out);
  return 0;
}

extern "C" int mvcuda_removegrain(unsigned long long src, const int mode[3], unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  if (!mode || !out)
    return -4;
  if (!mvc::use_device())
    return -1;
  auto fs = mvc::frame_get(src);
  if (!fs)
    return MVCUDA_TOKEN_GONE;
  if (!frame_8bit(*fs))
    return -4;
  for (int p = 0; p < fs->f.planes; p++)
    if (mode[p] != -1 && mode[p] != 0 && mode[p] != 11 && mode[p] != 12 && mode[p] != 20)
      return -4;
  auto dst = mvc::frame_alloc(fs->f);
  if (!dst)
    return -2;
  RgPlanes g = {};
  int max_w = 0, max_h = 0;
  for (int p = 0; p < fs->f.planes; p++)
  {
    g.s[p] = plane_of(*fs, p);
    g.d[p] = plane_of(*dst, p);
    g.mode[p] = mode[p];
    max_w = fs->w[p] > max_w ? fs->w[p] : max_w;
    max_h = fs->h[p] > max_h ? fs->h[p] : max_h;
  }
  const dim3 grid((max_w + 127) / 128, max_h, fs->f.planes);
  mvc::pipe_submit([=](cudaStream_t st) { k_rg<<<grid, 128, 0, st>>>(g); }); // issued by the pipeline submitter (mvcuda_internal.h)
  mvc::frame_put(dst, producer, frame, *out);
  return 0;
}

// ---- single masktools-style operations per plane (CudaLogic / CudaClamp / CudaExpand / CudaInpand / CudaLutxy) ---------
namespace
{
  struct MtPlanes
  {
    Plane a[3], b[3], c[3], d[3];
    int op[3];
    const unsigned char *lut;
    int p1, p2;
  };

  // op: MVCUDA_MT_* (mvcuda.h)
  __global__ void k_mt(MtPlanes m)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y, p = blockIdx.z;
    const Plane &d = m.d[p];
    if (x >= d.w || y >= d.h)
      return;
    const Plane &a = m.a[p];
    int r;
    switch (m.op[p])
    {
    case MVCUDA_MT_MIN: r = min((int)a.at(x, y), (int)m.b[p].at(x, y)); break;
    case MVCUDA_MT_MAX: r = max((int)a.at(x, y), (int)m.b[p].at(x, y)); break;
    case MVCUDA_MT_CLAMP:
    {
      // clamp_sse2: min with (bright +sat overshoot), then max with (dark -sat undershoot)
      int up = (int)m.b[p].at(x, y) + m.p1, lo = (int)m.c[p].at(x, y) - m.p2;
      up = up > 255 ? 255 : up;
      lo = lo < 0 ? 0 : lo;
      r = max(min((int)a.at(x, y), up), lo);
      break;
    }
    case MVCUDA_MT_VEXPAND:
    case MVCUDA_MT_VINPAND:
    {
      const int up = a.at(x, y > 0 ? y - 1 : y), c = a.at(x, y), dn = a.at(x, y < a.h - 1 ? y + 1 : y);
      r = m.op[p] == MVCUDA_MT_VEXPAND ? max(c, max(up, dn)) : min(c, min(up, dn));
      break;
    }
    case MVCUDA_MT_LUT: r = m.lut[a.at(x, y) * 256 + m.b[p].at(x, y)]; break;
    case MVCUDA_MT_COPY_B: r = m.b[p].at(x, y); break;
    default: r = a.at(x, y); break;
    }
    d.p[(size_t)y * d.pitch + x] = (unsigned char)r;
  }
}

extern "C" int mvcuda_mt(const int op[3], unsigned long long a, unsigned long long b, unsigned long long c, unsigned long long lut,
  int p1, int p2, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  if (!op || !out)
    return -4;
  if (!mvc::use_device())
    return -1;
  auto fa = mvc::frame_get(a), fb = mvc::frame_get(b), fc = mvc::frame_get(c);
  if (!fa || !fb || !fc)
    return MVCUDA_TOKEN_GONE;
  for (const auto *f : { fb.get(), fc.get() })
    if (f->f.width != fa->f.width || f->f.height != fa->f.height || f->f.planes != fa->f.planes || f->f.x_ratio_uv != fa->f.x_ratio_uv
      || f->f.y_ratio_uv != fa->f.y_ratio_uv || !frame_8bit(*f))
      return -4;
  if (!frame_8bit(*fa))
    return -4;
  std::shared_ptr<DevLut> l;
  bool uses_lut = false;
  for (int p = 0; p < fa->f.planes; p++)
    uses_lut = uses_lut || op[p] == MVCUDA_MT_LUT;
  if (uses_lut && !(l = lut_get(lut)))
    return -4;
  auto dst = mvc::frame_alloc(fa->f);
  if (!dst)
    return -2;
  MtPlanes m = {};
  int max_w = 0, max_h = 0;
  for (int p = 0; p < fa->f.planes; p++)
  {
    m.a[p] = plane_of(*fa, p);
    m.b[p] = plane_of(*fb, p);
    m.c[p] = plane_of(*fc, p);
    m.d[p] = plane_of(*dst, p);
    m.op[p] = op[p];
    max_w = fa->w[p] > max_w ? fa->w[p] : max_w;
    max_h = fa->h[p] > max_h ? fa->h[p] : max_h;
  }
  m.lut = l ? l->table : nullptr;
  m.p1 = p1;
  m.p2 = p2;
  const dim3 grid((max_w + 127) / 128, max_h, fa->f.planes);
  mvc::pipe_submit([=](cudaStream_t s) { k_mt<<<grid, 128, 0, s>>>(m); });
  mvc::frame_put(dst, producer, frame, *out);
  return 0;
}

// ---- QTGMC's binomial smooth, resharp and temporal sharpness limit, fused (CudaQtgmcSharpLimit) ------------------------
// The chain of QTGMC.avsi "Create basic output" .. sharpLimit1 (see mvcuda_qtgmc_sharp in mvcuda.h; all planes, U = V = 3)
// in one kernel instead of up to seventeen filters, with the integer steps of the single kernels (k_merge, k_mt, k_rg).
namespace
{
  const int SL_X = 64, SL_Y = 16;   // output tile
  const int SL_REFS = 5;

  struct SlPlanes
  {
    Plane edi[3], d1[3], d2[3], ref[SL_REFS][3], d[3];
    const unsigned char *lut;       // the resharp mt_lutxy table
    const unsigned char *lutp;      // Precise: the vresharp limit table, else nullptr
    int tr1, w1, inv1, w2, inv2;    // binomial: Merge weights (mvcuda_merge.cu weighted mode)
    int smode, nref, sovs;
  };

  __device__ __forceinline__ int rg12(int a1, int a2, int a3, int a4, int c, int a5, int a6, int a7, int a8)
  {
    const int a123 = avg(a2, avg(a1, a3));
    const int a678 = avg(a7, avg(a6, a8));
    const int a4c5 = avg(c, avg(a4, a5));
    int a123678 = avg(a123, a678);
    a123678 = a123678 > 0 ? a123678 - 1 : 0;
    return avg(a4c5, a123678);
  }

  __global__ void __launch_bounds__(256) k_qtgmc_sharp(SlPlanes k)
  {
    // binomial on rows y0 - 2 .. y0 + SL_Y + 1, columns x0 - 1 .. x0 + SL_X; vresharp on rows y0 - 1 .. y0 + SL_Y
    __shared__ unsigned char bin[SL_Y + 4][SL_X + 2], vr[SL_Y + 2][SL_X + 2];
    const int p = blockIdx.z, tid = threadIdx.x, nt = blockDim.x;
    const Plane &d = k.d[p], &ed = k.edi[p];
    const int w = d.w, h = d.h, x0 = blockIdx.x * SL_X, y0 = blockIdx.y * SL_Y;
    if (x0 >= w || y0 >= h)
      return;
    // the samples are those of the clamped coordinates: the vertical morphology replicates the edge rows; clamped
    // columns are read only for border samples, which RemoveGrain copies
    for (int i = tid; i < (SL_Y + 4) * (SL_X + 2); i += nt)
    {
      const int r = i / (SL_X + 2), c = i % (SL_X + 2);
      const int gy = min(max(y0 - 2 + r, 0), h - 1), gx = min(max(x0 - 1 + c, 0), w - 1);
      int b = ed.at(gx, gy);
      if (k.tr1 > 0)
      {
        int m = k.d1[p].at(gx, gy);
        if (k.tr1 > 1)
          m = (m * k.inv1 + (int)k.d2[p].at(gx, gy) * k.w1 + 16384) >> 15; // Merge(degrain1, degrain2, 0.2)
        b = (m * k.inv2 + b * k.w2 + 16384) >> 15;                         // Merge(.., edi, 0.25 / 0.0625)
      }
      bin[r][c] = (unsigned char)b;
    }
    __syncthreads();
    if (k.smode == 2)
    {
      for (int i = tid; i < (SL_Y + 2) * (SL_X + 2); i += nt)
      {
        const int r = i / (SL_X + 2), c = i % (SL_X + 2);
        const int u = bin[r][c], m = bin[r + 1][c], dn = bin[r + 2][c];
        int v = avg(max(m, max(u, dn)), min(m, min(u, dn)));
        if (k.lutp)
          v = k.lutp[v * 256 + m]; // Precise: mt_lutxy(vresharp, binomial)
        vr[r][c] = (unsigned char)v;
      }
      __syncthreads();
    }
    for (int i = tid; i < SL_X * SL_Y; i += nt)
    {
      const int x = x0 + i % SL_X, y = y0 + i / SL_X;
      if (x >= w || y >= h)
        continue;
      const int r = 1 + i / SL_X, c = 1 + i % SL_X; // (x, y) in vr; bin row r + 1
      int res = bin[r + 1][c];
      if (k.smode)
      {
        // RemoveGrain 12 (k_rg; RgTools' SSE4.1 mode 11 is the same code) of vresharp (SMode 2) or of the binomial (SMode 1)
        int rg;
        if (k.smode == 2)
        {
          rg = vr[r][c];
          if (x > 0 && y > 0 && x < w - 1 && y < h - 1)
            rg = rg12(vr[r - 1][c - 1], vr[r - 1][c], vr[r - 1][c + 1], vr[r][c - 1], vr[r][c], vr[r][c + 1], vr[r + 1][c - 1], vr[r + 1][c], vr[r + 1][c + 1]);
        }
        else
        {
          const int q = r + 1;
          rg = bin[q][c];
          if (x > 0 && y > 0 && x < w - 1 && y < h - 1)
            rg = rg12(bin[q - 1][c - 1], bin[q - 1][c], bin[q - 1][c + 1], bin[q][c - 1], bin[q][c], bin[q][c + 1], bin[q + 1][c - 1], bin[q + 1][c], bin[q + 1][c + 1]);
        }
        res = k.lut[res * 256 + rg];
      }
      if (k.nref)
      {
        int up = 0, lo = 255;
        for (int j = 0; j < k.nref; j++)
        {
          const int v = k.ref[j][p].at(x, y);
          up = max(up, v);
          lo = min(lo, v);
        }
        up = min(up + k.sovs, 255);
        lo = max(lo - k.sovs, 0);
        res = max(min(res, up), lo); // clamp_sse2: min with the bright limit, then max with the dark one
      }
      d.p[(size_t)y * d.pitch + x] = (unsigned char)res;
    }
  }

  bool same_geom(const mvc::DevFrame &a, const mvc::DevFrame &b)
  {
    return a.f.width == b.f.width && a.f.height == b.f.height && a.f.planes == b.f.planes && a.f.x_ratio_uv == b.f.x_ratio_uv
      && a.f.y_ratio_uv == b.f.y_ratio_uv && frame_8bit(b);
  }
}

extern "C" int mvcuda_qtgmc_sharp(const MvCudaQtgmcSharp *q, unsigned long long edi, unsigned long long deg1, unsigned long long deg2,
  const unsigned long long *refs, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  if (!q || !out || q->sovs < 0 || q->sovs > 255 || q->tr1 < 0 || q->tr1 > 2 || q->smode < 0 || q->smode > 2 || q->nref < 0
    || q->nref > SL_REFS || (q->nref && !refs))
    return -4;
  if (!mvc::use_device())
    return -1;
  auto fe = mvc::frame_get(edi);
  std::shared_ptr<mvc::DevFrame> f1, f2, fr[SL_REFS];
  if (!fe || (q->tr1 > 0 && !(f1 = mvc::frame_get(deg1))) || (q->tr1 > 1 && !(f2 = mvc::frame_get(deg2))))
    return MVCUDA_TOKEN_GONE;
  for (int j = 0; j < q->nref; j++)
    if (!(fr[j] = mvc::frame_get(refs[j])))
      return MVCUDA_TOKEN_GONE;
  if (!frame_8bit(*fe) || (f1 && !same_geom(*fe, *f1)) || (f2 && !same_geom(*fe, *f2)))
    return -4;
  for (int j = 0; j < q->nref; j++)
    if (!same_geom(*fe, *fr[j]))
      return -4;
  std::shared_ptr<DevLut> l, lp;
  if (q->smode && !(l = lut_get(q->lut)))
    return -4;
  if (q->smode == 2 && q->lut_precise && !(lp = lut_get(q->lut_precise)))
    return -4;
  auto dst = mvc::frame_alloc(fe->f);
  if (!dst)
    return -2;
  SlPlanes k = {};
  int max_w = 0, max_h = 0;
  for (int p = 0; p < fe->f.planes; p++)
  {
    k.edi[p] = plane_of(*fe, p);
    k.d1[p] = f1 ? plane_of(*f1, p) : k.edi[p];
    k.d2[p] = f2 ? plane_of(*f2, p) : k.edi[p];
    for (int j = 0; j < q->nref; j++)
      k.ref[j][p] = plane_of(*fr[j], p);
    k.d[p] = plane_of(*dst, p);
    max_w = fe->w[p] > max_w ? fe->w[p] : max_w;
    max_h = fe->h[p] > max_h ? fe->h[p] : max_h;
  }
  k.lut = l ? l->table : nullptr;
  k.lutp = lp ? lp->table : nullptr;
  k.tr1 = q->tr1;
  k.w1 = q->w1;
  k.inv1 = q->inv1;
  k.w2 = q->w2;
  k.inv2 = q->inv2;
  k.smode = q->smode;
  k.nref = q->nref;
  k.sovs = q->sovs;
  const dim3 grid((max_w + SL_X - 1) / SL_X, (max_h + SL_Y - 1) / SL_Y, fe->f.planes);
  mvc::pipe_submit([=](cudaStream_t s) { k_qtgmc_sharp<<<grid, 256, 0, s>>>(k); }); // issued by the pipeline submitter
  mvc::frame_put(dst, producer, frame, *out);
  return 0;
}