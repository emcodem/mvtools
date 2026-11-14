// SplatFps picture build on the GPU: Laplacian-pyramid forward splatting, i.e. motion-compensated
// projection of both source pictures to the output time, band by band, with a normalised weighted sum, an in-place
// fallback blend and a cross-fade in holes. Kernels and the host-memory driver. Included twice, by
// mvcuda_splat_exact.cu (SPLAT_EXACT, compiled with -fmad=false) and mvcuda_splat_fast.cu, inside namespace SPLAT_NS;
// mvcuda_splat.cu exports the C entry and picks one.
//
// The algorithm and its exact CPU path: splatfps/core/splat_core.cpp. The exact path is the deterministic scalar
// path: float32 in a fixed evaluation order, no contraction, IEEE division and sqrt, exp through double; the fast CPU
// path and CUDA (exact=true) are checked bit for bit against it. Exact variant: every per-sample value is computed
// with the exact path's operations in the exact path's order, and the splats are added to each cell in the exact
// path's order (sorted passes, k_ordered), so it is bit-identical. Fast variant: FMA contraction, the single-precision exp and atomic splats (any order).
//
// Data: planes of float32, packed rows. A picture is 3 planes, a field 4 (vx, vy, err, local-fallback weight).
// Full-resolution fields are never stored: finest levels evaluate the x scale bilinear up-sampling of the
// ME-resolution projection field at the sample they need.
//
// Settings: splat::Config (splatfps/core/splat_config.h), from MvCudaSplatParams::config or the defaults of the
// picture size; the kernels get the values they use as arguments, the fixed filter tables come from splat_const.h.

#include "mvcuda_internal.h"
#include "../../splatfps/core/splat_config.h"

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#ifdef SPLAT_EXACT
#include <cub/device/device_radix_sort.cuh>
#endif

namespace SPLAT_NS
{
  constexpr int MAXL = splat::CFG_MAX_LEVELS + 1;
  // pyramid filters (splat_const.h) as scalars, so device code can use them
  // pyramid reduce taps (centre, distance 1, 2, 3), from splat::Config::pyr_reduce
  struct RTaps { float d0, d1, d2, d3; };
  constexpr float EA0 = splat::PYR_EXPAND_A[0], EA1 = splat::PYR_EXPAND_A[1], EA2 = splat::PYR_EXPAND_A[2];
  constexpr float EB0 = splat::PYR_EXPAND_B[0], EB1 = splat::PYR_EXPAND_B[1], EB2 = splat::PYR_EXPAND_B[2],
    EB3 = splat::PYR_EXPAND_B[3];
  constexpr float EC0 = splat::PYR_EXPAND_C[0], EC1 = splat::PYR_EXPAND_C[1], EC2 = splat::PYR_EXPAND_C[2];
  constexpr float VMIN = splat::SAMPLE_MIN, VMAX = splat::SAMPLE_MAX;

  // the settings of a parameter set: its Config, or the defaults of its picture size
  static splat::Config cfg_of(const MvCudaSplatParams &p)
  {
    return p.config ? *static_cast<const splat::Config *>(p.config) : splat::config_for_size(p.width, p.height);
  }

  __device__ __forceinline__ int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

  // edge-clamped sample of plane p (int offsets: a plane has fewer than 2^31 samples)
  __device__ __forceinline__ float ldc(const float *p, int w, int h, int x, int y)
  {
    return p[clampi(y, 0, h - 1) * w + clampi(x, 0, w - 1)];
  }

  __device__ __forceinline__ float exp_ref(float x)
  {
#ifdef SPLAT_EXACT
    return (float)exp((double)x);
#else
    return __expf(x);
#endif
  }

  // ---- pyramid ---------------------------------------------------------------------------------------------------
  // picture W x H -> internal grid GW x GH with an edge-repeat border of px / py (3 planes along blockIdx.z)
  __global__ void k_pad(const float *src, int W, int H, float *dst, int GW, int GH, int px, int py)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.z;
    if (x >= GW || y >= GH)
      return;
    dst[(size_t)c * GW * GH + (size_t)y * GW + x] = ldc(src + (size_t)c * W * H, W, H, x - px, y - py);
  }

  // pyramid reduce filter: symmetric 7 taps (centre, pairs at distance 1, 2, 3); k_reduce_v / k_reduce_h keep even samples
  __device__ __forceinline__ float dn_tap(const RTaps &D, float g0, float gm1, float gp1, float gm2, float gp2,
    float gm3, float gp3)
  {
    float r = D.d0 * g0;
    r = r + D.d1 * (gm1 + gp1);
    r = r + D.d2 * (gm2 + gp2);
    r = r + D.d3 * (gm3 + gp3);
    return r;
  }

  // vertical pass: src w x h -> dst w x oh (even rows)
  __global__ void k_reduce_v(const float *src, int w, int h, float *dst, int oh, RTaps D)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.z;
    if (x >= w || j >= oh)
      return;
    const float *s = src + (size_t)c * w * h;
    const int y = 2 * j;
    float *o = dst + (size_t)c * w * oh + (size_t)j * w + x;
    if (y >= 3 && y + 3 <= h - 1)
    {
      const float *p = s + y * w + x; // all taps inside: no clamps
      *o = dn_tap(D, p[0], p[-w], p[w], p[-2 * w], p[2 * w], p[-3 * w], p[3 * w]);
      return;
    }
    *o = dn_tap(D, ldc(s, w, h, x, y), ldc(s, w, h, x, y - 1), ldc(s, w, h, x, y + 1), ldc(s, w, h, x, y - 2),
      ldc(s, w, h, x, y + 2), ldc(s, w, h, x, y - 3), ldc(s, w, h, x, y + 3));
  }

  // the vertical pass of two pyramids with different taps from the same level (the two-build blend's level 0): each
  // source sample read once, both results as k_reduce_v computes them
  __global__ void k_reduce_v2(const float *src, int w, int h, float *dst, float *dst2, int oh, RTaps D, RTaps D2)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.z;
    if (x >= w || j >= oh)
      return;
    const float *s = src + (size_t)c * w * h;
    const int y = 2 * j;
    const size_t oi = (size_t)c * w * oh + (size_t)j * w + x;
    float g0, gm1, gp1, gm2, gp2, gm3, gp3;
    if (y >= 3 && y + 3 <= h - 1)
    {
      const float *p = s + y * w + x; // all taps inside: no clamps
      g0 = p[0]; gm1 = p[-w]; gp1 = p[w]; gm2 = p[-2 * w]; gp2 = p[2 * w]; gm3 = p[-3 * w]; gp3 = p[3 * w];
    }
    else
    {
      g0 = ldc(s, w, h, x, y); gm1 = ldc(s, w, h, x, y - 1); gp1 = ldc(s, w, h, x, y + 1);
      gm2 = ldc(s, w, h, x, y - 2); gp2 = ldc(s, w, h, x, y + 2); gm3 = ldc(s, w, h, x, y - 3);
      gp3 = ldc(s, w, h, x, y + 3);
    }
    dst[oi] = dn_tap(D, g0, gm1, gp1, gm2, gp2, gm3, gp3);
    dst2[oi] = dn_tap(D2, g0, gm1, gp1, gm2, gp2, gm3, gp3);
  }

  // horizontal pass: src w x oh -> dst ow x oh (even columns)
  __global__ void k_reduce_h(const float *src, int w, int oh, float *dst, int ow, RTaps D)
  {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.z;
    if (i >= ow || y >= oh)
      return;
    const float *s = src + (size_t)c * w * oh;
    const int x = 2 * i;
    float *o = dst + (size_t)c * ow * oh + (size_t)y * ow + i;
    if (x >= 3 && x + 3 <= w - 1)
    {
      const float *p = s + y * w + x; // all taps inside: no clamps
      *o = dn_tap(D, p[0], p[-1], p[1], p[-2], p[2], p[-3], p[3]);
      return;
    }
    *o = dn_tap(D, ldc(s, w, oh, x, y), ldc(s, w, oh, x - 1, y), ldc(s, w, oh, x + 1, y), ldc(s, w, oh, x - 2, y),
      ldc(s, w, oh, x + 2, y), ldc(s, w, oh, x - 3, y), ldc(s, w, oh, x + 3, y));
  }

  // sample (X, Y) of the 2x up-sampling of l (lw x lh, edge clamp): pyramid expand, one 2-D kernel per output phase
  // (X, Y even / odd), around low sample (X / 2, Y / 2)
  __device__ float up_at(const float *l, int lw, int lh, int X, int Y)
  {
    const int x = X >> 1, y = Y >> 1;
    auto C = [&](int dy, int dx) { return ldc(l, lw, lh, x + dx, y + dy); };
    const float c = C(0, 0), n = C(-1, 0), wst = C(0, -1), e = C(0, 1), s = C(1, 0);
    const float nw = C(-1, -1), ne = C(-1, 1), sw = C(1, -1), se = C(1, 1);
    float r;
    if (!(Y & 1) && !(X & 1))
    {
      r = EA0 * c;
      r = r + EA1 * (((n + wst) + e) + s);
      r = r + EA2 * (((nw + ne) + sw) + se);
    }
    else if (!(Y & 1))
    {
      r = EB0 * (c + e);
      r = r + EB1 * (wst + C(0, 2));
      r = r + EB2 * (((n + ne) + s) + se);
      r = r + EB3 * (((nw + C(-1, 2)) + sw) + C(1, 2));
    }
    else if (!(X & 1))
    {
      r = EB0 * (c + s);
      r = r + EB1 * (n + C(2, 0));
      r = r + EB2 * (((wst + e) + sw) + se);
      r = r + EB3 * (((nw + ne) + C(2, -1)) + C(2, 1));
    }
    else
    {
      r = EC0 * (((c + e) + s) + se);
      r = r + EC1 * (((((((n + ne) + C(2, 0)) + C(2, 1)) + wst) + C(0, 2)) + sw) + C(1, 2));
      r = r + EC2 * (((nw + C(-1, 2)) + C(2, -1)) + C(2, 2));
    }
    return r;
  }

#ifndef SPLAT_EXACT
  // Fast variant of up_at for a 32 x 8 thread block covering 32 x 16 outputs (each thread rows y and y + 8): the expand
  // kernel is separable, taps (1/8, 3/4, 1/8) around an even output sample and (1/2, 1/2) at an odd one, so each
  // output reads only its 3 x 3 (or fewer) low samples, from a shared tile the block loads once (edge clamp as ldc).
  // The exact variant keeps up_at's full evaluation order.
  constexpr int UP_BW = 32, UP_TH = 8, UP_BH = 2 * UP_TH;
  constexpr int UPT_W = UP_BW / 2 + 2, UPT_H = UP_BH / 2 + 2;
  constexpr float UP_E0 = 0.75f, UP_E1 = 0.125f, UP_O = 0.5f;
  static_assert(EA0 == UP_E0 * UP_E0 && EA1 == UP_E0 * UP_E1 && EA2 == UP_E1 * UP_E1, "expand kernel A not separable");
  static_assert(EB0 == UP_E0 * UP_O && EB1 == 0.0f && EB2 == UP_E1 * UP_O && EB3 == 0.0f, "expand kernel B not separable");
  static_assert(EC0 == UP_O * UP_O && EC1 == 0.0f && EC2 == 0.0f, "expand kernel C not separable");

  // the block's low samples: tile [ty][tx] = l(clamp(bx + tx), clamp(by + ty)), bx / by one left of / above the
  // block's first centre sample
  __device__ __forceinline__ void up_tile_load(float (&t)[UPT_H][UPT_W], const float *l, int lw, int lh)
  {
    const int bx = (int)(blockIdx.x * UP_BW / 2) - 1, by = (int)(blockIdx.y * UP_BH / 2) - 1;
    for (int k = threadIdx.y * UP_BW + threadIdx.x; k < UPT_H * UPT_W; k += UP_BW * UP_TH)
    {
      const int ty = k / UPT_W, tx = k - ty * UPT_W;
      t[ty][tx] = l[clampi(by + ty, 0, lh - 1) * lw + clampi(bx + tx, 0, lw - 1)];
    }
  }

  // the up-sampling at this thread's sample (X, Y) = (blockIdx.x * 32 + threadIdx.x, blockIdx.y * 16 + threadIdx.y + 8 u)
  __device__ __forceinline__ float up_tile_at(const float (&t)[UPT_H][UPT_W], int u)
  {
    const int cx = (threadIdx.x >> 1) + 1, cy = (threadIdx.y >> 1) + 1 + u * (UP_TH / 2);
    const bool ox = threadIdx.x & 1, oy = threadIdx.y & 1;
    const float x0 = ox ? 0.0f : UP_E1, x1 = ox ? UP_O : UP_E0, x2 = ox ? UP_O : UP_E1;
    const float y0 = oy ? 0.0f : UP_E1, y1 = oy ? UP_O : UP_E0, y2 = oy ? UP_O : UP_E1;
    auto row = [&](int r) { return x0 * t[r][cx - 1] + x1 * t[r][cx] + x2 * t[r][cx + 1]; };
    return y0 * row(cy - 1) + y1 * row(cy) + y2 * row(cy + 1);
  }
#endif

  // Laplacian band = low - up(low1), low1 the next coarser low-pass level
  // fast variant: block 32 x 8 over 32 x 16 samples (up_tile_at)
  __global__ void k_band(const float *low, int w, int h, const float *low1, int lw, int lh, float *band)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, c = blockIdx.z;
#ifdef SPLAT_EXACT
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
      return;
    const size_t i = (size_t)c * w * h + (size_t)y * w + x;
    band[i] = low[i] - up_at(low1 + (size_t)c * lw * lh, lw, lh, x, y);
#else
    __shared__ float t[UPT_H][UPT_W];
    up_tile_load(t, low1 + (size_t)c * lw * lh, lw, lh);
    __syncthreads();
    if (x >= w)
      return;
    for (int u = 0; u < 2; u++)
    {
      const int yy = blockIdx.y * UP_BH + threadIdx.y + u * UP_TH;
      if (yy >= h)
        break;
      const size_t i = (size_t)c * w * h + (size_t)yy * w + x;
      band[i] = low[i] - up_tile_at(t, u);
    }
#endif
  }

  // the bands of two pyramids of the same picture at the same level (the two-build blend's level 0): each sample of
  // low read once, both bands as k_band computes them (low1 / low1b the two next coarser levels)
  __global__ void k_band2(const float *low, int w, int h, const float *low1, const float *low1b, int lw, int lh, float *band,
    float *bandb)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, c = blockIdx.z;
    const size_t cl = (size_t)c * lw * lh;
#ifdef SPLAT_EXACT
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
      return;
    const size_t i = (size_t)c * w * h + (size_t)y * w + x;
    const float v = low[i];
    band[i] = v - up_at(low1 + cl, lw, lh, x, y);
    bandb[i] = v - up_at(low1b + cl, lw, lh, x, y);
#else
    __shared__ float t[UPT_H][UPT_W], tb[UPT_H][UPT_W];
    up_tile_load(t, low1 + cl, lw, lh);
    up_tile_load(tb, low1b + cl, lw, lh);
    __syncthreads();
    if (x >= w)
      return;
    for (int u = 0; u < 2; u++)
    {
      const int yy = blockIdx.y * UP_BH + threadIdx.y + u * UP_TH;
      if (yy >= h)
        break;
      const size_t i = (size_t)c * w * h + (size_t)yy * w + x;
      const float v = low[i];
      band[i] = v - up_tile_at(t, u);
      bandb[i] = v - up_tile_at(tb, u);
    }
#endif
  }

  // ---- fields ----------------------------------------------------------------------------------------------------
  // projection field of one side at ME resolution (vectors scaled to the output time, weights, border): P (n_side 0)
  // moves by d of its forward vector, N by 1 - d of its backward vector; pr = 1 - that fraction is the side's temporal
  // proximity. Splat weight gain * pr, or gain * max(min_w, exp(-err / err_scale)) * pr with err_scale > 0; fallback
  // weight * pr. The border xb x yb ME pixels (the padding area) takes the nearest cell inside the picture. edge: the
  // fallback weight is dropped where the cell's own vector carries it out of the picture (splat::fallback_edge_keep_at,
  // margin edge_margin).
  struct PrepArgs { int xb, yb; float err_scale, min_w, gain; int edge; float edge_margin; };
  __global__ void k_splat_prep(const float *F, int mw, int mh, float d, int n_side, PrepArgs a, float *G)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= mw || y >= mh)
      return;
    const size_t n = (size_t)mw * mh, i = (size_t)y * mw + x;
    const float g = n_side ? 1.0f - d : d;
    const float pr = 1.0f - g;
    const int sx = min(max(x, a.xb), mw - 1 - a.xb), sy = min(max(y, a.yb), mh - 1 - a.yb);
    const size_t s = (size_t)sy * mw + sx;
    G[i] = F[s] * g;
    G[n + i] = F[n + s] * g;
    const float m = fmaxf(a.min_w, a.err_scale > 0.0f ? exp_ref(-F[2 * n + s] / a.err_scale) : 1.0f);
    G[2 * n + i] = (a.gain * m) * pr;
    const float keep = a.edge ? splat::fallback_edge_keep_at(a.xb, a.yb, a.edge_margin, sx, sy, F[s], F[n + s], mw, mh)
                              : 1.0f;
    G[3 * n + i] = F[3 * n + s] * pr * keep;
  }

  static PrepArgs prep_args(const splat::Config &c)
  {
    return { c.pad_x / c.scale, c.pad_y / c.scale, c.splat_err_scale, c.splat_min_weight, c.splat_gain, c.fallback_edge,
      c.fallback_edge_margin };
  }

  // 3x3 [1 2 1] reduction (edge clamp), vectors / 2
  __global__ void k_field_reduce2(const float *F, int w, int h, float *o, int ow, int oh)
  {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.z;
    if (i >= ow || j >= oh)
      return;
    const float *s = F + (size_t)c * w * h;
    auto g = [&](int y, int x) { return s[(size_t)min(max(y, 0), h - 1) * w + min(max(x, 0), w - 1)]; };
    const int y = 2 * j, x = 2 * i;
    float r = 0.0625f * (((g(y - 1, x - 1) + g(y - 1, x + 1)) + g(y + 1, x - 1)) + g(y + 1, x + 1));
    r = r + 0.125f * (((g(y - 1, x) + g(y, x - 1)) + g(y, x + 1)) + g(y + 1, x));
    r = r + 0.25f * g(y, x);
    if (c < 2)
      r = r / 2.0f;
    o[(size_t)c * ow * oh + (size_t)j * ow + i] = r;
  }

  // the field of one level: the full-resolution field read with a stride, or the level's own coarse field
  struct LevelField
  {
    const float *G;  // ME-resolution projection field (stride > 0) or the coarse field (stride 0)
    int gw, gh;
    int stride;
    int sh;          // log2 of the ME scale (stride > 0)
    float still;     // vectors up to this long stay in place
  };

  // one side's field at a band sample: vector in the level's pixels, splat weight mc, in-place weight st, zero flag
  struct SideSample { float vx, vy, mc, st; bool zero; };

  // vectors / stride; a vector of length <= still (Config::still_vector) is not splatted, its splat weight joins the in-place weight
  // 2^-k for 0 <= k < 127: x / 2^k and x * 2^-k round the same exact value, so they are the same float
  __device__ __forceinline__ float pow2_recip(int k) { return __int_as_float((127 - k) << 23); }

  __device__ __forceinline__ SideSample side_at(const LevelField &f, int x, int y)
  {
    float v[4];
    float is = 1.0f; // 1 / stride (a power of two)
    if (f.stride)
    {
      // the x (1 << sh) bilinear up-sampling of G (vectors * scale) at full-resolution sample (X, Y), edge clamp; the
      // sample geometry once for the 4 channels
      const int X = x * f.stride, Y = y * f.stride, sh = f.sh, msk = (1 << sh) - 1;
      const float fs = (float)(1 << sh), ifs = pow2_recip(sh);
      const int ix = X >> sh, iy = Y >> sh;
      const int c0 = clampi(ix, 0, f.gw - 1), c1 = clampi(ix + 1, 0, f.gw - 1);
      const int r0 = clampi(iy, 0, f.gh - 1) * f.gw, r1 = clampi(iy + 1, 0, f.gh - 1) * f.gw;
      const float tx = (float)(X & msk) * ifs, ty = (float)(Y & msk) * ifs;
      const int n = f.gw * f.gh;
#pragma unroll
      for (int c = 0; c < 4; c++)
      {
        const float *p = f.G + c * n;
        float a = p[r0 + c0], b = p[r0 + c1], cc = p[r1 + c0], e = p[r1 + c1];
        if (c < 2)
        {
          a = a * fs; b = b * fs; cc = cc * fs; e = e * fs;
        }
        const float l6 = a * (1.0f - tx) + b * tx;
        const float l18 = cc * (1.0f - tx) + e * tx;
        v[c] = l6 * (1.0f - ty) + l18 * ty;
      }
      is = pow2_recip(__ffs(f.stride) - 1);
    }
    else
    {
      // edge clamp: a coarse level can be one sample larger than its field (the picture pyramid rounds up)
      const size_t n = (size_t)f.gw * f.gh, i = (size_t)min(y, f.gh - 1) * f.gw + min(x, f.gw - 1);
      for (int c = 0; c < 4; c++)
        v[c] = f.G[c * n + i];
    }
    SideSample r;
    r.vx = v[0] * is;
    r.vy = v[1] * is;
    r.mc = v[2];
    r.st = v[3];
#ifdef SPLAT_EXACT
    r.zero = sqrtf(r.vx * r.vx + r.vy * r.vy) <= f.still;
#else
    r.zero = f.still >= 0.0f && r.vx * r.vx + r.vy * r.vy <= f.still * f.still;
#endif
    if (r.zero)
      r.st = r.st + r.mc;
    return r;
  }

  // ---- one level ---------------------------------------------------------------------------------------------------
  // Accumulator layout. Vector accumulation (fast variant compiled for compute capability 9.0+, where float4 atomicAdd
  // exists): a cell's four values (3 channels, weight) interleaved, one vector atomic per contribution, and k_splat adds
  // the in-place parts itself. Otherwise planes (the exact variant's ordered splats add plane by plane; scalar atomics
  // on interleaved cells measured several times slower) and k_inplace writes the in-place parts first.
#if !defined(SPLAT_EXACT) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define SPLAT_VEC_ACC 1
#endif
  // index of value c (0..2 channels, 3 weight) of accumulator cell ai (an cells)
  __device__ __forceinline__ size_t acc_ix(int c, size_t ai, size_t an)
  {
#ifdef SPLAT_VEC_ACC
    return 4 * ai + (size_t)c;
#else
    return (size_t)c * an + ai;
#endif
  }

  // accumulator: 4 planes (3 channels, weight) of aw x ah = (w + 1) x (h + 1) (the extra row / column takes the corner
  // splats of targets on the last row / column). k_inplace writes the in-place part P st_P + N st_N: the
  // non-motion-compensated blend that the local fallback asks for
  __global__ void k_inplace(const float *P, const float *N, int w, int h, LevelField fp, LevelField fn, float *acc)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
      return;
    const int aw = w + 1, ah = h + 1;
    const size_t an = (size_t)aw * ah, ai = (size_t)y * aw + x, n = (size_t)w * h, i = (size_t)y * w + x;
    const SideSample a = side_at(fp, x, y), b = side_at(fn, x, y);
    for (int c = 0; c < 3; c++)
      acc[acc_ix(c, ai, an)] = P[c * n + i] * a.st + N[c * n + i] * b.st;
    acc[acc_ix(3, ai, an)] = a.st + b.st;
  }

  // bilinear forward splats of one side (blockIdx.z: 0 = P, 1 = N): the sample moves to (x + vx, y + vy) and adds
  // sample * mc and mc to the 4 surrounding cells with the bilinear weights; targets outside the band are dropped.
  // Fast variant only (atomic adds, any order)
#ifdef SPLAT_VEC_ACC
  // Vector accumulation (fast variant, compute capability 9.0+): one float4 atomic per cell contribution, the in-place
  // part (k_inplace's term) added here too (the accumulator starts at zero). A block row is one warp (32 x 8 blocks).
  // When all 32 samples of the warp move by the same whole-pixel offset (pans, and smooth motion in general), lane k's
  // right-hand corners are lane k + 1's left-hand cells: each lane adds its left neighbour's right-hand corners to its
  // own left-hand ones (warp shuffles) and only lane 31 adds right-hand corners, so a sample costs 2 atomics instead of
  // 4 (plus the in-place part, folded in when the offset is zero). When the next block row has the same offset, the
  // bottom-row cells are handed to it through shared memory: about 1 atomic per sample. Otherwise each sample adds its
  // 4 corners. Launch with 32 x SPLAT_BH blocks.
  constexpr int SPLAT_BH = 8;
  __device__ __forceinline__ float4 add4(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
  __device__ __forceinline__ float4 shfl_up4(float4 v)
  {
    return make_float4(__shfl_up_sync(0xffffffffu, v.x, 1), __shfl_up_sync(0xffffffffu, v.y, 1),
      __shfl_up_sync(0xffffffffu, v.z, 1), __shfl_up_sync(0xffffffffu, v.w, 1));
  }
  __global__ void k_splat(const float *P, const float *N, int w, int h, LevelField fp, LevelField fn, float *acc)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, side = blockIdx.z;
    const int lane = threadIdx.x; // blockDim.x == 32
    const bool in = x < w && y < h;
    const int aw = w + 1, ah = h + 1;
    const size_t an = (size_t)aw * ah, n = (size_t)w * h, i = (size_t)y * w + x;
    const float *pic = side ? N : P;
    SideSample t = {};
    bool mv = false;
    int ix = 0, iy = 0;
    float fx = 0.0f, fy = 0.0f;
    if (in)
    {
      t = side_at(side ? fn : fp, x, y);
      if (!t.zero)
      {
        const float px = (float)x + t.vx, py = (float)y + t.vy;
        if (px >= 0.0f && px <= (float)(w - 1) && py >= 0.0f && py <= (float)(h - 1))
        {
          const float fix = floorf(px), fiy = floorf(py);
          ix = (int)fix;
          iy = (int)fiy;
          fx = px - fix;
          fy = py - fiy;
          mv = true;
        }
      }
    }
    // every lane takes part in both shuffles and the vote (a short-circuited shuffle would leave lanes out)
    const int dx = ix - x, dy = iy - y;
    const int dx0 = __shfl_sync(0xffffffffu, dx, 0), dy0 = __shfl_sync(0xffffffffu, dy, 0);
    const bool uni = __all_sync(0xffffffffu, mv && dx == dx0 && dy == dy0);
    float4 inpl = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (in && t.st != 0.0f)
      inpl = make_float4(pic[i] * t.st, pic[n + i] * t.st, pic[2 * n + i] * t.st, t.st);
    float4 c[4] = {};
    if (mv)
    {
      const float wa[4] = { (1.0f - fx) * (1.0f - fy), fx * (1.0f - fy), (1.0f - fx) * fy, fx * fy };
      float lt[3];
      for (int k = 0; k < 3; k++)
        lt[k] = pic[k * n + i] * t.mc;
      for (int k = 0; k < 4; k++)
        c[k] = make_float4(lt[0] * wa[k], lt[1] * wa[k], lt[2] * wa[k], wa[k] * t.mc);
    }
    auto put = [&](int cx, int cy, float4 v) {
      atomicAdd(reinterpret_cast<float4 *>(&acc[acc_ix(0, (size_t)cy * aw + cx, an)]), v);
    };
    // per warp of the block: its whole-pixel offset when uniform; the bottom-row cells handed to the warp below
    __shared__ int2 offs[SPLAT_BH];
    __shared__ float4 bot[SPLAT_BH][33];
    const int wy = threadIdx.y;
    const bool fold = dx == 0 && dy == 0; // uniform and offset zero: the sample's own cell is its top-left target
    if (uni)
    {
      // the left neighbour's top-right / bottom-right corners land on this lane's top-left / bottom-left cells
      const float4 l1 = shfl_up4(c[1]), l3 = shfl_up4(c[3]);
      if (lane > 0)
      {
        c[0] = add4(c[0], l1);
        c[2] = add4(c[2], l3);
      }
      if (fold)
        c[0] = add4(c[0], inpl);
    }
    if (lane == 0)
      offs[wy] = uni ? make_int2(dx, dy) : make_int2(INT_MIN, INT_MIN);
    __syncthreads();
    // with the same offset in the next block row, this warp's bottom-row cells are that warp's top-row cells: hand
    // them over (one atomic per cell instead of two)
    const int2 me = offs[wy];
    const bool up = uni && wy > 0 && offs[wy - 1].x == me.x && offs[wy - 1].y == me.y;
    const bool down = uni && wy < SPLAT_BH - 1 && offs[wy + 1].x == me.x && offs[wy + 1].y == me.y;
    if (down)
    {
      bot[wy][lane] = c[2];
      if (lane == 31)
        bot[wy][32] = c[3];
    }
    __syncthreads();
    if (uni)
    {
      if (up)
      {
        c[0] = add4(c[0], bot[wy - 1][lane]);
        if (lane == 31)
          c[1] = add4(c[1], bot[wy - 1][32]);
      }
      if (!fold && t.st != 0.0f)
        put(x, y, inpl);
      put(ix, iy, c[0]);
      if (lane == 31)
        put(ix + 1, iy, c[1]);
      if (!down)
      {
        put(ix, iy + 1, c[2]);
        if (lane == 31)
          put(ix + 1, iy + 1, c[3]);
      }
      return;
    }
    if (!in)
      return;
    if (t.st != 0.0f)
      put(x, y, inpl);
    if (mv)
      for (int k = 0; k < 4; k++)
        put(ix + (k & 1), iy + (k >> 1), c[k]);
  }
#else
  __global__ void k_splat(const float *P, const float *N, int w, int h, LevelField fp, LevelField fn, float *acc)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, side = blockIdx.z;
    if (x >= w || y >= h)
      return;
    const SideSample t = side_at(side ? fn : fp, x, y);
    const int aw = w + 1, ah = h + 1;
    const size_t an = (size_t)aw * ah, n = (size_t)w * h, i = (size_t)y * w + x;
    const float *pic = side ? N : P;
    if (t.zero)
      return;
    const float px = (float)x + t.vx, py = (float)y + t.vy;
    if (!(px >= 0.0f && px <= (float)(w - 1) && py >= 0.0f && py <= (float)(h - 1)))
      return;
    const float ix = floorf(px), iy = floorf(py);
    const float fx = px - ix, fy = py - iy;
    const float wa[4] = { (1.0f - fx) * (1.0f - fy), fx * (1.0f - fy), (1.0f - fx) * fy, fx * fy };
    const size_t base = (size_t)iy * aw + (size_t)ix;
    const size_t off[4] = { 0, 1, (size_t)aw, (size_t)aw + 1 };
    float lt[3];
    for (int c = 0; c < 3; c++)
      lt[c] = pic[c * n + i] * t.mc;
    for (int k = 0; k < 4; k++)
    {
      for (int c = 0; c < 3; c++)
        atomicAdd(&acc[acc_ix(c, base + off[k], an)], lt[c] * wa[k]);
      atomicAdd(&acc[acc_ix(3, base + off[k], an)], wa[k] * t.mc);
    }
  }
#endif

#ifdef SPLAT_EXACT
  // Ordered splats (exact variant): the exact path adds the contributions side by side, corner by corner, source pixels
  // in raster order. One pass per side and corner: k_keys gives every source pixel its target cell (none = `none`), a
  // stable radix sort groups them by cell keeping the raster order, and k_ordered adds each cell's group in that order.
  struct Splat { float lt[3], wt; size_t cell; bool ok; };

  __device__ __forceinline__ Splat splat_of(const float *pic, int w, int h, const LevelField &f, int x, int y, int corner)
  {
    Splat r;
    r.ok = false;
    const SideSample t = side_at(f, x, y);
    if (t.zero)
      return r;
    const float px = (float)x + t.vx, py = (float)y + t.vy;
    if (!(px >= 0.0f && px <= (float)(w - 1) && py >= 0.0f && py <= (float)(h - 1)))
      return r;
    const float ix = floorf(px), iy = floorf(py);
    const float fx = px - ix, fy = py - iy;
    float a;
    int ox = 0, oy = 0;
    switch (corner)
    {
    case 0: a = (1.0f - fx) * (1.0f - fy); break;
    case 1: a = fx * (1.0f - fy); ox = 1; break;
    case 2: a = (1.0f - fx) * fy; oy = 1; break;
    default: a = fx * fy; ox = 1; oy = 1; break;
    }
    const size_t n = (size_t)w * h, i = (size_t)y * w + x;
    for (int c = 0; c < 3; c++)
      r.lt[c] = (pic[c * n + i] * t.mc) * a;
    r.wt = a * t.mc;
    r.cell = ((size_t)iy + oy) * (w + 1) + (size_t)ix + ox;
    r.ok = true;
    return r;
  }

  __global__ void k_keys(const float *pic, int w, int h, LevelField f, int corner, unsigned none, unsigned *key, unsigned *idx)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
      return;
    const Splat s = splat_of(pic, w, h, f, x, y, corner);
    const size_t i = (size_t)y * w + x;
    key[i] = s.ok ? (unsigned)s.cell : none;
    idx[i] = (unsigned)i;
  }

  // one thread per sorted position; the first of each cell's group adds the whole group in order
  __global__ void k_ordered(const float *pic, int w, int h, LevelField f, int corner, unsigned none, const unsigned *key,
    const unsigned *idx, float *acc)
  {
    const size_t n = (size_t)w * h;
    const size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n)
      return;
    const unsigned k = key[j];
    if (k == none || (j > 0 && key[j - 1] == k))
      return;
    const size_t an = (size_t)(w + 1) * (h + 1);
    float s0 = acc[k], s1 = acc[an + k], s2 = acc[2 * an + k], s3 = acc[3 * an + k];
    for (size_t q = j; q < n && key[q] == k; q++)
    {
      const unsigned src = idx[q];
      const Splat s = splat_of(pic, w, h, f, (int)(src % (unsigned)w), (int)(src / (unsigned)w), corner);
      s0 = s0 + s.lt[0];
      s1 = s1 + s.lt[1];
      s2 = s2 + s.lt[2];
      s3 = s3 + s.wt;
    }
    acc[k] = s0; acc[an + k] = s1; acc[2 * an + k] = s2; acc[3 * an + k] = s3;
  }
#endif

  // a whole accumulator cell (3 channels, weight)
  __device__ __forceinline__ float4 acc_cell(const float *acc, int ai, size_t an)
  {
#ifdef SPLAT_VEC_ACC
    return *reinterpret_cast<const float4 *>(&acc[acc_ix(0, ai, an)]);
#else
    return make_float4(acc[acc_ix(0, ai, an)], acc[acc_ix(1, ai, an)], acc[acc_ix(2, ai, an)], acc[acc_ix(3, ai, an)]);
#endif
  }
#ifdef SPLAT_EXACT
  constexpr int NORM_Z = 3;
#else
  constexpr int NORM_Z = 1;
#endif

  // normalisation (or cross-fade where nothing landed) plus the reconstruction step: out = built + up(coarser out);
  // coarser == nullptr at the coarsest level; clamp at the finest; hole: total weight below this (Config::hole_weight)
  // fast variant: block 32 x 8 over 32 x 16 samples (up_tile_at), all 3 channels per thread (one launch with
  // NORM_Z = 1: each accumulator cell read once); the exact variant one channel per blockIdx.z (NORM_Z = 3).
  // out: ow x oh planes holding sample (ox + x', oy + y') at (x', y'); the finest level writes the cropped picture
  // this way, the others their whole level (ow = w, ox = 0)
  __global__ void k_norm(const float *acc, const float *P, const float *N, int w, int h, float d, const float *coarser,
    int cw, int ch, int clamp, float hole, float *out, int ow, int oh, int ox, int oy)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, c = blockIdx.z;
    const int aw = w + 1, ah = h + 1;
    const size_t an = (size_t)aw * ah, n = (size_t)w * h;
#ifdef SPLAT_EXACT
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h || x < ox || x >= ox + ow || y < oy || y >= oy + oh)
      return;
    const size_t ai = (size_t)y * aw + x, i = (size_t)y * w + x;
    const float sw = acc[acc_ix(3, ai, an)];
    float b = sw < hole ? P[c * n + i] * (1.0f - d) + N[c * n + i] * d : acc[acc_ix(c, ai, an)] / sw;
    if (coarser)
    {
      b = b + up_at(coarser + (size_t)c * cw * ch, cw, ch, x, y);
      if (clamp)
        b = fminf(fmaxf(b, VMIN), VMAX);
    }
    out[c * (size_t)ow * oh + (size_t)(y - oy) * ow + (x - ox)] = b;
#else
    (void)c;
    __shared__ float t[3][UPT_H][UPT_W];
    const size_t cn = (size_t)cw * ch, on = (size_t)ow * oh;
    if (coarser) // uniform over the launch
    {
      for (int k = 0; k < 3; k++)
        up_tile_load(t[k], coarser + k * cn, cw, ch);
      __syncthreads();
    }
    // the output window lies inside the level (ox + ow <= w, oy + oh <= h)
    if ((unsigned)(x - ox) >= (unsigned)ow)
      return;
#pragma unroll
    for (int u = 0; u < 2; u++)
    {
      const int y = blockIdx.y * UP_BH + threadIdx.y + u * UP_TH;
      if ((unsigned)(y - oy) < (unsigned)oh)
      {
        const int ai = y * aw + x, i = y * w + x;
        const float4 A = acc_cell(acc, ai, an);
        const float av[3] = { A.x, A.y, A.z };
        float *o = out + (size_t)(y - oy) * ow + (x - ox);
#pragma unroll
        for (int k = 0; k < 3; k++)
        {
          float b = A.w < hole ? P[k * n + i] * (1.0f - d) + N[k * n + i] * d : __fdividef(av[k], A.w);
          if (coarser)
          {
            b = b + up_tile_at(t[k], u);
            if (clamp)
              b = fminf(fmaxf(b, VMIN), VMAX);
          }
          o[k * on] = b;
        }
      }
    }
#endif
  }

  // ---- two-build blend -----------------------------------------------------------------------------------------
  // The operations and their order of splat::blend_error_map and splat::blend_mix_rows (splat_core.h).
  __device__ __forceinline__ float max_ref(float a, float b) { return a < b ? b : a; } // std::max

  // error map on the ME grid: e = (3 x 3 sum of max(errP, errN), edge clamp, dy outer, dx inner) / 9
  __global__ void k_blend_err(const float *fP, const float *fN, int mw, int mh, float *e)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= mw || y >= mh)
      return;
    const size_t n = (size_t)mw * mh;
    const float *eP = fP + 2 * n, *eN = fN + 2 * n;
    float s = 0.0f;
    for (int dy = -1; dy <= 1; dy++)
      for (int dx = -1; dx <= 1; dx++)
      {
        const size_t k = (size_t)clampi(y + dy, 0, mh - 1) * mw + clampi(x + dx, 0, mw - 1);
        s = s + max_ref(eP[k], eN[k]);
      }
    e[(size_t)y * mw + x] = s / 9.0f;
  }

  // the mix, in place on main (3 planes W x H): w = smoothstep of the bilinear error at the pixel, out = w main +
  // (1 - w) second
  struct BlendArgs { int sc, off_x, off_y; float elo, einv; };
  static BlendArgs blend_args(const splat::Config &c)
  {
    return { c.scale, c.pad_x, c.pad_y, (float)c.blend_err_lo, (float)(1.0 / (c.blend_err_hi - c.blend_err_lo)) };
  }
  __device__ __forceinline__ float clampf_ref(float v, float lo, float hi) // std::min(std::max(v, lo), hi)
  {
    const float m = v < lo ? lo : v;
    return hi < m ? hi : m;
  }
  __global__ void k_blend_mix(float *main, const float *second, int W, int H, const float *e, int mw, int mh,
    BlendArgs a)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
      return;
    const float my = clampf_ref((y + a.off_y + 0.5f) / a.sc - 0.5f, 0.0f, (float)(mh - 1));
    const int ya = (int)my, yb = min(ya + 1, mh - 1);
    const float fy = my - ya;
    const float mx = clampf_ref((x + a.off_x + 0.5f) / a.sc - 0.5f, 0.0f, (float)(mw - 1));
    const int xa = (int)mx, xb = min(xa + 1, mw - 1);
    const float fx = mx - xa;
    const float *r0 = e + (size_t)ya * mw, *r1 = e + (size_t)yb * mw;
    const float ev = (1 - fy) * ((1 - fx) * r0[xa] + fx * r0[xb]) + fy * ((1 - fx) * r1[xa] + fx * r1[xb]);
    float t = (ev - a.elo) * a.einv;
    t = clampf_ref(t, 0.0f, 1.0f);
    const float w = t * t * (3.0f - 2.0f * t);
    const size_t n = (size_t)W * H, i = (size_t)y * W + x;
    for (int c = 0; c < 3; c++)
    {
      const float av = main[c * n + i];
      main[c * n + i] = w * av + (1.0f - w) * second[c * n + i];
    }
  }

#ifndef SPLAT_EXACT
  // ---- two-build blend, fast variant ----------------------------------------------------------------------------
  // the finest level of both builds and the mix in one pass: each build's normalisation plus reconstruction (k_norm's
  // operations, clamped), then w main + (1 - w) second with w from the error map (k_blend_mix's operations); the
  // second build's picture never goes to memory. All 3 channels per thread: each accumulator cell (interleaved with
  // vector accumulation) and the mix weight once per sample. Block 32 x 8 over 32 x 16 samples; out W x H at (ox, oy).
  __global__ void k_norm_mix(const float *acc, const float *P, const float *N, const float *coarser, const float *acc2,
    const float *P2, const float *N2, const float *coarser2, int w, int h, float d, int cw, int ch, float hole, float *out,
    int ow, int oh, int ox, int oy, const float *e, int mw, int mh, BlendArgs a)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int aw = w + 1, ah = h + 1;
    const size_t an = (size_t)aw * ah, n = (size_t)w * h, cn = (size_t)cw * ch, on = (size_t)ow * oh;
    __shared__ float t[3][UPT_H][UPT_W], t2[3][UPT_H][UPT_W];
    for (int c = 0; c < 3; c++)
    {
      up_tile_load(t[c], coarser + c * cn, cw, ch);
      up_tile_load(t2[c], coarser2 + c * cn, cw, ch);
    }
    __syncthreads();
    if ((unsigned)(x - ox) >= (unsigned)ow)
      return;
    // the error map's horizontal sample position of this output column
    const float mx = clampf_ref((x - ox + a.off_x + 0.5f) / a.sc - 0.5f, 0.0f, (float)(mw - 1));
    const int xa = (int)mx, xb = min(xa + 1, mw - 1);
    const float fx = mx - xa;
#pragma unroll
    for (int u = 0; u < 2; u++)
    {
      const int y = blockIdx.y * UP_BH + threadIdx.y + u * UP_TH;
      if ((unsigned)(y - oy) < (unsigned)oh)
      {
        const int ai = y * aw + x, i = y * w + x;
        const float my = clampf_ref((y - oy + a.off_y + 0.5f) / a.sc - 0.5f, 0.0f, (float)(mh - 1));
        const int ya = (int)my, yb = min(ya + 1, mh - 1);
        const float fy = my - ya;
        const float *r0 = e + (size_t)ya * mw, *r1 = e + (size_t)yb * mw;
        const float ev = (1 - fy) * ((1 - fx) * r0[xa] + fx * r0[xb]) + fy * ((1 - fx) * r1[xa] + fx * r1[xb]);
        float tt = (ev - a.elo) * a.einv;
        tt = clampf_ref(tt, 0.0f, 1.0f);
        const float wm = tt * tt * (3.0f - 2.0f * tt);
        const float4 A = acc_cell(acc, ai, an), A2 = acc_cell(acc2, ai, an);
        const float av[3] = { A.x, A.y, A.z }, av2[3] = { A2.x, A2.y, A2.z };
        float *o = out + (size_t)(y - oy) * ow + (x - ox);
#pragma unroll
        for (int c = 0; c < 3; c++)
        {
          float b = A.w < hole ? P[c * n + i] * (1.0f - d) + N[c * n + i] * d : __fdividef(av[c], A.w);
          b = fminf(fmaxf(b + up_tile_at(t[c], u), VMIN), VMAX);
          float b2 = A2.w < hole ? P2[c * n + i] * (1.0f - d) + N2[c * n + i] * d : __fdividef(av2[c], A2.w);
          b2 = fminf(fmaxf(b2 + up_tile_at(t2[c], u), VMIN), VMAX);
          o[c * on] = wm * b + (1.0f - wm) * b2;
        }
      }
    }
  }
#endif

  // ---- host driver -------------------------------------------------------------------------------------------------
  struct Bufs
  {
    std::vector<std::pair<void *, size_t>> v;
    bool failed = false;
    float *get(size_t floats)
    {
      void *p = mvc::dev_alloc(floats * sizeof(float));
      if (!p)
        failed = true;
      else
        v.push_back({ p, floats * sizeof(float) });
      return (float *)p;
    }
    ~Bufs()
    {
      for (auto &b : v)
        mvc::dev_free(b.first, b.second);
    }
  };

  static dim3 grid(int w, int h, int z, dim3 b) { return dim3((w + b.x - 1) / b.x, (h + b.y - 1) / b.y, z); }
  // the samples one 32 x 8 block of k_band / k_norm covers (for grid())
  static dim3 up_cover()
  {
#ifdef SPLAT_EXACT
    return dim3(32, 8);
#else
    return dim3(UP_BW, UP_BH);
#endif
  }

  // Scratch of the picture build for one parameter set. A device context keeps one for its lifetime (no pool traffic and
  // no wait per frame); build_dev makes one per call.
  struct BuildScratch
  {
    Bufs B;
    splat::Config cfg;
    int W = 0, H = 0, L = 0, GW = 0, GH = 0, MW = 0, MH = 0;
    int lw[MAXL + 1], lh[MAXL + 1];
    float *proj[2], *low[2][MAXL + 1], *band[2][MAXL + 1], *tmp, *acc, *rec[2];
    std::vector<float *> red[2]; // coarse fields made by repeated reduction, in the order build_run uses them
#ifdef SPLAT_EXACT
    unsigned *keys[2], *idxs[2];
    void *sort_tmp = nullptr;
    size_t sort_bytes = 0;
#endif
    // -4 bad parameters, -2 out of device memory
    int alloc(const splat::Config &c, int width, int height, cudaStream_t st)
    {
      cfg = c;
      W = width;
      H = height;
      L = c.levels;
      if (W < 16 || H < 16 || L < 0 || L >= MAXL || c.stride_max < 1)
        return -4;
      GW = W + 2 * c.pad_x;
      GH = H + 2 * c.pad_y;
      MW = GW / c.scale;
      MH = GH / c.scale;
      lw[0] = GW;
      lh[0] = GH;
      for (int i = 0; i < L; i++)
      {
        lw[i + 1] = (lw[i] + 1) / 2;
        lh[i + 1] = (lh[i] + 1) / 2;
      }
      for (int s = 0; s < 2; s++)
      {
        proj[s] = B.get((size_t)4 * MW * MH);
        for (int i = 0; i <= L; i++)
        {
          low[s][i] = B.get((size_t)3 * lw[i] * lh[i]);
          band[s][i] = i < L ? B.get((size_t)3 * lw[i] * lh[i]) : low[s][i];
        }
      }
      tmp = B.get((size_t)3 * GW * ((GH + 1) / 2));
      acc = B.get((size_t)4 * (GW + 1) * (GH + 1));
      // reconstruction of the levels above 0 (level 0 goes straight to the output): rec[i & 1], largest i = 2 / 1
      rec[0] = B.get((size_t)3 * (L >= 2 ? lw[2] * lh[2] : 1));
      rec[1] = B.get((size_t)3 * (L >= 1 ? lw[1] * lh[1] : 1));
      // the reductions build_run makes (same walk over the levels)
      const int log2max = splat::ilog2(c.scale);
      bool have = false;
      int cw = MW, chh = MH;
      for (int i = 0; i <= L; i++)
      {
        if ((1 << i) <= c.stride_max)
          continue;
        const int nred = have ? 1 : i - log2max;
        for (int r = 0; r < nred; r++)
        {
          cw /= 2;
          chh /= 2;
          for (int s = 0; s < 2; s++)
            red[s].push_back(B.get((size_t)4 * cw * chh));
        }
        have = true;
      }
#ifdef SPLAT_EXACT
      const size_t npix = (size_t)GW * GH;
      for (int k = 0; k < 2; k++)
      {
        keys[k] = (unsigned *)B.get(npix);
        idxs[k] = (unsigned *)B.get(npix);
      }
      cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, keys[0], keys[1], idxs[0], idxs[1], (int)npix, 0, 32, st);
      sort_tmp = B.get((sort_bytes + 3) / 4);
#endif
      return B.failed ? -2 : 0;
    }
    // a picture's own pyramid (bands 0 .. L - 1 and the coarsest low-pass level L) in one buffer of pyr_floats()
    size_t pyr_floats() const
    {
      size_t n = 0;
      for (int i = 0; i <= L; i++)
        n += (size_t)3 * lw[i] * lh[i];
      return n;
    }
    void pyr_ptrs(float *base, float **band) const
    {
      for (int i = 0; i <= L; i++)
      {
        band[i] = base;
        base += (size_t)3 * lw[i] * lh[i];
      }
    }
  };

  // L2 cache size of the current device in bytes (queried once)
  static size_t l2_bytes()
  {
    static size_t v = 0;
    if (!v)
    {
      int dev = 0, b = 0;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&b, cudaDevAttrL2CacheSize, dev);
      v = b > 0 ? (size_t)b : (size_t)1 << 20;
    }
    return v;
  }

  // pad P (3 planes W x H) and decompose it into a Laplacian pyramid: low[0 .. L] the low-pass levels (low[L] is
  // band[L], the coarsest level), band[0 .. L - 1] the bands; S.tmp is the reduce scratch. padded: low[0] already
  // holds P padded with S's border (another pyramid of the same picture made it), so P is not read. v0: the vertical
  // pass of level 0 is already there (k_reduce_v2, made together with another pyramid of the picture).
  // band0 false: the level-0 band is left out (k_band2 makes it together with another pyramid's).
  static void pyramid_run(cudaStream_t st, const BuildScratch &S, const float *P, float *const *low, float *const *band,
    bool padded = false, const float *v0 = nullptr, bool band0 = true)
  {
    const splat::Config &cfg = S.cfg;
    const int L = S.L;
    const int *lw = S.lw, *lh = S.lh;
    const RTaps rt = { cfg.pyr_reduce[0], cfg.pyr_reduce[1], cfg.pyr_reduce[2], cfg.pyr_reduce[3] };
    const dim3 blk(32, 8), blk_up = up_cover();
    if (!padded)
      k_pad<<<grid(S.GW, S.GH, 3, blk), blk, 0, st>>>(P, S.W, S.H, low[0], S.GW, S.GH, cfg.pad_x, cfg.pad_y);
    for (int i = 0; i < L; i++)
    {
      const bool have = i == 0 && v0;
      const float *v = have ? v0 : S.tmp;
      if (!have)
        k_reduce_v<<<grid(lw[i], lh[i + 1], 3, blk), blk, 0, st>>>(low[i], lw[i], lh[i], S.tmp, lh[i + 1], rt);
      k_reduce_h<<<grid(lw[i + 1], lh[i + 1], 3, blk), blk, 0, st>>>(v, lw[i], lh[i + 1], low[i + 1], lw[i + 1], rt);
    }
    for (int i = band0 ? 0 : 1; i < L; i++)
      k_band<<<grid(lw[i], lh[i], 3, blk_up), blk, 0, st>>>(low[i], lw[i], lh[i], low[i + 1], lw[i + 1], lh[i + 1], band[i]);
  }

#ifndef SPLAT_EXACT
  // true when the loaded k_splat was compiled with vector accumulation (SPLAT_VEC_ACC): decided by the PTX version of
  // the code the driver runs, which can be older than the device (PTX JIT-compiled for a newer GPU)
  static bool splat_vec_acc()
  {
    static int v = -1;
    if (v < 0)
    {
      cudaFuncAttributes a = {};
      v = cudaFuncGetAttributes(&a, k_splat) == cudaSuccess && a.ptxVersion >= 90 ? 1 : 0;
    }
    return v == 1;
  }
#endif

  // The picture build on device buffers: P, N 3 planes W x H, fwdP / bwdN 4 planes of the ME grid, dout 3 planes W x H.
  // Steps: pad and decompose both pictures into Laplacian pyramids, projection fields of both sides, per-level fields,
  // then per level from coarse to fine: in-place weights, splats, normalisation plus reconstruction; finally crop.
  // Only queues work on st (S belongs to st until it is done). -2 on a launch failure.
  // pyrP / pyrN: the pictures' pyramids made by pyramid_run (BuildScratch::pyr_ptrs layout), or null: made here.
  // the per-level fields of both sides into lf: projection fields at ME resolution (k_splat_prep), then per level a
  // stride into them, or own coarse fields made by repeated reduction (the first coarse level from the ME-resolution
  // field, 1 / scale)
  static void level_fields(cudaStream_t st, BuildScratch &S, const float *fwdP, const float *bwdN, float d,
    LevelField (&lf)[2][MAXL + 1])
  {
    const splat::Config &cfg = S.cfg;
    const int L = S.L, MW = S.MW, MH = S.MH, sh = splat::ilog2(cfg.scale);
    const PrepArgs pa = prep_args(cfg);
    const dim3 blk(32, 8);
    const float *me[2] = { fwdP, bwdN };
    for (int s = 0; s < 2; s++)
      k_splat_prep<<<grid(MW, MH, 1, blk), blk, 0, st>>>(me[s], MW, MH, d, s, pa, S.proj[s]);
    const int log2max = sh;
    const float *cur[2] = { nullptr, nullptr };
    size_t ri = 0;
    int cw = MW, chh = MH;
    for (int i = 0; i <= L; i++)
    {
      const int stv = 1 << i;
      if (stv <= cfg.stride_max)
      {
        for (int s = 0; s < 2; s++)
          lf[s][i] = { S.proj[s], MW, MH, stv, sh, cfg.still_vector };
        continue;
      }
      const int nred = cur[0] ? 1 : i - log2max;
      for (int r = 0; r < nred; r++, ri++)
      {
        const int ow = cw / 2, oh = chh / 2;
        for (int s = 0; s < 2; s++)
        {
          const float *from = cur[s] ? cur[s] : S.proj[s];
          float *o = S.red[s][ri];
          k_field_reduce2<<<grid(ow, oh, 4, blk), blk, 0, st>>>(from, cw, chh, o, ow, oh);
          cur[s] = o;
        }
        cw = ow;
        chh = oh;
      }
      for (int s = 0; s < 2; s++)
        lf[s][i] = { cur[s], cw, chh, 0, sh, cfg.still_vector };
    }
  }

  static int build_run(cudaStream_t st, BuildScratch &S, const float *P, const float *N, const float *fwdP,
    const float *bwdN, float d, float *dout, float *const *pyrP = nullptr, float *const *pyrN = nullptr)
  {
    const splat::Config &cfg = S.cfg;
    const int W = S.W, H = S.H, L = S.L;
    const int *lw = S.lw, *lh = S.lh;
    const dim3 blk(32, 8), blk_up = up_cover();
    const float *src[2] = { P, N };
    float *const *pyr[2] = { pyrP, pyrN };
    const float *band[2][MAXL + 1];
    for (int s = 0; s < 2; s++)
    {
      if (!pyr[s])
        pyramid_run(st, S, src[s], S.low[s], S.band[s]);
      for (int i = 0; i <= L; i++)
        band[s][i] = pyr[s] ? pyr[s][i] : S.band[s][i];
    }
    LevelField lf[2][MAXL + 1];
    level_fields(st, S, fwdP, bwdN, d, lf);
    // levels from coarse to fine, each reconstructed on the way
    const float *prev = nullptr;
    int pw = 0, ph = 0;
    for (int i = L; i >= 0; i--)
    {
      const int w = lw[i], h = lh[i];
#ifdef SPLAT_EXACT
      cudaMemsetAsync(S.acc, 0, sizeof(float) * 4 * (w + 1) * (h + 1), st);
      k_inplace<<<grid(w, h, 1, blk), blk, 0, st>>>(band[0][i], band[1][i], w, h, lf[0][i], lf[1][i], S.acc);
      {
        const size_t n = (size_t)w * h;
        const unsigned none = (unsigned)((size_t)(w + 1) * (h + 1));
        int bits = 1;
        while ((1ull << bits) <= none)
          bits++;
        for (int side = 0; side < 2; side++)
          for (int corner = 0; corner < 4; corner++)
          {
            k_keys<<<grid(w, h, 1, blk), blk, 0, st>>>(band[side][i], w, h, lf[side][i], corner, none, S.keys[0],
              S.idxs[0]);
            size_t tb = S.sort_bytes;
            if (cub::DeviceRadixSort::SortPairs(S.sort_tmp, tb, S.keys[0], S.keys[1], S.idxs[0], S.idxs[1], (int)n, 0,
                  bits, st) != cudaSuccess)
              return -2;
            k_ordered<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(band[side][i], w, h, lf[side][i], corner, none,
              S.keys[1], S.idxs[1], S.acc);
          }
      }
#else
      // with vector accumulation k_splat adds the in-place parts too (side by side, into a zeroed accumulator)
      cudaMemsetAsync(S.acc, 0, sizeof(float) * 4 * (w + 1) * (h + 1), st);
      if (!splat_vec_acc())
        k_inplace<<<grid(w, h, 1, blk), blk, 0, st>>>(band[0][i], band[1][i], w, h, lf[0][i], lf[1][i], S.acc);
      k_splat<<<grid(w, h, 2, blk), blk, 0, st>>>(band[0][i], band[1][i], w, h, lf[0][i], lf[1][i], S.acc);
#endif
      // the finest level goes straight into dout without the border (W x H at pad_x, pad_y)
      if (i == 0)
        k_norm<<<grid(w, h, NORM_Z, blk_up), blk, 0, st>>>(S.acc, band[0][i], band[1][i], w, h, d, prev, pw, ph, 1,
          cfg.hole_weight, dout, W, H, cfg.pad_x, cfg.pad_y);
      else
      {
        float *o = S.rec[i & 1];
        k_norm<<<grid(w, h, NORM_Z, blk_up), blk, 0, st>>>(S.acc, band[0][i], band[1][i], w, h, d, prev, pw, ph, 0,
          cfg.hole_weight, o, w, h, 0, 0);
        prev = o;
        pw = w;
        ph = h;
      }
    }
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -2;
  }

  // the two-build blend after both builds: error map of the ME fields into e (MW x MH), then the mix into main
  // (3 planes W x H, in place) with second
  static void blend_run(cudaStream_t st, const splat::Config &cfg, const float *fwdP, const float *bwdN, int MW, int MH,
    float *e, float *main, const float *second, int W, int H)
  {
    const dim3 blk(32, 8);
    k_blend_err<<<grid(MW, MH, 1, blk), blk, 0, st>>>(fwdP, bwdN, MW, MH, e);
    k_blend_mix<<<grid(W, H, 1, blk), blk, 0, st>>>(main, second, W, H, e, MW, MH, blend_args(cfg));
  }

#ifndef SPLAT_EXACT
  // true when the two builds can share their fields and finest pass (build_run_blend): vector accumulation (k_splat
  // adds the in-place parts), the same level count and sizes (the second build differs only in its pyramid filter)
  // and at least one level above the finest
  static bool blend_fusable(const BuildScratch &S, const BuildScratch &S2)
  {
    if (!splat_vec_acc() || S.L != S2.L || S.L < 1 || S.GW != S2.GW || S.GH != S2.GH || S.MW != S2.MW || S.MH != S2.MH)
      return false;
    return S.cfg.scale == S2.cfg.scale && S.cfg.stride_max == S2.cfg.stride_max;
  }

  // Both builds of the two-build blend and the mix (fast variant): the fields once (S's scratch), per level a splat
  // pass per build, the coarser levels normalised per build, the finest level of both normalised and mixed straight
  // into dout. pyr / pyr2: the pictures' pyramids of the two builds (null: made here,
  // the second from the first one's padded picture). e: the error map (S.MW x S.MH).
  static int build_run_blend(cudaStream_t st, BuildScratch &S, BuildScratch &S2, const float *P, const float *N,
    const float *fwdP, const float *bwdN, float d, float *dout, float *e, float *const *pyrP, float *const *pyrN,
    float *const *pyr2P, float *const *pyr2N)
  {
    const splat::Config &cfg = S.cfg;
    const int W = S.W, H = S.H, L = S.L;
    const int *lw = S.lw, *lh = S.lh;
    const dim3 blk(32, 8), blk_up = up_cover();
    const float *src[2] = { P, N };
    float *const *pyr[2][2] = { { pyrP, pyrN }, { pyr2P, pyr2N } };
    const float *band[2][2][MAXL + 1]; // [build][side][level]
    for (int s = 0; s < 2; s++)
    {
      if (!pyr[0][s])
        pyramid_run(st, S, src[s], S.low[s], S.band[s]);
      if (!pyr[1][s])
      {
        // the second pyramid from the first one's padded picture (same border)
        float *low[MAXL + 1];
        for (int i = 0; i <= L; i++)
          low[i] = S2.low[s][i];
        low[0] = pyr[0][s] ? nullptr : S.low[s][0];
        pyramid_run(st, S2, src[s], low[0] ? low : S2.low[s], S2.band[s], low[0] != nullptr);
      }
      for (int i = 0; i <= L; i++)
      {
        band[0][s][i] = pyr[0][s] ? pyr[0][s][i] : S.band[s][i];
        band[1][s][i] = pyr[1][s] ? pyr[1][s][i] : S2.band[s][i];
      }
    }
    LevelField lf[2][MAXL + 1];
    level_fields(st, S, fwdP, bwdN, d, lf);
    k_blend_err<<<grid(S.MW, S.MH, 1, blk), blk, 0, st>>>(fwdP, bwdN, S.MW, S.MH, e);
    const BlendArgs ba = blend_args(cfg);
    const float *prev[2] = { nullptr, nullptr };
    int pw = 0, ph = 0;
    for (int i = L; i >= 0; i--)
    {
      const int w = lw[i], h = lh[i];
      const size_t abytes = sizeof(float) * 4 * (w + 1) * (h + 1);
      // one splat pass per build, each right after its accumulator is cleared (the memset leaves it in L2, where the
      // atomics then hit; clearing it in the previous normalisation instead measured 14 % slower at HD); a single
      // pass into both accumulators measured slower too (twice the atomic working set in L2 at once)
      cudaMemsetAsync(S.acc, 0, abytes, st);
      k_splat<<<grid(w, h, 2, blk), blk, 0, st>>>(band[0][0][i], band[0][1][i], w, h, lf[0][i], lf[1][i], S.acc);
      cudaMemsetAsync(S2.acc, 0, abytes, st);
      k_splat<<<grid(w, h, 2, blk), blk, 0, st>>>(band[1][0][i], band[1][1][i], w, h, lf[0][i], lf[1][i], S2.acc);
      if (i == 0)
        k_norm_mix<<<grid(w, h, 1, blk_up), blk, 0, st>>>(S.acc, band[0][0][i], band[0][1][i], prev[0], S2.acc,
          band[1][0][i], band[1][1][i], prev[1], w, h, d, pw, ph, cfg.hole_weight, dout, W, H, cfg.pad_x, cfg.pad_y, e,
          S.MW, S.MH, ba);
      else
      {
        float *o = S.rec[i & 1], *o2 = S2.rec[i & 1];
        k_norm<<<grid(w, h, NORM_Z, blk_up), blk, 0, st>>>(S.acc, band[0][0][i], band[0][1][i], w, h, d, prev[0], pw, ph, 0,
          cfg.hole_weight, o, w, h, 0, 0);
        k_norm<<<grid(w, h, NORM_Z, blk_up), blk, 0, st>>>(S2.acc, band[1][0][i], band[1][1][i], w, h, d, prev[1], pw, ph, 0,
          S2.cfg.hole_weight, o2, w, h, 0, 0);
        prev[0] = o;
        prev[1] = o2;
        pw = w;
        ph = h;
      }
    }
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -2;
  }
#endif

  // build_run with its own scratch; waits for st before returning (the scratch goes back to the pool). part -1: the
  // whole build (with the two-build blend when the settings have it); 0: the main build alone; 1: the second build
  // alone (splat::blend_config).
  static int build_dev(const MvCudaSplatParams &p, cudaStream_t st, const float *P, const float *N, const float *fwdP,
    const float *bwdN, float d, float *dout, int part = -1)
  {
    const splat::Config cfg = cfg_of(p);
    const bool blend = part < 0 && cfg.blend_sigma > 0.0;
    int rc;
    {
      BuildScratch S, S2;
      Bufs B;
      rc = S.alloc(part == 1 ? splat::blend_config(cfg) : cfg, p.width, p.height, st);
#ifndef SPLAT_EXACT
      if (!rc && blend)
      {
        float *e = B.get((size_t)S.MW * S.MH);
        rc = B.failed ? -2 : S2.alloc(splat::blend_config(cfg), p.width, p.height, st);
        if (!rc && blend_fusable(S, S2))
        {
          rc = build_run_blend(st, S, S2, P, N, fwdP, bwdN, d, dout, e, nullptr, nullptr, nullptr, nullptr);
          if (cudaStreamSynchronize(st) != cudaSuccess || cudaGetLastError() != cudaSuccess)
            return -2;
          return rc;
        }
      }
#endif
      if (!rc)
        rc = build_run(st, S, P, N, fwdP, bwdN, d, dout);
      if (!rc && blend)
      {
        float *second = B.get((size_t)3 * p.width * p.height), *e = B.get((size_t)S.MW * S.MH);
        rc = B.failed ? -2 : S2.alloc(splat::blend_config(cfg), p.width, p.height, st);
        if (!rc)
          rc = build_run(st, S2, P, N, fwdP, bwdN, d, second);
        if (!rc)
          blend_run(st, cfg, fwdP, bwdN, S.MW, S.MH, e, dout, second, p.width, p.height);
      }
      if (cudaStreamSynchronize(st) != cudaSuccess || cudaGetLastError() != cudaSuccess)
        return -2;
    }
    return rc;
  }

  int build_host(const MvCudaSplatParams &p, const float *P, const float *N, const float *fwdP, const float *bwdN, float d,
    float *out, int part)
  {
    const splat::Config cfg = cfg_of(p);
    const int W = p.width, H = p.height, MW = splat::me_width(cfg, W), MH = splat::me_height(cfg, H);
    mvc::StreamLease lease;
    cudaStream_t st = lease.s;
    Bufs B;
    float *src[2] = { B.get((size_t)3 * W * H), B.get((size_t)3 * W * H) };
    float *me[2] = { B.get((size_t)4 * MW * MH), B.get((size_t)4 * MW * MH) };
    float *dout = B.get((size_t)3 * W * H);
    if (B.failed)
      return -2;
    const float *host_src[2] = { P, N }, *host_me[2] = { fwdP, bwdN };
    for (int s = 0; s < 2; s++)
    {
      cudaMemcpyAsync(src[s], host_src[s], sizeof(float) * 3 * W * H, cudaMemcpyHostToDevice, st);
      cudaMemcpyAsync(me[s], host_me[s], sizeof(float) * 4 * MW * MH, cudaMemcpyHostToDevice, st);
    }
    int rc = build_dev(p, st, src[0], src[1], me[0], me[1], d, dout, part);
    if (rc)
      return rc;
    cudaMemcpyAsync(out, dout, sizeof(float) * 3 * W * H, cudaMemcpyDeviceToHost, st);
    if (cudaStreamSynchronize(st) != cudaSuccess || cudaGetLastError() != cudaSuccess)
      return -2;
    return 0;
  }

  // ---- host-memory entries of single build stages (tests)
  // one data set: host -> device for every input, the kernels, device -> host for the output
  struct HostRun
  {
    mvc::StreamLease lease;
    Bufs B;
    ~HostRun() { cudaStreamSynchronize(lease.s); } // before the buffers go back (an early return may leave copies queued)
    float *up(const float *h, size_t n)
    {
      float *d = B.get(n);
      if (d)
        cudaMemcpyAsync(d, h, sizeof(float) * n, cudaMemcpyHostToDevice, lease.s);
      return d;
    }
    int down(float *h, const float *d, size_t n)
    {
      cudaMemcpyAsync(h, d, sizeof(float) * n, cudaMemcpyDeviceToHost, lease.s);
      return (cudaStreamSynchronize(lease.s) != cudaSuccess || cudaGetLastError() != cudaSuccess) ? -2 : 0;
    }
  };

  // one pyramid reduce step of one plane (w x h) with the main taps or (second) those of the second build
  int pyr_reduce_host(const MvCudaSplatParams &p, const float *src, int w, int h, int second, float *dst)
  {
    const splat::Config cfg = cfg_of(p);
    const float *t = second ? cfg.blend_reduce : cfg.pyr_reduce;
    const RTaps rt = { t[0], t[1], t[2], t[3] };
    const int oh = (h + 1) / 2, ow = (w + 1) / 2;
    HostRun R;
    float *s = R.up(src, (size_t)w * h), *tmp = R.B.get((size_t)w * oh), *o = R.B.get((size_t)ow * oh);
    if (R.B.failed)
      return -2;
    const dim3 blk(32, 8);
    k_reduce_v<<<grid(w, oh, 1, blk), blk, 0, R.lease.s>>>(s, w, h, tmp, oh, rt);
    k_reduce_h<<<grid(ow, oh, 1, blk), blk, 0, R.lease.s>>>(tmp, w, oh, o, ow, rt);
    return R.down(dst, o, (size_t)ow * oh);
  }

  // the projection field of one side (k_splat_prep) on a w x h field
  int border_weight_host(const MvCudaSplatParams &p, const float *F, int w, int h, float d, int n_side, float *G)
  {
    const splat::Config cfg = cfg_of(p);
    HostRun R;
    const size_t n = (size_t)4 * w * h;
    float *f = R.up(F, n), *g = R.B.get(n);
    if (R.B.failed)
      return -2;
    const dim3 blk(32, 8);
    k_splat_prep<<<grid(w, h, 1, blk), blk, 0, R.lease.s>>>(f, w, h, d, n_side, prep_args(cfg), g);
    return R.down(G, g, n);
  }

  int blend_error_map_host(const MvCudaSplatParams &p, const float *fwdP, const float *bwdN, float *e)
  {
    const splat::Config cfg = cfg_of(p);
    const int MW = splat::me_width(cfg, p.width), MH = splat::me_height(cfg, p.height);
    HostRun R;
    const size_t n = (size_t)MW * MH;
    float *a = R.up(fwdP, 4 * n), *b = R.up(bwdN, 4 * n), *o = R.B.get(n);
    if (R.B.failed)
      return -2;
    const dim3 blk(32, 8);
    k_blend_err<<<grid(MW, MH, 1, blk), blk, 0, R.lease.s>>>(a, b, MW, MH, o);
    return R.down(e, o, n);
  }

  int blend_mix_host(const MvCudaSplatParams &p, const float *main, const float *second, const float *e, float *out)
  {
    const splat::Config cfg = cfg_of(p);
    const int W = p.width, H = p.height, MW = splat::me_width(cfg, W), MH = splat::me_height(cfg, H);
    HostRun R;
    const size_t n = (size_t)3 * W * H;
    float *m = R.up(main, n), *s = R.up(second, n), *ed = R.up(e, (size_t)MW * MH);
    if (R.B.failed)
      return -2;
    const dim3 blk(32, 8);
    k_blend_mix<<<grid(W, H, 1, blk), blk, 0, R.lease.s>>>(m, s, W, H, ed, MW, MH, blend_args(cfg));
    return R.down(out, m, n);
  }
}
