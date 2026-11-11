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

  __device__ __forceinline__ float ldc(const float *p, int w, int h, int x, int y)
  {
    return p[(size_t)clampi(y, 0, h - 1) * w + clampi(x, 0, w - 1)];
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
    dst[(size_t)c * w * oh + (size_t)j * w + x] = dn_tap(D, ldc(s, w, h, x, y), ldc(s, w, h, x, y - 1), ldc(s, w, h, x, y + 1),
      ldc(s, w, h, x, y - 2), ldc(s, w, h, x, y + 2), ldc(s, w, h, x, y - 3), ldc(s, w, h, x, y + 3));
  }

  // horizontal pass: src w x oh -> dst ow x oh (even columns)
  __global__ void k_reduce_h(const float *src, int w, int oh, float *dst, int ow, RTaps D)
  {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.z;
    if (i >= ow || y >= oh)
      return;
    const float *s = src + (size_t)c * w * oh;
    const int x = 2 * i;
    dst[(size_t)c * ow * oh + (size_t)y * ow + i] = dn_tap(D, ldc(s, w, oh, x, y), ldc(s, w, oh, x - 1, y),
      ldc(s, w, oh, x + 1, y), ldc(s, w, oh, x - 2, y), ldc(s, w, oh, x + 2, y), ldc(s, w, oh, x - 3, y),
      ldc(s, w, oh, x + 3, y));
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

  // Laplacian band = low - up(low1), low1 the next coarser low-pass level
  __global__ void k_band(const float *low, int w, int h, const float *low1, int lw, int lh, float *band)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.z;
    if (x >= w || y >= h)
      return;
    const size_t i = (size_t)c * w * h + (size_t)y * w + x;
    band[i] = low[i] - up_at(low1 + (size_t)c * lw * lh, lw, lh, x, y);
  }

  // ---- fields ----------------------------------------------------------------------------------------------------
  // projection field of one side at ME resolution (vectors scaled to the output time, weights, border): P (n_side 0)
  // moves by d of its forward vector, N by 1 - d of its backward vector; pr = 1 - that fraction is the side's temporal
  // proximity. Splat weight gain * pr, or gain * max(min_w, exp(-err / err_scale)) * pr with err_scale > 0; fallback
  // weight * pr. The border xb x yb ME pixels (the padding area) takes the nearest cell inside the picture.
  struct PrepArgs { int xb, yb; float err_scale, min_w, gain; };
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
    G[3 * n + i] = F[3 * n + s] * pr;
  }

  // the x (1 << sh) bilinear up-sampling of G (vectors * scale) at full-resolution sample (X, Y), channel c
  __device__ __forceinline__ float full_at(const float *G, int mw, int mh, int X, int Y, int c, int sh)
  {
    const float *p = G + (size_t)c * mw * mh;
    const int x = X >> sh, y = Y >> sh, msk = (1 << sh) - 1;
    const float fs = (float)(1 << sh);
    float a = ldc(p, mw, mh, x, y), b = ldc(p, mw, mh, x + 1, y), cc = ldc(p, mw, mh, x, y + 1),
      e = ldc(p, mw, mh, x + 1, y + 1);
    if (c < 2)
    {
      a = a * fs; b = b * fs; cc = cc * fs; e = e * fs;
    }
    const float tx = (float)(X & msk) / fs, ty = (float)(Y & msk) / fs;
    const float l6 = a * (1.0f - tx) + b * tx;
    const float l18 = cc * (1.0f - tx) + e * tx;
    return l6 * (1.0f - ty) + l18 * ty;
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
  __device__ __forceinline__ SideSample side_at(const LevelField &f, int x, int y)
  {
    float v[4];
    if (f.stride)
    {
      for (int c = 0; c < 4; c++)
        v[c] = full_at(f.G, f.gw, f.gh, x * f.stride, y * f.stride, c, f.sh);
    }
    else
    {
      // edge clamp: a coarse level can be one sample larger than its field (the picture pyramid rounds up)
      const size_t n = (size_t)f.gw * f.gh, i = (size_t)min(y, f.gh - 1) * f.gw + min(x, f.gw - 1);
      for (int c = 0; c < 4; c++)
        v[c] = f.G[c * n + i];
    }
    const float s = (float)(f.stride ? f.stride : 1);
    SideSample r;
    r.vx = v[0] / s;
    r.vy = v[1] / s;
    r.mc = v[2];
    r.st = v[3];
    r.zero = sqrtf(r.vx * r.vx + r.vy * r.vy) <= f.still;
    if (r.zero)
      r.st = r.st + r.mc;
    return r;
  }

  // ---- one level ---------------------------------------------------------------------------------------------------
  // index of value c (0..2 channels, 3 weight) of accumulator cell ai (an cells): planes in the exact variant (the
  // ordered splats add plane by plane); the fast variant interleaves a cell's four values so one vector atomic adds
  // them together
#ifdef SPLAT_EXACT
  __device__ __forceinline__ size_t acc_ix(int c, size_t ai, size_t an) { return (size_t)c * an + ai; }
#else
  __device__ __forceinline__ size_t acc_ix(int c, size_t ai, size_t) { return 4 * ai + (size_t)c; }
#endif

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
  __global__ void k_splat(const float *P, const float *N, int w, int h, LevelField fp, LevelField fn, float *acc)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, side = blockIdx.z;
    if (x >= w || y >= h)
      return;
    const SideSample t = side_at(side ? fn : fp, x, y);
    const int aw = w + 1, ah = h + 1;
    const size_t an = (size_t)aw * ah, n = (size_t)w * h, i = (size_t)y * w + x;
    const float *pic = side ? N : P;
    // this side's in-place part (k_inplace's term) into the sample's own cell; the accumulator starts at zero
    if (t.st != 0.0f)
    {
      const size_t ai = (size_t)y * aw + x;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
      atomicAdd(reinterpret_cast<float4 *>(&acc[acc_ix(0, ai, an)]),
        make_float4(pic[i] * t.st, pic[n + i] * t.st, pic[2 * n + i] * t.st, t.st));
#else
      for (int c = 0; c < 3; c++)
        atomicAdd(&acc[acc_ix(c, ai, an)], pic[c * n + i] * t.st);
      atomicAdd(&acc[acc_ix(3, ai, an)], t.st);
#endif
    }
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
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
      atomicAdd(reinterpret_cast<float4 *>(&acc[acc_ix(0, base + off[k], an)]),
        make_float4(lt[0] * wa[k], lt[1] * wa[k], lt[2] * wa[k], wa[k] * t.mc));
#else
      for (int c = 0; c < 3; c++)
        atomicAdd(&acc[acc_ix(c, base + off[k], an)], lt[c] * wa[k]);
      atomicAdd(&acc[acc_ix(3, base + off[k], an)], wa[k] * t.mc);
#endif
    }
  }

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

  // normalisation (or cross-fade where nothing landed) plus the reconstruction step: out = built + up(coarser out);
  // coarser == nullptr at the coarsest level; clamp at the finest; hole: total weight below this (Config::hole_weight)
  __global__ void k_norm(const float *acc, const float *P, const float *N, int w, int h, float d, const float *coarser,
    int cw, int ch, int clamp, float hole, float *out)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.z;
    if (x >= w || y >= h)
      return;
    const int aw = w + 1, ah = h + 1;
    const size_t an = (size_t)aw * ah, ai = (size_t)y * aw + x, n = (size_t)w * h, i = (size_t)y * w + x;
    const float sw = acc[acc_ix(3, ai, an)];
    float b = sw < hole ? P[c * n + i] * (1.0f - d) + N[c * n + i] * d : acc[acc_ix(c, ai, an)] / sw;
    if (coarser)
    {
      b = b + up_at(coarser + (size_t)c * cw * ch, cw, ch, x, y);
      if (clamp)
        b = fminf(fmaxf(b, VMIN), VMAX);
    }
    out[c * n + i] = b;
  }

  // internal grid -> picture: drop the px / py border
  __global__ void k_crop(const float *src, int GW, int GH, float *dst, int W, int H, int px, int py)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.z;
    if (x >= W || y >= H)
      return;
    dst[(size_t)c * W * H + (size_t)y * W + x] = src[(size_t)c * GW * GH + (size_t)(y + py) * GW + x + px];
  }

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
      rec[0] = B.get((size_t)3 * GW * GH);
      rec[1] = B.get((size_t)3 * GW * GH);
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

  // pad P (3 planes W x H) and decompose it into a Laplacian pyramid: low[0 .. L] the low-pass levels (low[L] is
  // band[L], the coarsest level), band[0 .. L - 1] the bands; S.tmp is the reduce scratch
  static void pyramid_run(cudaStream_t st, const BuildScratch &S, const float *P, float *const *low, float *const *band)
  {
    const splat::Config &cfg = S.cfg;
    const int L = S.L;
    const int *lw = S.lw, *lh = S.lh;
    const RTaps rt = { cfg.pyr_reduce[0], cfg.pyr_reduce[1], cfg.pyr_reduce[2], cfg.pyr_reduce[3] };
    const dim3 blk(32, 8);
    k_pad<<<grid(S.GW, S.GH, 3, blk), blk, 0, st>>>(P, S.W, S.H, low[0], S.GW, S.GH, cfg.pad_x, cfg.pad_y);
    for (int i = 0; i < L; i++)
    {
      k_reduce_v<<<grid(lw[i], lh[i + 1], 3, blk), blk, 0, st>>>(low[i], lw[i], lh[i], S.tmp, lh[i + 1], rt);
      k_reduce_h<<<grid(lw[i + 1], lh[i + 1], 3, blk), blk, 0, st>>>(S.tmp, lw[i], lh[i + 1], low[i + 1], lw[i + 1], rt);
    }
    for (int i = 0; i < L; i++)
      k_band<<<grid(lw[i], lh[i], 3, blk), blk, 0, st>>>(low[i], lw[i], lh[i], low[i + 1], lw[i + 1], lh[i + 1], band[i]);
  }

  // The picture build on device buffers: P, N 3 planes W x H, fwdP / bwdN 4 planes of the ME grid, dout 3 planes W x H.
  // Steps: pad and decompose both pictures into Laplacian pyramids, projection fields of both sides, per-level fields,
  // then per level from coarse to fine: in-place weights, splats, normalisation plus reconstruction; finally crop.
  // Only queues work on st (S belongs to st until it is done). -2 on a launch failure.
  // pyrP / pyrN: the pictures' pyramids made by pyramid_run (BuildScratch::pyr_ptrs layout), or null: made here.
  static int build_run(cudaStream_t st, BuildScratch &S, const float *P, const float *N, const float *fwdP,
    const float *bwdN, float d, float *dout, float *const *pyrP = nullptr, float *const *pyrN = nullptr)
  {
    const splat::Config &cfg = S.cfg;
    const int W = S.W, H = S.H, L = S.L, GW = S.GW, GH = S.GH, MW = S.MW, MH = S.MH;
    const int *lw = S.lw, *lh = S.lh, sh = splat::ilog2(cfg.scale);
    const PrepArgs pa = { cfg.pad_x / cfg.scale, cfg.pad_y / cfg.scale, cfg.splat_err_scale, cfg.splat_min_weight,
      cfg.splat_gain };
    const dim3 blk(32, 8);
    const float *src[2] = { P, N };
    const float *me[2] = { fwdP, bwdN };
    float *const *pyr[2] = { pyrP, pyrN };
    const float *band[2][MAXL + 1];
    for (int s = 0; s < 2; s++)
    {
      if (!pyr[s])
        pyramid_run(st, S, src[s], S.low[s], S.band[s]);
      for (int i = 0; i <= L; i++)
        band[s][i] = pyr[s] ? pyr[s][i] : S.band[s][i];
      k_splat_prep<<<grid(MW, MH, 1, blk), blk, 0, st>>>(me[s], MW, MH, d, s, pa, S.proj[s]);
    }
    // per-level fields: stride into the full-resolution field, or own coarse fields made by repeated reduction (the
    // first coarse level from the ME-resolution field, 1 / scale)
    LevelField lf[2][MAXL + 1];
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
      // k_splat adds the in-place parts too (side by side, into a zeroed accumulator)
      cudaMemsetAsync(S.acc, 0, sizeof(float) * 4 * (w + 1) * (h + 1), st);
      k_splat<<<grid(w, h, 2, blk), blk, 0, st>>>(band[0][i], band[1][i], w, h, lf[0][i], lf[1][i], S.acc);
#endif
      float *o = S.rec[i & 1];
      k_norm<<<grid(w, h, 3, blk), blk, 0, st>>>(S.acc, band[0][i], band[1][i], w, h, d, prev, pw, ph, i == 0,
        cfg.hole_weight, o);
      prev = o;
      pw = w;
      ph = h;
    }
    k_crop<<<grid(W, H, 3, blk), blk, 0, st>>>(prev, GW, GH, dout, W, H, cfg.pad_x, cfg.pad_y);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -2;
  }

  // build_run with its own scratch; waits for st before returning (the scratch goes back to the pool).
  static int build_dev(const MvCudaSplatParams &p, cudaStream_t st, const float *P, const float *N, const float *fwdP,
    const float *bwdN, float d, float *dout)
  {
    BuildScratch S;
    int rc = S.alloc(cfg_of(p), p.width, p.height, st);
    if (!rc)
      rc = build_run(st, S, P, N, fwdP, bwdN, d, dout);
    if (cudaStreamSynchronize(st) != cudaSuccess || cudaGetLastError() != cudaSuccess)
      return -2;
    return rc;
  }

  int build_host(const MvCudaSplatParams &p, const float *P, const float *N, const float *fwdP, const float *bwdN, float d,
    float *out)
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
    int rc = build_dev(p, st, src[0], src[1], me[0], me[1], d, dout);
    if (rc)
      return rc;
    cudaMemcpyAsync(out, dout, sizeof(float) * 3 * W * H, cudaMemcpyDeviceToHost, st);
    if (cudaStreamSynchronize(st) != cudaSuccess || cudaGetLastError() != cudaSuccess)
      return -2;
    return 0;
  }
}
