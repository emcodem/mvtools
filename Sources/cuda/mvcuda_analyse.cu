// GPU version of MAnalyse's hierarchical search (GPU_PLAN.md M3), bit-exact against GroupOfPlanes::SearchMVs /
// PlaneOfBlocks::search_mv_slice for the production configuration: exhaustive search at every level, truemotion
// parameters, global MV, meander, 16-bit samples, chroma on, dct 0, divide 0, no temporal predictor, tryMany off.
//
// Parallelism. Inside a plane the block search is a chain: with meander a block depends on its left neighbour in scan
// order and on the block above, which snakes through the whole plane. The CPU code removes the chain by cutting the
// plane into horizontal slices that are searched independently (the "up" predictor and nLambda are dropped on the
// first row of a slice, the "bottom-right" predictor on the last row; see the "mt-inconsistency" comments in
// PlaneOfBlocks.cpp). That is what the multi-threaded CPU result already contains, and what is reproduced here:
// one CUDA block per slice, the slices of a plane run concurrently. slice_threads = 1 reproduces the single-thread
// CPU result, slice_threads = N the CPU result with N avstp threads (slices = min(N, 64, blkY / 4), at least 1).
// Inside a slice the blocks are processed one after the other, the candidate vectors of a block in parallel.
//
// Blocks whose result would trigger the CPU's "bad vector" wide search (UMH, depends on a counter shared by all
// slices) are not supported: the call returns MVCUDA_NEEDS_CPU and the caller falls back to the CPU.

#include "mvcuda_internal.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace
{
  typedef mvc_pix pix;
  static constexpr int SPW = 4 / (int)sizeof(mvc_pix); // samples per 4-byte word (cp.async granularity)

  const int FREQ_HALF = 8192; // PlaneOfBlocks: freqArray size 8192 * pel * 2, coarse levels have pel 1
  const int MAX_CAND = 64;

  struct LevelArgs
  {
    int level, n_blk_x, n_blk_y;
    int pel, log_pel, n_scale, log_scale, smallest;
    int bs_x, bs_y, ov_x, ov_y, cw, ch, log_x_uv, log_y_uv;
    int ext_w, ext_h, ew[3], hpad[3], vpad[3];
    const pix *src[3];
    const pix *ref[3][4];
    int lambda_level, lsad, pnew, pzero, pglobal, bits, chroma_scale, radius, meander, use_global;
    int bad_sad, n_slices;
    int async_src;   // 1: the source block can be copied with 4-byte cp.async (all plane rows 4-byte aligned)
    const int *glob;   // device: doubled global MV estimate of this level (x, y), already in units of this level's pel
    int *vec;          // device: (vx, vy, sad) triples of this level
    int *bad_flag;
  };

  __device__ __forceinline__ int warp_sum(int v)
  {
    for (int o = 16; o > 0; o >>= 1)
      v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
  }

  __device__ __forceinline__ int scale_chroma(int sad, int scale)
  {
    if (scale == 0)
      return sad;
    return scale > 0 ? sad >> scale : sad << (-scale);
  }

  __device__ int sad_block(const LevelArgs &a, int p, int x0, int y0, int w, int h, int xp, int yp, int lane)
  {
    const int sub = (xp & (a.pel - 1)) | ((yp & (a.pel - 1)) << a.log_pel);
    const pix *rp = a.ref[p][sub];
    const int rx = xp >> a.log_pel, ry = yp >> a.log_pel;
    const int ew = a.ew[p];
    int s = 0;
    for (int i = lane; i < w * h; i += 32)
    {
      const int r = i / w, c = i - r * w;
      s += abs((int)a.src[p][(size_t)(y0 + r) * ew + x0 + c] - (int)rp[(size_t)(ry + r) * ew + rx + c]);
    }
    return warp_sum(s);
  }

  // InterpolatePrediction<bigsad_t, bigsad_t> for 16-bit data
  __global__ void k_interp(LevelArgs a, const int *coarse, int c_blk_x, int c_blk_y, int log_pel_coarse)
  {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= a.n_blk_x * a.n_blk_y)
      return;
    int norm_factor = 3 - a.log_pel + log_pel_coarse;
    const int mul_factor = norm_factor < 0 ? -norm_factor : 0;
    norm_factor = norm_factor < 0 ? 0 : norm_factor;
    const int normov = (a.bs_x - a.ov_x) * (a.bs_y - a.ov_y);
    const int aoddx = a.bs_x * 3 - a.ov_x * 2, aevenx = a.bs_x * 3 - a.ov_x * 4;
    const int aoddy = a.bs_y * 3 - a.ov_y * 2, aeveny = a.bs_y * 3 - a.ov_y * 4;
    const bool no_overlap = (a.ov_x == 0 && a.ov_y == 0);
    const bool small_overlap = a.ov_x <= (a.bs_x >> 1) && a.ov_y <= (a.bs_y >> 1);

    const int k = index % a.n_blk_x, l = index / a.n_blk_x;
    int i = k, j = l;
    if (i >= 2 * c_blk_x)
      i = 2 * c_blk_x - 1;
    if (j >= 2 * c_blk_y)
      j = 2 * c_blk_y - 1;
    const int offy = -1 + 2 * (j % 2), offx = -1 + 2 * (i % 2);
    const int iper2 = i / 2, jper2 = j / 2;
    const int *v1, *v2, *v3, *v4;
#define CV(ix, iy) (coarse + 3 * ((ix) + (iy) * c_blk_x))
    if (i == 0 || i >= 2 * c_blk_x - 1)
    {
      if (j == 0 || j >= 2 * c_blk_y - 1)
        v1 = v2 = v3 = v4 = CV(iper2, jper2);
      else
      {
        v1 = v2 = CV(iper2, jper2);
        v3 = v4 = CV(iper2, jper2 + offy);
      }
    }
    else if (j == 0 || j >= 2 * c_blk_y - 1)
    {
      v1 = v2 = CV(iper2, jper2);
      v3 = v4 = CV(iper2 + offx, jper2);
    }
    else
    {
      v1 = CV(iper2, jper2);
      v2 = CV(iper2 + offx, jper2);
      v3 = CV(iper2, jper2 + offy);
      v4 = CV(iper2 + offx, jper2 + offy);
    }
#undef CV
    int x, y;
    long long tmp_sad;
    if (no_overlap)
    {
      x = 9 * v1[0] + 3 * v2[0] + 3 * v3[0] + v4[0];
      y = 9 * v1[1] + 3 * v2[1] + 3 * v3[1] + v4[1];
      tmp_sad = 9 * (long long)v1[2] + 3 * (long long)v2[2] + 3 * (long long)v3[2] + (long long)v4[2] + 8;
    }
    else if (small_overlap)
    {
      const int ax1 = offx > 0 ? aoddx : aevenx;
      const int ax2 = (a.bs_x - a.ov_x) * 4 - ax1;
      const int ay1 = offy > 0 ? aoddy : aeveny;
      const int ay2 = (a.bs_y - a.ov_y) * 4 - ay1;
      const int a11 = ax1 * ay1, a12 = ax1 * ay2, a21 = ax2 * ay1, a22 = ax2 * ay2;
      x = (a11 * v1[0] + a21 * v2[0] + a12 * v3[0] + a22 * v4[0]) / normov;
      y = (a11 * v1[1] + a21 * v2[1] + a12 * v3[1] + a22 * v4[1]) / normov;
      tmp_sad = ((long long)a11 * v1[2] + (long long)a21 * v2[2] + (long long)a12 * v3[2] + (long long)a22 * v4[2]) / normov;
    }
    else
    {
      x = (v1[0] + v2[0] + v3[0] + v4[0]) << 2;
      y = (v1[1] + v2[1] + v3[1] + v4[1]) << 2;
      tmp_sad = ((long long)v1[2] + v2[2] + v3[2] + v4[2] + 2) << 2;
    }
    a.vec[3 * index + 0] = (x >> norm_factor) << mul_factor;
    a.vec[3 * index + 1] = (y >> norm_factor) << mul_factor;
    a.vec[3 * index + 2] = (int)(tmp_sad >> 4);
  }

  // PlaneOfBlocks::estimate_global_mv_doubled_slice on the vectors of the coarser plane; one CUDA block.
  // freq: device scratch of 2 * 2 * FREQ_HALF ints.
  __global__ void k_global(const int *vec, int n_blk, int *freq, int *glob_out, int use_global)
  {
    __shared__ int s_min[2], s_max[2], s_best[2], s_sum[2], s_num;
    const int freq_size = 2 * FREQ_HALF; // coarse planes have pel 1
    if (!use_global)
    {
      if (threadIdx.x == 0)
      {
        glob_out[0] = 0;
        glob_out[1] = 0;
      }
      return;
    }
    for (int i = threadIdx.x; i < 2 * freq_size; i += blockDim.x)
      freq[i] = 0;
    if (threadIdx.x < 2)
    {
      s_min[threadIdx.x] = freq_size - 1;
      s_max[threadIdx.x] = 0;
      s_sum[threadIdx.x] = 0;
    }
    if (threadIdx.x == 0)
      s_num = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < n_blk; i += blockDim.x)
    {
      for (int c = 0; c < 2; c++)
      {
        const int ind = (freq_size >> 1) + vec[3 * i + c];
        if (ind >= 0 && ind < freq_size)
        {
          atomicAdd(&freq[c * freq_size + ind], 1);
          atomicMax(&s_max[c], ind);
          atomicMin(&s_min[c], ind);
        }
      }
    }
    __syncthreads();
    if (threadIdx.x < 2)
    {
      const int c = threadIdx.x;
      const int *f = freq + c * freq_size;
      int count = f[s_min[c]], index = s_min[c];
      for (int i = s_min[c] + 1; i <= s_max[c]; i++)
        if (f[i] > count)
        {
          count = f[i];
          index = i;
        }
      s_best[c] = index - (freq_size >> 1);
    }
    __syncthreads();
    const int medx = s_best[0], medy = s_best[1];
    for (int i = threadIdx.x; i < n_blk; i += blockDim.x)
    {
      if (abs(vec[3 * i] - medx) < 6 && abs(vec[3 * i + 1] - medy) < 6)
      {
        atomicAdd(&s_sum[0], vec[3 * i]);
        atomicAdd(&s_sum[1], vec[3 * i + 1]);
        atomicAdd(&s_num, 1);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
      if (s_num > 0)
      {
        glob_out[0] = 2 * s_sum[0] / s_num;
        glob_out[1] = 2 * s_sum[1] / s_num;
      }
      else
      {
        glob_out[0] = 2 * medx;
        glob_out[1] = 2 * medy;
      }
    }
  }

  struct Vec
  {
    int x, y, sad;
  };

  __device__ __forceinline__ int median3(int a, int b, int c)
  {
    if (a < b)
    {
      if (b < c) return b;
      else if (a < c) return c;
      else return a;
    }
    else
    {
      if (a < c) return a;
      else if (b < c) return c;
      else return b;
    }
  }

  __device__ __forceinline__ Vec clip_mv(int dx_min, int dx_max, int dy_min, int dy_max, int x, int y, int sad)
  {
    Vec v;
    v.x = x < dx_min ? dx_min : (x >= dx_max ? dx_max - 1 : x);
    v.y = y < dy_min ? dy_min : (y >= dy_max ? dy_max - 1 : y);
    v.sad = sad;
    return v;
  }

  struct BlockShared
  {
    int sl[MAX_CAND], sc[MAX_CAND];
    int ring_dx[MAX_CAND], ring_dy[MAX_CAND];
    int nlambda;
    int prev_x, prev_y, prev_sad; // result of the previous block of the row (left predictor), passed without a global round trip
  };

  // 4-byte asynchronous global -> shared copy (no registers, completes in the background)
  __device__ __forceinline__ void cp_async4(void *dst_shared, const void *src_global)
  {
    const unsigned s = (unsigned)__cvta_generic_to_shared(dst_shared);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(s), "l"(src_global));
  }
  __device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::); }
  __device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::); }
  __device__ __forceinline__ void prefetch_l1(const void *p) { asm volatile("prefetch.global.L1 [%0];" :: "l"(p)); }

  // Luma and chroma SAD of one candidate vector for the source block in shared memory. Lane = column. BSY / CHH are
  // the compile-time row counts of the luma block and of one chroma block (0 = unknown, runtime loops): with known
  // counts every reference sample of the candidate is loaded into registers before the arithmetic starts, so all loads
  // are in flight together (the measured cost of one candidate is dominated by the load latency). The two warp
  // reductions come last. sc is scaled like ScaleSadChroma. For cw <= 16 U and V share the warp (lanes [0, cw) U,
  // [cw, 2 cw) V).
  template <int BSY, int CHH>
  __device__ __forceinline__ void eval_candidate(const LevelArgs &a, const pix *s_y, const pix *s_u, const pix *s_v,
    int x0, int y0, int x1, int y1, int cx, int cy, int lane, bool pair, int &sl, int &sc)
  {
    const int xp = x0 * a.pel + cx, yp = y0 * a.pel + cy;
    const int sub = (xp & (a.pel - 1)) | ((yp & (a.pel - 1)) << a.log_pel);
    const int rx = xp >> a.log_pel, ry = yp >> a.log_pel;
    int sum_l = 0, sum_c = 0;
    const int rows_l = BSY ? BSY : a.bs_y, rows_c = CHH ? CHH : a.ch;
    const int xpc = x1 * a.pel + (cx >> a.log_x_uv), ypc = y1 * a.pel + (cy >> a.log_y_uv);
    const int subc = (xpc & (a.pel - 1)) | ((ypc & (a.pel - 1)) << a.log_pel);
    const int rxc = xpc >> a.log_pel, ryc = ypc >> a.log_pel;
    if constexpr (CHH > 0)
    {
      // the chroma rows are consumed after the luma rows: pull them into L1 now, without using registers, so that
      // the second batch of loads is an L1 hit instead of a second round trip
      if (pair)
      {
        const int pl = lane < a.cw ? 1 : 2, col = lane < a.cw ? lane : lane - a.cw;
        if (lane < 2 * a.cw)
        {
          const pix *r = a.ref[pl][subc] + (size_t)ryc * a.ew[pl] + rxc + col;
          #pragma unroll
          for (int row = 0; row < CHH; row++)
            prefetch_l1(r + (size_t)row * a.ew[pl]);
        }
      }
    }
    if (lane < a.bs_x)
    {
      const pix *r = a.ref[0][sub] + (size_t)ry * a.ew[0] + rx + lane;
      const pix *q = s_y + lane;
      if constexpr (BSY > 0)
      {
        int rv[BSY];
        #pragma unroll
        for (int row = 0; row < BSY; row++)
          rv[row] = r[(size_t)row * a.ew[0]];
        #pragma unroll
        for (int row = 0; row < BSY; row++)
          sum_l += abs((int)q[row * a.bs_x] - rv[row]);
      }
      else
      {
        #pragma unroll 32
        for (int row = 0; row < rows_l; row++)
          sum_l += abs((int)q[row * a.bs_x] - (int)r[(size_t)row * a.ew[0]]);
      }
    }
    if (pair)
    {
      const int pl = lane < a.cw ? 1 : 2, col = lane < a.cw ? lane : lane - a.cw;
      if (lane < 2 * a.cw)
      {
        const pix *r = a.ref[pl][subc] + (size_t)ryc * a.ew[pl] + rxc + col;
        const pix *q = (pl == 1 ? s_u : s_v) + col;
        if constexpr (CHH > 0)
        {
          int rv[CHH];
          #pragma unroll
          for (int row = 0; row < CHH; row++)
            rv[row] = r[(size_t)row * a.ew[pl]];
          #pragma unroll
          for (int row = 0; row < CHH; row++)
            sum_c += abs((int)q[row * a.cw] - rv[row]);
        }
        else
        {
          #pragma unroll 32
          for (int row = 0; row < rows_c; row++)
            sum_c += abs((int)q[row * a.cw] - (int)r[(size_t)row * a.ew[pl]]);
        }
      }
    }
    else if (lane < a.cw)
    {
      for (int pl = 1; pl < 3; pl++)
      {
        const pix *r = a.ref[pl][subc] + (size_t)ryc * a.ew[pl] + rxc + lane;
        const pix *q = (pl == 1 ? s_u : s_v) + lane;
        #pragma unroll 32
        for (int row = 0; row < rows_c; row++)
          sum_c += abs((int)q[row * a.cw] - (int)r[(size_t)row * a.ew[pl]]);
      }
    }
    sl = warp_sum(sum_l);
    sc = scale_chroma(warp_sum(sum_c), a.chroma_scale);
  }

  // picks the compile-time row counts for the common block sizes
  __device__ __forceinline__ void eval_dispatch(const LevelArgs &a, const pix *s_y, const pix *s_u, const pix *s_v,
    int x0, int y0, int x1, int y1, int cx, int cy, int lane, bool pair, int &sl, int &sc)
  {
    if (a.bs_y == 32 && a.ch == 32)
      eval_candidate<32, 32>(a, s_y, s_u, s_v, x0, y0, x1, y1, cx, cy, lane, pair, sl, sc);
    else if (a.bs_y == 16 && a.ch == 16)
      eval_candidate<16, 16>(a, s_y, s_u, s_v, x0, y0, x1, y1, cx, cy, lane, pair, sl, sc);
    else if (a.bs_y == 32 && a.ch == 16)
      eval_candidate<32, 16>(a, s_y, s_u, s_v, x0, y0, x1, y1, cx, cy, lane, pair, sl, sc);
    else if (a.bs_y == 16 && a.ch == 8)
      eval_candidate<16, 8>(a, s_y, s_u, s_v, x0, y0, x1, y1, cx, cy, lane, pair, sl, sc);
    else
      eval_candidate<0, 0>(a, s_y, s_u, s_v, x0, y0, x1, y1, cx, cy, lane, pair, sl, sc);
  }

#ifdef MVC_PHASE_PROF
  // per-phase cycle counters of the block loop (development aid, build with -DMVC_PHASE_PROF): [log2(blocks per row)][phase]
  __device__ unsigned long long g_prof[16][8];
#define PH_START() ph_t = clock64()
#define PH(i) do { if (tid == 0) { long long n_ = clock64(); atomicAdd(&g_prof[bk][i], (unsigned long long)(n_ - ph_t)); ph_t = n_; } } while (0)
#else
#define PH_START() do {} while (0)
#define PH(i) do {} while (0)
#endif

  // one CUDA block per slice; every warp evaluates candidate vectors, the block order inside the slice is serial
  __global__ void __launch_bounds__(768, 1) k_search(LevelArgs a)
  {
    extern __shared__ __align__(16) pix s_dyn[];
    __shared__ BlockShared b;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nwarps = blockDim.x >> 5;
    const int slice = blockIdx.x;
#ifdef MVC_PHASE_PROF
    const int bk = 31 - __clz(a.n_blk_x);
    long long ph_t = 0;
#endif
    const int beg = slice * a.n_blk_y / a.n_slices, end = (slice + 1) * a.n_blk_y / a.n_slices;
    const int step_x = a.bs_x - a.ov_x, step_y = a.bs_y - a.ov_y;
    const int h_scaled = a.hpad[0] >> a.log_scale, v_scaled = a.vpad[0] >> a.log_scale;
    const int ring_n = 4 * a.radius * (a.radius + 1);
    const int blk_el = a.bs_x * a.bs_y + 2 * a.cw * a.ch;
    pix *s_y = s_dyn, *s_u = s_dyn, *s_v = s_dyn;
    // asynchronous copy of the source block of block column bx in row by into the shared buffer dst
    auto issue_src = [&](pix *dst, int bx, int by) {
      const int xs = a.hpad[0] + bx * step_x, ys = a.vpad[0] + by * step_y;
      const int xc = a.hpad[1] + bx * (step_x >> a.log_x_uv), yc = a.vpad[1] + by * (step_y >> a.log_y_uv);
      if (!a.async_src)
      {
        // plane rows not 4-byte aligned (8-bit, small coarse levels): plain copy
        for (int i = tid; i < a.bs_x * a.bs_y; i += blockDim.x)
        {
          const int row = i / a.bs_x, col = i - row * a.bs_x;
          dst[i] = a.src[0][(size_t)(ys + row) * a.ew[0] + xs + col];
        }
        for (int i = tid; i < a.cw * a.ch; i += blockDim.x)
        {
          const int row = i / a.cw, col = i - row * a.cw;
          dst[a.bs_x * a.bs_y + i] = a.src[1][(size_t)(yc + row) * a.ew[1] + xc + col];
          dst[a.bs_x * a.bs_y + a.cw * a.ch + i] = a.src[2][(size_t)(yc + row) * a.ew[2] + xc + col];
        }
        return;
      }
      const int wl = (a.bs_x * a.bs_y) / SPW, wc = (a.cw * a.ch) / SPW;     // 4-byte words
      const int rl = a.bs_x / SPW, rc = a.cw / SPW;                          // words per row
      for (int i = tid; i < wl + 2 * wc; i += blockDim.x)
      {
        if (i < wl)
        {
          const int row = i / rl, col = SPW * (i - row * rl);
          cp_async4(dst + row * a.bs_x + col, a.src[0] + (size_t)(ys + row) * a.ew[0] + xs + col);
        }
        else
        {
          int j = i - wl;
          const int pl = j >= wc ? 2 : 1;
          if (j >= wc)
            j -= wc;
          const int row = j / rc, col = SPW * (j - row * rc);
          pix *d = dst + a.bs_x * a.bs_y + (pl - 1) * a.cw * a.ch;
          cp_async4(d + row * a.cw + col, a.src[pl] + (size_t)(yc + row) * a.ew[pl] + (pl == 1 ? 0 : 0) + xc + col);
        }
      }
      cp_async_commit();
    };
    const bool pair_chroma = (2 * a.cw <= 32);

    // ring offsets in ExpandingSearch order (sides without corners, then the corners), relative to the centre
    if (tid == 0)
    {
      int n = 0;
      for (int r = 1; r <= a.radius; r++)
      {
        for (int i = -r + 1; i < r; i++)
        {
          b.ring_dx[n] = i; b.ring_dy[n++] = -r;
          b.ring_dx[n] = i; b.ring_dy[n++] = r;
        }
        for (int j = -r + 1; j < r; j++)
        {
          b.ring_dx[n] = -r; b.ring_dy[n++] = j;
          b.ring_dx[n] = r; b.ring_dy[n++] = j;
        }
        b.ring_dx[n] = -r; b.ring_dy[n++] = -r;
        b.ring_dx[n] = -r; b.ring_dy[n++] = r;
        b.ring_dx[n] = r; b.ring_dy[n++] = -r;
        b.ring_dx[n] = r; b.ring_dy[n++] = r;
      }
    }

    for (int blky = beg; blky < end; blky++)
    {
      const int dir = (blky % 2 == 0 || !a.meander) ? 1 : -1;
      const int start = (dir == 1) ? 0 : a.n_blk_x - 1;
      const bool is_top = (blky == beg), is_bottom = (blky == end - 1);

      // The neighbours that do not depend on the previous block of the row (own prediction, up, bottom-right or
      // up-right) are loaded one block ahead; only the left predictor comes from the block just searched.
      int own_n[3] = { 0, 0, 0 }, up_n[3] = { 0, 0, 0 }, p3_n[3] = { 0, 0, 0 };
      bool have_up = false, have_p3 = false;
      auto prefetch = [&](int it) {
        const int bx = start + it * dir;
        const int id = blky * a.n_blk_x + bx;
        const bool side = (dir == 1 && bx < a.n_blk_x - 1) || (dir == -1 && bx > 0);
        const int *o = a.vec + 3 * id;
        own_n[0] = o[0]; own_n[1] = o[1]; own_n[2] = o[2];
        have_up = !is_top;
        if (have_up)
        {
          const int *u = a.vec + 3 * (id - a.n_blk_x);
          up_n[0] = u[0]; up_n[1] = u[1]; up_n[2] = u[2];
        }
        have_p3 = false;
        const int *q = nullptr;
        if (!is_bottom && !a.smallest && side)
          q = a.vec + 3 * (id + a.n_blk_x + dir);
        else if (!is_top && side)
          q = a.vec + 3 * (id - a.n_blk_x + dir);
        if (q)
        {
          have_p3 = true;
          p3_n[0] = q[0]; p3_n[1] = q[1]; p3_n[2] = q[2];
        }
      };
      prefetch(0);
      issue_src(s_dyn, start, blky); // the first block of the row (buffer 0); later ones are issued a block ahead

      for (int it = 0; it < a.n_blk_x; it++)
      {
        PH_START();
        const int blkx = start + it * dir;
        const int idx = blky * a.n_blk_x + blkx;
        // values for this block (prefetched), then start loading the next block's constants
        const int own_v[3] = { own_n[0], own_n[1], own_n[2] };
        const int up_v[3] = { up_n[0], up_n[1], up_n[2] };
        const int p3_v[3] = { p3_n[0], p3_n[1], p3_n[2] };
        const bool use_up = have_up, use_p3 = have_p3;
        if (it + 1 < a.n_blk_x)
          prefetch(it + 1);

        // ---- geometry (identical in every thread) and the source block into shared memory
        int x0[3], y0[3];
        x0[0] = a.hpad[0] + blkx * step_x;
        y0[0] = a.vpad[0] + blky * step_y;
        for (int p = 1; p < 3; p++)
        {
          x0[p] = a.hpad[p] + blkx * (step_x >> a.log_x_uv);
          y0[p] = a.vpad[p] + blky * (step_y >> a.log_y_uv);
        }
        // the source block of this block was copied by cp.async during the previous block (or before the row);
        // wait for our own copies, the barrier below makes everybody's visible
        cp_async_wait_all();
        s_y = s_dyn + (it & 1) * blk_el;
        s_u = s_y + a.bs_x * a.bs_y;
        s_v = s_u + a.cw * a.ch;
        if (it + 1 < a.n_blk_x)
          issue_src(s_dyn + ((it + 1) & 1) * blk_el, start + (it + 1) * dir, blky);

        const int dx_max = a.pel * (a.ext_w - x0[0] - a.bs_x - a.hpad[0] + h_scaled);
        const int dy_max = a.pel * (a.ext_h - y0[0] - a.bs_y - a.vpad[0] + v_scaled);
        const int dx_min = -a.pel * (x0[0] - a.hpad[0] + h_scaled);
        const int dy_min = -a.pel * (y0[0] - a.vpad[0] + v_scaled);

        // ---- predictors (FetchPredictors), identical in every thread
        const bool side_ok = (dir == 1 && blkx < a.n_blk_x - 1) || (dir == -1 && blkx > 0);
        (void)side_ok;
        const Vec own_c = clip_mv(dx_min, dx_max, dy_min, dy_max, own_v[0], own_v[1], own_v[2]);
        const Vec zero_shifted = clip_mv(dx_min, dx_max, dy_min, dy_max, 0, 0, 0); // field shift 0
        Vec p1 = zero_shifted, p2 = zero_shifted, p3 = zero_shifted, p0;
        if (it > 0)
          p1 = clip_mv(dx_min, dx_max, dy_min, dy_max, b.prev_x, b.prev_y, b.prev_sad);
        if (use_up)
          p2 = clip_mv(dx_min, dx_max, dy_min, dy_max, up_v[0], up_v[1], up_v[2]);
        if (use_p3)
          p3 = clip_mv(dx_min, dx_max, dy_min, dy_max, p3_v[0], p3_v[1], p3_v[2]);
        if (!is_top)
        {
          p0.x = median3(p1.x, p2.x, p3.x);
          p0.y = median3(p1.y, p2.y, p3.y);
          p0.sad = max(p1.sad, max(p2.sad, p3.sad));
        }
        else
          p0 = p1;
        const Vec pred = a.smallest ? p0 : own_c;
        if (tid == 0)
        {
          int nlambda = is_top ? 0 : a.lambda_level;
          if (a.lsad == 0)
            nlambda = 0;
          else
          {
            // double arithmetic, IEEE exact like the CPU code (no fused operations in these lines); FP64 is slow on
            // GeForce, so this is done by one thread only
            const double factor = __ddiv_rn((double)a.lsad, (double)(a.lsad + (pred.sad >> 1)));
            nlambda = (int)__dmul_rn(__dmul_rn((double)nlambda, factor), factor);
          }
          b.nlambda = nlambda;
        }
        const Vec glob = clip_mv(dx_min, dx_max, dy_min, dy_max, a.glob[0] * a.pel, a.glob[1] * a.pel, 0);
        __syncthreads(); // source block and nLambda in shared memory; the previous block's result is visible
        const int nlambda = b.nlambda;
        PH(0);

        // ---- phase A: SADs of the 7 candidates (zero, global, predictor, predictors[0..3]), one warp each
        for (int c = warp; c < 7; c += nwarps)
        {
          int cx, cy;
          switch (c)
          {
          case 0: cx = 0; cy = 0; break;
          case 1: cx = glob.x; cy = glob.y; break;
          case 2: cx = pred.x; cy = pred.y; break;
          case 3: cx = p0.x; cy = p0.y; break;
          case 4: cx = p1.x; cy = p1.y; break;
          case 5: cx = p2.x; cy = p2.y; break;
          default: cx = p3.x; cy = p3.y; break;
          }
          int sl, sc;
          eval_dispatch(a, s_y, s_u, s_v, x0[0], y0[0], x0[1], y0[1], cx, cy, lane, pair_chroma, sl, sc);
          if (lane == 0)
          {
            b.sl[c] = sl;
            b.sc[c] = sc;
          }
        }
        __syncthreads();
        PH(1);

        // ---- fold phase A (every thread, identical): same order and comparisons as PseudoEPZSearch
        int best_x, best_y, best_sad, min_cost;
        {
          const int sad = b.sl[0] + b.sc[0];
          best_x = 0;
          best_y = 0;
          best_sad = sad;
          min_cost = sad + (int)(((long long)a.pzero * sad) >> 8);
          const int sadg = b.sl[1] + b.sc[1];
          const int costg = sadg + (int)(((long long)a.pglobal * sadg) >> 8);
          if (costg < min_cost)
          {
            best_x = glob.x;
            best_y = glob.y;
            best_sad = sadg;
            min_cost = costg;
          }
          const int sadp = b.sl[2] + b.sc[2];
          if (sadp < min_cost)
          {
            best_x = pred.x;
            best_y = pred.y;
            best_sad = sadp;
            min_cost = sadp;
          }
          const Vec pv[4] = { p0, p1, p2, p3 };
          for (int i = 0; i < 4; i++) // CheckMV0
          {
            const int cx = pv[i].x, cy = pv[i].y;
            if (!(cx >= dx_min && cy >= dy_min && cx < dx_max && cy < dy_max))
              continue;
            const int dxp = pred.x - cx, dyp = pred.y - cy;
            const int dist = (int)((unsigned)(dxp * dxp) + (unsigned)(dyp * dyp));
            int cost = (int)((unsigned)nlambda * (unsigned)dist) >> (16 - a.bits);
            if (cost >= min_cost)
              continue;
            cost += b.sl[3 + i];
            if (cost >= min_cost)
              continue;
            cost += b.sc[3 + i];
            if (cost >= min_cost)
              continue;
            best_x = cx;
            best_y = cy;
            best_sad = b.sl[3 + i] + b.sc[3 + i];
            min_cost = cost;
          }
        }
        __syncthreads(); // everybody has read the phase A results before phase B overwrites them
        PH(2);

        // ---- phase B: SADs of the ring candidates around the best vector so far
        for (int c = warp; c < ring_n; c += nwarps)
        {
          const int cx = best_x + b.ring_dx[c], cy = best_y + b.ring_dy[c];
          if (!(cx >= dx_min && cy >= dy_min && cx < dx_max && cy < dy_max))
            continue; // never read: the fold below skips invalid vectors
          int sl, sc;
          eval_dispatch(a, s_y, s_u, s_v, x0[0], y0[0], x0[1], y0[1], cx, cy, lane, pair_chroma, sl, sc);
          if (lane == 0)
          {
            b.sl[c] = sl;
            b.sc[c] = sc;
          }
        }
        __syncthreads();
        PH(3);

        // ---- fold the ring (CheckMV, with penaltyNew) in warp 0, one lane per candidate. The sequential CPU rule
        // "accept if cost < running minimum" ends at the earliest candidate with the smallest cost below the
        // minimum before the ring, so a minimum over (cost, index) is equivalent (the partial sums only grow, hence
        // the cost chain with early exits accepts exactly when its final sum is below the minimum).
        if (warp == 0)
        {
          long long key = 0x7fffffffffffffffLL;
          for (int c = lane; c < ring_n; c += 32)
          {
            const int cx = best_x + b.ring_dx[c], cy = best_y + b.ring_dy[c];
            if (!(cx >= dx_min && cy >= dy_min && cx < dx_max && cy < dy_max))
              continue;
            const int dxp = pred.x - cx, dyp = pred.y - cy;
            const int dist = (int)((unsigned)(dxp * dxp) + (unsigned)(dyp * dyp));
            int cost = (int)((unsigned)nlambda * (unsigned)dist) >> (16 - a.bits);
            if (cost >= min_cost)
              continue;
            cost += b.sl[c] + (int)(((long long)a.pnew * b.sl[c]) >> 8);
            if (cost >= min_cost)
              continue;
            cost += b.sc[c] + (int)(((long long)a.pnew * b.sc[c]) >> 8);
            if (cost >= min_cost)
              continue;
            const long long k = (long long)cost * 128 + c;
            key = k < key ? k : key;
          }
          for (int o = 16; o > 0; o >>= 1)
          {
            const long long other = __shfl_xor_sync(0xffffffffu, key, o);
            key = other < key ? other : key;
          }
          if (lane == 0)
          {
            if (key != 0x7fffffffffffffffLL)
            {
              const int c = (int)(key & 127);
              best_x += b.ring_dx[c];
              best_y += b.ring_dy[c];
              best_sad = b.sl[c] + b.sc[c];
            }
            // the CPU runs a wide UMH search for very bad vectors (not reproduced, see the file header)
            if (idx > 1 + beg * a.n_blk_x && best_sad > a.bad_sad)
              atomicExch(a.bad_flag, 1);
            a.vec[3 * idx + 0] = best_x;
            a.vec[3 * idx + 1] = best_y;
            a.vec[3 * idx + 2] = best_sad;
            b.prev_x = best_x; // left predictor of the next block
            b.prev_y = best_y;
            b.prev_sad = best_sad;
          }
        }
        __syncthreads(); // the next block reads b.prev_*; the global vector is read by later rows / levels
        PH(4);
#ifdef MVC_PHASE_PROF
        if (tid == 0)
          atomicAdd(&g_prof[bk][5], 1ull);
#endif
      }
    }
  }

  int ilog2(int v)
  {
    int l = 0;
    while ((1 << l) < v)
      l++;
    return l;
  }
}

extern "C" int MVC_API(mvcuda_analyse_host)(const MvCudaSuperParams *sp, const MvCudaAnalyseParams *ap, const void *const src[3],
  const int src_pitch_bytes[3], const void *const ref[3], const int ref_pitch_bytes[3], int *payload, int payload_words)
{
  if (!sp || !ap || !src || !ref || !payload)
    return -4;
  // supported: chroma on, pel 1 or 2, exhaustive search everywhere, field shift 0
  if (!sp->chroma || sp->levels != ap->n_levels || ap->n_levels < 1 || ap->n_levels > MVCUDA_MAX_LEVELS
    || ap->search_param < 1 || ap->pel_search < 1 || ap->search_param > 3 || ap->pel_search > 3 || ap->n_blk_x < 1
    || ap->n_blk_y < 1 || ap->slice_threads < 1 || ap->plevel < 0 || ap->plevel > 2 || ap->blk_x > 32)
    return -4;
  if (!mvc::use_device())
    return -1;

  // the asynchronous source-block copy moves 4-byte words: everything it touches must be a multiple of SPW samples
  {
    const int lx = ilog2(sp->x_ratio_uv), step_x = ap->blk_x - ap->overlap_x;
    bool ok = ap->blk_x % SPW == 0 && (ap->blk_x >> lx) % SPW == 0 && step_x % SPW == 0 && (step_x >> lx) % SPW == 0
      && sp->hpad % SPW == 0 && (sp->hpad / sp->x_ratio_uv) % SPW == 0
      && ((size_t)ap->blk_x * ap->blk_y) % SPW == 0 && ((size_t)(ap->blk_x >> lx) * (ap->blk_y >> ilog2(sp->y_ratio_uv))) % SPW == 0;
    if (!ok)
      return -4;
  }

  const int nl = ap->n_levels;
  int nbx[MVCUDA_MAX_LEVELS], nby[MVCUDA_MAX_LEVELS];
  const int w_b = (ap->blk_x - ap->overlap_x) * ap->n_blk_x + ap->overlap_x;
  const int h_b = (ap->blk_y - ap->overlap_y) * ap->n_blk_y + ap->overlap_y;
  int words = 2;
  int offs[MVCUDA_MAX_LEVELS]; // word offset of the first triple of each level inside the payload
  for (int l = nl - 1; l >= 0; l--)
  {
    nbx[l] = ((w_b >> l) - ap->overlap_x) / (ap->blk_x - ap->overlap_x);
    nby[l] = ((h_b >> l) - ap->overlap_y) / (ap->blk_y - ap->overlap_y);
    if (nbx[l] < 1 || nby[l] < 1)
      return -4;
    offs[l] = words + 1;
    words += 1 + nbx[l] * nby[l] * 3;
  }
  if (payload_words < words)
    return -4;

  const bool timing = getenv("MVCUDA_TIMING") != nullptr;
  auto t0 = std::chrono::steady_clock::now();
  auto lap = [&](const char *what) {
    if (!timing)
      return;
    auto t1 = std::chrono::steady_clock::now();
    fprintf(stderr, "mvcuda analyse: %-22s %7.2f ms\n", what, std::chrono::duration<double, std::milli>(t1 - t0).count());
    t0 = t1;
  };

  int rc = 0;
  mvc::StreamLease lease;
  cudaStream_t st = lease.s;
  std::shared_ptr<mvc::DevSuper> s_src, s_ref;
  int *d_pay = nullptr, *d_freq = nullptr, *d_glob = nullptr, *d_bad = nullptr;
  const size_t pay_bytes = (size_t)words * sizeof(int), freq_bytes = 2 * 2 * FREQ_HALF * sizeof(int);
  std::vector<int> host;

  rc = mvc::super_get(&MVC_NS::super_build, *sp, ap->src_key, src, src_pitch_bytes, st, s_src);
  if (!rc)
    rc = mvc::super_get(&MVC_NS::super_build, *sp, ap->ref_key, ref, ref_pitch_bytes, st, s_ref);
  lap("2 x super (cached)");
  if (rc)
    return rc;

#define AN_CHECK(call) \
  do { \
    cudaError_t e_ = (call); \
    if (e_ != cudaSuccess) { \
      fprintf(stderr, "mvcuda: %s failed: %s\n", #call, cudaGetErrorString(e_)); \
      rc = -2; \
      goto done; \
    } \
  } while (0)

  d_pay = (int *)mvc::dev_alloc(pay_bytes);
  d_freq = (int *)mvc::dev_alloc(freq_bytes);
  d_glob = (int *)mvc::dev_alloc(2 * sizeof(int));
  d_bad = (int *)mvc::dev_alloc(sizeof(int));
  if (!d_pay || !d_freq || !d_glob || !d_bad)
  {
    rc = -2;
    goto done;
  }
  {
    host.assign(words, 0);
    host[0] = words;
    host[1] = 1;
    for (int l = nl - 1; l >= 0; l--)
      host[offs[l] - 1] = nbx[l] * nby[l] * 3 + 1;
    AN_CHECK(cudaMemcpyAsync(d_pay, host.data(), pay_bytes, cudaMemcpyHostToDevice, st));
    AN_CHECK(cudaMemsetAsync(d_bad, 0, sizeof(int), st));
    AN_CHECK(cudaMemsetAsync(d_glob, 0, 2 * sizeof(int), st));

    for (int l = nl - 1; l >= 0; l--)
    {
      LevelArgs a = {};
      a.level = l;
      a.n_blk_x = nbx[l];
      a.n_blk_y = nby[l];
      a.pel = (l == 0) ? sp->pel : 1;
      a.log_pel = ilog2(a.pel);
      a.n_scale = 1 << l;
      a.log_scale = l;
      a.smallest = (l == nl - 1) ? 1 : 0;
      a.bs_x = ap->blk_x;
      a.bs_y = ap->blk_y;
      a.ov_x = ap->overlap_x;
      a.ov_y = ap->overlap_y;
      a.log_x_uv = ilog2(sp->x_ratio_uv);
      a.log_y_uv = ilog2(sp->y_ratio_uv);
      a.cw = a.bs_x >> a.log_x_uv;
      a.ch = a.bs_y >> a.log_y_uv;
      for (int p = 0; p < 3; p++)
      {
        const mvc::DevPlane &dr = s_ref->pl[l * 3 + p];
        a.src[p] = (const pix *)s_src->pl[l * 3 + p].sub[0];
        for (int i = 0; i < 4; i++)
          a.ref[p][i] = (const pix *)dr.sub[i];
        a.ew[p] = dr.d.ew;
        a.hpad[p] = dr.d.hpad;
        a.vpad[p] = dr.d.vpad;
      }
      a.async_src = (a.ew[0] % SPW == 0 && a.ew[1] % SPW == 0 && a.ew[2] % SPW == 0) ? 1 : 0;
      a.ext_w = s_ref->pl[l * 3].d.ew;
      a.ext_h = s_ref->pl[l * 3].d.eh;
      // _lambda_level = lambda / (pel * pel) * nScale (plevel 1) or * nScale^2 (plevel 2)
      a.lambda_level = ap->lambda / (a.pel * a.pel);
      if (ap->plevel == 1)
        a.lambda_level *= a.n_scale;
      else if (ap->plevel == 2)
        a.lambda_level *= a.n_scale * a.n_scale;
      a.lsad = ap->lsad;
      a.pnew = ap->pnew;
      a.pzero = ap->pzero;
      a.pglobal = ap->use_global ? ap->pglobal : ap->pzero;
      a.bits = sp->bits_per_pixel;
      a.chroma_scale = ap->chroma_sad_scale;
      a.radius = (l == 0) ? ap->pel_search : ap->search_param;
      a.meander = ap->meander;
      a.use_global = ap->use_global;
      a.bad_sad = ap->bad_sad;
      a.glob = d_glob;
      a.vec = d_pay + offs[l];
      a.bad_flag = d_bad;
      int slices = ap->slice_threads;
      if (slices > 64)
        slices = 64;
      if (slices > a.n_blk_y / 4)
        slices = a.n_blk_y / 4;
      a.n_slices = slices < 1 ? 1 : slices;

      if (l < nl - 1)
      {
        // global MV estimate from the coarser plane (doubled), then the interpolated prediction for this plane
        k_global<<<1, 1024, 0, st>>>(d_pay + offs[l + 1], nbx[l + 1] * nby[l + 1], d_freq, d_glob, ap->use_global);
        const int total = a.n_blk_x * a.n_blk_y;
        k_interp<<<(total + 127) / 128, 128, 0, st>>>(a, d_pay + offs[l + 1], nbx[l + 1], nby[l + 1], 0);
      }
      else
        AN_CHECK(cudaMemsetAsync(d_glob, 0, 2 * sizeof(int), st));
      {
        const size_t shm = 2 * ((size_t)a.bs_x * a.bs_y + 2 * (size_t)a.cw * a.ch) * sizeof(pix); // double-buffered source block
        k_search<<<a.n_slices, 768, shm, st>>>(a);
      }
      AN_CHECK(cudaGetLastError());
    }
    AN_CHECK(cudaMemcpyAsync(payload, d_pay, pay_bytes, cudaMemcpyDeviceToHost, st));
    int bad = 0;
    AN_CHECK(cudaMemcpyAsync(&bad, d_bad, sizeof(int), cudaMemcpyDeviceToHost, st));
    AN_CHECK(cudaStreamSynchronize(st));
    lap("analysis kernels");
#ifdef MVC_PHASE_PROF
    {
      unsigned long long h[16][8];
      cudaMemcpyFromSymbol(h, g_prof, sizeof(h));
      for (int k = 0; k < 16; k++)
        if (h[k][5])
          std::fprintf(stderr, "phase cycles/block, blocks-per-row 2^%d: wait+pred %.0f  A %.0f  foldA %.0f  B %.0f  foldB %.0f  (n=%llu)\n", k,
            (double)h[k][0] / h[k][5], (double)h[k][1] / h[k][5], (double)h[k][2] / h[k][5], (double)h[k][3] / h[k][5], (double)h[k][4] / h[k][5], h[k][5]);
    }
#endif
    if (bad)
      rc = MVCUDA_NEEDS_CPU;
  }

done:
  if (rc)
    cudaStreamSynchronize(st);
  mvc::dev_free(d_pay, pay_bytes);
  mvc::dev_free(d_freq, freq_bytes);
  mvc::dev_free(d_glob, 2 * sizeof(int));
  mvc::dev_free(d_bad, sizeof(int));
  return rc;
}
