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
#include "mvcuda_refine.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <type_traits>
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
    int search_type;  // mvc_refine::S_*: the refine step of this level (exhaustive levels use the ring fold below)
    int has_chroma;   // 0: luma-only search, cw = ch = 0
    int async_src;  // 1: the source block can be copied with 4-byte cp.async (all plane rows 4-byte aligned)
    int packed;     // 1: SADs in the packed form of k_wave (pk_load / pk_sum), see analyse_impl
    int packed_s;   // 1: eval_dispatch may use eval_candidate_pk (reference rows aligned for the packed reads)
    const int *glob;   // device: doubled global MV estimate of this level (x, y), already in units of this level's pel
    int *vec;          // device: (vx, vy, sad) triples of this level
    int *bad_flag;
    // wavefront kernel (k_wave) only
    int *prog;         // device: per block row, number of finished blocks of that row
    int *ticket;       // device: next block row to hand out
    int *nlam;         // device: nLambda per block (k_interp writes it, k_wave reads it); unused on the smallest level
    int *rowbad;       // device: per block row, number of blocks that took the wide "bad vector" search
    int bad_range;     // MAnalyse badrange (> 0 UMH, < 0 exhaustive rings, 0 none)
  };

  // One whole analysis call for the wavefront kernel (one launch for all levels): the rows of all levels, coarsest level
  // first, form one ticket sequence. A row of a finer level starts when the coarser level is complete and its global MV
  // estimate is known; the CTA that finishes the last row of a level makes that estimate. Predictors are interpolated
  // per block from the coarser level's vectors as they are needed (no separate pass).
  struct WaveCall
  {
    LevelArgs lv[MVCUDA_MAX_LEVELS];
    int nl;            // number of levels, lv[nl - 1] is the coarsest
    int use_global;
    int *ticket;       // device: next row ticket
    int *level_done;   // device, per level: finished rows
    int *level_ready;  // device, per level: 1 once lv[l].glob is written (not used for the coarsest level)
    int *freq;         // device scratch of glob_estimate
    int *payload;      // device payload: its header words are written here (total words, 1, per level the size word)
    int words;
    unsigned long long *tim; // MVC_WAVE_TIMING builds: per level first row start, last row end (%globaltimer ns)
    // When host_out is set, the CTA that finishes last copies the payload and the bad flag (state[bad_word]) to host_out
    // (pinned host memory, same layout as payload + state) and clears the state (state_words, finished included) for the
    // next call on this buffer: no memset before and no copy after the launch.
    int *host_out;
    int *state;
    unsigned *finished; // device: CTAs done (in the state)
    int state_words, bad_word;
  };

  __device__ __forceinline__ unsigned long long global_ns()
  {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
  }

  // nLambda of a block whose predictor SAD is pred_sad (PlaneOfBlocks::search_mv_slice): 0 on the first row of a slice,
  // else lambda_level * (lsad / (lsad + pred_sad / 2))^2 in IEEE double (no fused operations, like the CPU code)
  __device__ __forceinline__ int calc_nlambda(int lambda_level, int lsad, int pred_sad, bool first_row)
  {
    int nlambda = first_row ? 0 : lambda_level;
    if (lsad == 0)
      return 0;
    const double factor = __ddiv_rn((double)lsad, (double)(lsad + (pred_sad >> 1)));
    return (int)__dmul_rn(__dmul_rn((double)nlambda, factor), factor);
  }

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

  struct Vec
  {
    int x, y, sad;
  };

  // InterpolatePrediction<bigsad_t, bigsad_t> for 16-bit data: the predictor of block index of this level from the
  // vectors of the coarser one (read with ld.global.cg: the wavefront kernel reads vectors other CTAs just wrote)
  __device__ __forceinline__ Vec interp_pred(const LevelArgs &a, const int *coarse, int c_blk_x, int c_blk_y, int log_pel_coarse, int index)
  {
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
    const int *q1, *q2, *q3, *q4;
#define CV(ix, iy) (coarse + 3 * ((ix) + (iy) * c_blk_x))
    if (i == 0 || i >= 2 * c_blk_x - 1)
    {
      if (j == 0 || j >= 2 * c_blk_y - 1)
        q1 = q2 = q3 = q4 = CV(iper2, jper2);
      else
      {
        q1 = q2 = CV(iper2, jper2);
        q3 = q4 = CV(iper2, jper2 + offy);
      }
    }
    else if (j == 0 || j >= 2 * c_blk_y - 1)
    {
      q1 = q2 = CV(iper2, jper2);
      q3 = q4 = CV(iper2 + offx, jper2);
    }
    else
    {
      q1 = CV(iper2, jper2);
      q2 = CV(iper2 + offx, jper2);
      q3 = CV(iper2, jper2 + offy);
      q4 = CV(iper2 + offx, jper2 + offy);
    }
#undef CV
    const int v1[3] = { __ldcg(q1), __ldcg(q1 + 1), __ldcg(q1 + 2) }, v2[3] = { __ldcg(q2), __ldcg(q2 + 1), __ldcg(q2 + 2) };
    const int v3[3] = { __ldcg(q3), __ldcg(q3 + 1), __ldcg(q3 + 2) }, v4[3] = { __ldcg(q4), __ldcg(q4 + 1), __ldcg(q4 + 2) };
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
    return Vec{ (x >> norm_factor) << mul_factor, (y >> norm_factor) << mul_factor, (int)(tmp_sad >> 4) };
  }

  __global__ void k_interp(LevelArgs a, const int *coarse, int c_blk_x, int c_blk_y, int log_pel_coarse)
  {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= a.n_blk_x * a.n_blk_y)
      return;
    const Vec v = interp_pred(a, coarse, c_blk_x, c_blk_y, log_pel_coarse, index);
    a.vec[3 * index + 0] = v.x;
    a.vec[3 * index + 1] = v.y;
    a.vec[3 * index + 2] = v.sad;
  }

  // PlaneOfBlocks::estimate_global_mv_doubled_slice on the vectors of the coarser plane, by one CTA of nt threads (tid =
  // thread index; all of them call it). freq: device scratch of 2 * 2 * FREQ_HALF ints. Vectors read with ld.global.cg.
  __device__ void glob_estimate(const int *vec, int n_blk, int *freq, int *glob_out, int use_global, int tid, int nt)
  {
    __shared__ int s_min[2], s_max[2], s_best[2], s_sum[2], s_num;
    const int freq_size = 2 * FREQ_HALF; // coarse planes have pel 1
    if (!use_global)
    {
      if (tid == 0)
      {
        glob_out[0] = 0;
        glob_out[1] = 0;
      }
      return;
    }
    for (int i = tid; i < 2 * freq_size; i += nt)
      freq[i] = 0;
    if (tid < 2)
    {
      s_min[tid] = freq_size - 1;
      s_max[tid] = 0;
      s_sum[tid] = 0;
    }
    if (tid == 0)
      s_num = 0;
    __syncthreads();
    for (int i = tid; i < n_blk; i += nt)
    {
      for (int c = 0; c < 2; c++)
      {
        const int ind = (freq_size >> 1) + __ldcg(vec + 3 * i + c);
        if (ind >= 0 && ind < freq_size)
        {
          atomicAdd(&freq[c * freq_size + ind], 1);
          atomicMax(&s_max[c], ind);
          atomicMin(&s_min[c], ind);
        }
      }
    }
    __syncthreads();
    // most frequent value, the first (lowest) one on ties, as the CPU's ascending scan: one warp per component, each
    // lane over a stride of the occupied range, then a warp reduction (the loads go to L2, a one-thread scan would take
    // an L2 round trip per entry)
    if (tid < 64)
    {
      const int c = tid >> 5, ln = tid & 31;
      const int *f = freq + c * freq_size;
      int count = -1, index = s_min[c];
      for (int i = s_min[c] + ln; i <= s_max[c]; i += 32)
      {
        const int v = __ldcg(f + i);
        if (v > count)
        {
          count = v;
          index = i;
        }
      }
      for (int o = 16; o > 0; o >>= 1)
      {
        const int oc = __shfl_xor_sync(0xffffffffu, count, o), oi = __shfl_xor_sync(0xffffffffu, index, o);
        if (oc > count || (oc == count && oi < index))
        {
          count = oc;
          index = oi;
        }
      }
      if (ln == 0)
        s_best[c] = index - (freq_size >> 1);
    }
    __syncthreads();
    const int medx = s_best[0], medy = s_best[1];
    for (int i = tid; i < n_blk; i += nt)
    {
      const int vx = __ldcg(vec + 3 * i), vy = __ldcg(vec + 3 * i + 1);
      if (abs(vx - medx) < 6 && abs(vy - medy) < 6)
      {
        atomicAdd(&s_sum[0], vx);
        atomicAdd(&s_sum[1], vy);
        atomicAdd(&s_num, 1);
      }
    }
    __syncthreads();
    if (tid == 0)
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
    __syncthreads(); // the shared state may be used again
  }

  __global__ void k_global(const int *vec, int n_blk, int *freq, int *glob_out, int use_global)
  {
    glob_estimate(vec, n_blk, freq, glob_out, use_global, threadIdx.x, blockDim.x);
  }

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
    int sl[mvc_refine::MAX_BATCH], sc[mvc_refine::MAX_BATCH]; // SADs of the candidates of one batch (phase A, ring, mvc_refine batches)
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

  // Packed SADs (k_wave pk_load / pk_sum, k_search eval_candidate_pk): a lane takes 4 columns of a row; their reference
  // samples are read as the aligned words around them (PK_WPR words: 2 x 4 bytes at 8 bit, 4 x 4 bytes at 16 bit).
  constexpr int PK_WPR = 2 * sizeof(pix);

  // SAD of the 4 samples of one row held in the words w[0 .. WPR) that start sh bytes into w; q = the 4 source samples
  template <int WPR>
  __device__ __forceinline__ unsigned pk_row(const unsigned *w, int sh, const unsigned *q, unsigned sum)
  {
    if constexpr (WPR == 2)
      return __vsadu4(__funnelshift_r(w[0], w[1], 8 * sh), q[0]) + sum;
    else
    {
      // sh in {0, 2, 4, 6}: the samples start in word sh / 4, at bit 8 * (sh & 3)
      const bool hi = sh >= 4;
      const unsigned a0 = hi ? w[1] : w[0], a1 = hi ? w[2] : w[1], a2 = hi ? w[3] : w[2];
      const int s = 8 * (sh & 3);
      sum = __vsadu2(__funnelshift_r(a0, a1, s), q[0]) + sum;
      return __vsadu2(__funnelshift_r(a1, a2, s), q[1]) + sum;
    }
  }

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

  // N rows of 4 samples, step samples apart, from r: the aligned words around each row into w (PK_WPR per row); returns
  // the byte offset of r in its first word (the same for every row: step keeps the alignment)
  template <int N>
  __device__ __forceinline__ int pk_fetch(const pix *r, size_t step, unsigned *w)
  {
    constexpr size_t AL = 4 * sizeof(pix) - 1;
    if constexpr (PK_WPR == 2)
    {
      const unsigned *b = (const unsigned *)((size_t)r & ~AL);
      const size_t sw = step / 4;
      #pragma unroll
      for (int i = 0; i < N; i++)
      {
        w[2 * i] = __ldg(b + i * sw);
        w[2 * i + 1] = __ldg(b + i * sw + 1);
      }
    }
    else
    {
      const uint2 *b = (const uint2 *)((size_t)r & ~AL);
      const size_t sw = step / 4;
      #pragma unroll
      for (int i = 0; i < N; i++)
      {
        const uint2 u0 = __ldg(b + i * sw), u1 = __ldg(b + i * sw + 1);
        w[4 * i] = u0.x;
        w[4 * i + 1] = u0.y;
        w[4 * i + 2] = u1.x;
        w[4 * i + 3] = u1.y;
      }
    }
    return (int)((size_t)r & AL);
  }

  // Packed form of eval_candidate (B5; LevelArgs::packed_s: reference rows and sub-plane bases 4-byte (8 bit) / 8-byte
  // (16 bit) aligned, paired or no chroma). Lane = 4 columns of every RP-th row: luma BX / 4 lanes per row, chroma U in
  // lanes 0..15 and V in 16..31 with CW / 4 lanes per row. Every reference word of the candidate is loaded before the
  // sums; the SADs are the same integers as eval_candidate's (only the order of the additions differs).
  template <int BX, int BSY, int CW, int CHH>
  __device__ __forceinline__ void eval_candidate_pk(const LevelArgs &a, const pix *s_y, const pix *s_u, const pix *s_v,
    int x0, int y0, int x1, int y1, int cx, int cy, int lane, int &sl, int &sc)
  {
    constexpr int RP = 32 / (BX / 4), NR = BSY / RP;
    constexpr int CRP = CW > 0 ? 16 / (CW / 4) : 1, NC = CHH / CRP;
    static_assert(BSY % RP == 0 && (CHH == 0 || (CW <= 16 && CHH % CRP == 0)), "packed layout");
    const int xp = x0 * a.pel + cx, yp = y0 * a.pel + cy;
    const int sub = (xp & (a.pel - 1)) | ((yp & (a.pel - 1)) << a.log_pel);
    const int rx = xp >> a.log_pel, ry = yp >> a.log_pel;
    unsigned wl[NR * PK_WPR], wc[NC > 0 ? NC * PK_WPR : 1];
    const int cg = lane % (BX / 4), rg = lane / (BX / 4);
    const int shl = pk_fetch<NR>(a.ref[0][sub] + (size_t)(ry + rg) * a.ew[0] + rx + 4 * cg, (size_t)RP * a.ew[0], wl);
    int shc = 0, pl = 1, ccg = 0, crg = 0;
    if constexpr (CHH > 0)
    {
      const int xpc = x1 * a.pel + (cx >> a.log_x_uv), ypc = y1 * a.pel + (cy >> a.log_y_uv);
      const int subc = (xpc & (a.pel - 1)) | ((ypc & (a.pel - 1)) << a.log_pel);
      const int rxc = xpc >> a.log_pel, ryc = ypc >> a.log_pel;
      pl = lane < 16 ? 1 : 2;
      ccg = (lane & 15) % (CW / 4);
      crg = (lane & 15) / (CW / 4);
      shc = pk_fetch<NC>(a.ref[pl][subc] + (size_t)(ryc + crg) * a.ew[pl] + rxc + 4 * ccg, (size_t)CRP * a.ew[pl], wc);
    }
    unsigned sum_l = 0, sum_c = 0;
    {
      const unsigned *q = (const unsigned *)(s_y + rg * BX + 4 * cg);
      #pragma unroll
      for (int i = 0; i < NR; i++)
        sum_l = pk_row<PK_WPR>(wl + PK_WPR * i, shl, q + i * RP * BX * sizeof(pix) / 4, sum_l);
    }
    if constexpr (CHH > 0)
    {
      const unsigned *q = (const unsigned *)((pl == 1 ? s_u : s_v) + crg * CW + 4 * ccg);
      #pragma unroll
      for (int i = 0; i < NC; i++)
        sum_c = pk_row<PK_WPR>(wc + PK_WPR * i, shc, q + i * CRP * CW * sizeof(pix) / 4, sum_c);
    }
    sl = warp_sum((int)sum_l);
    sc = scale_chroma(warp_sum((int)sum_c), a.chroma_scale);
  }

  // picks the compile-time row counts for the common block sizes
  __device__ __forceinline__ void eval_dispatch(const LevelArgs &a, const pix *s_y, const pix *s_u, const pix *s_v,
    int x0, int y0, int x1, int y1, int cx, int cy, int lane, bool pair, int &sl, int &sc)
  {
    if (a.packed_s)
    {
      // FRC (4:2:2 / 4:2:0, blksize 32 / 16) and the luma-only searches
      if (a.bs_x == 32 && a.bs_y == 32 && a.cw == 16 && a.ch == 32)
        return eval_candidate_pk<32, 32, 16, 32>(a, s_y, s_u, s_v, x0, y0, x1, y1, cx, cy, lane, sl, sc);
      if (a.bs_x == 32 && a.bs_y == 32 && a.cw == 16 && a.ch == 16)
        return eval_candidate_pk<32, 32, 16, 16>(a, s_y, s_u, s_v, x0, y0, x1, y1, cx, cy, lane, sl, sc);
      if (a.bs_x == 16 && a.bs_y == 16 && a.cw == 8 && a.ch == 16)
        return eval_candidate_pk<16, 16, 8, 16>(a, s_y, s_u, s_v, x0, y0, x1, y1, cx, cy, lane, sl, sc);
      if (a.bs_x == 16 && a.bs_y == 16 && a.cw == 8 && a.ch == 8)
        return eval_candidate_pk<16, 16, 8, 8>(a, s_y, s_u, s_v, x0, y0, x1, y1, cx, cy, lane, sl, sc);
      if (a.bs_x == 32 && a.bs_y == 32 && a.ch == 0)
        return eval_candidate_pk<32, 32, 0, 0>(a, s_y, s_u, s_v, x0, y0, x1, y1, cx, cy, lane, sl, sc);
      if (a.bs_x == 16 && a.bs_y == 16 && a.ch == 0)
        return eval_candidate_pk<16, 16, 0, 0>(a, s_y, s_u, s_v, x0, y0, x1, y1, cx, cy, lane, sl, sc);
    }
    if (a.bs_y == 32 && a.ch == 32)
      eval_candidate<32, 32>(a, s_y, s_u, s_v, x0, y0, x1, y1, cx, cy, lane, pair, sl, sc);
    else if (a.bs_y == 16 && a.ch == 16)
      eval_candidate<16, 16>(a, s_y, s_u, s_v, x0, y0, x1, y1, cx, cy, lane, pair, sl, sc);
    else if (a.bs_y == 32 && a.ch == 16)
      eval_candidate<32, 16>(a, s_y, s_u, s_v, x0, y0, x1, y1, cx, cy, lane, pair, sl, sc);
    else if (a.bs_y == 16 && a.ch == 8)
      eval_candidate<16, 8>(a, s_y, s_u, s_v, x0, y0, x1, y1, cx, cy, lane, pair, sl, sc);
    else if (a.ch == 0 && a.bs_y == 16)
      eval_candidate<16, 0>(a, s_y, s_u, s_v, x0, y0, x1, y1, cx, cy, lane, pair, sl, sc);
    else if (a.ch == 0 && a.bs_y == 32)
      eval_candidate<32, 0>(a, s_y, s_u, s_v, x0, y0, x1, y1, cx, cy, lane, pair, sl, sc);
    else if (a.ch == 0 && a.bs_y == 8)
      eval_candidate<8, 0>(a, s_y, s_u, s_v, x0, y0, x1, y1, cx, cy, lane, pair, sl, sc);
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

  // Candidate checker for mvc_refine (PlaneOfBlocks::CheckMV / CheckMV2 / CheckMVdir) on the block currently in shared
  // memory. Run by one warp: eval_dispatch spreads the SAD of one candidate over the lanes and returns it in every lane.
  struct AnaChecker
  {
    int bx, by, min_cost, best_sad;
    int dx_min, dx_max, dy_min, dy_max, pred_x, pred_y, nlambda;
    const LevelArgs *a;
    const pix *s_y, *s_u, *s_v;
    int x0, y0, x1, y1;
    int lane, warp, nwarps;
    bool pair;
    BlockShared *b;
    __device__ bool ok(int cx, int cy) const { return cx >= dx_min && cy >= dy_min && cx < dx_max && cy < dy_max; }

    // One batch of candidates (see mvcuda_refine.cuh); called by every thread of the CUDA block with identical arguments.
    // The SADs of the valid candidates are computed by the warps in parallel, then every thread folds them in the CPU's
    // order: a candidate is accepted when its full cost chain (vector cost, luma, chroma, each with penaltyNew) stays
    // strictly below the minimum so far. The partial sums only grow, so this accepts exactly what the CPU's early-exit
    // chain accepts. Not inlined: mvc_refine calls it from many places and the SAD code behind it is large.
    __device__ __noinline__ void run_batch(int mode, int n, const int *xs, const int *ys, const int *vals, int &dir)
    {
      for (int c = warp; c < n; c += nwarps)
      {
        const int cx = xs[c], cy = ys[c];
        if (!ok(cx, cy))
          continue;
        int sl, sc;
        eval_dispatch(*a, s_y, s_u, s_v, x0, y0, x1, y1, cx, cy, lane, pair, sl, sc);
        if (lane == 0)
        {
          b->sl[c] = sl;
          b->sc[c] = sc;
        }
      }
      __syncthreads();
      for (int c = 0; c < n; c++)
      {
        const int cx = xs[c], cy = ys[c];
        if (!ok(cx, cy))
          continue;
        const int dxp = pred_x - cx, dyp = pred_y - cy;
        const int dist = (int)((unsigned)(dxp * dxp) + (unsigned)(dyp * dyp));
        int cost = (int)((unsigned)nlambda * (unsigned)dist) >> (16 - a->bits);
        if (cost >= min_cost)
          continue;
        const int sl = b->sl[c], sc = b->sc[c];
        cost += sl + (int)(((long long)a->pnew * sl) >> 8);
        if (cost >= min_cost)
          continue;
        cost += sc + (int)(((long long)a->pnew * sc) >> 8);
        if (cost >= min_cost)
          continue;
        if (mode != mvc_refine::M_CHECKDIR)
        {
          bx = cx;
          by = cy;
        }
        min_cost = cost;
        best_sad = sl + sc;
        if (mode != mvc_refine::M_CHECK)
          dir = vals ? vals[c] : c;
      }
      __syncthreads(); // the next batch overwrites b->sl / b->sc
    }
  };

  // one CUDA block per slice; every warp evaluates candidate vectors, the block order inside the slice is serial.
  // Threads per block: with 768 the kernel was capped at 80 registers and the SAD code spilled 2.7 KB per thread, which
  // made the candidate rounds limited by local-memory traffic; 256 compiles without spills (GPU_PIPELINE_PLAN.md
  // section 25, S1: FRC chain -9%).
  constexpr int SEARCH_THREADS = 256;
  __global__ void __launch_bounds__(SEARCH_THREADS, 1) k_search(LevelArgs a)
  {
    extern __shared__ __align__(16) pix s_dyn[];
    __shared__ BlockShared b;
    __shared__ LevelArgs s_args; // copy of the kernel argument for the mvc_refine path: taking the address of the by-value argument would make every field access a slow generic load
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nwarps = blockDim.x >> 5;
    if (a.search_type != mvc_refine::S_EXHAUSTIVE && tid == 0)
      s_args = a;
    const int slice = blockIdx.x;
#ifdef MVC_PHASE_PROF
    const int bk = 31 - __clz(a.n_blk_x);
    long long ph_t = 0;
#endif
    const int beg = slice * a.n_blk_y / a.n_slices, end = (slice + 1) * a.n_blk_y / a.n_slices;
    const int step_x = a.bs_x - a.ov_x, step_y = a.bs_y - a.ov_y;
    const int h_scaled = a.hpad[0] >> a.log_scale, v_scaled = a.vpad[0] >> a.log_scale;
    const int ring_radius = (a.search_type == mvc_refine::S_EXHAUSTIVE) ? a.radius : 0; // the ring tables are for the exhaustive fold only
    const int ring_n = 4 * ring_radius * (ring_radius + 1);
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
      for (int r = 1; r <= ring_radius; r++)
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

        if (a.search_type != mvc_refine::S_EXHAUSTIVE)
        {
          // any other search type: the CPU control flow of PlaneOfBlocks::Refine, run by every thread of the block with
          // identical scalar state; the candidate SADs of each batch are spread over the warps, see mvcuda_refine.cuh
          AnaChecker ck;
          ck.bx = best_x;
          ck.by = best_y;
          ck.min_cost = min_cost;
          ck.best_sad = best_sad;
          ck.dx_min = dx_min;
          ck.dx_max = dx_max;
          ck.dy_min = dy_min;
          ck.dy_max = dy_max;
          ck.pred_x = pred.x;
          ck.pred_y = pred.y;
          ck.nlambda = nlambda;
          ck.a = &s_args; // written by thread 0 before the first barrier of the block loop
          ck.s_y = s_y;
          ck.s_u = s_u;
          ck.s_v = s_v;
          ck.x0 = x0[0];
          ck.y0 = y0[0];
          ck.x1 = x0[1];
          ck.y1 = y0[1];
          ck.lane = lane;
          ck.warp = warp;
          ck.nwarps = nwarps;
          ck.pair = pair_chroma;
          ck.b = &b;
          mvc_refine::refine(ck, a.search_type, a.radius);
          best_x = ck.bx;
          best_y = ck.by;
          best_sad = ck.best_sad;
          if (tid == 0)
          {
            if (idx > 1 + beg * a.n_blk_x && best_sad > a.bad_sad)
              atomicExch(a.bad_flag, 1);
            a.vec[3 * idx + 0] = best_x;
            a.vec[3 * idx + 1] = best_y;
            a.vec[3 * idx + 2] = best_sad;
            b.prev_x = best_x;
            b.prev_y = best_y;
            b.prev_sad = best_sad;
          }
          __syncthreads(); // the next block reads b.prev_*
          PH(4);
#ifdef MVC_PHASE_PROF
          if (tid == 0)
            atomicAdd(&g_prof[bk][5], 1ull);
#endif
          continue;
        }

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

  // ================================================================================================================
  // Wavefront search (GPU_PIPELINE_PLAN.md W): meander=false and one slice, i.e. the CPU result of
  // MAnalyse(meander=false, mt=false). Without meander a block depends on its left neighbour, on the block above and,
  // on the smallest level and on the last row, on the block above-right (FetchPredictors); the bottom-right predictor
  // of the other rows is the not yet searched (interpolated) vector. So block (x, y) can run as soon as row y - 1 has
  // finished block x (x + 1 for the up-right predictor): one warp per block row, rows handed out in order by an atomic
  // ticket, a progress counter per row (release / acquire). A row only waits for a row with a smaller ticket, whose warp
  // is already running, so this cannot deadlock for any grid size. The critical path is about n_blk_x + n_blk_y blocks
  // instead of a whole slice.
  //
  // Inside a warp the candidates of a batch are spread over groups of GW lanes (GW = block width rounded up to a power
  // of two; lane = column), 32 / GW candidates at a time; no block-wide barriers. The fold over the candidates runs
  // in every lane with identical values, in the CPU order (same code as the slice kernel and AnaChecker).

  __device__ __forceinline__ int ld_acquire(const int *p)
  {
    int v;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
  }
  __device__ __forceinline__ void st_release(int *p, int v)
  {
    asm volatile("st.release.gpu.global.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
  }

  // Waits until *p >= need (acquire loads) and returns the value read; seen: a value read before (with acquire
  // semantics) or 0. A relaxed poll followed by one fence.acq_rel was measured slower: the fence sits on the wavefront's
  // critical path at almost every block (GPU_PIPELINE_PLAN.md section 24). MVC_WAVE_SLEEP: optional pause between polls.
#ifndef MVC_WAVE_SLEEP
#define MVC_WAVE_SLEEP 0
#endif
  __device__ __forceinline__ int wait_at_least(const int *p, int need, int seen)
  {
    while (seen < need)
    {
      seen = ld_acquire(p);
      if (MVC_WAVE_SLEEP > 0 && seen < need)
        __nanosleep(MVC_WAVE_SLEEP);
    }
    return seen;
  }

  // SAD of one candidate, computed by the GW lanes of a group (col = lane inside the group) and summed over the group;
  // every lane of the group returns the sums. Lanes of an invalid candidate contribute 0. Chroma: for cw <= GW / 2 the
  // lanes [0, cw) take U and [cw, 2 cw) V, else each lane < cw takes both. BSY / CHH: compile-time row counts of the
  // luma block and of one chroma block (0 = runtime loop), as in eval_candidate. The reference planes are read with
  // __ldg: the pointers come from shared memory (WaveState), and plain dereferences compiled to generic loads that the
  // compiler issued one at a time (each row waited for the previous one); explicit global read-only loads go out together.
  template <int GW, int BSY, int CHH>
  __device__ __forceinline__ void eval_group(const LevelArgs &a, const pix *s_y, const pix *s_u, const pix *s_v,
    int x0, int y0, int x1, int y1, int cx, int cy, int col, bool valid, int &sl, int &sc)
  {
    int sum_l = 0, sum_c = 0;
    if (valid)
    {
      const int xp = x0 * a.pel + cx, yp = y0 * a.pel + cy;
      const int sub = (xp & (a.pel - 1)) | ((yp & (a.pel - 1)) << a.log_pel);
      const int rx = xp >> a.log_pel, ry = yp >> a.log_pel;
      if (col < a.bs_x)
      {
        const pix *r = a.ref[0][sub] + (size_t)ry * a.ew[0] + rx + col;
        const pix *q = s_y + col;
        if constexpr (BSY > 0)
        {
          int rv[BSY];
          #pragma unroll
          for (int row = 0; row < BSY; row++)
            rv[row] = __ldg(r + (size_t)row * a.ew[0]);
          #pragma unroll
          for (int row = 0; row < BSY; row++)
            sum_l += abs((int)q[row * a.bs_x] - rv[row]);
        }
        else
        {
          for (int row = 0; row < a.bs_y; row++)
            sum_l += abs((int)q[row * a.bs_x] - (int)__ldg(r + (size_t)row * a.ew[0]));
        }
      }
      if (a.cw)
      {
        const int xpc = x1 * a.pel + (cx >> a.log_x_uv), ypc = y1 * a.pel + (cy >> a.log_y_uv);
        const int subc = (xpc & (a.pel - 1)) | ((ypc & (a.pel - 1)) << a.log_pel);
        const int rxc = xpc >> a.log_pel, ryc = ypc >> a.log_pel;
        if (2 * a.cw <= GW)
        {
          if (col < 2 * a.cw)
          {
            const int pl = col < a.cw ? 1 : 2, cc = col < a.cw ? col : col - a.cw;
            const pix *r = a.ref[pl][subc] + (size_t)ryc * a.ew[pl] + rxc + cc;
            const pix *q = (pl == 1 ? s_u : s_v) + cc;
            if constexpr (CHH > 0)
            {
              int rv[CHH];
              #pragma unroll
              for (int row = 0; row < CHH; row++)
                rv[row] = __ldg(r + (size_t)row * a.ew[pl]);
              #pragma unroll
              for (int row = 0; row < CHH; row++)
                sum_c += abs((int)q[row * a.cw] - rv[row]);
            }
            else
            {
              for (int row = 0; row < a.ch; row++)
                sum_c += abs((int)q[row * a.cw] - (int)__ldg(r + (size_t)row * a.ew[pl]));
            }
          }
        }
        else if (col < a.cw)
        {
          for (int pl = 1; pl < 3; pl++)
          {
            const pix *r = a.ref[pl][subc] + (size_t)ryc * a.ew[pl] + rxc + col;
            const pix *q = (pl == 1 ? s_u : s_v) + col;
            for (int row = 0; row < a.ch; row++)
              sum_c += abs((int)q[row * a.cw] - (int)__ldg(r + (size_t)row * a.ew[pl]));
          }
        }
      }
    }
    #pragma unroll
    for (int o = GW / 2; o > 0; o >>= 1)
    {
      sum_l += __shfl_xor_sync(0xffffffffu, sum_l, o);
      sum_c += __shfl_xor_sync(0xffffffffu, sum_c, o);
    }
    sl = sum_l;
    sc = scale_chroma(sum_c, a.chroma_scale);
  }

  // State of the one warp of a k_wave CTA, in static shared memory: the candidate list, the SADs and the geometry are
  // exchanged through these __shared__ variables (plain shared loads, no generic pointers). The SAD evaluator is inlined
  // everywhere: as a noinline function ptxas issued its loads one at a time (about 5x slower). Each k_wave instance has
  // exactly one SAD variant, which keeps the inlined copies in the mvc_refine call sites affordable to compile.
  const int WAVE_SRC_MAX = 32 * 32 * 3; // a 32x32 luma block and two 32x32 chroma blocks

  struct WaveState
  {
    LevelArgs a;                              // copy of the kernel argument
    int x0, y0, x1, y1;                       // source block position (luma, chroma) in the padded planes
    int dx_min, dx_max, dy_min, dy_max;       // search range of the current block
    int n;                                    // number of candidates in xs / ys
    int xs[mvc_refine::MAX_BATCH], ys[mvc_refine::MAX_BATCH];
    int sl[mvc_refine::MAX_BATCH], sc[mvc_refine::MAX_BATCH];
    int ring_dx[48], ring_dy[48];             // exhaustive search offsets (ExpandingSearch order, radius <= 3)
    int fix_dx[26], fix_dy[26];               // offsets of the fixed batches of wave_plan: UMH cross + grid (20), hexagon (6)
    int bc[20];                               // values thread 0 hands to the whole CTA (row, neighbours, nLambda, level)
    long long key[4];                         // per-warp minimum of the parallel fold
    unsigned long long kmin;                  // fused fold (wave_eval_inl with FOLD): minimum of (cost << 8 | index)
    int odd;                                  // fused fold: some cost part was negative or the sum overflowed
  };
  // register cap of the fixed-plan k_wave kernels and the registers of reference samples one lane keeps in flight in
  // wave_eval_inl. Measured on the RTX 5090 (GPU_PIPELINE_PLAN.md section 24): 96 / 24 beat 255 / 64 by 17-19% in QTGMC
  // Fast / Slower (more resident CTAs per SM with 6-8 analyses in flight, and less spilling); 80 and 128 were slower.
#ifndef MVC_WAVE_REGS
#define MVC_WAVE_REGS 96
#endif
#ifndef MVC_WAVE_RSAMP
#define MVC_WAVE_RSAMP 24
#endif
  const int WAVE_NW = 4;                      // warps per block row (one CTA)
  const int WAVE_NT = 32 * WAVE_NW;
  __shared__ WaveState g_ws;
  __shared__ __align__(16) pix g_wsrc[WAVE_SRC_MAX]; // source block: luma, then U, then V
  // interpolated predictors of the current row and the row below, made by the whole CTA at the start of a row (one
  // block's inline interpolation would put two L2 round trips on the wavefront's critical path); rows of more blocks
  // interpolate inline
  const int WAVE_IP_MAX = 512;
  __shared__ int g_ip[2][WAVE_IP_MAX][3];          // (x, y, sad): [0] this row, [1] the row below
  __shared__ int g_ipl[WAVE_IP_MAX];               // nLambda of this row's blocks

  // SADs of the g_ws.n candidates g_ws.xs / ys into g_ws.sl / sc, 32 / GW candidates at a time. With range_test,
  // candidates outside the search range are skipped (their entries are never read). Called by all 32 lanes.
  // Split form of eval_group for the compile-time sizes with paired chroma (CHH == 0 or 2 * cw <= GW): eg_load reads the
  // reference rows of one candidate into registers, eg_sum forms the sums. wave_eval_inl loads several candidates per
  // lane before summing any of them, so the memory latency is paid once per R rounds instead of once per round.
  template <int BSY, int CHH>
  struct EvalRegs
  {
    int l[BSY];
    int c[CHH > 0 ? CHH : 1];
  };

  template <int GW, int BSY, int CHH>
  __device__ __forceinline__ void eg_load(const LevelArgs &a, int x0, int y0, int x1, int y1, int cx, int cy, int col,
    bool valid, EvalRegs<BSY, CHH> &e)
  {
    if (!valid)
      return;
    const int xp = x0 * a.pel + cx, yp = y0 * a.pel + cy;
    const int sub = (xp & (a.pel - 1)) | ((yp & (a.pel - 1)) << a.log_pel);
    const int rx = xp >> a.log_pel, ry = yp >> a.log_pel;
    if (col < a.bs_x)
    {
      const pix *r = a.ref[0][sub] + (size_t)ry * a.ew[0] + rx + col;
      #pragma unroll
      for (int row = 0; row < BSY; row++)
        e.l[row] = __ldg(r + (size_t)row * a.ew[0]);
    }
    if constexpr (CHH > 0)
    {
      if (col < 2 * a.cw)
      {
        const int xpc = x1 * a.pel + (cx >> a.log_x_uv), ypc = y1 * a.pel + (cy >> a.log_y_uv);
        const int subc = (xpc & (a.pel - 1)) | ((ypc & (a.pel - 1)) << a.log_pel);
        const int rxc = xpc >> a.log_pel, ryc = ypc >> a.log_pel;
        const int pl = col < a.cw ? 1 : 2, cc = col < a.cw ? col : col - a.cw;
        const pix *r = a.ref[pl][subc] + (size_t)ryc * a.ew[pl] + rxc + cc;
        #pragma unroll
        for (int row = 0; row < CHH; row++)
          e.c[row] = __ldg(r + (size_t)row * a.ew[pl]);
      }
    }
  }

  template <int GW, int BSY, int CHH>
  __device__ __forceinline__ void eg_sum(const LevelArgs &a, const pix *s_y, const pix *s_u, const pix *s_v, int col,
    bool valid, const EvalRegs<BSY, CHH> &e, int &sl, int &sc)
  {
    int sum_l = 0, sum_c = 0;
    if (valid)
    {
      if (col < a.bs_x)
      {
        const pix *q = s_y + col;
        #pragma unroll
        for (int row = 0; row < BSY; row++)
          sum_l += abs((int)q[row * a.bs_x] - e.l[row]);
      }
      if constexpr (CHH > 0)
      {
        if (col < 2 * a.cw)
        {
          const int pl = col < a.cw ? 1 : 2, cc = col < a.cw ? col : col - a.cw;
          const pix *q = (pl == 1 ? s_u : s_v) + cc;
          #pragma unroll
          for (int row = 0; row < CHH; row++)
            sum_c += abs((int)q[row * a.cw] - e.c[row]);
        }
      }
    }
    #pragma unroll
    for (int o = GW / 2; o > 0; o >>= 1)
    {
      sum_l += __shfl_xor_sync(0xffffffffu, sum_l, o);
      if constexpr (CHH > 0)
        sum_c += __shfl_xor_sync(0xffffffffu, sum_c, o);
    }
    sl = sum_l;
    sc = scale_chroma(sum_c, a.chroma_scale);
  }

  // Packed form for 8-bit samples (LevelArgs::packed: bs_x == GW, chroma absent or cw == GW / 2, all reference pitches
  // and sub-plane bases 4-byte aligned): a lane takes 4 columns of every 4th row (luma: GW / 4 lanes per row; chroma:
  // the first GW / 2 lanes U, the others V, GW / 8 lanes per row), reads each row as the two aligned words around it
  // and forms the SAD of 4 samples with one __vsadu4. Half the loads and half the registers of the byte form; the sums
  // are the same integers (only the order of the additions differs).
  // 16-bit samples (B5): the same lane layout; the 4 samples of a row (8 bytes) are read as the two aligned 8-byte words
  // around them (reference pitches and sub-plane bases 8-byte aligned), picked out by funnel shifts and summed with two
  // __vsadu2. Half the load instructions of the split form, the same registers.
  template <int BSY, int CHH>
  struct PackRegs
  {
    static constexpr int WPR = PK_WPR;
    unsigned l[BSY / 4 * WPR];
    unsigned c[CHH > 0 ? CHH / 4 * WPR : 1];
    int shl, shc;                        // byte offset of the row start in its first word / 8-byte word (same for every row)
  };

  template <int GW, int BSY, int CHH>
  __device__ __forceinline__ void pk_load(const LevelArgs &a, int x0, int y0, int x1, int y1, int cx, int cy, int col,
    bool valid, PackRegs<BSY, CHH> &e)
  {
    if (!valid)
      return;
    const int xp = x0 * a.pel + cx, yp = y0 * a.pel + cy;
    const int sub = (xp & (a.pel - 1)) | ((yp & (a.pel - 1)) << a.log_pel);
    const int rx = xp >> a.log_pel, ry = yp >> a.log_pel;
    constexpr int WPR = PackRegs<BSY, CHH>::WPR;
    constexpr size_t AL = 4 * sizeof(pix) - 1; // alignment mask of the words read: 4 (8 bit) or 8 bytes (16 bit)
    {
      const int cg = col % (GW / 4), rg = col / (GW / 4);
      const pix *r = a.ref[0][sub] + (size_t)(ry + rg) * a.ew[0] + rx + 4 * cg;
      e.shl = (int)((size_t)r & AL);
      const size_t step = (size_t)a.ew[0] * sizeof(pix); // 4 rows in words
      if constexpr (WPR == 2)
      {
        const unsigned *w = (const unsigned *)((size_t)r & ~AL);
        #pragma unroll
        for (int i = 0; i < BSY / 4; i++)
        {
          e.l[2 * i] = __ldg(w + i * step);
          e.l[2 * i + 1] = __ldg(w + i * step + 1);
        }
      }
      else
      {
        const uint2 *w = (const uint2 *)((size_t)r & ~AL);
        #pragma unroll
        for (int i = 0; i < BSY / 4; i++)
        {
          const uint2 u0 = __ldg(w + i * (step / 2)), u1 = __ldg(w + i * (step / 2) + 1);
          e.l[4 * i] = u0.x;
          e.l[4 * i + 1] = u0.y;
          e.l[4 * i + 2] = u1.x;
          e.l[4 * i + 3] = u1.y;
        }
      }
    }
    if constexpr (CHH > 0)
    {
      const int xpc = x1 * a.pel + (cx >> a.log_x_uv), ypc = y1 * a.pel + (cy >> a.log_y_uv);
      const int subc = (xpc & (a.pel - 1)) | ((ypc & (a.pel - 1)) << a.log_pel);
      const int rxc = xpc >> a.log_pel, ryc = ypc >> a.log_pel;
      const int pl = col < GW / 2 ? 1 : 2, cc = col % (GW / 2), cg = cc % (GW / 8), rg = cc / (GW / 8);
      const pix *r = a.ref[pl][subc] + (size_t)(ryc + rg) * a.ew[pl] + rxc + 4 * cg;
      e.shc = (int)((size_t)r & AL);
      const size_t step = (size_t)a.ew[pl] * sizeof(pix);
      if constexpr (WPR == 2)
      {
        const unsigned *w = (const unsigned *)((size_t)r & ~AL);
        #pragma unroll
        for (int i = 0; i < CHH / 4; i++)
        {
          e.c[2 * i] = __ldg(w + i * step);
          e.c[2 * i + 1] = __ldg(w + i * step + 1);
        }
      }
      else
      {
        const uint2 *w = (const uint2 *)((size_t)r & ~AL);
        #pragma unroll
        for (int i = 0; i < CHH / 4; i++)
        {
          const uint2 u0 = __ldg(w + i * (step / 2)), u1 = __ldg(w + i * (step / 2) + 1);
          e.c[4 * i] = u0.x;
          e.c[4 * i + 1] = u0.y;
          e.c[4 * i + 2] = u1.x;
          e.c[4 * i + 3] = u1.y;
        }
      }
    }
  }

  template <int GW, int BSY, int CHH>
  __device__ __forceinline__ void pk_sum(const LevelArgs &a, int col, bool valid, const PackRegs<BSY, CHH> &e, int &sl, int &sc)
  {
    constexpr int WPR = PackRegs<BSY, CHH>::WPR;
    unsigned sum_l = 0, sum_c = 0;
    if (valid)
    {
      {
        const int cg = col % (GW / 4), rg = col / (GW / 4);
        const unsigned *q = (const unsigned *)(g_wsrc + rg * GW + 4 * cg);
        #pragma unroll
        for (int i = 0; i < BSY / 4; i++) // 4 rows down = 4 * GW samples = GW * sizeof(pix) words
          sum_l = pk_row<WPR>(e.l + WPR * i, e.shl, q + i * GW * sizeof(pix), sum_l);
      }
      if constexpr (CHH > 0)
      {
        const int pl = col < GW / 2 ? 1 : 2, cc = col % (GW / 2), cg = cc % (GW / 8), rg = cc / (GW / 8);
        const unsigned *q = (const unsigned *)(g_wsrc + GW * BSY + (pl - 1) * (GW / 2) * CHH + rg * (GW / 2) + 4 * cg);
        #pragma unroll
        for (int i = 0; i < CHH / 4; i++)
          sum_c = pk_row<WPR>(e.c + WPR * i, e.shc, q + i * (GW / 2) * sizeof(pix), sum_c);
      }
    }
    int il = (int)sum_l, ic = (int)sum_c;
    #pragma unroll
    for (int o = GW / 2; o > 0; o >>= 1)
    {
      il += __shfl_xor_sync(0xffffffffu, il, o);
      if constexpr (CHH > 0)
        ic += __shfl_xor_sync(0xffffffffu, ic, o);
    }
    sl = il;
    sc = scale_chroma(ic, a.chroma_scale);
  }

  // Fused fold: the arguments of the CPU cost of a candidate (vector cost against the predictor, CheckMV's minimum before
  // the batch). The leader of each candidate group adds (cost << 8 | index) to the CTA-wide minimum when the cost is
  // below that minimum, and flags the batch when a cost part is negative or the sum overflows an int, exactly the
  // conditions under which wave_fold_par falls back to the sequential fold; wave_fold_take reads the result after the
  // barrier at the end of wave_eval_inl. Before such a batch g_ws.kmin = ~0 and g_ws.odd = 0 (wave_fold_init).
  struct FoldArgs
  {
    int pred_x, pred_y, nlambda, min_cost;
  };

  template <bool FOLD>
  __device__ __forceinline__ void wave_put(int c, int sl, int sc, const FoldArgs &f)
  {
    g_ws.sl[c] = sl;
    g_ws.sc[c] = sc;
    if constexpr (FOLD)
    {
      const LevelArgs &a = g_ws.a;
      const int dxp = f.pred_x - g_ws.xs[c], dyp = f.pred_y - g_ws.ys[c];
      const int dist = (int)((unsigned)(dxp * dxp) + (unsigned)(dyp * dyp));
      const int lam = (int)((unsigned)f.nlambda * (unsigned)dist) >> (16 - a.bits);
      const int tl = (int)(((long long)a.pnew * sl) >> 8), tc = (int)(((long long)a.pnew * sc) >> 8);
      const long long cost = (long long)lam + sl + tl + sc + tc;
      if (lam < 0 || tl < 0 || tc < 0 || cost > 0x7fffffffLL)
        atomicOr(&g_ws.odd, 1);
      else if (cost < f.min_cost)
        atomicMin(&g_ws.kmin, ((unsigned long long)cost << 8) | (unsigned)c);
    }
  }

  __device__ __forceinline__ void wave_fold_init()
  {
    g_ws.kmin = ~0ull;
    g_ws.odd = 0;
  }

  // Rounds of the packed (PK) or the split byte form: R rounds of G candidates per pass with all their loads in flight
  // before any sum. R is a compile-time count (registers); wave_eval_inl takes R = 1 for a batch that fits one round
  // (most batches: phase A, the hexagon, radius-1 rings), so those do not carry the registers of R rounds.
  template <int GW, int BSY, int CHH, bool FOLD, bool PK, int R>
  __device__ __forceinline__ void wave_rounds(const LevelArgs &a, int n, int grp, int col, bool range_test, const FoldArgs &fa)
  {
    constexpr int G = WAVE_NT / GW;
    const pix *s_y = g_wsrc, *s_u = g_wsrc + a.bs_x * a.bs_y, *s_v = s_u + a.cw * a.ch;
    for (int base = 0; base < n; base += G * R)
    {
      typename std::conditional<PK, PackRegs<BSY, CHH>, EvalRegs<BSY, CHH>>::type e[R];
      bool v[R];
      int cs[R];
      #pragma unroll
      for (int r = 0; r < R; r++)
      {
        const int c = base + r * G + grp;
        cs[r] = c;
        const int cx = c < n ? g_ws.xs[c] : 0, cy = c < n ? g_ws.ys[c] : 0;
        v[r] = c < n && (!range_test || (cx >= g_ws.dx_min && cy >= g_ws.dy_min && cx < g_ws.dx_max && cy < g_ws.dy_max));
        if constexpr (PK)
          pk_load<GW, BSY, CHH>(a, g_ws.x0, g_ws.y0, g_ws.x1, g_ws.y1, cx, cy, col, v[r], e[r]);
        else
          eg_load<GW, BSY, CHH>(a, g_ws.x0, g_ws.y0, g_ws.x1, g_ws.y1, cx, cy, col, v[r], e[r]);
      }
      #pragma unroll
      for (int r = 0; r < R; r++)
      {
        if (base + r * G < n) // some group has a candidate in this round (uniform over the warp; no break, so the loop
        {                     // stays unrolled and e[] stays in registers)
          int sl, sc;
          if constexpr (PK)
            pk_sum<GW, BSY, CHH>(a, col, v[r], e[r], sl, sc);
          else
            eg_sum<GW, BSY, CHH>(a, s_y, s_u, s_v, col, v[r], e[r], sl, sc);
          if (v[r] && col == 0)
            wave_put<FOLD>(cs[r], sl, sc, fa);
        }
      }
    }
  }

  template <int GW, int BSY, int CHH, bool FOLD = false>
  __device__ __forceinline__ void wave_eval_inl(int lane, bool range_test, const FoldArgs &fa = FoldArgs{})
  {
    const LevelArgs &a = g_ws.a;
    constexpr int G = WAVE_NT / GW; // candidate groups in the CTA
    const int grp = lane / GW, col = lane % GW;
    const pix *s_y = g_wsrc, *s_u = g_wsrc + a.bs_x * a.bs_y, *s_v = s_u + a.cw * a.ch;
    const int n = g_ws.n;
    if constexpr (BSY > 0)
    if (a.packed)
    {
      constexpr int W = (BSY + CHH) * sizeof(pix) / 2; // registers per candidate
      constexpr int R = (MVC_WAVE_RSAMP / W) < 1 ? 1 : MVC_WAVE_RSAMP / W;
      if (n <= G)
        wave_rounds<GW, BSY, CHH, FOLD, true, 1>(a, n, grp, col, range_test, fa);
      else
        wave_rounds<GW, BSY, CHH, FOLD, true, R>(a, n, grp, col, range_test, fa);
      __syncthreads();
      return;
    }
    // the split form needs paired chroma (U and V columns side by side in one group): no chroma, 4:2:0, 4:2:2
    if constexpr (BSY > 0)
    if (CHH == 0 || 2 * a.cw <= GW)
    {
      constexpr int R = (MVC_WAVE_RSAMP / (BSY + CHH)) < 1 ? 1 : MVC_WAVE_RSAMP / (BSY + CHH);
      if (n <= G)
        wave_rounds<GW, BSY, CHH, FOLD, false, 1>(a, n, grp, col, range_test, fa);
      else
        wave_rounds<GW, BSY, CHH, FOLD, false, R>(a, n, grp, col, range_test, fa);
      __syncthreads();
      return;
    }
    for (int base = 0; base < n; base += G)
    {
      const int c = base + grp;
      const int cx = c < n ? g_ws.xs[c] : 0, cy = c < n ? g_ws.ys[c] : 0;
      const bool valid = c < n
        && (!range_test || (cx >= g_ws.dx_min && cy >= g_ws.dy_min && cx < g_ws.dx_max && cy < g_ws.dy_max));
      int sl, sc;
      eval_group<GW, BSY, CHH>(a, s_y, s_u, s_v, g_ws.x0, g_ws.y0, g_ws.x1, g_ws.y1, cx, cy, col, valid, sl, sc);
      if (valid && col == 0)
        wave_put<FOLD>(c, sl, sc, fa);
    }
    __syncthreads();
  }
  // Folds the evaluated candidates in the CPU order with the semantics of CheckMV / CheckMV2 / CheckMVdir (see
  // AnaChecker::run_batch); every lane runs it with identical values.
  __device__ __forceinline__ void wave_fold(int mode, int n, const int *vals, int &dir, int &bx, int &by, int &min_cost,
    int &best_sad, int pred_x, int pred_y, int nlambda)
  {
    const LevelArgs &a = g_ws.a;
    for (int c = 0; c < n; c++)
    {
      const int cx = g_ws.xs[c], cy = g_ws.ys[c];
      if (!(cx >= g_ws.dx_min && cy >= g_ws.dy_min && cx < g_ws.dx_max && cy < g_ws.dy_max))
        continue;
      const int dxp = pred_x - cx, dyp = pred_y - cy;
      const int dist = (int)((unsigned)(dxp * dxp) + (unsigned)(dyp * dyp));
      int cost = (int)((unsigned)nlambda * (unsigned)dist) >> (16 - a.bits);
      if (cost >= min_cost)
        continue;
      const int sl = g_ws.sl[c], sc = g_ws.sc[c];
      cost += sl + (int)(((long long)a.pnew * sl) >> 8);
      if (cost >= min_cost)
        continue;
      cost += sc + (int)(((long long)a.pnew * sc) >> 8);
      if (cost >= min_cost)
        continue;
      if (mode != mvc_refine::M_CHECKDIR)
      {
        bx = cx;
        by = cy;
      }
      min_cost = cost;
      best_sad = sl + sc;
      if (mode != mvc_refine::M_CHECK)
        dir = vals ? vals[c] : c;
    }
  }

  // Parallel form of wave_fold, called by every thread of the CTA. When every part of every candidate's cost (vector
  // cost, luma term, chroma term) is non-negative and the sum fits an int, the CPU's sequential "accept while the cost
  // chain stays below the running minimum" ends at the first candidate with the smallest final cost below the minimum
  // before the batch, so a minimum over (cost, index) gives exactly its result. Otherwise (an overflowed vector cost, a
  // negative pnew) the sequential fold runs.
  __device__ __forceinline__ void wave_fold_par(int tid, int mode, int n, const int *vals, int &dir, int &bx, int &by,
    int &min_cost, int &best_sad, int pred_x, int pred_y, int nlambda)
  {
    const LevelArgs &a = g_ws.a;
    const long long none = 0x7fffffffffffffffLL;
    long long key = none;
    int odd = 0;
    for (int c = tid; c < n; c += WAVE_NT)
    {
      const int cx = g_ws.xs[c], cy = g_ws.ys[c];
      if (!(cx >= g_ws.dx_min && cy >= g_ws.dy_min && cx < g_ws.dx_max && cy < g_ws.dy_max))
        continue;
      const int dxp = pred_x - cx, dyp = pred_y - cy;
      const int dist = (int)((unsigned)(dxp * dxp) + (unsigned)(dyp * dyp));
      const int lam = (int)((unsigned)nlambda * (unsigned)dist) >> (16 - a.bits);
      const int sl = g_ws.sl[c], sc = g_ws.sc[c];
      const int tl = (int)(((long long)a.pnew * sl) >> 8), tc = (int)(((long long)a.pnew * sc) >> 8);
      const long long cost = (long long)lam + sl + tl + sc + tc;
      if (lam < 0 || tl < 0 || tc < 0 || cost > 0x7fffffffLL)
        odd = 1;
      if (cost < min_cost)
      {
        const long long k = (cost << 8) | c;
        key = k < key ? k : key;
      }
    }
    if (__syncthreads_or(odd))
    {
      wave_fold(mode, n, vals, dir, bx, by, min_cost, best_sad, pred_x, pred_y, nlambda);
      return;
    }
    for (int o = 16; o > 0; o >>= 1)
    {
      const long long other = __shfl_xor_sync(0xffffffffu, key, o);
      key = other < key ? other : key;
    }
    if ((tid & 31) == 0)
      g_ws.key[tid >> 5] = key;
    __syncthreads();
    key = g_ws.key[0];
    for (int w = 1; w < WAVE_NW; w++)
      key = g_ws.key[w] < key ? g_ws.key[w] : key;
    if (key != none)
    {
      const int c = (int)(key & 255);
      if (mode != mvc_refine::M_CHECKDIR)
      {
        bx = g_ws.xs[c];
        by = g_ws.ys[c];
      }
      min_cost = (int)(key >> 8);
      best_sad = g_ws.sl[c] + g_ws.sc[c];
      if (mode != mvc_refine::M_CHECK)
        dir = vals ? vals[c] : c;
    }
  }

  // result of a batch evaluated with the fused fold (wave_eval_inl<..., true>), read by every thread after its barrier;
  // the same result as wave_fold_par
  __device__ __forceinline__ void wave_fold_take(int mode, int n, const int *vals, int &dir, int &bx, int &by,
    int &min_cost, int &best_sad, int pred_x, int pred_y, int nlambda)
  {
    if (g_ws.odd)
    {
      wave_fold(mode, n, vals, dir, bx, by, min_cost, best_sad, pred_x, pred_y, nlambda);
      return;
    }
    const unsigned long long key = g_ws.kmin;
    if (key != ~0ull)
    {
      const int c = (int)(key & 255);
      if (mode != mvc_refine::M_CHECKDIR)
      {
        bx = g_ws.xs[c];
        by = g_ws.ys[c];
      }
      min_cost = (int)(key >> 8);
      best_sad = g_ws.sl[c] + g_ws.sc[c];
      if (mode != mvc_refine::M_CHECK)
        dir = vals ? vals[c] : c;
    }
  }

  // evaluation and fold of one batch for the mvc_refine kernels: one copy per kernel instance instead of one per call
  // site of mvc_refine (inlining it there made those kernels take about 20 minutes to compile per architecture). With
  // compile-time sizes (BSY > 0) the evaluation inside still has its loads in flight together.
  template <int GW, int BSY, int CHH>
  __device__ __noinline__ void wave_batch_call(int tid, int mode, int n, const int *vals, int &dir, int &bx, int &by,
    int &min_cost, int &best_sad, int pred_x, int pred_y, int nlambda)
  {
    wave_eval_inl<GW, BSY, CHH>(tid, true);
    wave_fold_par(tid, mode, n, vals, dir, bx, by, min_cost, best_sad, pred_x, pred_y, nlambda);
  }

  // Candidate checker for mvc_refine run by the threads of a k_wave CTA
  template <int GW, int BSY, int CHH>
  struct WaveChecker
  {
    int bx, by, min_cost, best_sad, pred_x, pred_y, nlambda, lane;
    __device__ bool ok(int cx, int cy) const
    {
      return cx >= g_ws.dx_min && cy >= g_ws.dy_min && cx < g_ws.dx_max && cy < g_ws.dy_max;
    }
    __device__ __forceinline__ void run_batch(int mode, int n, const int *xs, const int *ys, const int *vals, int &dir)
    {
      if (xs != g_ws.xs) // wave_bad_search writes its lists into g_ws itself (lane 0, published by the barrier below)
        for (int c = lane; c < n; c += WAVE_NT)
        {
          g_ws.xs[c] = xs[c];
          g_ws.ys[c] = ys[c];
        }
      if (lane == 0)
        g_ws.n = n;
      __syncthreads();
      wave_batch_call<GW, BSY, CHH>(lane, mode, n, vals, dir, bx, by, min_cost, best_sad, pred_x, pred_y, nlambda);
      __syncthreads(); // the next batch overwrites the candidate list and the SADs
    }
  };

  // The wide search PseudoEPZSearch runs for a "bad" block (SAD above the badSAD threshold): UMHSearch(badrange * pel)
  // around (0, 0) for badrange > 0, ExpandingSearch rings around (0, 0) until a good vector is found for badrange < 0,
  // then the small sub-pel rings around the result. Rare, so it uses the runtime-size SAD code and is not inlined
  // (one copy per GW). The candidate lists are cut into batches of at most MAX_BATCH in the CPU order, which keeps
  // the sequential semantics (the UMH list exceeds one batch for pel 2). Called by every thread of the CTA.
  template <int GW>
  __device__ __noinline__ void wave_bad_search(int tid, int &bx, int &by, int &min_cost, int &best_sad, int pred_x,
    int pred_y, int nlambda)
  {
    const LevelArgs &a = g_ws.a;
    WaveChecker<GW, 0, 0> ck;
    ck.bx = bx;
    ck.by = by;
    ck.min_cost = min_cost;
    ck.best_sad = best_sad;
    ck.pred_x = pred_x;
    ck.pred_y = pred_y;
    ck.nlambda = nlambda;
    ck.lane = tid;
    const int found = best_sad;
    const int range = a.bad_range * a.pel; // badrange * nPel (negative for the exhaustive variant)
    // the candidate lists go straight into the shared list (lane 0; run_batch's barrier publishes them): local arrays
    // of MAX_BATCH entries made k_wave's stack frame, and so the device's local-memory reservation, 1.3 KB per thread larger
    int n = 0, dummy = 0;
    auto flush = [&]() {
      if (n)
        ck.run_batch(mvc_refine::M_CHECK, n, g_ws.xs, g_ws.ys, nullptr, dummy);
      n = 0;
    };
    auto add = [&](int x, int y) {
      if (tid == 0)
      {
        g_ws.xs[n] = x;
        g_ws.ys[n] = y;
      }
      n++;
      if (n == mvc_refine::MAX_BATCH)
        flush();
    };
    // ExpandingSearch(r, s, mvx, mvy) in the order of mvc_refine::add_expanding
    auto ring = [&](int r, int s, int mvx, int mvy) {
      for (int i = -r + s; i < r; i += s)
      {
        add(mvx + i, mvy - r);
        add(mvx + i, mvy + r);
      }
      for (int j = -r + s; j < r; j += s)
      {
        add(mvx - r, mvy + j);
        add(mvx + r, mvy + j);
      }
      add(mvx - r, mvy - r);
      add(mvx - r, mvy + r);
      add(mvx + r, mvy - r);
      add(mvx + r, mvy + r);
      flush();
    };
    if (a.bad_range > 0)
    {
      // UMHSearch(range, 0, 0) as mvc_refine::umh: CrossSearch, the hex grid rings, then Hex2Search(range)
      for (int i = 1; i < range; i += 2)
      {
        add(-i, 0);
        add(i, 0);
      }
      for (int j = 1; j < range; j += 2)
      {
        add(0, j);
        add(0, -j);
      }
      const int hex4[16][2] = { {-4, 2}, {-4, 1}, {-4, 0}, {-4,-1}, {-4,-2}, { 4,-2}, { 4,-1}, { 4, 0}, { 4, 1}, { 4, 2},
                                { 2, 3}, { 0, 4}, {-2, 3}, {-2,-3}, { 0,-4}, { 2,-3} };
      int i = 1;
      do
      {
        for (int j = 0; j < 16; j++)
          add(hex4[j][0] * i, hex4[j][1] * i);
      } while (++i <= range / 4);
      flush();
      mvc_refine::hex2(ck, range);
    }
    else if (a.bad_range < 0)
    {
      for (int i = 1; i < -range; i += a.pel)
      {
        ring(i, a.pel, 0, 0);
        if (ck.best_sad < found / 4)
          break; // a rather good vector is found
      }
    }
    const int mvx = ck.bx, mvy = ck.by; // refine in a small area
    for (int i = 1; i < a.pel; i++)
      ring(i, 1, mvx, mvy);
    bx = ck.bx;
    by = ck.by;
    min_cost = ck.min_cost;
    best_sad = ck.best_sad;
  }

#ifdef MVC_PHASE_PROF
  // k_wave phase cycles (development aid): [log2(blocks per row)][issue, wait, predictors, A eval, A fold, ring / refine eval,
  // ring fold, store, count]
  __device__ unsigned long long g_wprof[16][12];
#define WPH_START() long long wt_ = clock64()
#define WPH(i) do { if (lane == 0) { long long n_ = clock64(); atomicAdd(&g_wprof[wbk][i], (unsigned long long)(n_ - wt_)); wt_ = n_; } } while (0)
#else
#define WPH_START() do {} while (0)
#define WPH(i) do {} while (0)
#endif

  // Radius of the refine step when it is a single batch of rings around the best vector so far, else 0: exhaustive search
  // (every coarse level) and Hex2Search with range <= 1, which is only its closing ExpandingSearch(1, 1) (QTGMC's
  // pelsearch=1 at level 0). Both are CheckMV batches in ExpandingSearch order.
  __host__ __device__ inline int ring_radius_of(int search_type, int radius)
  {
    if (search_type == mvc_refine::S_EXHAUSTIVE)
      return radius;
    if (search_type == mvc_refine::S_HEX2 && radius <= 1)
      return 1;
    return 0;
  }

  // Refine steps the ring-only kernel (GEN = false) runs as a fixed sequence of batches, else 0 (mvc_refine needed):
  // WP_RING: the rings of ring_radius_of; WP_HEX: Hex2Search(2..3) = the radius-2 hexagon (CheckMVdir) around the best
  // vector, the centre moved to the winner, ExpandingSearch(1, 1) around it (the hexagon walk of larger ranges runs
  // range / 2 - 1 = 0 times); WP_UMH: UMHSearch(2..3) = CrossSearch (+-1 only) and one hex grid ring around the best
  // vector in one batch, then WP_HEX. QTGMC's pel-search 2 levels (Slow .. Placebo) are WP_HEX / WP_UMH.
  enum { WP_GEN = 0, WP_RING = 1, WP_HEX = 2, WP_UMH = 3 };
  __host__ __device__ inline int wave_plan(int search_type, int radius)
  {
    if (ring_radius_of(search_type, radius) > 0)
      return WP_RING;
    if (search_type == mvc_refine::S_HEX2 && radius >= 2 && radius <= 3)
      return WP_HEX;
    if (search_type == mvc_refine::S_UMH && radius >= 2 && radius <= 3)
      return WP_UMH;
    return WP_GEN;
  }
  // candidate offsets of the fixed batches in the CPU order (mvc_refine::umh with range 2..3, mvc_refine::hex2) and the
  // values CheckMVdir hands over (the hexagon direction = index)
  // (copied to shared memory per level: g_ws.fix_dx / fix_dy, UMH first, then the hexagon)
  __device__ const int g_fix_dx[26] = { -1, 1, 0, 0, -4, -4, -4, -4, -4, 4, 4, 4, 4, 4, 2, 0, -2, -2, 0, 2, -2, -1, 1, 2, 1, -1 };
  __device__ const int g_fix_dy[26] = { 0, 0, 1, -1, 2, 1, 0, -1, -2, -2, -1, 0, 1, 2, 3, 4, 3, -3, -4, -3, 0, 2, 2, 0, -2, -2 };

  // GEN = false: only the fixed batches of wave_plan (no mvc_refine code: small and fast to compile);
  // GEN = true: every search type through mvc_refine
  template <int GW, int BSY, int CHH, bool GEN>
  __global__ void __maxnreg__(GEN ? 255 : MVC_WAVE_REGS) k_wave(const __grid_constant__ WaveCall c)
  {
    const int lane = threadIdx.x; // thread index in the CTA (all WAVE_NW warps work on one block row)
    int cur = -1;                 // level of the previous row of this CTA
    int ring_n = 0;
    int plan = WP_GEN;            // wave_plan of that level

    for (;;)
    {
      if (lane == 0)
      {
        const int t = atomicAdd(c.ticket, 1);
        if (t == 0)
        {
          // payload header: total words, 1, and per level the size word in front of its vectors
          c.payload[0] = c.words;
          c.payload[1] = 1;
          for (int l = 0; l < c.nl; l++)
            c.lv[l].vec[-1] = c.lv[l].n_blk_x * c.lv[l].n_blk_y * 3 + 1;
        }
        int level = -1, row = 0;
        for (int l = c.nl - 1, first = 0; l >= 0; first += c.lv[l].n_blk_y, l--)
          if (t < first + c.lv[l].n_blk_y)
          {
            level = l;
            row = t - first;
            break;
          }
        g_ws.bc[0] = row;
        g_ws.bc[13] = level;
      }
      __syncthreads();
      const int row = g_ws.bc[0], level = g_ws.bc[13];
      if (level < 0)
      {
        if (c.host_out)
        {
          if (lane == 0)
          {
            __threadfence(); // this CTA's vectors before its count
            g_ws.bc[17] = atomicAdd(c.finished, 1u) == gridDim.x - 1 ? 1 : 0;
            __threadfence();
          }
          __syncthreads();
          if (g_ws.bc[17])
          {
            // the last CTA: every vector of every CTA is written (fenced before their counts)
            for (int i = lane; i < c.words; i += WAVE_NT)
              c.host_out[i] = __ldcg(c.payload + i);
            if (lane == 0)
              c.host_out[c.words + c.bad_word] = __ldcg(c.state + c.bad_word);
            for (int i = lane; i < c.state_words; i += WAVE_NT)
              c.state[i] = 0;
          }
        }
        return;
      }
      // the level's arguments, copied to shared memory once per level: c.lv[level] with a run-time index would be a
      // memory load at every use instead of a constant-bank operand
      const LevelArgs &a = g_ws.a;
      if (level != cur)
      {
        // a new level for this CTA: wait until the coarser level is complete and the global MV estimate is written
        if (lane == 0)
        {
          if (level < c.nl - 1)
            while (ld_acquire(c.level_ready + level) == 0)
              __nanosleep(256);
          g_ws.a = c.lv[level];
          const int pl = wave_plan(a.search_type, a.radius);
          const int ring_r = pl == WP_RING ? ring_radius_of(a.search_type, a.radius) : pl == WP_GEN ? 0 : 1;
          int n = 0;
          for (int r = 1; r <= ring_r; r++)
            n = mvc_refine::add_expanding(g_ws.ring_dx, g_ws.ring_dy, n, r, 1, 0, 0);
          g_ws.bc[14] = 4 * ring_r * (ring_r + 1);
          g_ws.bc[18] = pl;
          for (int i = 0; i < 26; i++)
          {
            g_ws.fix_dx[i] = g_fix_dx[i];
            g_ws.fix_dy[i] = g_fix_dy[i];
          }
        }
        __syncthreads();
        ring_n = g_ws.bc[14];
        plan = g_ws.bc[18];
        cur = level;
      }
#ifdef MVC_PHASE_PROF
      const int wbk = 31 - __clz(a.n_blk_x);
#endif
      const int step_x = a.bs_x - a.ov_x, step_y = a.bs_y - a.ov_y;
      const int h_scaled = a.hpad[0] >> a.log_scale, v_scaled = a.vpad[0] >> a.log_scale;
      pix *const s_y = g_wsrc, *const s_u = g_wsrc + a.bs_x * a.bs_y, *const s_v = s_u + a.cw * a.ch;
      const int nbx = a.n_blk_x;
      const bool coarser = level < c.nl - 1; // predictors interpolated from the coarser level (else zero)
      const LevelArgs &ca = c.lv[coarser ? level + 1 : level];
      const bool is_top = row == 0, is_bottom = row == a.n_blk_y - 1;
      const int lag = (a.smallest || is_bottom) ? 2 : 1;
      int seen = 0; // progress of the row above as last read (lane 0)
      int left_x = 0, left_y = 0, left_sad = 0;
      int row_bad = 0; // blocks of this row that took the wide search (same value in every thread)
#ifdef MVC_WAVE_TIMING
      if (lane == 0)
        atomicMin(c.tim + 2 * level, global_ns());
#endif
      const bool ip_shared = coarser && nbx <= WAVE_IP_MAX;
      if (ip_shared)
      {
        const int rows = is_bottom ? 1 : 2;
        for (int i = lane; i < rows * nbx; i += WAVE_NT)
        {
          const int below = i >= nbx ? 1 : 0, bx = i - below * nbx;
          const Vec v = interp_pred(a, ca.vec, ca.n_blk_x, ca.n_blk_y, 0, (row + below) * nbx + bx);
          g_ip[below][bx][0] = v.x;
          g_ip[below][bx][1] = v.y;
          g_ip[below][bx][2] = v.sad;
          if (!below)
            g_ipl[bx] = calc_nlambda(a.lambda_level, a.lsad, v.sad, is_top);
        }
        __syncthreads();
      }

      for (int blkx = 0; blkx < nbx; blkx++)
      {
        WPH_START();
        const int idx = row * nbx + blkx;
        int x0[3], y0[3];
        x0[0] = a.hpad[0] + blkx * step_x;
        y0[0] = a.vpad[0] + row * step_y;
        for (int p = 1; p < 3; p++)
        {
          x0[p] = a.hpad[p] + blkx * (step_x >> a.log_x_uv);
          y0[p] = a.vpad[p] + row * (step_y >> a.log_y_uv);
        }
        // source block into shared memory (asynchronous where the geometry allows it); completes while we wait
        if (a.async_src)
        {
          const int wl = (a.bs_x * a.bs_y) / SPW, wc = (a.cw * a.ch) / SPW;
          const int rl = a.bs_x / SPW, rc = a.cw / SPW;
          for (int i = lane; i < wl + 2 * wc; i += WAVE_NT)
          {
            if (i < wl)
            {
              const int r = i / rl, c = SPW * (i - r * rl);
              cp_async4(s_y + r * a.bs_x + c, a.src[0] + (size_t)(y0[0] + r) * a.ew[0] + x0[0] + c);
            }
            else
            {
              int j = i - wl;
              const int pl = j >= wc ? 2 : 1;
              if (j >= wc)
                j -= wc;
              const int r = j / rc, c = SPW * (j - r * rc);
              cp_async4((pl == 1 ? s_u : s_v) + r * a.cw + c, a.src[pl] + (size_t)(y0[pl] + r) * a.ew[pl] + x0[pl] + c);
            }
          }
          cp_async_commit();
        }
        else
        {
          for (int i = lane; i < a.bs_x * a.bs_y; i += WAVE_NT)
          {
            const int r = i / a.bs_x, c = i - r * a.bs_x;
            s_y[i] = a.src[0][(size_t)(y0[0] + r) * a.ew[0] + x0[0] + c];
          }
          for (int i = lane; i < a.cw * a.ch; i += WAVE_NT)
          {
            const int r = i / a.cw, c = i - r * a.cw;
            s_u[i] = a.src[1][(size_t)(y0[1] + r) * a.ew[1] + x0[1] + c];
            s_v[i] = a.src[2][(size_t)(y0[2] + r) * a.ew[2] + x0[2] + c];
          }
        }

        // neighbours: own and bottom-right are not searched yet (interpolated, or zero on the smallest level), the
        // up / up-right ones come from the row above once its counter has passed them
        const bool side = blkx < nbx - 1;
        int v[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 }; // own, up, p3
        int use_p3 = 0;
        int own_nlambda = 0;
        if (lane == 0)
        {
          // not the smallest level: the predictor is this block's own interpolated vector (clipping changes x / y only),
          // so nLambda does not depend on the search
          if (ip_shared)
          {
            v[0] = g_ip[0][blkx][0]; v[1] = g_ip[0][blkx][1]; v[2] = g_ip[0][blkx][2];
            own_nlambda = g_ipl[blkx];
          }
          else if (coarser)
          {
            const Vec o = interp_pred(a, ca.vec, ca.n_blk_x, ca.n_blk_y, 0, idx);
            v[0] = o.x; v[1] = o.y; v[2] = o.sad;
            own_nlambda = calc_nlambda(a.lambda_level, a.lsad, o.sad, is_top);
          }
          const bool p3_below = !is_bottom && !a.smallest && side;
          if (p3_below)
          {
            if (ip_shared)
            {
              v[6] = g_ip[1][blkx + 1][0]; v[7] = g_ip[1][blkx + 1][1]; v[8] = g_ip[1][blkx + 1][2];
            }
            else
            {
              const Vec q = interp_pred(a, ca.vec, ca.n_blk_x, ca.n_blk_y, 0, idx + nbx + 1);
              v[6] = q.x; v[7] = q.y; v[8] = q.sad;
            }
            use_p3 = 1;
          }
          WPH(0);
          if (!is_top)
          {
            const int need = blkx + lag < nbx ? blkx + lag : nbx;
            seen = wait_at_least(a.prog + row - 1, need, seen);
            WPH(1);
            const int *u = a.vec + 3 * (idx - nbx);
            v[3] = __ldcg(u); v[4] = __ldcg(u + 1); v[5] = __ldcg(u + 2);
            if (!p3_below && side)
            {
              const int *q = a.vec + 3 * (idx - nbx + 1);
              v[6] = __ldcg(q); v[7] = __ldcg(q + 1); v[8] = __ldcg(q + 2);
              use_p3 = 1;
            }
          }
        }
        if (lane == 0)
        {
          for (int i = 0; i < 9; i++)
            g_ws.bc[1 + i] = v[i];
          g_ws.bc[10] = use_p3;
          g_ws.bc[15] = own_nlambda;
        }
        __syncthreads();
        for (int i = 0; i < 9; i++)
          v[i] = g_ws.bc[1 + i];
        use_p3 = g_ws.bc[10];
        own_nlambda = g_ws.bc[15];

        const int dx_max = a.pel * (a.ext_w - x0[0] - a.bs_x - a.hpad[0] + h_scaled);
        const int dy_max = a.pel * (a.ext_h - y0[0] - a.bs_y - a.vpad[0] + v_scaled);
        const int dx_min = -a.pel * (x0[0] - a.hpad[0] + h_scaled);
        const int dy_min = -a.pel * (y0[0] - a.vpad[0] + v_scaled);

        // predictors (FetchPredictors), identical in every lane
        const Vec own_c = clip_mv(dx_min, dx_max, dy_min, dy_max, v[0], v[1], v[2]);
        const Vec zero_shifted = clip_mv(dx_min, dx_max, dy_min, dy_max, 0, 0, 0);
        Vec p1 = zero_shifted, p2 = zero_shifted, p3 = zero_shifted, p0;
        if (blkx > 0)
          p1 = clip_mv(dx_min, dx_max, dy_min, dy_max, left_x, left_y, left_sad);
        if (!is_top)
          p2 = clip_mv(dx_min, dx_max, dy_min, dy_max, v[3], v[4], v[5]);
        if (use_p3)
          p3 = clip_mv(dx_min, dx_max, dy_min, dy_max, v[6], v[7], v[8]);
        if (!is_top)
        {
          p0.x = median3(p1.x, p2.x, p3.x);
          p0.y = median3(p1.y, p2.y, p3.y);
          p0.sad = max(p1.sad, max(p2.sad, p3.sad));
        }
        else
          p0 = p1;
        const Vec pred = a.smallest ? p0 : own_c;
        int nlambda;
        if (!a.smallest)
          nlambda = own_nlambda;
        else
        {
          if (lane == 0)
            g_ws.bc[11] = calc_nlambda(a.lambda_level, a.lsad, pred.sad, is_top);
          __syncthreads();
          nlambda = g_ws.bc[11];
        }
        // the global estimate was written by another CTA of this launch (level_ready orders it before this row)
        const Vec glob = clip_mv(dx_min, dx_max, dy_min, dy_max, __ldcg(a.glob) * a.pel, __ldcg(a.glob + 1) * a.pel, 0);

        WPH(2);
        // phase A: zero, global, predictor, predictors[0..3], all evaluated without a range test as in the slice
        // kernel (the CPU uses the first three unconditionally; CheckMV0 below tests the other four)
        if (lane < 7)
        {
          const Vec cv = lane == 0 ? Vec{ 0, 0, 0 } : lane == 1 ? glob : lane == 2 ? pred : lane == 3 ? p0 : lane == 4 ? p1 : lane == 5 ? p2 : p3;
          g_ws.xs[lane] = cv.x;
          g_ws.ys[lane] = cv.y;
        }
        if (lane == 0)
        {
          g_ws.n = 7;
          g_ws.x0 = x0[0];
          g_ws.y0 = y0[0];
          g_ws.x1 = x0[1];
          g_ws.y1 = y0[1];
          g_ws.dx_min = dx_min;
          g_ws.dx_max = dx_max;
          g_ws.dy_min = dy_min;
          g_ws.dy_max = dy_max;
        }
        if (a.async_src)
          cp_async_wait_all();
        __syncthreads(); // source block, candidate list and geometry visible to every lane
        wave_eval_inl<GW, BSY, CHH>(lane, false);
        WPH(3);

        int best_x, best_y, best_sad, min_cost;
        {
          const int sad = g_ws.sl[0] + g_ws.sc[0];
          best_x = 0;
          best_y = 0;
          best_sad = sad;
          min_cost = sad + (int)(((long long)a.pzero * sad) >> 8);
          const int sadg = g_ws.sl[1] + g_ws.sc[1];
          const int costg = sadg + (int)(((long long)a.pglobal * sadg) >> 8);
          if (costg < min_cost)
          {
            best_x = glob.x;
            best_y = glob.y;
            best_sad = sadg;
            min_cost = costg;
          }
          const int sadp = g_ws.sl[2] + g_ws.sc[2];
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
            cost += g_ws.sl[3 + i];
            if (cost >= min_cost)
              continue;
            cost += g_ws.sc[3 + i];
            if (cost >= min_cost)
              continue;
            best_x = cx;
            best_y = cy;
            best_sad = g_ws.sl[3 + i] + g_ws.sc[3 + i];
            min_cost = cost;
          }
        }
        // no barrier here: the phase A fold only reads the SADs, and the next writes to them (the refine batch) follow
        // the barrier after that batch's candidate list
        WPH(4);

        if (!GEN || plan != WP_GEN)
        {
          // fixed batches (wave_plan), one SAD call site: [WP_UMH: cross + hex grid] [WP_HEX, WP_UMH: hexagon] rings
          // around the vector found so far (all rings of an exhaustive level in one batch: the hot path of every coarse
          // level; radius 1 after a hexagon)
          for (int stage = plan == WP_UMH ? 0 : plan == WP_HEX ? 1 : 2;; stage++)
          {
            const int n = stage == 0 ? 20 : stage == 1 ? 6 : ring_n;
            for (int c = lane; c < n; c += WAVE_NT)
            {
              g_ws.xs[c] = best_x + (stage == 2 ? g_ws.ring_dx[c] : g_ws.fix_dx[c + (stage == 1 ? 20 : 0)]);
              g_ws.ys[c] = best_y + (stage == 2 ? g_ws.ring_dy[c] : g_ws.fix_dy[c + (stage == 1 ? 20 : 0)]);
            }
            if (lane == 0)
            {
              g_ws.n = n;
              wave_fold_init();
            }
            __syncthreads();
            wave_eval_inl<GW, BSY, CHH, true>(lane, true, FoldArgs{ pred.x, pred.y, nlambda, min_cost });
            WPH(5);
            int dir = -2;
            wave_fold_take(stage == 1 ? mvc_refine::M_CHECKDIR : mvc_refine::M_CHECK, n, nullptr, dir, best_x, best_y,
              min_cost, best_sad, pred.x, pred.y, nlambda);
            WPH(6);
            if (stage == 2)
              break; // the end-of-block barrier comes before anything overwrites the candidates and the SADs
            __syncthreads(); // the next batch overwrites the candidate list and the SADs
            if (stage == 1 && dir != -2)
            {
              // CheckMVdir kept the cost and the SAD of the winner; Hex2Search moves its centre there
              best_x += g_ws.fix_dx[20 + dir];
              best_y += g_ws.fix_dy[20 + dir];
            }
          }
        }
        else if constexpr (GEN)
        {
          WaveChecker<GW, BSY, CHH> ck;
          ck.bx = best_x;
          ck.by = best_y;
          ck.min_cost = min_cost;
          ck.best_sad = best_sad;
          ck.pred_x = pred.x;
          ck.pred_y = pred.y;
          ck.nlambda = nlambda;
          ck.lane = lane;
          mvc_refine::refine(ck, a.search_type, a.radius);
          best_x = ck.bx;
          best_y = ck.by;
          best_sad = ck.best_sad;
          min_cost = ck.min_cost;
          WPH(5);
        }
        // Bad vector (PseudoEPZSearch): the threshold grows with badcount, the number of earlier bad blocks in raster order
        // (one slice). Only a block above badSAD can be bad; such a block waits until the rows above are finished, so
        // their bad counts are known, and then decides exactly as the CPU does.
        if (idx > 1 && best_sad > a.bad_sad)
        {
          if (lane == 0)
          {
            int cnt = row_bad;
            if (row > 0)
            {
              wait_at_least(a.prog + row - 1, nbx, 0); // rows finish in order: then every row above is complete
              for (int r = 0; r < row; r++)
                cnt += __ldcg(a.rowbad + r);
            }
            g_ws.bc[12] = cnt;
          }
          __syncthreads();
          const int badcount = g_ws.bc[12];
          if (best_sad > a.bad_sad + a.bad_sad * badcount / 16) // BADCOUNT_LIMIT
          {
            row_bad++;
            wave_bad_search<GW>(lane, best_x, best_y, min_cost, best_sad, pred.x, pred.y, nlambda);
          }
        }
        left_x = best_x;
        left_y = best_y;
        left_sad = best_sad;
        WPH(7);
#ifdef MVC_PHASE_PROF
        if (lane == 0)
          atomicAdd(&g_wprof[wbk][8], 1ull);
#endif
#ifndef MVC_WAVE_STORE_LANE
#define MVC_WAVE_STORE_LANE 32
#endif
        if (MVC_WAVE_STORE_LANE != 0)
          __syncthreads(); // the next block overwrites the shared source block
        // warp 1 publishes the vector while warp 0 starts on the next block (its predictors and the wait for the row
        // above): the release fence is off this row's own chain; the next block's left predictor is in the registers
        if (lane == MVC_WAVE_STORE_LANE)
        {
          if (blkx == nbx - 1)
            a.rowbad[row] = row_bad; // published by the release below (the row's last one)
          int *o = a.vec + 3 * idx;
          o[0] = best_x;
          o[1] = best_y;
          o[2] = best_sad;
          st_release(a.prog + row, blkx + 1); // orders the vector stores above before the counter
          if (blkx == nbx - 1)
            __threadfence(); // the row's vectors before the level count below (made by lane 0 after the barrier)
        }
        if (MVC_WAVE_STORE_LANE == 0)
          __syncthreads(); // the next block overwrites the shared source block
      }
      __syncthreads(); // warp 1's last stores of this row are fenced

      // the CTA that finishes the last row of a level estimates the global MV of the next finer level from it
      if (lane == 0)
      {
#ifdef MVC_WAVE_TIMING
        atomicMax(c.tim + 2 * level + 1, global_ns());
#endif
        __threadfence(); // this row's vectors before the count
        g_ws.bc[16] = atomicAdd(c.level_done + level, 1) == a.n_blk_y - 1 ? 1 : 0;
        __threadfence(); // every other row's vectors (fenced before their counts) after it
      }
      __syncthreads();
      if (g_ws.bc[16] && level > 0)
      {
        glob_estimate(a.vec, a.n_blk_x * a.n_blk_y, c.freq, const_cast<int *>(c.lv[level - 1].glob), c.use_global, lane, WAVE_NT);
        if (lane == 0)
        {
          __threadfence();
          st_release(c.level_ready + level - 1, 1);
        }
      }
    }
  }

  // Picks the k_wave instance: the ring-only kernels (no mvc_refine) with compile-time row counts for the square 16 and
  // 32 blocks (the QTGMC sizes); the general kernels (mvc_refine inlined at about 20 call sites) only with runtime row
  // loops, which keeps their compile time bounded. gen: some level needs mvc_refine. One CTA per row of the finest level
  // (every level has at most as many rows); CTAs take rows by ticket until all levels are done.
  template <int GW>
  void launch_wave(const WaveCall &c, bool gen, cudaStream_t st)
  {
    const LevelArgs &a = c.lv[0];
    const int grid = a.n_blk_y; // fewer CTAs (40-70% of the rows) were measured slower, section 24
#ifdef MVC_DEV_FASTONLY
    // development builds (-DMVC_DEV_FASTONLY): no general kernels, the host refuses such levels before launching
    if (gen)
      return;
    if constexpr (GW < 16)
      k_wave<GW, 0, 0, false><<<grid, WAVE_NT, 0, st>>>(c);
#else
    if constexpr (GW < 16)
    {
      if (gen)
        k_wave<GW, 0, 0, true><<<grid, WAVE_NT, 0, st>>>(c);
      else
        k_wave<GW, 0, 0, false><<<grid, WAVE_NT, 0, st>>>(c);
    }
    else if (gen)
      k_wave<GW, 0, 0, true><<<grid, WAVE_NT, 0, st>>>(c);
#endif
    else if (a.bs_y == GW && a.ch == 0)
      k_wave<GW, GW, 0, false><<<grid, WAVE_NT, 0, st>>>(c);
    else if (a.bs_y == GW && a.ch == GW / 2)
      k_wave<GW, GW, GW / 2, false><<<grid, WAVE_NT, 0, st>>>(c);
    else if (a.bs_y == GW && a.ch == GW)
      k_wave<GW, GW, GW, false><<<grid, WAVE_NT, 0, st>>>(c);
    else
      k_wave<GW, 0, 0, false><<<grid, WAVE_NT, 0, st>>>(c);
  }

  int ilog2(int v)
  {
    int l = 0;
    while ((1 << l) < v)
      l++;
    return l;
  }
}

// Payload + wave-state buffers whose state part is zero (k_wave clears it at the end of every completed call), by size.
// fresh: a new buffer, its state still to be cleared once.
namespace
{
  std::mutex g_wave_mutex;
  std::multimap<size_t, void *> *g_wave_free = new std::multimap<size_t, void *>; // never destroyed (static destruction order)

  int *wave_buf_get(size_t bytes, bool &fresh)
  {
    {
      std::lock_guard<std::mutex> lk(g_wave_mutex);
      auto it = g_wave_free->find(bytes);
      if (it != g_wave_free->end())
      {
        void *p = it->second;
        g_wave_free->erase(it);
        fresh = false;
        return (int *)p;
      }
    }
    fresh = true;
    return (int *)mvc::dev_alloc(bytes);
  }

  void wave_buf_put(int *p, size_t bytes)
  {
    std::lock_guard<std::mutex> lk(g_wave_mutex);
    g_wave_free->emplace(bytes, p);
  }
}

// src_device: src / ref are the device planes of two device frames (tokens, see mvcuda_analyse_token); otherwise host
// planes that the super builder uploads
static int analyse_impl(const MvCudaSuperParams *sp, const MvCudaAnalyseParams *ap, const void *const src[3],
  const int src_pitch_bytes[3], const void *const ref[3], const int ref_pitch_bytes[3], int *payload, int payload_words, bool src_device,
  unsigned long long dev_seq = 0)
{
  if (!sp || !ap || !src || !ref || !payload)
    return -4;
  // supported: luma only or luma + chroma, pel 1 or 2, field shift 0, every search type of mvcuda_refine.cuh
  // (the coarse levels are exhaustive, as on the CPU, except for the horizontal / vertical types)
  const int user_type = ap->search_type ? ap->search_type : mvc_refine::S_EXHAUSTIVE;
  if (sp->levels != ap->n_levels || ap->n_levels < 1 || ap->n_levels > MVCUDA_MAX_LEVELS
    || ap->search_param < 1 || ap->pel_search < 1 || ap->n_blk_x < 1 || !mvc_refine::supported_type(user_type)
    || ap->n_blk_y < 1 || ap->slice_threads < 1 || ap->plevel < 0 || ap->plevel > 2 || ap->blk_x > 32)
    return -4;
  const bool horv = user_type == mvc_refine::S_HORIZONTAL || user_type == mvc_refine::S_VERTICAL;
  int lvl_type[MVCUDA_MAX_LEVELS];
  for (int l = 0; l < ap->n_levels; l++)
  {
    lvl_type[l] = (l == 0 || horv) ? user_type : mvc_refine::S_EXHAUSTIVE;
    if ((l == 0 ? ap->pel_search : ap->search_param) > mvc_refine::max_param(lvl_type[l]))
      return -4;
  }
  if (!mvc::use_device())
    return -1;

  // the asynchronous source-block copy moves 4-byte words: everything it touches must be a multiple of SPW samples;
  // otherwise the kernel copies the block with plain loads
  bool async_geo = true;
  {
    const int step_x = ap->blk_x - ap->overlap_x;
    async_geo = ap->blk_x % SPW == 0 && step_x % SPW == 0 && sp->hpad % SPW == 0 && ((size_t)ap->blk_x * ap->blk_y) % SPW == 0;
    if (sp->chroma)
    {
      const int lx = ilog2(sp->x_ratio_uv);
      async_geo = async_geo && (ap->blk_x >> lx) % SPW == 0 && (step_x >> lx) % SPW == 0 && (sp->hpad / sp->x_ratio_uv) % SPW == 0
        && ((size_t)(ap->blk_x >> lx) * (ap->blk_y >> ilog2(sp->y_ratio_uv))) % SPW == 0;
    }
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
  // meander=false: wavefront kernel = CPU result with meander=false, mt=false (slice_threads is ignored); meander=true:
  // slice kernel = CPU result with slice_threads threads. MVCUDA_WAVE=0 forces the slice kernel (comparisons only).
  static const bool wave_allowed = [] { const char *e = getenv("MVCUDA_WAVE"); return !(e && e[0] == '0'); }();
  const bool wave = !ap->meander && wave_allowed;
  int row_off[MVCUDA_MAX_LEVELS], rows_total = 0;
  for (int l = 0; l < nl; l++)
  {
    row_off[l] = rows_total;
    rows_total += nby[l];
  }
  int wave_gw = 4;
  while (wave_gw < ap->blk_x)
    wave_gw *= 2;
  if (wave)
  {
    // the source block must fit the static shared buffer of k_wave (always true for blk_x <= 32 and blk_y <= 32)
    const int cw = sp->chroma ? ap->blk_x / sp->x_ratio_uv : 0, ch = sp->chroma ? ap->blk_y / sp->y_ratio_uv : 0;
    if (ap->blk_x * ap->blk_y + 2 * cw * ch > WAVE_SRC_MAX)
      return -4;
#ifdef MVC_DEV_FASTONLY
    for (int l = 0; l < nl; l++)
      if (wave_plan(lvl_type[l], l == 0 ? ap->pel_search : ap->search_param) == WP_GEN)
        return -4;
#endif
  }

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
  int *d_pay = nullptr, *d_freq = nullptr, *d_glob = nullptr, *d_bad = nullptr, *d_wave = nullptr;
  const size_t pay_bytes = (size_t)words * sizeof(int), freq_bytes = 2 * 2 * FREQ_HALF * sizeof(int);
  // wavefront, all cleared per call: [row progress, all levels][bad blocks per row, all levels][ticket][finished rows per
  // level][global estimate ready per level][global estimate (x, y) per level][bad flag]
  const size_t w_rowbad = rows_total, w_ticket = 2 * (size_t)rows_total, w_done = w_ticket + 1, w_ready = w_done + MVCUDA_MAX_LEVELS;
  const size_t w_glob = w_ready + MVCUDA_MAX_LEVELS, w_bad = w_glob + 2 * MVCUDA_MAX_LEVELS, w_fin = w_bad + 1, wave_words = w_fin + 1;
  const size_t wave_bytes = wave ? wave_words * sizeof(int) : 0;
  // wavefront: the wave state follows the payload in one buffer, so one copy brings back the vectors and the bad flag
  const size_t pay_alloc = pay_bytes + wave_bytes;
  int *h_stage = nullptr; // pinned copy of that buffer (a copy to pageable memory blocks inside the driver)
  std::vector<int> host;
  WaveCall wc = {};

  int bad = 0;
  bool self_finish = false, fresh = false;
  // Everything up to the copy of the result, queued without a host wait: the super frames (cache), the buffers, the
  // kernels, the copy. 0 or the error.
  auto issue = [&]() -> int {
#define AN_CHECK(call) \
  do { \
    cudaError_t e_ = (call); \
    if (e_ != cudaSuccess) { \
      fprintf(stderr, "mvcuda: %s failed: %s\n", #call, cudaGetErrorString(e_)); \
      return -2; \
    } \
  } while (0)
  {
    const mvc::SuperBuildFn build = src_device ? &MVC_NS::super_build_dev : &MVC_NS::super_build;
    int r = mvc::super_get(build, *sp, ap->src_key, src, src_pitch_bytes, st, s_src);
    if (!r)
      r = mvc::super_get(build, *sp, ap->ref_key, ref, ref_pitch_bytes, st, s_ref);
    if (r)
      return r;
  }
  lap("2 x super (cached)");

  d_freq = (int *)mvc::dev_alloc(freq_bytes);
  if (wave)
  {
    // one launch for the whole call (WaveCall): the kernel writes the payload header and every vector, and with the
    // pinned buffer also hands them over and clears its state (self_finish)
    h_stage = (int *)mvc::pin_alloc(pay_alloc);
    self_finish = h_stage != nullptr;
    d_pay = self_finish ? wave_buf_get(pay_alloc, fresh) : (int *)mvc::dev_alloc(pay_alloc);
    d_wave = d_pay ? d_pay + words : nullptr;
    if (d_wave)
    {
      d_glob = d_wave + w_glob;
      d_bad = d_wave + w_bad;
    }
  }
  else
  {
    d_pay = (int *)mvc::dev_alloc(pay_alloc);
    d_glob = (int *)mvc::dev_alloc(2 * sizeof(int));
    d_bad = (int *)mvc::dev_alloc(sizeof(int));
  }
  if (!d_pay || !d_freq || !d_glob || !d_bad || (wave && !d_wave))
    return -2;
  {
    if (wave)
    {
      if (!self_finish || fresh) // a self-finishing call leaves the state cleared for the next one
        AN_CHECK(cudaMemsetAsync(d_wave, 0, wave_bytes, st));
    }
    else
    {
      host.assign(words, 0);
      host[0] = words;
      host[1] = 1;
      for (int l = nl - 1; l >= 0; l--)
        host[offs[l] - 1] = nbx[l] * nby[l] * 3 + 1;
      AN_CHECK(cudaMemcpyAsync(d_pay, host.data(), pay_bytes, cudaMemcpyHostToDevice, st));
      AN_CHECK(cudaMemsetAsync(d_bad, 0, sizeof(int), st));
      AN_CHECK(cudaMemsetAsync(d_glob, 0, 2 * sizeof(int), st));
    }
    bool wave_gen = false;

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
      a.has_chroma = sp->chroma ? 1 : 0;
      a.search_type = lvl_type[l];
      a.log_x_uv = a.has_chroma ? ilog2(sp->x_ratio_uv) : 0;
      a.log_y_uv = a.has_chroma ? ilog2(sp->y_ratio_uv) : 0;
      a.cw = a.has_chroma ? a.bs_x >> a.log_x_uv : 0;
      a.ch = a.has_chroma ? a.bs_y >> a.log_y_uv : 0;
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
      a.async_src = (async_geo && a.ew[0] % SPW == 0 && a.ew[1] % SPW == 0 && a.ew[2] % SPW == 0) ? 1 : 0;
      {
        // packed SADs: the block fills the lane group (bs_x == GW), chroma absent or paired with cw == GW / 2, rows in
        // groups of 4, every reference row 4-byte (8 bit) / 8-byte (16 bit) aligned (word reads around each row)
        static const bool pack_allowed = [] { const char *e = getenv("MVCUDA_WAVE_PACK"); return !(e && e[0] == '0'); }();
        bool pk = pack_allowed && wave && wave_gw >= 16 && ap->blk_x == wave_gw && a.bs_y % 4 == 0
          && (!a.has_chroma || (2 * a.cw == wave_gw && a.ch % 4 == 0));
        for (int p = 0; p < (a.has_chroma ? 3 : 1) && pk; p++)
        {
          pk = a.ew[p] % 4 == 0;
          for (int i = 0; i < 4; i++)
            pk = pk && ((size_t)a.ref[p][i] & (4 * sizeof(pix) - 1)) == 0;
        }
        a.packed = pk ? 1 : 0;
        // the slice kernel's packed form (eval_candidate_pk picks the block geometries it has): alignment only
        static const bool spack_allowed = [] { const char *e = getenv("MVCUDA_SEARCH_PACK"); return !(e && e[0] == '0'); }();
        bool pks = spack_allowed;
        for (int p = 0; p < (a.has_chroma ? 3 : 1) && pks; p++)
        {
          pks = a.ew[p] % 4 == 0;
          for (int i = 0; i < 4; i++)
            pks = pks && ((size_t)a.ref[p][i] & (4 * sizeof(pix) - 1)) == 0;
        }
        a.packed_s = pks ? 1 : 0;
      }
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
      a.glob = wave ? d_glob + 2 * l : d_glob;
      a.vec = d_pay + offs[l];
      a.bad_flag = d_bad;
      int slices = ap->slice_threads;
      if (slices > 64)
        slices = 64;
      if (slices > a.n_blk_y / 4)
        slices = a.n_blk_y / 4;
      a.n_slices = slices < 1 ? 1 : slices;
      if (wave)
      {
        a.prog = d_wave + row_off[l];
        a.rowbad = d_wave + w_rowbad + row_off[l];
        a.bad_range = ap->bad_range;
        wave_gen = wave_gen || wave_plan(a.search_type, a.radius) == WP_GEN;
        wc.lv[l] = a;
        continue; // launched once for all levels below
      }

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
        k_search<<<a.n_slices, SEARCH_THREADS, shm, st>>>(a);
      }
      AN_CHECK(cudaGetLastError());
    }
    if (wave)
    {
      wc.nl = nl;
      wc.use_global = ap->use_global;
      wc.ticket = d_wave + w_ticket;
      wc.level_done = d_wave + w_done;
      wc.level_ready = d_wave + w_ready;
      wc.freq = d_freq;
      wc.payload = d_pay;
      wc.words = words;
      if (self_finish)
      {
        wc.host_out = h_stage;
        wc.state = d_wave;
        wc.finished = (unsigned *)(d_wave + w_fin);
        wc.state_words = (int)wave_words;
        wc.bad_word = (int)w_bad;
      }
#ifdef MVC_WAVE_TIMING
      unsigned long long *d_tim = nullptr;
      {
        unsigned long long h_tim[2 * MVCUDA_MAX_LEVELS];
        for (int l = 0; l < MVCUDA_MAX_LEVELS; l++)
        {
          h_tim[2 * l] = ~0ull;
          h_tim[2 * l + 1] = 0;
        }
        cudaMalloc(&d_tim, sizeof(h_tim));
        cudaMemcpy(d_tim, h_tim, sizeof(h_tim), cudaMemcpyHostToDevice);
        wc.tim = d_tim;
      }
#endif
      switch (wave_gw)
      {
      case 4: launch_wave<4>(wc, wave_gen, st); break;
      case 8: launch_wave<8>(wc, wave_gen, st); break;
      case 16: launch_wave<16>(wc, wave_gen, st); break;
      default: launch_wave<32>(wc, wave_gen, st); break;
      }
      AN_CHECK(cudaGetLastError());
#ifdef MVC_WAVE_TIMING
      {
        // per level: duration (first row start to last row end) and the gap before it, averaged, every 200 calls
        unsigned long long h_tim[2 * MVCUDA_MAX_LEVELS];
        cudaStreamSynchronize(st);
        cudaMemcpy(h_tim, d_tim, sizeof(h_tim), cudaMemcpyDeviceToHost);
        cudaFree(d_tim);
        static std::mutex m;
        static double dur[MVCUDA_MAX_LEVELS], gap[MVCUDA_MAX_LEVELS], total;
        static long calls;
        std::lock_guard<std::mutex> lk(m);
        for (int l = 0; l < nl; l++)
        {
          dur[l] += (double)(h_tim[2 * l + 1] - h_tim[2 * l]) / 1000.0;
          if (l < nl - 1)
            gap[l] += ((double)h_tim[2 * l] - (double)h_tim[2 * (l + 1) + 1]) / 1000.0;
        }
        total += (double)(h_tim[1] - h_tim[2 * (nl - 1)]) / 1000.0;
        if (++calls % 200 == 0)
        {
          fprintf(stderr, "wave timing, %ld calls, us per call: total %.1f |", calls, total / calls);
          for (int l = nl - 1; l >= 0; l--)
            fprintf(stderr, " L%d %.1f (gap %.1f)", l, dur[l] / calls, l < nl - 1 ? gap[l] / calls : 0.0);
          fprintf(stderr, "\n");
        }
      }
#endif
    }
    if (self_finish)
      ; // k_wave hands the result over itself
    else if (h_stage)
      AN_CHECK(cudaMemcpyAsync(h_stage, d_pay, pay_alloc, cudaMemcpyDeviceToHost, st));
    else
    {
      AN_CHECK(cudaMemcpyAsync(payload, d_pay, pay_bytes, cudaMemcpyDeviceToHost, st));
      AN_CHECK(cudaMemcpyAsync(&bad, d_bad, sizeof(int), cudaMemcpyDeviceToHost, st));
    }
  }
  return 0;
#undef AN_CHECK
  };

  if (src_device && wave)
  {
    // The device frames are results of pipeline ops (up to dev_seq). The calls are issued by the pipeline submitter (one
    // thread for all launches, mvcuda_internal.h), this stream ordered after the op that made the frames.
    mvc::mark_request(dev_seq);
    mvc::pipe_call([&](cudaStream_t ps) {
      cudaEvent_t e = mvc::mark_in_op(dev_seq, ps);
      rc = (e && cudaStreamWaitEvent(st, e, 0) != cudaSuccess) ? -2 : issue();
    });
  }
  else
  {
    if (src_device && !mvc::pipeline_order(st, dev_seq))
      return -2;
    rc = issue();
  }
  if (rc)
    goto done;
  if (cudaStreamSynchronize(st) != cudaSuccess)
  {
    fprintf(stderr, "mvcuda: analysis failed: %s\n", cudaGetErrorString(cudaGetLastError()));
    rc = -2;
    goto done;
  }
  if (h_stage)
  {
    memcpy(payload, h_stage, pay_bytes);
    bad = h_stage[words + w_bad];
  }
  lap("analysis kernels");
  {
#ifdef MVC_PHASE_PROF
    {
      unsigned long long h[16][8];
      cudaMemcpyFromSymbol(h, g_prof, sizeof(h));
      for (int k = 0; k < 16; k++)
        if (h[k][5])
          std::fprintf(stderr, "phase cycles/block, blocks-per-row 2^%d: wait+pred %.0f  A %.0f  foldA %.0f  B %.0f  foldB %.0f  (n=%llu)\n", k,
            (double)h[k][0] / h[k][5], (double)h[k][1] / h[k][5], (double)h[k][2] / h[k][5], (double)h[k][3] / h[k][5], (double)h[k][4] / h[k][5], h[k][5]);
      unsigned long long hw[16][12];
      cudaMemcpyFromSymbol(hw, g_wprof, sizeof(hw));
      for (int k = 0; k < 16; k++)
        if (hw[k][8])
        {
          const double n = (double)hw[k][8];
          std::fprintf(stderr, "wave cycles/block, blocks-per-row 2^%d: issue %.0f  wait %.0f  pred %.0f  A-eval %.0f  A-fold %.0f  R-eval %.0f  R-fold %.0f  store %.0f  (n=%llu)\n",
            k, hw[k][0] / n, hw[k][1] / n, hw[k][2] / n, hw[k][3] / n, hw[k][4] / n, hw[k][5] / n, hw[k][6] / n, hw[k][7] / n, hw[k][8]);
        }
    }
#endif
    if (bad)
      rc = MVCUDA_NEEDS_CPU;
  }

done:
  if (rc)
    cudaStreamSynchronize(st); // queued work may still use the buffers (and the super frames) released below
  // a buffer of a completed self-finishing call has its state cleared: back to the wave pool (else the general pool)
  if (self_finish && d_pay && (rc == 0 || rc == MVCUDA_NEEDS_CPU))
    wave_buf_put(d_pay, pay_alloc);
  else
    mvc::dev_free(d_pay, pay_alloc);
  mvc::dev_free(d_freq, freq_bytes);
  mvc::pin_free(h_stage, pay_alloc);
  if (!wave)
  {
    mvc::dev_free(d_glob, 2 * sizeof(int));
    mvc::dev_free(d_bad, sizeof(int));
  }
  return rc;
}

extern "C" int MVC_API(mvcuda_analyse_host)(const MvCudaSuperParams *sp, const MvCudaAnalyseParams *ap, const void *const src[3],
  const int src_pitch_bytes[3], const void *const ref[3], const int ref_pitch_bytes[3], int *payload, int payload_words)
{
  return analyse_impl(sp, ap, src, src_pitch_bytes, ref, ref_pitch_bytes, payload, payload_words, false);
}

// The same search with the source and reference pictures given as device frames (tokens of the cudafilters pipeline):
// the super frames are built from device memory, no upload. MVCUDA_TOKEN_GONE when a token was evicted.
extern "C" int MVC_API(mvcuda_analyse_token)(const MvCudaSuperParams *sp, const MvCudaAnalyseParams *ap, unsigned long long src_token,
  unsigned long long ref_token, int *payload, int payload_words)
{
  if (!sp || !ap || !payload)
    return -4;
  if (!mvc::use_device())
    return -1;
  auto fs = mvc::frame_get(src_token), fr = mvc::frame_get(ref_token);
  if (!fs || !fr)
    return MVCUDA_TOKEN_GONE;
  // the frames must be the pictures the super parameters describe
  auto fits = [&](const mvc::DevFrame &f) {
    return f.f.width == sp->width && f.f.height == sp->height && f.f.bits_per_sample == sp->bits_per_pixel
      && f.f.bytes_per_sample == (int)sizeof(mvc_pix)
      && (!sp->chroma || (f.f.planes == 3 && f.f.x_ratio_uv == sp->x_ratio_uv && f.f.y_ratio_uv == sp->y_ratio_uv));
  };
  if (!fits(*fs) || !fits(*fr))
    return -4;
  const void *src[3] = {}, *ref[3] = {};
  int src_pitch[3] = {}, ref_pitch[3] = {};
  for (int p = 0; p < fs->f.planes; p++)
  {
    src[p] = fs->plane[p];
    src_pitch[p] = (int)fs->pitch[p];
    ref[p] = fr->plane[p];
    ref_pitch[p] = (int)fr->pitch[p];
  }
  return analyse_impl(sp, ap, src, src_pitch, ref, ref_pitch, payload, payload_words, true, fs->seq > fr->seq ? fs->seq : fr->seq);
}
