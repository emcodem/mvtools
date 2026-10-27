// GPU version of MRecalculate (GPU_PLAN.md M4), bit-exact against PlaneOfBlocks::recalculate_mv_slice for the
// production configuration: exhaustive search (search=3), smooth 0/1, 16-bit samples, luma + chroma SAD.
//
// Every block is independent (it starts from the interpolated vector of the previous analysis and never reads
// neighbour results of the current pass), so one warp handles one block. The CPU code evaluates candidates
// with early exits (cost so far >= best cost => skip); all partial costs are non-negative, so "full cost < best cost"
// selects exactly the same candidate, which is what is done here, with the same early exits only to save work.

#include "mvcuda_internal.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace
{
  using mvc::pix;

  struct KArgs
  {
    const pix *ref[3][4];
    const pix *src[3];
    int ew[3];
    int hpad[3], vpad[3];
    int ext_w0, ext_h0;       // luma extended size (search bounds)
    int pel, log_pel;
    int bits;
    int log_x_uv, log_y_uv;
    int blk_x, blk_y, ov_x, ov_y, n_blk_x, n_blk_y;
    int old_n_x, old_n_y, old_size_x, old_size_y, old_ov_x, old_ov_y, log_pel_old;
    int lambda_level, pnew, th_sad, smooth, radius, chroma_scale, n_slices;
    const int *old_vec; // 3 ints per block
    int *out;           // 3 ints per block
  };

  __device__ __forceinline__ int warp_sum(int v)
  {
    for (int o = 16; o > 0; o >>= 1)
      v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
  }

  // sum of absolute differences of a w x h block of src (plane p, top-left x0,y0) against the reference block displaced
  // by the sub-pel position (xp, yp) in units of 1/pel of a sample (absolute, padding included)
  __device__ int block_sad(const KArgs &a, int p, int x0, int y0, int w, int h, int xp, int yp)
  {
    const int sub = (xp & (a.pel - 1)) | ((yp & (a.pel - 1)) << a.log_pel);
    const pix *rp = a.ref[p][sub];
    const int rx = xp >> a.log_pel, ry = yp >> a.log_pel;
    const int ew = a.ew[p];
    int s = 0;
    for (int i = threadIdx.x & 31; i < w * h; i += 32)
    {
      const int r = i / w, c = i - r * w;
      s += abs((int)a.src[p][(size_t)(y0 + r) * ew + x0 + c] - (int)rp[(size_t)(ry + r) * ew + rx + c]);
    }
    return warp_sum(s);
  }

  __device__ __forceinline__ int scale_chroma(int sad, int scale)
  {
    if (scale == 0)
      return sad;
    return scale > 0 ? sad >> scale : sad << (-scale);
  }

  __global__ void k_recalc(KArgs a)
  {
    const int idx = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (idx >= a.n_blk_x * a.n_blk_y)
      return;
    const int lane = threadIdx.x & 31;
    const int blkx = idx % a.n_blk_x, blky = idx / a.n_blk_x;

    const int step_x = a.blk_x - a.ov_x, step_y = a.blk_y - a.ov_y;
    int x0[3], y0[3];
    x0[0] = a.hpad[0] + blkx * step_x;
    y0[0] = a.vpad[0] + blky * step_y;
    for (int p = 1; p < 3; p++)
    {
      x0[p] = a.hpad[p] + blkx * (step_x >> a.log_x_uv);
      y0[p] = a.vpad[p] + blky * (step_y >> a.log_y_uv);
    }
    const int cw = a.blk_x >> a.log_x_uv, ch = a.blk_y >> a.log_y_uv;

    // search boundaries (recalculate_mv_slice)
    const int dx_max = a.pel * (a.ext_w0 - x0[0] - a.blk_x);
    const int dy_max = a.pel * (a.ext_h0 - y0[0] - a.blk_y);
    const int dx_min = -a.pel * x0[0];
    const int dy_min = -a.pel * y0[0];

    // interpolated / nearest old vector
    const int step_x_old = a.old_size_x - a.old_ov_x, step_y_old = a.old_size_y - a.old_ov_y;
    const int center_x = a.blk_x / 2 + step_x * blkx;
    const int blkxold = (center_x - a.old_size_x / 2) / step_x_old;
    const int center_y = a.blk_y / 2 + step_y * blky;
    const int blkyold = (center_y - a.old_size_y / 2) / step_y_old;
    const int delta_x = max(0, center_x - (a.old_size_x / 2 + step_x_old * blkxold));
    const int delta_y = max(0, center_y - (a.old_size_y / 2 + step_y_old * blkyold));
    const int bx1 = min(a.old_n_x - 1, max(0, blkxold)), bx2 = min(a.old_n_x - 1, max(0, blkxold + 1));
    const int by1 = min(a.old_n_y - 1, max(0, blkyold)), by2 = min(a.old_n_y - 1, max(0, blkyold + 1));

    int vx, vy, vsad;
    if (a.smooth == 1)
    {
      const int *v1 = a.old_vec + 3 * (bx1 + by1 * a.old_n_x), *v2 = a.old_vec + 3 * (bx2 + by1 * a.old_n_x);
      const int *v3 = a.old_vec + 3 * (bx1 + by2 * a.old_n_x), *v4 = a.old_vec + 3 * (bx2 + by2 * a.old_n_x);
      const int v1x = v1[0] * step_x_old + delta_x * (v2[0] - v1[0]);
      const int v1y = v1[1] * step_x_old + delta_x * (v2[1] - v1[1]);
      const long long v1s = (long long)v1[2] * step_x_old + delta_x * ((long long)v2[2] - v1[2]);
      const int v2x = v3[0] * step_x_old + delta_x * (v4[0] - v3[0]);
      const int v2y = v3[1] * step_x_old + delta_x * (v4[1] - v3[1]);
      const long long v2s = (long long)v3[2] * step_x_old + delta_x * ((long long)v4[2] - v3[2]);
      vx = (v1x + delta_y * (v2x - v1x) / step_y_old) / step_x_old;
      vy = (v1y + delta_y * (v2y - v1y) / step_y_old) / step_x_old;
      vsad = (int)((v1s + delta_y * (v2s - v1s) / step_y_old) / step_x_old);
    }
    else
    {
      const int *v;
      if (delta_x * 2 < step_x_old && delta_y * 2 < step_y_old)
        v = a.old_vec + 3 * (bx1 + by1 * a.old_n_x);
      else if (delta_x * 2 >= step_x_old && delta_y * 2 < step_y_old)
        v = a.old_vec + 3 * (bx2 + by1 * a.old_n_x);
      else if (delta_x * 2 < step_x_old && delta_y * 2 >= step_y_old)
        v = a.old_vec + 3 * (bx1 + by2 * a.old_n_x);
      else
        v = a.old_vec + 3 * (bx2 + by2 * a.old_n_x);
      vx = v[0];
      vy = v[1];
      vsad = v[2];
    }
    // to the new pel, clip to the search area
    vx = (vx << a.log_pel) >> a.log_pel_old;
    vy = (vy << a.log_pel) >> a.log_pel_old;
    const int pred_x = vx < dx_min ? dx_min : (vx >= dx_max ? dx_max - 1 : vx);
    const int pred_y = vy < dy_min ? dy_min : (vy >= dy_max ? dy_max - 1 : vy);
    (void)vsad; // the normalised old SAD is overwritten by the real SAD below

    // nLambda is 0 on the first block row of every slice (the CPU cuts the plane into min(threads, 64, rows / 4)
    // slices; with one slice that is row 0 only)
    bool first_row = false;
    for (int t = 0; t < a.n_slices; t++)
      first_row |= (t * a.n_blk_y / a.n_slices == blky);
    const int nlambda = first_row ? 0 : a.lambda_level;

    // SAD (luma, and scaled chroma) of a candidate vector
    auto luma_sad = [&](int cx, int cy) {
      return block_sad(a, 0, x0[0], y0[0], a.blk_x, a.blk_y, x0[0] * a.pel + cx, y0[0] * a.pel + cy);
    };
    auto chroma_sad = [&](int cx, int cy) {
      int s = 0;
      for (int p = 1; p < 3; p++)
        s += block_sad(a, p, x0[p], y0[p], cw, ch, x0[p] * a.pel + (cx >> a.log_x_uv), y0[p] * a.pel + (cy >> a.log_y_uv));
      return scale_chroma(s, a.chroma_scale);
    };

    int best_x = pred_x, best_y = pred_y;
    int best_sad = luma_sad(pred_x, pred_y) + chroma_sad(pred_x, pred_y);
    int min_cost = best_sad;

    if (best_sad > a.th_sad)
    {
      const int mvx = best_x, mvy = best_y;
      auto check = [&](int cx, int cy) {
        if (!(cx >= dx_min && cy >= dy_min && cx < dx_max && cy < dy_max))
          return;
        const int dxp = pred_x - cx, dyp = pred_y - cy;
        const int dist = (int)((unsigned)(dxp * dxp) + (unsigned)(dyp * dyp));
        int cost = (int)((unsigned)nlambda * (unsigned)dist) >> (16 - a.bits);
        if (cost >= min_cost)
          return;
        const int sl = luma_sad(cx, cy);
        cost += sl + (int)(((long long)a.pnew * sl) >> 8);
        if (cost >= min_cost)
          return;
        const int sc = chroma_sad(cx, cy);
        cost += sc + (int)(((long long)a.pnew * sc) >> 8);
        if (cost >= min_cost)
          return;
        best_x = cx;
        best_y = cy;
        min_cost = cost;
        best_sad = sl + sc;
      };
      // ExpandingSearch: sides without corners, then the corners
      for (int r = 1; r <= a.radius; r++)
      {
        for (int i = -r + 1; i < r; i++)
        {
          check(mvx + i, mvy - r);
          check(mvx + i, mvy + r);
        }
        for (int j = -r + 1; j < r; j++)
        {
          check(mvx - r, mvy + j);
          check(mvx + r, mvy + j);
        }
        check(mvx - r, mvy - r);
        check(mvx - r, mvy + r);
        check(mvx + r, mvy - r);
        check(mvx + r, mvy + r);
      }
    }

    if (lane == 0)
    {
      a.out[3 * idx + 0] = best_x;
      a.out[3 * idx + 1] = best_y;
      a.out[3 * idx + 2] = best_sad;
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

extern "C" int mvcuda_recalc_host(const MvCudaSuperParams *sp, const MvCudaRecalcParams *rp, const void *const src[3],
  const int src_pitch_bytes[3], const void *const ref[3], const int ref_pitch_bytes[3], const int *old_vectors,
  int *out_vectors)
{
  if (!sp || !rp || !src || !ref || !old_vectors || !out_vectors)
    return -4;
  // supported: level-0-only super, chroma on, pel 1 or 2, exhaustive search, 16-bit, smooth 0/1
  if (sp->levels != 1 || !sp->chroma || rp->smooth < 0 || rp->smooth > 1 || rp->search_radius < 1 || rp->old_pel < 1
    || rp->n_blk_x < 1 || rp->n_blk_y < 1 || rp->old_blk_x < 1 || rp->old_blk_y < 1 || rp->lambda_level < 0)
    return -4;
  if (mvcuda_device_count() <= 0)
    return -1;

  const bool timing = getenv("MVCUDA_TIMING") != nullptr;
  auto t0 = std::chrono::steady_clock::now();
  auto lap = [&](const char *what) {
    if (!timing)
      return;
    auto t1 = std::chrono::steady_clock::now();
    fprintf(stderr, "mvcuda recalc: %-22s %7.2f ms\n", what, std::chrono::duration<double, std::milli>(t1 - t0).count());
    t0 = t1;
  };

  int rc = 0;
  mvc::StreamLease lease;
  cudaStream_t st = lease.s;
  std::shared_ptr<mvc::DevSuper> s_src, s_ref;
  int *d_old = nullptr, *d_out = nullptr;
  const size_t n_old = (size_t)rp->old_blk_x * rp->old_blk_y, n_new = (size_t)rp->n_blk_x * rp->n_blk_y;
  const size_t old_bytes = n_old * 3 * sizeof(int), new_bytes = n_new * 3 * sizeof(int);

  rc = mvc::super_get(*sp, rp->src_key, src, src_pitch_bytes, st, s_src);
  if (!rc)
    rc = mvc::super_get(*sp, rp->ref_key, ref, ref_pitch_bytes, st, s_ref);
  lap("2 x super (cached)");
  if (rc)
    return rc;

  KArgs a = {};
  for (int p = 0; p < 3; p++)
  {
    a.src[p] = s_src->pl[p].sub[0];
    for (int i = 0; i < 4; i++)
      a.ref[p][i] = s_ref->pl[p].sub[i];
    a.ew[p] = s_ref->pl[p].d.ew;
    a.hpad[p] = s_ref->pl[p].d.hpad;
    a.vpad[p] = s_ref->pl[p].d.vpad;
  }
  a.ext_w0 = s_ref->pl[0].d.ew;
  a.ext_h0 = s_ref->pl[0].d.eh;
  a.pel = sp->pel;
  a.log_pel = ilog2(sp->pel);
  a.bits = sp->bits_per_pixel;
  a.log_x_uv = ilog2(sp->x_ratio_uv);
  a.log_y_uv = ilog2(sp->y_ratio_uv);
  a.blk_x = rp->blk_x;
  a.blk_y = rp->blk_y;
  a.ov_x = rp->overlap_x;
  a.ov_y = rp->overlap_y;
  a.n_blk_x = rp->n_blk_x;
  a.n_blk_y = rp->n_blk_y;
  a.old_n_x = rp->old_blk_x;
  a.old_n_y = rp->old_blk_y;
  a.old_size_x = rp->old_size_x;
  a.old_size_y = rp->old_size_y;
  a.old_ov_x = rp->old_overlap_x;
  a.old_ov_y = rp->old_overlap_y;
  a.log_pel_old = ilog2(rp->old_pel);
  a.lambda_level = rp->lambda_level;
  a.pnew = rp->pnew;
  a.th_sad = rp->th_sad;
  a.smooth = rp->smooth;
  a.radius = rp->search_radius;
  a.chroma_scale = rp->chroma_sad_scale;
  {
    int slices = rp->slice_threads < 1 ? 1 : rp->slice_threads;
    if (slices > 64)
      slices = 64;
    if (slices > rp->n_blk_y / 4)
      slices = rp->n_blk_y / 4;
    a.n_slices = slices < 1 ? 1 : slices;
  }

#define RC_CHECK(call) \
  do { \
    cudaError_t e_ = (call); \
    if (e_ != cudaSuccess) { \
      fprintf(stderr, "mvcuda: %s failed: %s\n", #call, cudaGetErrorString(e_)); \
      rc = -2; \
      goto done; \
    } \
  } while (0)

  d_old = (int *)mvc::dev_alloc(old_bytes);
  d_out = (int *)mvc::dev_alloc(new_bytes);
  if (!d_old || !d_out)
  {
    rc = -2;
    goto done;
  }
  RC_CHECK(cudaMemcpyAsync(d_old, old_vectors, old_bytes, cudaMemcpyHostToDevice, st));
  a.old_vec = d_old;
  a.out = d_out;
  {
    const int warps_per_block = 4;
    const int blocks = (int)((n_new + warps_per_block - 1) / warps_per_block);
    k_recalc<<<blocks, 32 * warps_per_block, 0, st>>>(a);
  }
  RC_CHECK(cudaGetLastError());
  RC_CHECK(cudaMemcpyAsync(out_vectors, d_out, new_bytes, cudaMemcpyDeviceToHost, st));
  RC_CHECK(cudaStreamSynchronize(st));
  lap("vectors + recalc kernel");

done:
  if (rc)
    cudaStreamSynchronize(st);
  mvc::dev_free(d_old, old_bytes);
  mvc::dev_free(d_out, new_bytes);
  return rc;
}
