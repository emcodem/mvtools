// Motion vectors from NVIDIA's optical flow accelerator (NVOFA) for cudafilters' CudaNVOFAnalyse (NVOF_FILTER_PLAN.md):
// one vector field in MAnalyse's payload format with a single level, from the flow of the source picture into the
// reference picture. NOT MAnalyse's search (another algorithm, not bit-exact with anything on the CPU); MAnalyse stays
// exact and does not use this unit.
//
// Per call: the luma of both pictures as 8-bit grayscale (9..16 bit rounded), NVOFA's flow on a 4x4 grid (1/32 pixel),
// then per block the median (x and y separately) of the grid cells under the block, rounded to 1/pel pixel and clamped
// to the area MAnalyse's search can reach, and its SAD (luma + scaled chroma) against the sub-pel planes of the super
// frame (MSuper(pel, hpad, vpad, levels=1, sharp=2) of the reference, kept with the device frame like MAnalyse's).
// The same computation as level 0 of the MVCUDA_NVOFA experiment in MAnalyse (commit 574cd22), which it replaces.

#include "mvcuda_internal.h"
#include "mvcuda_nvofa.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>

namespace
{
  typedef mvc_pix pix;

  struct NvofArgs
  {
    int n_blk_x, n_blk_y, bs_x, bs_y, ov_x, ov_y, cw, ch, log_x_uv, log_y_uv;
    int pel, log_pel, ext_w, ext_h, ew[3], hpad[3], vpad[3];
    const pix *src[3];
    const pix *ref[3][4];
    int has_chroma, chroma_scale;
    int *vec; // (vx, vy, sad) per block
  };

  // Luma (inside the padding of the super plane) as 8-bit grayscale: 9..16-bit samples rounded to 8 bit.
  __global__ void k_of_gray(const pix *src, int ew, int hpad, int vpad, int w, int h, int shift, uint8_t *dst, size_t dpitch)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
      return;
    int v = src[(size_t)(vpad + y) * ew + hpad + x];
    if (shift > 0)
      v = (v + (1 << (shift - 1))) >> shift;
    dst[(size_t)y * dpitch + x] = (uint8_t)min(v, 255);
  }

  // 1/32-pixel flow -> 1/pel vector, rounded half away from zero
  __device__ __forceinline__ int of_to_vec(int f, int pel)
  {
    const int den = 32, num = f * pel;
    return num >= 0 ? (num + den / 2) / den : -((-num + den / 2) / den);
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

  // SAD of the w x h block at (x0, y0) of plane p of the source against the reference at (xp, yp) in 1/pel units
  __device__ int sad_block(const NvofArgs &a, int p, int x0, int y0, int w, int h, int xp, int yp, int lane)
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

  // One warp per block: the lower median (x and y separately) of the flow cells under the block, clamped to MAnalyse's
  // search area, then its SAD. Blocks up to 64x64 cover at most 17 x 17 cells.
  __global__ void k_of_blocks(NvofArgs a, const short2 *flow, size_t flow_pitch, int gw, int gh, int grid)
  {
    __shared__ short2 cells[4][17 * 17];
    const int wib = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int idx = blockIdx.x * 4 + wib;
    if (idx >= a.n_blk_x * a.n_blk_y)
      return;
    const int bx = idx % a.n_blk_x, by = idx / a.n_blk_x;
    const int xi = bx * (a.bs_x - a.ov_x), yi = by * (a.bs_y - a.ov_y); // block origin inside the padding
    const int gx0 = min(xi / grid, gw - 1), gx1 = min((xi + a.bs_x - 1) / grid, gw - 1);
    const int gy0 = min(yi / grid, gh - 1), gy1 = min((yi + a.bs_y - 1) / grid, gh - 1);
    const int cw_ = gx1 - gx0 + 1, n = min(cw_ * (gy1 - gy0 + 1), 17 * 17);
    for (int i = lane; i < n; i += 32)
    {
      const int r = i / cw_, c = i - r * cw_;
      cells[wib][i] = *(const short2 *)((const char *)flow + (size_t)(gy0 + r) * flow_pitch + (size_t)(gx0 + c) * sizeof(short2));
    }
    __syncwarp();
    // lower median by rank: the candidate with (count < c) <= k < (count <= c)
    const int k = (n - 1) / 2;
    int med[2] = { 0, 0 };
    for (int comp = 0; comp < 2; comp++)
    {
      int found = 0, val = 0;
      for (int j = lane; j < n; j += 32)
      {
        const int c = comp ? cells[wib][j].y : cells[wib][j].x;
        int lt = 0, le = 0;
        for (int i = 0; i < n; i++)
        {
          const int v = comp ? cells[wib][i].y : cells[wib][i].x;
          lt += v < c;
          le += v <= c;
        }
        if (!found && lt <= k && k < le)
        {
          found = 1;
          val = c;
        }
      }
      const unsigned b = __ballot_sync(0xffffffffu, found);
      med[comp] = __shfl_sync(0xffffffffu, val, __ffs(b) - 1);
    }
    const int x0 = a.hpad[0] + xi, y0 = a.vpad[0] + yi;
    const int dx_min = -a.pel * x0, dy_min = -a.pel * y0;
    const int dx_max = a.pel * (a.ext_w - x0 - a.bs_x), dy_max = a.pel * (a.ext_h - y0 - a.bs_y);
    const int vx = max(dx_min, min(dx_max - 1, of_to_vec(med[0], a.pel)));
    const int vy = max(dy_min, min(dy_max - 1, of_to_vec(med[1], a.pel)));
    int sad = sad_block(a, 0, x0, y0, a.bs_x, a.bs_y, x0 * a.pel + vx, y0 * a.pel + vy, lane);
    if (a.has_chroma)
    {
      int sc = 0;
      for (int p = 1; p < 3; p++)
      {
        const int cx0 = a.hpad[p] + (xi >> a.log_x_uv), cy0 = a.vpad[p] + (yi >> a.log_y_uv);
        sc += sad_block(a, p, cx0, cy0, a.cw, a.ch, cx0 * a.pel + (vx >> a.log_x_uv), cy0 * a.pel + (vy >> a.log_y_uv), lane);
      }
      sad += scale_chroma(sc, a.chroma_scale);
    }
    if (lane == 0)
    {
      a.vec[3 * idx + 0] = vx;
      a.vec[3 * idx + 1] = vy;
      a.vec[3 * idx + 2] = sad;
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

static int nvof_dev(const MvCudaSuperParams *sp, const MvCudaNvofParams *np, const std::shared_ptr<mvc::DevFrame> &fs,
  const std::shared_ptr<mvc::DevFrame> &fr, int *payload, int payload_words)
{
  if (sp->levels != 1 || (sp->pel != 1 && sp->pel != 2) || np->preset < 1 || np->preset > 3 || np->n_blk_x < 1
    || np->n_blk_y < 1 || np->blk_x < 1 || np->blk_y < 1 || np->blk_x > 64 || np->blk_y > 64)
    return -4;
  const int nblk = np->n_blk_x * np->n_blk_y, words = 3 + 3 * nblk;
  if (payload_words < words)
    return -4;
  const size_t vec_bytes = (size_t)nblk * 3 * sizeof(int);
  int *d_vec = (int *)mvc::dev_alloc(vec_bytes);
  // pinned staging for the result: a pageable copy would block the submitter thread inside the driver
  int *h_stage = (int *)mvc::pin_alloc(vec_bytes);
  if (!d_vec || !h_stage)
  {
    mvc::dev_free(d_vec, vec_bytes);
    mvc::pin_free(h_stage, vec_bytes);
    return -2;
  }
  std::shared_ptr<mvc::DevSuper> s_src, s_ref;
  mvc_nvofa::Session *of = nullptr;
  auto issue = [&](cudaStream_t st) -> int {
#define NV_CHECK(call) \
  do { \
    cudaError_t e_ = (call); \
    if (e_ != cudaSuccess) { \
      fprintf(stderr, "mvcuda: %s failed: %s\n", #call, cudaGetErrorString(e_)); \
      return -2; \
    } \
  } while (0)
    int r = mvc::frame_super(&MVC_NS::super_build_dev, fs, *sp, st, s_src);
    if (!r)
      r = mvc::frame_super(&MVC_NS::super_build_dev, fr, *sp, st, s_ref);
    if (r)
      return r;
    of = mvc_nvofa::acquire(sp->width, sp->height, np->preset);
    if (!of)
      return MVCUDA_NEEDS_CPU;
    const int shift = sp->bits_per_pixel > 8 ? sp->bits_per_pixel - 8 : 0;
    const dim3 gb(32, 8), gg((sp->width + 31) / 32, (sp->height + 7) / 8);
    const mvc::DevPlane *pl[2] = { &s_src->pl[0], &s_ref->pl[0] };
    for (int i = 0; i < 2; i++)
      k_of_gray<<<gg, gb, 0, st>>>((const pix *)pl[i]->sub[0], pl[i]->d.ew, pl[i]->d.hpad, pl[i]->d.vpad, sp->width, sp->height,
        shift, of->in[i], of->in_pitch);
    NV_CHECK(cudaGetLastError());
    if (mvc_nvofa::execute(of, st))
      return MVCUDA_NEEDS_CPU;
    NvofArgs a = {};
    a.n_blk_x = np->n_blk_x;
    a.n_blk_y = np->n_blk_y;
    a.bs_x = np->blk_x;
    a.bs_y = np->blk_y;
    a.ov_x = np->overlap_x;
    a.ov_y = np->overlap_y;
    a.pel = sp->pel;
    a.log_pel = ilog2(sp->pel);
    a.has_chroma = sp->chroma ? 1 : 0;
    a.log_x_uv = a.has_chroma ? ilog2(sp->x_ratio_uv) : 0;
    a.log_y_uv = a.has_chroma ? ilog2(sp->y_ratio_uv) : 0;
    a.cw = a.has_chroma ? a.bs_x >> a.log_x_uv : 0;
    a.ch = a.has_chroma ? a.bs_y >> a.log_y_uv : 0;
    a.chroma_scale = np->chroma_sad_scale;
    for (int p = 0; p < (a.has_chroma ? 3 : 1); p++)
    {
      const mvc::DevPlane &dr = s_ref->pl[p];
      a.src[p] = (const pix *)s_src->pl[p].sub[0];
      for (int i = 0; i < 4; i++)
        a.ref[p][i] = (const pix *)dr.sub[i < dr.nsub ? i : 0];
      a.ew[p] = dr.d.ew;
      a.hpad[p] = dr.d.hpad;
      a.vpad[p] = dr.d.vpad;
    }
    a.ext_w = s_ref->pl[0].d.ew;
    a.ext_h = s_ref->pl[0].d.eh;
    a.vec = d_vec;
    k_of_blocks<<<(nblk + 3) / 4, 128, 0, st>>>(a, of->flow, of->flow_pitch, of->gw, of->gh, of->grid);
    NV_CHECK(cudaGetLastError());
    NV_CHECK(cudaMemcpyAsync(h_stage, d_vec, vec_bytes, cudaMemcpyDeviceToHost, st));
    return 0;
#undef NV_CHECK
  };
  const mvc::DevFrame *in[2] = { fs.get(), fr.get() };
  const int rc = mvc::leased_call(in, 2, issue);
  // leased_call has waited for the stream: the session and the buffer can go back
  mvc_nvofa::release(of);
  if (rc == 0)
    std::memcpy(payload + 3, h_stage, vec_bytes);
  mvc::dev_free(d_vec, vec_bytes);
  mvc::pin_free(h_stage, vec_bytes);
  if (rc)
    return rc;
  payload[0] = words;
  payload[1] = 1;
  payload[2] = 1 + 3 * nblk;
  return 0;
}

extern "C" int MVC_API(mvcuda_nvof_token)(const MvCudaSuperParams *sp, const MvCudaNvofParams *np, unsigned long long src_token,
  unsigned long long ref_token, int *payload, int payload_words)
{
  if (!sp || !np || !payload)
    return -4;
  if (!mvc::use_device())
    return -1;
  auto fs = mvc::frame_get(src_token), fr = mvc::frame_get(ref_token);
  if (!fs || !fr)
    return MVCUDA_TOKEN_GONE;
  if (!mvc::super_fits(*fs, *sp, (int)sizeof(mvc_pix)) || !mvc::super_fits(*fr, *sp, (int)sizeof(mvc_pix)))
    return -4;
  return nvof_dev(sp, np, fs, fr, payload, payload_words);
}
