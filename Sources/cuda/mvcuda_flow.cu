// GPU version of MFlowFps (mask=2, the "extra frames" interpolation), bit-exact against MVFlowFps.cpp / MaskFun.cpp /
// SimpleResize.cpp for 8-bit and 9..16-bit samples, pel 1 or 2, planar YUV with 4:2:0 / 4:2:2 / 4:4:4 chroma.
//
// Stages (all integer arithmetic except the occlusion norm, which is IEEE double like the CPU):
//   1. small vector fields (nBlkXP x nBlkYP shorts) of the four vector frames, padded like CheckAndPadSmallY, and their
//      chroma versions (VectorSmallMaskYToHalfUV)
//   2. occlusion masks (MakeVectorOcclusionMaskTime + CheckAndPadMaskSmall): per-block maximum accumulation, so the
//      result does not depend on the order of the blocks
//   3. per output pixel: bilinear upsizing of the small arrays (SimpleResize, evaluated on the fly instead of being
//      materialised: 10 arrays of frame size would cost about 220 MB per frame) and the FlowInterExtra blend, reading the
//      reference frames through the pel-interleaved "finest" layout (MVFinest: sample (X, Y) of the finest plane is
//      sub-plane (Y mod pel) * pel + (X mod pel) at (X / pel, Y / pel))
// The same device functions can materialise the full-size arrays for the golden-data tests (dbg slots in the header).

#include "mvcuda_internal.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
  typedef mvc_pix pix;

  // ---- SimpleResize (C reference semantics) ---------------------------------------------------------------------------
  struct UpGeo
  {
    int new_w, new_h, old_w, old_h;
  };

  // horizontal control of output column x: source column and the two weights (hControl)
  __device__ __forceinline__ void h_ctl(const UpGeo &g, int x, int &offs, int &w1, int &w2)
  {
    int j = x * 256 * g.old_w / g.new_w + (128 * g.old_w) / g.new_w - 128;
    if (j < 0)
      j = 0;
    const int k = j >> 8;
    w2 = j - (k << 8);
    w1 = 256 - w2;
    if (k > g.old_w - 2)
    {
      offs = g.old_w - 1;
      w1 = 256;
      w2 = 0;
    }
    else
      offs = k;
  }

  // vertical control of output row y: source row, weight of the next row, and the last row offset
  __device__ __forceinline__ void v_ctl(const UpGeo &g, int y, int &off, int &w2, int &last_off)
  {
    auto one = [&](int yy, int &o, int &w) {
      int j = yy * 256 * g.old_h / g.new_h + (128 * g.old_h) / g.new_h - 128;
      if (j < 0)
        j = 0;
      int k = j >> 8;
      if (k > g.old_h - 2)
      {
        k = g.old_h - 1;
        j = k << 8;
      }
      o = k;
      w = j - (k << 8);
    };
    int dummy;
    one(y, off, w2);
    one(g.new_h - 1, last_off, dummy);
  }

  struct UpPos
  {
    int row1, row2, vw2;   // source rows and the weight of the second
    int hoff, hw1, hw2;    // source column and weights
    int old_w;
  };

  __device__ __forceinline__ UpPos up_pos(const UpGeo &g, int x, int y)
  {
    UpPos p;
    int voff, last;
    v_ctl(g, y, voff, p.vw2, last);
    p.row1 = voff;
    p.row2 = (voff < last) ? voff + 1 : 0;
    h_ctl(g, x, p.hoff, p.hw1, p.hw2);
    p.old_w = g.old_w;
    return p;
  }

  // bilinear value of a small array (vertical pass stored in the element type, then horizontal)
  template <typename T>
  __device__ __forceinline__ int up_value(const T *s, const UpPos &p)
  {
    const int inv = 256 - p.vw2;
    const int c0 = p.hoff, c1 = min(p.hoff + 1, p.old_w - 1); // the second column has weight 0 at the right edge
    const T a = (T)((s[p.row1 * p.old_w + c0] * inv + s[p.row2 * p.old_w + c0] * p.vw2 + 128) >> 8);
    const T b = (T)((s[p.row1 * p.old_w + c1] * inv + s[p.row2 * p.old_w + c1] * p.vw2 + 128) >> 8);
    return (a * p.hw1 + b * p.hw2 + 128) >> 8;
  }

  // vector limiting after the resize (MakeVectorsSafe): relative position must stay inside the frame
  __device__ __forceinline__ int limit_x(int v, int x, int real_w, int lp)
  {
    if (x >= real_w)
      return (short)v;
    const int maxr = ((real_w - x) << lp) - 1, minr = -(x << lp);
    return max(min((short)v, (short)maxr), (short)minr);
  }
  __device__ __forceinline__ int limit_y(int v, int y, int real_h, int lp)
  {
    if (y >= real_h)
      return (short)v;
    const int maxr = ((real_h - y) << lp) - 1, minr = -(y << lp);
    return max(min((short)v, (short)maxr), (short)minr);
  }

  // ---- stage 1: small vector fields ----------------------------------------------------------------------------------
  struct SmallArgs
  {
    int blk_x, blk_y, blk_xp, blk_yp, lx, ly, distinct;
  };

  __device__ __forceinline__ void small_value(const int *vec, const SmallArgs &a, int cx, int cy, int &vx, int &vy)
  {
    if (cy < a.blk_y)
    {
      const int bx = min(cx, a.blk_x - 1);
      vx = vec[3 * (cy * a.blk_x + bx)];
      vy = vec[3 * (cy * a.blk_x + bx) + 1];
      if (cx >= a.blk_x)
        vx = min(vx, 0); // right padding: not positive
    }
    else
    {
      const int bx = min(cx, a.blk_x - 1), by = a.blk_y - 1;
      vx = vec[3 * (by * a.blk_x + bx)];
      vy = vec[3 * (by * a.blk_x + bx) + 1];
      if (cx >= a.blk_x)
        vx = min(vx, 0);
      vy = min(vy, 0); // bottom padding: not positive
    }
  }

  __global__ void k_small(const int *vec, SmallArgs a, short *vx, short *vy, short *vxuv, short *vyuv)
  {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.blk_xp * a.blk_yp)
      return;
    const int cx = i % a.blk_xp, cy = i / a.blk_xp;
    int x, y;
    small_value(vec, a, cx, cy, x, y);
    vx[i] = (short)x;
    vy[i] = (short)y;
    if (a.distinct)
    {
      vxuv[i] = (short)(a.lx ? (short)x >> 1 : x);
      vyuv[i] = (short)(a.ly ? (short)y >> 1 : y);
    }
  }

  // ---- stage 2: occlusion masks -----------------------------------------------------------------------------------------
  __global__ void k_occ_zero(int *acc, int n)
  {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
      acc[i] = 0;
  }

  __device__ __forceinline__ void occ_add(int *acc, int idx, int occlusion, double occ_norm)
  {
    // int(255 * occlusion * occnorm), min 255 (gamma 1.0): the product 255 * occlusion is an int, then a double multiply
    int v = (int)__dmul_rn((double)(255 * occlusion), occ_norm);
    v = min(v, 255);
    atomicMax(&acc[idx], v);
  }

  __global__ void k_occ(const int *vec, SmallArgs a, int *acc, int time256, int step_x, int step_y, double occ_norm)
  {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.blk_x * a.blk_y)
      return;
    const int bx = i % a.blk_x, by = i / a.blk_x;
    const int time4096x = time256 * 16 / step_x, time4096y = time256 * 16 / step_y;
    const int vx = vec[3 * i], vy = vec[3 * i + 1];
    if (bx < a.blk_x - 1)
    {
      const int vx1 = vec[3 * (i + 1)];
      if (vx1 < vx)
      {
        const int occlusion = vx - vx1;
        for (int bxi = bx + vx1 * time4096x / 4096; bxi <= bx + vx * time4096x / 4096 + 1 && bxi >= 0 && bxi < a.blk_x; bxi++)
          occ_add(acc, bxi + by * a.blk_xp, occlusion, occ_norm);
      }
    }
    if (by < a.blk_y - 1)
    {
      const int vy1 = vec[3 * (i + a.blk_x) + 1];
      if (vy1 < vy)
      {
        const int occlusion = vy - vy1;
        for (int byi = by + vy1 * time4096y / 4096; byi <= by + vy * time4096y / 4096 + 1 && byi >= 0 && byi < a.blk_y; byi++)
          occ_add(acc, bx + byi * a.blk_xp, occlusion, occ_norm);
      }
    }
  }

  // CheckAndPadMaskSmall: copy the last column to the right padding, then the last row (padding included) downwards
  __global__ void k_occ_pack(const int *acc, SmallArgs a, unsigned char *mask)
  {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.blk_xp * a.blk_yp)
      return;
    const int cx = i % a.blk_xp, cy = i / a.blk_xp;
    const int sx = min(cx, a.blk_x - 1), sy = min(cy, a.blk_y - 1);
    mask[i] = (unsigned char)acc[sy * a.blk_xp + sx];
  }

  // ---- stage 3: interpolation ----------------------------------------------------------------------------------------
  struct FlowArgs
  {
    int pel, lp, time256, plane;
    int w[3], h[3];          // plane sizes
    int hp[3], vp[3];        // padding of the super planes (samples)
    int ew[3];               // extended width of the super planes
    const pix *left[3][4];   // sub-planes of the left frame (prefF)
    const pix *right[3][4];  // sub-planes of the right frame (prefB)
    const short *svx[2][4], *svy[2][4];   // [kind][B, F, BB, FF]
    const unsigned char *smask[2];        // [B, F]
    UpGeo geo[2];
    int real_w[2], real_h[2];
    int kind[3];
    pix *dst[3];
  };

  __device__ __forceinline__ int fetch(const pix *const sub[4], int ew, int pel_mask, int lp, int xa, int ya)
  {
    const int idx = (xa & pel_mask) | ((ya & pel_mask) << lp);
    return sub[idx][(size_t)(ya >> lp) * ew + (xa >> lp)];
  }

  __device__ __forceinline__ int median3r(int a, int b, int c)
  {
    // reduced median, a <= c
    if (b <= a)
      return a;
    if (c <= b)
      return c;
    return b;
  }

  __global__ void k_flow(FlowArgs a)
  {
    const int p = a.plane;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= a.w[p] || y >= a.h[p])
      return;
    const int kind = a.kind[p];
    const UpGeo &g = a.geo[kind];
    const UpPos pos = up_pos(g, x, y);
    const int rw = a.real_w[kind], rh = a.real_h[kind];
    const int t = a.time256;

    // full-size vectors and masks at this pixel (index 0 = B, 1 = F, 2 = BB, 3 = FF)
    int vx[4], vy[4];
    for (int k = 0; k < 4; k++)
    {
      vx[k] = limit_x(up_value<short>(a.svx[kind][k], pos), x, rw, a.lp);
      vy[k] = limit_y(up_value<short>(a.svy[kind][k], pos), y, rh, a.lp);
    }
    const int maskB = up_value<unsigned char>(a.smask[0], pos) & 255;
    const int maskF = up_value<unsigned char>(a.smask[1], pos) & 255;

    const int pm = a.pel - 1;
    const int x0 = a.hp[p] * a.pel + (x << a.lp), y0 = a.vp[p] * a.pel + (y << a.lp); // finest coordinates incl. padding
    const int vxF = (vx[1] * t) >> 8, vyF = (vy[1] * t) >> 8;
    const int dstF = fetch(a.left[p], a.ew[p], pm, a.lp, x0 + vxF, y0 + vyF);
    const int vxFF = (vx[3] * t) >> 8, vyFF = (vy[3] * t) >> 8;
    const int dstFF = fetch(a.left[p], a.ew[p], pm, a.lp, x0 + vxFF, y0 + vyFF);
    const int vxB = (vx[0] * (256 - t)) >> 8, vyB = (vy[0] * (256 - t)) >> 8;
    const int dstB = fetch(a.right[p], a.ew[p], pm, a.lp, x0 + vxB, y0 + vyB);
    const int vxBB = (vx[2] * (256 - t)) >> 8, vyBB = (vy[2] * (256 - t)) >> 8;
    const int dstBB = fetch(a.right[p], a.ew[p], pm, a.lp, x0 + vxBB, y0 + vyBB);

    int minfb, maxfb;
    if (dstF > dstB)
    {
      minfb = dstB;
      maxfb = dstF;
    }
    else
    {
      maxfb = dstB;
      minfb = dstF;
    }
    const int r = (((median3r(minfb, dstBB, maxfb) * maskF + dstF * (255 - maskF) + 255) >> 8) * (256 - t)
      + ((median3r(minfb, dstFF, maxfb) * maskB + dstB * (255 - maskB) + 255) >> 8) * t) >> 8;
    a.dst[p][(size_t)y * a.w[p] + x] = (pix)r;
  }

  // ---- debug: materialise the full-size arrays (same device functions) -------------------------------------------------
  struct MatArgs
  {
    UpGeo geo;
    int real_w, real_h, lp, pitch;
  };

  __global__ void k_mat_vec(const short *small, MatArgs m, short *out, bool is_x)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= m.geo.new_w || y >= m.geo.new_h)
      return;
    const UpPos pos = up_pos(m.geo, x, y);
    const int v = up_value<short>(small, pos);
    out[(size_t)y * m.pitch + x] = (short)(is_x ? limit_x(v, x, m.real_w, m.lp) : limit_y(v, y, m.real_h, m.lp));
  }

  __global__ void k_mat_mask(const unsigned char *small, MatArgs m, unsigned char *out)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= m.geo.new_w || y >= m.geo.new_h)
      return;
    out[(size_t)y * m.pitch + x] = (unsigned char)up_value<unsigned char>(small, up_pos(m.geo, x, y));
  }

  int ilog2(int v)
  {
    int l = 0;
    while ((1 << l) < v)
      l++;
    return l;
  }

  int align16(int v) { return (v + 15) & ~15; }
}

// One interpolated frame from two device frames (left / right picture): a new device frame (out), or with host_dst the
// picture in host memory. Issued by the submitter on a leased stream ordered after the work that made the pictures
// (leased_call), not on the pipeline stream: there it would wait for every upload queued before it. The super frames are
// kept with the pictures (frame_super); the vector fields go up and the picture comes back through pinned memory.
static int flow_dev(const MvCudaSuperParams *sp, const MvCudaFlowParams *fp, const std::shared_ptr<mvc::DevFrame> &fl,
  const std::shared_ptr<mvc::DevFrame> &fr, const int *vec_b, const int *vec_f, const int *vec_bb, const int *vec_ff,
  void *const dbg[MVCUDA_FLOW_DEBUG_SLOTS], unsigned long long producer, int frame, MvCudaTokenHeader *out,
  void *const host_dst[3] = nullptr, const int host_pitch[3] = nullptr)
{
  if (!sp || !fp || !fl || !fr || !vec_b || !vec_f || !vec_bb || !vec_ff || (!out && !host_dst))
    return -4;
  if (fp->mode != 2 || sp->levels != 1 || !sp->chroma || fp->time256 < 1 || fp->time256 > 255 || fp->width != sp->width
    || fp->height != sp->height || (sp->pel != 1 && sp->pel != 2))
    return -4;
  if (!mvc::super_fits(*fl, *sp, (int)sizeof(pix)) || !mvc::super_fits(*fr, *sp, (int)sizeof(pix)))
    return -4;
  if (!mvc::use_device())
    return -1;

  const int nb = fp->n_blk_xp * fp->n_blk_yp;
  const size_t small_bytes = (size_t)nb * sizeof(short), mask_bytes = (size_t)nb;
  const size_t vec_bytes = (size_t)fp->n_blk_x * fp->n_blk_y * 3 * sizeof(int), acc_bytes = (size_t)nb * sizeof(int);
  const int distinct = fp->chroma_distinct ? 1 : 0;
  UpGeo geo[2];
  geo[0] = { fp->width_p, fp->height_p, fp->n_blk_xp, fp->n_blk_yp };
  geo[1] = { fp->width_p_uv, fp->height_p_uv, fp->n_blk_xp, fp->n_blk_yp };
  const int uw[2] = { fp->width, fp->width / fp->x_ratio_uv }, uh[2] = { fp->height, fp->height / fp->y_ratio_uv };
  const int lp = ilog2(sp->pel);

  // the output frame from the upload pool (no queued pipeline work uses its buffer: another stream may write it); one
  // scratch buffer, released after the host wait: the 4 vector fields, small x / y arrays [kind][field], 2 masks, the
  // occlusion accumulator (each part 256-byte aligned)
  auto dst = mvc::frame_alloc(fl->f, true);
  if (!dst)
    return -2;
  auto al = [](size_t b) { return (b + 255) & ~(size_t)255; };
  const size_t vecs_bytes = 4 * vec_bytes;
  const size_t tmp_bytes = al(vecs_bytes) + 16 * al(small_bytes) + 2 * al(mask_bytes) + al(acc_bytes);
  unsigned char *d_tmp = (unsigned char *)mvc::dev_alloc(tmp_bytes);
  if (!d_tmp)
    return -2;
  int *d_vecs = (int *)d_tmp;
  short *d_svx[2][4] = {}, *d_svy[2][4] = {};
  unsigned char *d_smask[2] = {};
  int *d_acc = nullptr;
  {
    size_t o = al(vecs_bytes);
    for (int kind = 0; kind < 2; kind++)
      for (int k = 0; k < 4; k++)
      {
        d_svx[kind][k] = (short *)(d_tmp + o);
        o += al(small_bytes);
        d_svy[kind][k] = (short *)(d_tmp + o);
        o += al(small_bytes);
      }
    for (int m = 0; m < 2; m++)
    {
      d_smask[m] = d_tmp + o;
      o += al(mask_bytes);
    }
    d_acc = (int *)(d_tmp + o);
  }
  // pinned staging: the 4 vector fields on the way up, the picture (host output) on the way back
  const size_t stage_bytes = host_dst && dst->bytes > vecs_bytes ? dst->bytes : vecs_bytes;
  unsigned char *pin = (unsigned char *)mvc::pin_alloc(stage_bytes);
  std::vector<unsigned char> pageable;
  unsigned char *stage = pin;
  if (!stage)
  {
    pageable.resize(stage_bytes);
    stage = pageable.data();
  }
  const int *fields[4] = { vec_b, vec_f, vec_bb, vec_ff };
  for (int k = 0; k < 4; k++)
    std::memcpy(stage + k * vec_bytes, fields[k], vec_bytes);

  SmallArgs sa;
  sa.blk_x = fp->n_blk_x;
  sa.blk_y = fp->n_blk_y;
  sa.blk_xp = fp->n_blk_xp;
  sa.blk_yp = fp->n_blk_yp;
  sa.lx = fp->x_ratio_uv == 2 ? 1 : 0;
  sa.ly = fp->y_ratio_uv == 2 ? 1 : 0;
  sa.distinct = distinct;
  const int tb = 256, gridn = (nb + tb - 1) / tb, gridb = (fp->n_blk_x * fp->n_blk_y + tb - 1) / tb;

  std::shared_ptr<mvc::DevSuper> s_left, s_right; // held until the host wait
  const mvc::DevFrame *in[2] = { fl.get(), fr.get() };
  int rc = mvc::leased_call(in, 2, [&](cudaStream_t st) -> int {
    int r = mvc::frame_super(&MVC_NS::super_build_dev, fl, *sp, st, s_left);
    if (!r)
      r = mvc::frame_super(&MVC_NS::super_build_dev, fr, *sp, st, s_right);
    if (r)
      return r;
    if (cudaMemcpyAsync(d_vecs, stage, vecs_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess)
      return -2;
    const int *d_vec[4];
    for (int k = 0; k < 4; k++)
    {
      d_vec[k] = d_vecs + k * (vec_bytes / sizeof(int));
      k_small<<<gridn, tb, 0, st>>>(d_vec[k], sa, d_svx[0][k], d_svy[0][k], d_svx[1][k], d_svy[1][k]);
    }
    // occlusion masks: B (vectors of the left frame) with time 256 - t, F (right frame) with t
    for (int m = 0; m < 2; m++)
    {
      const int vecidx = m == 0 ? 0 : 1;
      const int tt = m == 0 ? 256 - fp->time256 : fp->time256;
      k_occ_zero<<<gridn, tb, 0, st>>>(d_acc, nb);
      k_occ<<<gridb, tb, 0, st>>>(d_vec[vecidx], sa, d_acc, tt, fp->blk_step_x, fp->blk_step_y, fp->occ_norm);
      k_occ_pack<<<gridn, tb, 0, st>>>(d_acc, sa, d_smask[m]);
    }

    FlowArgs fa = {};
    fa.pel = sp->pel;
    fa.lp = lp;
    fa.time256 = fp->time256;
    for (int p = 0; p < 3; p++)
    {
      const mvc::DevPlane &dl = s_left->pl[p], &dr = s_right->pl[p];
      const bool uvp = (p > 0 && distinct);
      fa.w[p] = p == 0 ? fp->width : (distinct ? uw[1] : fp->width);
      fa.h[p] = p == 0 ? fp->height : (distinct ? uh[1] : fp->height);
      fa.hp[p] = dl.d.hpad;
      fa.vp[p] = dl.d.vpad;
      fa.ew[p] = dl.d.ew;
      for (int i = 0; i < 4; i++)
      {
        fa.left[p][i] = (const pix *)dl.sub[i];
        fa.right[p][i] = (const pix *)dr.sub[i];
      }
      fa.kind[p] = uvp ? 1 : 0;
      fa.dst[p] = (pix *)dst->plane[p]; // packed rows of fa.w[p] samples, as k_flow writes them
    }
    // 4:4:4: chroma planes have the luma size (and use the luma arrays)
    for (int kind = 0; kind <= distinct; kind++)
    {
      for (int k = 0; k < 4; k++)
      {
        fa.svx[kind][k] = d_svx[kind][k];
        fa.svy[kind][k] = d_svy[kind][k];
      }
      fa.geo[kind] = geo[kind];
      fa.real_w[kind] = uw[kind];
      fa.real_h[kind] = uh[kind];
    }
    fa.smask[0] = d_smask[0];
    fa.smask[1] = d_smask[1];
    for (int p = 0; p < 3; p++)
    {
      const dim3 blk(32, 8), grd((fa.w[p] + 31) / 32, (fa.h[p] + 7) / 8, 1);
      fa.plane = p;
      k_flow<<<grd, blk, 0, st>>>(fa);
    }
    if (cudaGetLastError() != cudaSuccess)
      return -2;

    // optional debug arrays (tests only: pageable copies, waited for here)
    if (dbg)
    {
      for (int k = 0; k < 4; k++)
      {
        if (dbg[k])
          cudaMemcpyAsync(dbg[k], d_svx[0][k], small_bytes, cudaMemcpyDeviceToHost, st);
        if (dbg[4 + k])
          cudaMemcpyAsync(dbg[4 + k], d_svy[0][k], small_bytes, cudaMemcpyDeviceToHost, st);
      }
      for (int m = 0; m < 2; m++)
        if (dbg[8 + m])
          cudaMemcpyAsync(dbg[8 + m], d_smask[m], mask_bytes, cudaMemcpyDeviceToHost, st);
      for (int kind = 0; kind <= distinct; kind++)
      {
        const int base = 10 + 10 * kind;
        MatArgs ma;
        ma.geo = geo[kind];
        ma.real_w = uw[kind];
        ma.real_h = uh[kind];
        ma.lp = lp;
        ma.pitch = align16(geo[kind].new_w);
        const size_t full_el = (size_t)ma.pitch * geo[kind].new_h;
        short *d_full = nullptr;
        unsigned char *d_fullm = nullptr;
        if (cudaMalloc(&d_full, full_el * sizeof(short)) != cudaSuccess || cudaMalloc(&d_fullm, full_el) != cudaSuccess)
        {
          cudaFree(d_full);
          return -2;
        }
        const dim3 blk(32, 8), grd((geo[kind].new_w + 31) / 32, (geo[kind].new_h + 7) / 8);
        // slot order within a kind: vxB vyB vxF vyF maskB maskF vxBB vyBB vxFF vyFF
        const int slot_vx[4] = { 0, 2, 6, 8 }, slot_vy[4] = { 1, 3, 7, 9 };
        for (int k = 0; k < 4; k++)
        {
          if (dbg[base + slot_vx[k]])
          {
            cudaMemsetAsync(d_full, 0, full_el * sizeof(short), st);
            k_mat_vec<<<grd, blk, 0, st>>>(d_svx[kind][k], ma, d_full, true);
            cudaMemcpyAsync(dbg[base + slot_vx[k]], d_full, full_el * sizeof(short), cudaMemcpyDeviceToHost, st);
            cudaStreamSynchronize(st);
          }
          if (dbg[base + slot_vy[k]])
          {
            cudaMemsetAsync(d_full, 0, full_el * sizeof(short), st);
            k_mat_vec<<<grd, blk, 0, st>>>(d_svy[kind][k], ma, d_full, false);
            cudaMemcpyAsync(dbg[base + slot_vy[k]], d_full, full_el * sizeof(short), cudaMemcpyDeviceToHost, st);
            cudaStreamSynchronize(st);
          }
        }
        for (int m = 0; m < 2; m++)
          if (dbg[base + 4 + m])
          {
            cudaMemsetAsync(d_fullm, 0, full_el, st);
            k_mat_mask<<<grd, blk, 0, st>>>(d_smask[m], ma, d_fullm);
            cudaMemcpyAsync(dbg[base + 4 + m], d_fullm, full_el, cudaMemcpyDeviceToHost, st);
            cudaStreamSynchronize(st);
          }
        cudaStreamSynchronize(st);
        cudaFree(d_full);
        cudaFree(d_fullm);
      }
      if (cudaStreamSynchronize(st) != cudaSuccess)
        return -2;
    }
    // host output: the whole frame buffer back in one copy
    if (host_dst && cudaMemcpyAsync(stage, dst->plane[0], dst->bytes, cudaMemcpyDeviceToHost, st) != cudaSuccess)
      return -2;
    return 0;
  });
  // leased_call has waited for the stream in every case
  mvc::dev_free(d_tmp, tmp_bytes);
  if (!rc && host_dst)
    for (int p = 0; p < dst->f.planes; p++)
    {
      const unsigned char *s = stage + ((unsigned char *)dst->plane[p] - (unsigned char *)dst->plane[0]);
      for (int y = 0; y < dst->h[p]; y++)
        std::memcpy((unsigned char *)host_dst[p] + (size_t)y * host_pitch[p], s + y * dst->pitch[p], dst->pitch[p]);
    }
  mvc::pin_free(pin, stage_bytes);
  if (rc || host_dst)
    return rc;
  mvc::frame_put(dst, producer, frame, *out);
  return 0;
}

extern "C" int MVC_API(mvcuda_flow_host)(const MvCudaSuperParams *sp, const MvCudaFlowParams *fp, const void *const src_raw[3],
  const int src_pitch_bytes[3], const void *const ref_raw[3], const int ref_pitch_bytes[3], const int *vec_b,
  const int *vec_f, const int *vec_bb, const int *vec_ff, void *const dst[3], const int dst_pitch_bytes[3],
  void *const dbg[MVCUDA_FLOW_DEBUG_SLOTS])
{
  if (!sp || !fp || !src_raw || !ref_raw || !src_pitch_bytes || !ref_pitch_bytes || !dst || !dst_pitch_bytes)
    return -4;
  if (!mvc::use_device())
    return -1;
  // test entry point: both pictures uploaded, the result downloaded
  auto fl = mvc::upload_picture(*sp, (int)sizeof(pix), src_raw, src_pitch_bytes);
  auto fr = mvc::upload_picture(*sp, (int)sizeof(pix), ref_raw, ref_pitch_bytes);
  if (!fl || !fr)
    return -4;
  return flow_dev(sp, fp, fl, fr, vec_b, vec_f, vec_bb, vec_ff, dbg, 0, 0, nullptr, dst, dst_pitch_bytes);
}

extern "C" int MVC_API(mvcuda_flow_token)(const MvCudaSuperParams *sp, const MvCudaFlowParams *fp, unsigned long long left,
  unsigned long long right, const int *vec_b, const int *vec_f, const int *vec_bb, const int *vec_ff, unsigned long long producer,
  int frame, MvCudaTokenHeader *out)
{
  if (!sp || !fp || !out)
    return -4;
  if (!mvc::use_device())
    return -1;
  auto fl = mvc::frame_get(left), fr = mvc::frame_get(right);
  if (!fl || !fr)
    return MVCUDA_TOKEN_GONE;
  return flow_dev(sp, fp, fl, fr, vec_b, vec_f, vec_bb, vec_ff, nullptr, producer, frame, out);
}

extern "C" int MVC_API(mvcuda_flow_token_host)(const MvCudaSuperParams *sp, const MvCudaFlowParams *fp, unsigned long long left,
  unsigned long long right, const int *vec_b, const int *vec_f, const int *vec_bb, const int *vec_ff, void *const dst[3],
  const int dst_pitch_bytes[3])
{
  if (!sp || !fp || !dst || !dst_pitch_bytes)
    return -4;
  if (!mvc::use_device())
    return -1;
  auto fl = mvc::frame_get(left), fr = mvc::frame_get(right);
  if (!fl || !fr)
    return MVCUDA_TOKEN_GONE;
  for (int p = 0; p < fl->f.planes; p++)
    if (!dst[p] || dst_pitch_bytes[p] < (int)fl->pitch[p])
      return -4;
  return flow_dev(sp, fp, fl, fr, vec_b, vec_f, vec_bb, vec_ff, nullptr, 0, 0, nullptr, dst, dst_pitch_bytes);
}
