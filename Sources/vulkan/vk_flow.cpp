// mvtools_vulkan: MFlowFps mask=2 ("extra frames") on device frames, port of Sources/cuda/mvcuda_flow.cu (bit for bit
// against MVFlowFps.cpp / MaskFun.cpp / SimpleResize.cpp; see that file for the stages). Kernels: shaders/flow_small,
// flow_occ, flow_occpack, flow_main (.pix), flow_mat (debug arrays of the golden tests). One call = one Cmd (after the
// super frames, which frame_super builds and keeps with the pictures): vector upload, small fields, occlusion masks,
// interpolation, and for host output the download of the picture.
//
// Difference to the CUDA code: the occlusion value min(int(255 * occlusion * occ_norm), 255) is a double multiply there.
// It is monotone in the (positive) occlusion, so the host computes, with the same double arithmetic, the smallest
// occlusion for each value 1..255 and the kernel counts thresholds: same results without float64 on the device.

#include "vk_internal.h"

#include <climits>
#include <cstring>

namespace
{
  struct SmallArgs
  {
    uint64_t vec[4];
    uint64_t svx[8], svy[8];
    int blk_x, blk_y, blk_xp, blk_yp, lx, ly, distinct, pad_;
  };

  struct OccArgs
  {
    uint64_t vec[2], acc[2];
    int tt[2];
    int blk_x, blk_y, blk_xp, step_x, step_y, pad_;
    int thr[256];
  };

  struct PackArgs
  {
    uint64_t acc[2], mask[2];
    int blk_x, blk_y, blk_xp, blk_yp;
  };

  struct FlowArgs
  {
    uint64_t left[48], right[48];
    uint64_t svx[8], svy[8];
    uint64_t smask[2];
    uint64_t dst[3];
    int dpitch[3];
    int w[3], h[3], hp[3], vp[3], ew[3], kind[3];
    int geo[8];
    int real_w[2], real_h[2];
    int pel, lp, time256, pad_;
  };

  struct MatArgs
  {
    uint64_t small_, out_;
    int geo[4];
    int real_w, real_h, lp, pitch, mode, pad_;
  };

  int ilog2(int v)
  {
    int l = 0;
    while ((1 << l) < v)
      l++;
    return l;
  }

  int align16(int v) { return (v + 15) & ~15; }
  size_t al256(size_t b) { return (b + 255) & ~(size_t)255; }

  // the CPU's occlusion value for occlusion > 0 (MakeVectorOcclusionMaskTime with gamma 1.0)
  int occ_cpu(int occlusion, double occ_norm)
  {
    const double d = (double)(255 * occlusion) * occ_norm;
    if (!(d < 256.0))
      return 255;
    const int v = (int)d;
    return v < 255 ? v : 255;
  }

  // thr[k - 1] = smallest occlusion o >= 1 with occ_cpu(o) >= k, INT_MAX when none (occlusion < 2^23 keeps 255 * o exact)
  void occ_thresholds(double occ_norm, int thr[256])
  {
    const int omax = 1 << 23;
    for (int k = 1; k <= 255; k++)
    {
      if (occ_cpu(omax, occ_norm) < k)
      {
        thr[k - 1] = INT_MAX;
        continue;
      }
      int lo = 1, hi = omax;
      while (lo < hi)
      {
        const int mid = lo + (hi - lo) / 2;
        if (occ_cpu(mid, occ_norm) >= k)
          hi = mid;
        else
          lo = mid + 1;
      }
      thr[k - 1] = lo;
    }
    thr[255] = INT_MAX;
  }

  // the same test as vk_core.cpp frame_super: a stored super frame e serves the parameters p
  bool super_covers(const MvCudaSuperParams &e, const MvCudaSuperParams &p)
  {
    return e.width == p.width && e.height == p.height && e.x_ratio_uv == p.x_ratio_uv && e.y_ratio_uv == p.y_ratio_uv
      && e.hpad == p.hpad && e.vpad == p.vpad && e.pel == p.pel && e.sharp == p.sharp && e.bits_per_pixel == p.bits_per_pixel
      && e.chroma >= p.chroma && e.levels >= p.levels && (p.levels == 1 || e.rfilter == p.rfilter);
  }

  // with fr.extra_m held: the stored super frame of fr for p, or a new one recorded into cmd (fresh = true; the caller
  // stores it after the submit succeeded)
  int find_or_record_super(mvk::DevFrame &fr, const MvCudaSuperParams &p, vkrt::Cmd &cmd, std::shared_ptr<mvk::DevSuper> &out,
    bool &fresh)
  {
    fresh = false;
    for (const auto &e : fr.supers)
      if (super_covers(e->p, p))
      {
        out = e;
        return 0;
      }
    auto s = std::make_shared<mvk::DevSuper>();
    const int rc = mvk::super_build(p, fr, *s, cmd);
    if (rc)
      return rc;
    out = s;
    fresh = true;
    return 0;
  }
}

// One interpolated frame from two device frames (left / right picture): a new device frame (out), or with host_dst the
// picture in host memory.
static int flow_dev(const MvCudaSuperParams *sp, const MvCudaFlowParams *fp, const std::shared_ptr<mvk::DevFrame> &fl,
  const std::shared_ptr<mvk::DevFrame> &fr, const int *vec_b, const int *vec_f, const int *vec_bb, const int *vec_ff,
  void *const dbg[MVCUDA_FLOW_DEBUG_SLOTS], unsigned long long producer, int frame, MvCudaTokenHeader *out,
  void *const host_dst[3] = nullptr, const int host_pitch[3] = nullptr)
{
  if (!sp || !fp || !fl || !fr || !vec_b || !vec_f || !vec_bb || !vec_ff || (!out && !host_dst))
    return -4;
  if (fp->mode != 2 || sp->levels != 1 || !sp->chroma || fp->time256 < 1 || fp->time256 > 255 || fp->width != sp->width
    || fp->height != sp->height || (sp->pel != 1 && sp->pel != 2 && sp->pel != 4))
    return -4;
  const int esize = sp->bits_per_pixel <= 8 ? 1 : 2;
  if (!mvk::super_fits(*fl, *sp, esize) || !mvk::super_fits(*fr, *sp, esize))
    return -4;
  if (fp->n_blk_x < 1 || fp->n_blk_y < 1 || fp->n_blk_xp < fp->n_blk_x || fp->n_blk_yp < fp->n_blk_y || fp->blk_step_x < 1
    || fp->blk_step_y < 1 || fp->width_p < 1 || fp->height_p < 1 || fp->width_p_uv < 1 || fp->height_p_uv < 1)
    return -4;
  if (!vkrt::ready())
    return -1;
  vkrt::Kernel *k_small = vkrt::kernel("flow_small"), *k_occ = vkrt::kernel("flow_occ"), *k_pack = vkrt::kernel("flow_occpack"),
    *k_flow = vkrt::kernel(esize == 1 ? "flow_main_8" : "flow_main_16"), *k_mat = dbg ? vkrt::kernel("flow_mat") : nullptr;
  if (!k_small || !k_occ || !k_pack || !k_flow || (dbg && !k_mat))
    return -2;

  const int nb = fp->n_blk_xp * fp->n_blk_yp, nbv = fp->n_blk_x * fp->n_blk_y;
  const size_t small_bytes = (size_t)nb * 2, mask_bytes = (size_t)nb;
  const size_t vec_bytes = (size_t)nbv * 3 * sizeof(int), acc_bytes = (size_t)nb * sizeof(int);
  const int distinct = fp->chroma_distinct ? 1 : 0;
  int geo[2][4] = { { fp->width_p, fp->height_p, fp->n_blk_xp, fp->n_blk_yp },
    { fp->width_p_uv, fp->height_p_uv, fp->n_blk_xp, fp->n_blk_yp } };
  const int uw[2] = { fp->width, fp->width / fp->x_ratio_uv }, uh[2] = { fp->height, fp->height / fp->y_ratio_uv };
  const int lp = ilog2(sp->pel);

  // super frames of both pictures: kept with the frames (as mvk::frame_super does); missing ones are built in this call's
  // Cmd (one submit) and stored after it succeeded. Both frames' locks are held until then (std::scoped_lock orders them),
  // so a concurrent request for the same frame waits for this build instead of making its own.
  std::shared_ptr<mvk::DevSuper> s_left, s_right;
  vkrt::Cmd cmd;
  const bool same = fl.get() == fr.get();
  std::unique_lock<std::mutex> lk_a, lk_b;
  if (same)
    lk_a = std::unique_lock<std::mutex>(fl->extra_m);
  else
  {
    std::lock(fl->extra_m, fr->extra_m);
    lk_a = std::unique_lock<std::mutex>(fl->extra_m, std::adopt_lock);
    lk_b = std::unique_lock<std::mutex>(fr->extra_m, std::adopt_lock);
  }
  bool fresh_l = false, fresh_r = false;
  int rc = find_or_record_super(*fl, *sp, cmd, s_left, fresh_l);
  if (!rc)
  {
    if (same)
      s_right = s_left;
    else
      rc = find_or_record_super(*fr, *sp, cmd, s_right, fresh_r);
  }
  if (rc)
    return rc;

  std::shared_ptr<mvk::DevFrame> dst;
  if (!host_dst)
  {
    dst = mvk::frame_alloc(fl->f);
    if (!dst)
      return -2;
  }
  // scratch: the 4 vector fields, small x / y arrays [kind][field], 2 masks, 2 occlusion accumulators, and for host output
  // the picture
  size_t o = 0;
  const size_t o_vecs = o;
  o += al256(4 * vec_bytes);
  size_t o_svx[8], o_svy[8];
  for (int i = 0; i < 8; i++)
  {
    o_svx[i] = o;
    o += al256(small_bytes);
    o_svy[i] = o;
    o += al256(small_bytes);
  }
  const size_t o_mask = o;
  o += 2 * al256(mask_bytes);
  const size_t o_acc = o;
  o += 2 * al256(acc_bytes);
  size_t o_pic[3] = {};
  int pw[3] = {}, ph[3] = {};
  for (int p = 0; p < 3; p++)
  {
    pw[p] = p == 0 ? fp->width : (distinct ? uw[1] : fp->width);
    ph[p] = p == 0 ? fp->height : (distinct ? uh[1] : fp->height);
    if (host_dst)
    {
      o_pic[p] = o;
      o += al256((size_t)pw[p] * ph[p] * esize);
    }
  }
  vkrt::DevBuf tmp(o);
  if (!tmp)
    return -2;
  const vkrt::dptr T = tmp.p;

  {
    const int *fields[4] = { vec_b, vec_f, vec_bb, vec_ff };
    for (int k = 0; k < 4; k++)
      cmd.upload(T + o_vecs + k * vec_bytes, fields[k], vec_bytes);
  }
  cmd.fill(T + o_acc, 2 * al256(acc_bytes), 0);

  SmallArgs sa = {};
  for (int k = 0; k < 4; k++)
    sa.vec[k] = T + o_vecs + k * vec_bytes;
  for (int i = 0; i < 8; i++)
  {
    sa.svx[i] = T + o_svx[i];
    sa.svy[i] = T + o_svy[i];
  }
  sa.blk_x = fp->n_blk_x;
  sa.blk_y = fp->n_blk_y;
  sa.blk_xp = fp->n_blk_xp;
  sa.blk_yp = fp->n_blk_yp;
  sa.lx = fp->x_ratio_uv == 2 ? 1 : 0;
  sa.ly = fp->y_ratio_uv == 2 ? 1 : 0;
  sa.distinct = distinct;
  const uint32_t gridn = (uint32_t)((nb + 255) / 256), gridb = (uint32_t)((nbv + 255) / 256);
  cmd.dispatch(k_small, gridn, 1, 4, sa);

  // occlusion masks: B (vectors of the left frame, field 0) with time 256 - t, F (right frame, field 1) with t
  OccArgs oa = {};
  oa.vec[0] = sa.vec[0];
  oa.vec[1] = sa.vec[1];
  oa.acc[0] = T + o_acc;
  oa.acc[1] = T + o_acc + al256(acc_bytes);
  oa.tt[0] = 256 - fp->time256;
  oa.tt[1] = fp->time256;
  oa.blk_x = fp->n_blk_x;
  oa.blk_y = fp->n_blk_y;
  oa.blk_xp = fp->n_blk_xp;
  oa.step_x = fp->blk_step_x;
  oa.step_y = fp->blk_step_y;
  occ_thresholds(fp->occ_norm, oa.thr);
  cmd.dispatch(k_occ, gridb, 1, 2, oa);

  PackArgs pa = {};
  pa.acc[0] = oa.acc[0];
  pa.acc[1] = oa.acc[1];
  pa.mask[0] = T + o_mask;
  pa.mask[1] = T + o_mask + al256(mask_bytes);
  pa.blk_x = fp->n_blk_x;
  pa.blk_y = fp->n_blk_y;
  pa.blk_xp = fp->n_blk_xp;
  pa.blk_yp = fp->n_blk_yp;
  cmd.dispatch(k_pack, gridn, 1, 2, pa);

  FlowArgs fa = {};
  fa.pel = sp->pel;
  fa.lp = lp;
  fa.time256 = fp->time256;
  int mw = 0, mh = 0;
  for (int p = 0; p < 3; p++)
  {
    const mvk::DevPlane &dl = s_left->pl[p], &dr = s_right->pl[p];
    fa.w[p] = pw[p];
    fa.h[p] = ph[p];
    fa.hp[p] = dl.d.hpad;
    fa.vp[p] = dl.d.vpad;
    fa.ew[p] = dl.d.ew;
    for (int i = 0; i < MVCUDA_MAX_SUB; i++)
    {
      fa.left[p * 16 + i] = dl.sub[i];
      fa.right[p * 16 + i] = dr.sub[i];
    }
    fa.kind[p] = (p > 0 && distinct) ? 1 : 0;
    if (host_dst)
    {
      fa.dst[p] = T + o_pic[p];
      fa.dpitch[p] = pw[p];
    }
    else
    {
      fa.dst[p] = dst->plane[p];
      fa.dpitch[p] = (int)(dst->pitch[p] / esize);
    }
    mw = pw[p] > mw ? pw[p] : mw;
    mh = ph[p] > mh ? ph[p] : mh;
  }
  // 4:4:4: the chroma planes have the luma size and use the luma arrays
  for (int kind = 0; kind <= distinct; kind++)
  {
    for (int k = 0; k < 4; k++)
    {
      fa.svx[kind * 4 + k] = sa.svx[kind * 4 + k];
      fa.svy[kind * 4 + k] = sa.svy[kind * 4 + k];
    }
    for (int i = 0; i < 4; i++)
      fa.geo[kind * 4 + i] = geo[kind][i];
    fa.real_w[kind] = uw[kind];
    fa.real_h[kind] = uh[kind];
  }
  fa.smask[0] = pa.mask[0];
  fa.smask[1] = pa.mask[1];
  cmd.dispatch(k_flow, (uint32_t)((mw + 31) / 32), (uint32_t)((mh + 7) / 8), 3, fa);

  // optional debug arrays (golden tests)
  std::vector<vkrt::DevBuf> dbg_bufs;
  if (dbg)
  {
    for (int k = 0; k < 4; k++)
    {
      if (dbg[k])
        cmd.download(dbg[k], sa.svx[k], small_bytes);
      if (dbg[4 + k])
        cmd.download(dbg[4 + k], sa.svy[k], small_bytes);
    }
    for (int m = 0; m < 2; m++)
      if (dbg[8 + m])
        cmd.download(dbg[8 + m], pa.mask[m], mask_bytes);
    for (int kind = 0; kind <= distinct; kind++)
    {
      const int base = 10 + 10 * kind;
      MatArgs ma = {};
      for (int i = 0; i < 4; i++)
        ma.geo[i] = geo[kind][i];
      ma.real_w = uw[kind];
      ma.real_h = uh[kind];
      ma.lp = lp;
      ma.pitch = align16(geo[kind][0]);
      const size_t full_el = (size_t)ma.pitch * geo[kind][1];
      const uint32_t gx = (uint32_t)((geo[kind][0] + 31) / 32), gy = (uint32_t)((geo[kind][1] + 7) / 8);
      // slot order within a kind: vxB vyB vxF vyF maskB maskF vxBB vyBB vxFF vyFF
      const int slot_vx[4] = { 0, 2, 6, 8 }, slot_vy[4] = { 1, 3, 7, 9 };
      auto one = [&](int slot, vkrt::dptr small_arr, int mode) {
        if (!dbg[base + slot])
          return;
        const size_t bytes = full_el * (mode == 2 ? 1 : 2);
        dbg_bufs.emplace_back((bytes + 3) & ~(size_t)3);
        const vkrt::DevBuf &b = dbg_bufs.back();
        if (!b)
          return;
        cmd.fill(b.p, b.bytes, 0);
        ma.small_ = small_arr;
        ma.out_ = b.p;
        ma.mode = mode;
        cmd.dispatch(k_mat, gx, gy, 1, ma);
        cmd.download(dbg[base + slot], b.p, bytes);
      };
      for (int k = 0; k < 4; k++)
      {
        one(slot_vx[k], sa.svx[kind * 4 + k], 0);
        one(slot_vy[k], sa.svy[kind * 4 + k], 1);
      }
      for (int m = 0; m < 2; m++)
        one(4 + m, pa.mask[m], 2);
    }
    for (const auto &b : dbg_bufs)
      if (!b)
        return -2;
  }
  if (host_dst)
    for (int p = 0; p < 3; p++)
      cmd.download2d(host_dst[p], (size_t)host_pitch[p], T + o_pic[p], (size_t)pw[p] * esize, (size_t)pw[p] * esize, (size_t)ph[p]);
  rc = cmd.submit_wait();
  if (!rc)
  {
    // keep the new super frames with their pictures (counted in the registry budget, outside the frame locks)
    if (fresh_l)
      fl->supers.push_back(s_left);
    if (fresh_r)
      fr->supers.push_back(s_right);
  }
  if (lk_b.owns_lock())
    lk_b.unlock();
  lk_a.unlock();
  if (!rc)
  {
    if (fresh_l)
      mvk::frame_account_extra(*fl, mvk::super_bytes(*sp, esize));
    if (fresh_r)
      mvk::frame_account_extra(*fr, mvk::super_bytes(*sp, esize));
  }
  if (rc || host_dst)
    return rc;
  mvk::frame_put(dst, producer, frame, *out);
  return 0;
}

extern "C" int mvcuda_flow_host(const MvCudaSuperParams *sp, const MvCudaFlowParams *fp, const void *const src_raw[3],
  const int src_pitch_bytes[3], const void *const ref_raw[3], const int ref_pitch_bytes[3], const int *vec_b, const int *vec_f,
  const int *vec_bb, const int *vec_ff, void *const dst[3], const int dst_pitch_bytes[3], void *const dbg[MVCUDA_FLOW_DEBUG_SLOTS])
{
  if (!sp || !fp || !src_raw || !ref_raw || !src_pitch_bytes || !ref_pitch_bytes || !dst || !dst_pitch_bytes)
    return -4;
  if (!vkrt::ready())
    return -1;
  // test entry point: both pictures uploaded, the result downloaded
  const int esize = sp->bits_per_pixel <= 8 ? 1 : 2;
  auto fl = mvk::upload_picture(*sp, esize, src_raw, src_pitch_bytes);
  auto fr = mvk::upload_picture(*sp, esize, ref_raw, ref_pitch_bytes);
  if (!fl || !fr)
    return -4;
  return flow_dev(sp, fp, fl, fr, vec_b, vec_f, vec_bb, vec_ff, dbg, 0, 0, nullptr, dst, dst_pitch_bytes);
}

extern "C" int mvcuda_flow_token(const MvCudaSuperParams *sp, const MvCudaFlowParams *fp, unsigned long long left,
  unsigned long long right, const int *vec_b, const int *vec_f, const int *vec_bb, const int *vec_ff, unsigned long long producer,
  int frame, MvCudaTokenHeader *out)
{
  if (!sp || !fp || !out)
    return -4;
  if (!vkrt::ready())
    return -1;
  auto fl = mvk::frame_get(left), fr = mvk::frame_get(right);
  if (!fl || !fr)
    return MVCUDA_TOKEN_GONE;
  return flow_dev(sp, fp, fl, fr, vec_b, vec_f, vec_bb, vec_ff, nullptr, producer, frame, out);
}

extern "C" int mvcuda_flow_token_host(const MvCudaSuperParams *sp, const MvCudaFlowParams *fp, unsigned long long left,
  unsigned long long right, const int *vec_b, const int *vec_f, const int *vec_bb, const int *vec_ff, void *const dst[3],
  const int dst_pitch_bytes[3])
{
  if (!sp || !fp || !dst || !dst_pitch_bytes)
    return -4;
  if (!vkrt::ready())
    return -1;
  auto fl = mvk::frame_get(left), fr = mvk::frame_get(right);
  if (!fl || !fr)
    return MVCUDA_TOKEN_GONE;
  for (int p = 0; p < fl->f.planes; p++)
    if (!dst[p] || dst_pitch_bytes[p] < (int)fl->pitch[p])
      return -4;
  return flow_dev(sp, fp, fl, fr, vec_b, vec_f, vec_bb, vec_ff, nullptr, 0, 0, nullptr, dst, dst_pitch_bytes);
}
