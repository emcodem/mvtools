// mvtools_vulkan: MRecalculate on the device, port of Sources/cuda/mvcuda_recalc.cu (bit for bit). Kernel:
// shaders/recalc.pix.comp (one workgroup per block; the search routines of mvcuda_refine.cuh in shaders/ana_common.glsl).

#include "vk_internal.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace
{
  struct RecalcArgs
  {
    uint64_t ref[3][MVCUDA_MAX_SUB];
    uint64_t src[3];
    uint64_t old_vec, out_vec;
    int ew[3], hpad[3], vpad[3];
    int ext_w0, ext_h0, pel, log_pel, bits, log_x_uv, log_y_uv;
    int blk_x, blk_y, ov_x, ov_y, n_blk_x, n_blk_y;
    int old_n_x, old_n_y, old_size_x, old_size_y, old_ov_x, old_ov_y, log_pel_old;
    int lambda_level, pnew, th_sad, smooth, radius, chroma_scale, n_slices, search_type, has_chroma, cw, ch, pad_;
  };

  enum { S_ONETIME = 1, S_NSTEP = 2, S_LOGARITHMIC = 4, S_EXHAUSTIVE = 8, S_HEX2 = 16, S_UMH = 32, S_HORIZONTAL = 64, S_VERTICAL = 128 };

  bool supported_type(int t)
  {
    return t == S_ONETIME || t == S_NSTEP || t == S_LOGARITHMIC || t == S_EXHAUSTIVE || t == S_HEX2 || t == S_UMH
      || t == S_HORIZONTAL || t == S_VERTICAL;
  }
  int max_param(int t) { return t == S_EXHAUSTIVE ? 64 : 24; }

  int ilog2(int v)
  {
    int l = 0;
    while ((1 << l) < v)
      l++;
    return l;
  }

  int recalc_dev(const MvCudaSuperParams *sp, const MvCudaRecalcParams *rp, const std::shared_ptr<mvk::DevFrame> &fs,
    const std::shared_ptr<mvk::DevFrame> &fr, const int *old_vectors, int *out_vectors)
  {
    if (!sp || !rp || !fs || !fr || !old_vectors || !out_vectors)
      return -4;
    const int esize = sp->bits_per_pixel <= 8 ? 1 : 2;
    const int search_type = rp->search_type ? rp->search_type : S_EXHAUSTIVE;
    if (sp->levels != 1 || rp->smooth < 0 || rp->smooth > 1 || rp->search_radius < 1 || rp->old_pel < 1
      || rp->n_blk_x < 1 || rp->n_blk_y < 1 || rp->old_blk_x < 1 || rp->old_blk_y < 1 || rp->lambda_level < 0
      || !supported_type(search_type) || rp->search_radius > max_param(search_type))
      return -4;
    if (!mvk::super_fits(*fs, *sp, esize) || !mvk::super_fits(*fr, *sp, esize))
      return -4;
    if (!vkrt::ready())
      return -1;
    // compile-time row counts of the luma / chroma block (shaders/ana_common.glsl SC_BSY, SC_CHH)
    const int sc_ch = sp->chroma ? rp->blk_y / sp->y_ratio_uv : 0;
    const bool sc_ok = rp->blk_y <= 32 && sc_ch <= 32;
    vkrt::Kernel *k = vkrt::kernel(esize == 1 ? "recalc_8" : "recalc_16", { sc_ok ? rp->blk_y : 0, sc_ok ? sc_ch : 0 });
    if (!k)
      return -2;

    const bool timing = getenv("MVCUDA_TIMING") != nullptr;
    auto t0 = std::chrono::steady_clock::now();

    std::shared_ptr<mvk::DevSuper> s_src, s_ref;
    int rc = mvk::frame_super(fs, *sp, s_src);
    if (!rc)
      rc = mvk::frame_super(fr, *sp, s_ref);
    if (rc)
      return rc;

    const size_t n_old = (size_t)rp->old_blk_x * rp->old_blk_y, n_new = (size_t)rp->n_blk_x * rp->n_blk_y;
    const size_t old_bytes = n_old * 3 * sizeof(int), new_bytes = n_new * 3 * sizeof(int);
    vkrt::DevBuf buf(old_bytes + new_bytes);
    if (!buf)
      return -2;

    RecalcArgs a = {};
    a.pel = sp->pel;
    a.log_pel = ilog2(sp->pel);
    a.bits = sp->bits_per_pixel;
    a.has_chroma = sp->chroma ? 1 : 0;
    a.log_x_uv = ilog2(sp->x_ratio_uv);
    a.log_y_uv = ilog2(sp->y_ratio_uv);
    a.blk_x = rp->blk_x;
    a.blk_y = rp->blk_y;
    a.cw = a.has_chroma ? rp->blk_x >> a.log_x_uv : 0;
    a.ch = a.has_chroma ? rp->blk_y >> a.log_y_uv : 0;
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
    a.search_type = search_type;
    a.chroma_scale = rp->chroma_sad_scale;
    {
      int slices = rp->slice_threads < 1 ? 1 : rp->slice_threads;
      if (slices > 64)
        slices = 64;
      if (slices > rp->n_blk_y / 4)
        slices = rp->n_blk_y / 4;
      a.n_slices = slices < 1 ? 1 : slices;
    }
    for (int p = 0; p < (a.has_chroma ? 3 : 1); p++)
    {
      a.src[p] = s_src->pl[p].sub[0];
      for (int i = 0; i < MVCUDA_MAX_SUB; i++)
        a.ref[p][i] = s_ref->pl[p].sub[i];
      a.ew[p] = s_ref->pl[p].d.ew;
      a.hpad[p] = s_ref->pl[p].d.hpad;
      a.vpad[p] = s_ref->pl[p].d.vpad;
    }
    a.ext_w0 = s_ref->pl[0].d.ew;
    a.ext_h0 = s_ref->pl[0].d.eh;
    a.old_vec = buf.p;
    a.out_vec = buf.p + old_bytes;

    const uint32_t gx = n_new < 32768 ? (uint32_t)n_new : 32768u, gy = (uint32_t)((n_new + gx - 1) / gx);
    vkrt::Cmd cmd;
    cmd.upload(buf.p, old_vectors, old_bytes);
    cmd.dispatch(k, gx, gy, 1, a);
    cmd.download(out_vectors, a.out_vec, new_bytes);
    rc = cmd.submit_wait();
    if (timing)
      fprintf(stderr, "mvvk recalc: %7.2f ms\n",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    if (vkrt::log_enabled())
      fprintf(stderr, "mvvk recalc: %dx%d blocks, type %d radius %d -> %d\n", rp->n_blk_x, rp->n_blk_y, search_type, rp->search_radius, rc);
    return rc;
  }
}

extern "C" int mvcuda_recalc_host(const MvCudaSuperParams *sp, const MvCudaRecalcParams *rp, const void *const src[3], const int src_pitch_bytes[3], const void *const ref[3], const int ref_pitch_bytes[3], const int *old_vectors, int *out_vectors)
{
  if (!sp || !rp || !src || !ref || !src_pitch_bytes || !ref_pitch_bytes || !old_vectors || !out_vectors)
    return -4;
  if (!vkrt::ready())
    return -1;
  const int esize = sp->bits_per_pixel <= 8 ? 1 : 2;
  auto fs = mvk::upload_picture(*sp, esize, src, src_pitch_bytes);
  auto fr = mvk::upload_picture(*sp, esize, ref, ref_pitch_bytes);
  if (!fs || !fr)
    return -4;
  return recalc_dev(sp, rp, fs, fr, old_vectors, out_vectors);
}

extern "C" int mvcuda_recalc_token(const MvCudaSuperParams *sp, const MvCudaRecalcParams *rp, unsigned long long src_token, unsigned long long ref_token, const int *old_vectors, int *out_vectors)
{
  if (!sp || !rp || !old_vectors || !out_vectors)
    return -4;
  if (!vkrt::ready())
    return -1;
  auto fs = mvk::frame_get(src_token), fr = mvk::frame_get(ref_token);
  if (!fs || !fr)
    return MVCUDA_TOKEN_GONE;
  return recalc_dev(sp, rp, fs, fr, old_vectors, out_vectors);
}
