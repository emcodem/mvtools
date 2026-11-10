// SplatFps on the GPU: the C entries (mvcuda.h) for all stages; the kernels are in mvcuda_splat.cuh (picture build),
// mvcuda_splat_me.cuh (ME pictures, match errors, local fallback, pair-statistic sums), mvcuda_splat_mo.cuh (phase
// correlation, refinement, assignment) and mvcuda_splat_ctx.cuh (input / output conversion and the per-filter device
// pipeline), built as an exact and a fast variant (mvcuda_splat_exact.cu, mvcuda_splat_fast.cu). The entries pick the variant by p->exact.
#include "mvcuda_internal.h"
#include "mvcuda_splat_ctx.h"

// the host-memory entries of one variant (both namespaces declare the same set)
#define SPLAT_VARIANT_DECLS                                                                                                \
  int build_host(const MvCudaSplatParams &p, const float *P, const float *N, const float *fwdP, const float *bwdN, float d, \
    float *out);                                                                                                           \
  int me_picture_host(const MvCudaSplatParams &p, const float *y, int fir, float *out);                                    \
  int errors_host(const MvCudaSplatParams &p, const float *cur, const float *other, const float *vecs, int n, float *errs); \
  int disorder_weight_host(const MvCudaSplatParams &p, const float *fv, int count, float *ch3);    \
  int phase_pull_host(const MvCudaSplatParams &p, const float *fP, const float *fN, double d, double strength, double *d_adj,  \
    double *info);                                                                                                         \
  int pcorr_host(const MvCudaSplatParams &p, const float *Bp, const float *Bc, float *cands, float *hp, float *glob);    \
  int refine_host(const MvCudaSplatParams &p, const float *cur, const float *other, const float *cands, int n, int neg,  \
    float *out);                                                                                                           \
  int assign_host(const MvCudaSplatParams &p, const float *rf, const float *ef, const float *rb, const float *eb, int n,  \
    const float *bias, float *fwd, float *bwd);                                                                            \
  int fields_host(const MvCudaSplatParams &p, const float *Ap, const float *Ac, const float *An, const float *cand_f,     \
    const float *cand_b, int n, int toward_b, int refine, const float *bias, float *fwd, float *bwd);                      \
  int fields_luma_host(const MvCudaSplatParams &p, const float *yp, const float *yc, const float *yn, float *fwd,          \
    float *bwd, float *ms);

namespace splat_exact
{
  SPLAT_VARIANT_DECLS
}
namespace splat_fast
{
  SPLAT_VARIANT_DECLS
}

extern "C" int mvcuda_splat_build_host(const MvCudaSplatParams *p, const float *P, const float *N, const float *fwd_P,
  const float *bwd_N, float d, float *out)
{
  if (!p || !P || !N || !fwd_P || !bwd_N || !out)
    return -4;
  if (!mvc::use_device())
    return -1;
  return p->exact ? splat_exact::build_host(*p, P, N, fwd_P, bwd_N, d, out)
                  : splat_fast::build_host(*p, P, N, fwd_P, bwd_N, d, out);
}

#define SPLAT_DISPATCH(fn, ...)                                                                                        \
  if (!p)                                                                                                              \
    return -4;                                                                                                         \
  if (!mvc::use_device())                                                                                              \
    return -1;                                                                                                         \
  return p->exact ? splat_exact::fn(*p, __VA_ARGS__) : splat_fast::fn(*p, __VA_ARGS__);

extern "C" int mvcuda_splat_me_picture_host(const MvCudaSplatParams *p, const float *y, int fir, float *out)
{
  SPLAT_DISPATCH(me_picture_host, y, fir, out)
}

extern "C" int mvcuda_splat_errors_host(const MvCudaSplatParams *p, const float *cur, const float *other, const float *vecs,
  int n, float *errs)
{
  SPLAT_DISPATCH(errors_host, cur, other, vecs, n, errs)
}

extern "C" int mvcuda_splat_disorder_weight_host(const MvCudaSplatParams *p, const float *fv, int count, float *ch3)
{
  SPLAT_DISPATCH(disorder_weight_host, fv, count, ch3)
}

extern "C" int mvcuda_splat_phase_pull_host(const MvCudaSplatParams *p, const float *fP, const float *fN, double d, double strength,
  double *d_adj, double *info)
{
  SPLAT_DISPATCH(phase_pull_host, fP, fN, d, strength, d_adj, info)
}

extern "C" int mvcuda_splat_pcorr_host(const MvCudaSplatParams *p, const float *B_prev, const float *B_cur, float *cands,
  float *hp, float *glob)
{
  if (!B_prev || !B_cur || !cands)
    return -4;
  SPLAT_DISPATCH(pcorr_host, B_prev, B_cur, cands, hp, glob)
}

extern "C" int mvcuda_splat_refine_host(const MvCudaSplatParams *p, const float *cur, const float *other, const float *cands,
  int n, int neg, float *out)
{
  if (!cur || !other || !cands || !out || n < 1)
    return -4;
  SPLAT_DISPATCH(refine_host, cur, other, cands, n, neg, out)
}

extern "C" int mvcuda_splat_assign_host(const MvCudaSplatParams *p, const float *rf, const float *ef, const float *rb,
  const float *eb, int n, const float *bias, float *fwd, float *bwd)
{
  if (!rf || !ef || (rb && !eb) || !bias || !fwd || !bwd || n < 1)
    return -4;
  SPLAT_DISPATCH(assign_host, rf, ef, rb, eb, n, bias, fwd, bwd)
}

extern "C" int mvcuda_splat_fields_host(const MvCudaSplatParams *p, const float *A_prev, const float *A_cur, const float *A_next,
  const float *cand_fwd, const float *cand_bwd, int n, int toward_b, int refine, const float *bias, float *fwd, float *bwd)
{
  if (!A_cur || !A_next || !cand_fwd || (A_prev && !cand_bwd) || !bias || !fwd || !bwd || n < 1)
    return -4;
  SPLAT_DISPATCH(fields_host, A_prev, A_cur, A_next, cand_fwd, cand_bwd, n, toward_b, refine, bias, fwd, bwd)
}

extern "C" int mvcuda_splat_fields_luma_host(const MvCudaSplatParams *p, const float *y_prev, const float *y_cur,
  const float *y_next, float *fwd, float *bwd, float *ms)
{
  if (!y_cur || !y_next || !fwd || !bwd)
    return -4;
  SPLAT_DISPATCH(fields_luma_host, y_prev, y_cur, y_next, fwd, bwd, ms)
}

// ---- device pipeline (one context per filter instance)
extern "C" void *mvcuda_splat_ctx_create(const MvCudaSplatParams *p)
{
  if (!p || p->bits < 8 || p->bits > 16 || p->sub_w < 0 || p->sub_w > 1 || p->sub_h < 0 || p->sub_h > p->sub_w ||
      !mvc::use_device())
    return nullptr;
  return p->exact ?splat_exact::ctx_new(*p) : splat_fast::ctx_new(*p);
}

extern "C" void mvcuda_splat_ctx_free(void *ctx)
{
  if (ctx)
  {
    mvc::use_device();
    delete static_cast<SplatCtx *>(ctx);
  }
}

extern "C" int mvcuda_splat_ctx_has_source(void *ctx, int q)
{
  return ctx && static_cast<SplatCtx *>(ctx)->has_source(q) ? 1 : 0;
}

extern "C" int mvcuda_splat_ctx_source(void *ctx, int q, const void *const planes[3], const int pitch_bytes[3])
{
  if (!ctx || !planes || !pitch_bytes)
    return -4;
  if (!mvc::use_device())
    return -1;
  return static_cast<SplatCtx *>(ctx)->source(q, planes, pitch_bytes);
}

extern "C" int mvcuda_splat_ctx_frame(void *ctx, int p, double d, double strength, void *const out[3],
  const int out_pitch_bytes[3], double *d_adj)
{
  if (!ctx || !out || !out_pitch_bytes)
    return -4;
  if (!mvc::use_device())
    return -1;
  return static_cast<SplatCtx *>(ctx)->frame(p, d, strength, out, out_pitch_bytes, d_adj);
}

extern "C" int mvcuda_splat_ctx_frame_start(void *ctx, int p, double d, double strength, double *d_adj)
{
  if (!ctx)
    return -4;
  if (!mvc::use_device())
    return -1;
  return static_cast<SplatCtx *>(ctx)->frame_start(p, d, strength, d_adj);
}

extern "C" int mvcuda_splat_ctx_frame_finish(void *ctx, void *const out[3], const int out_pitch_bytes[3])
{
  if (!ctx || (out && !out_pitch_bytes))
    return -4;
  if (!mvc::use_device())
    return -1;
  return static_cast<SplatCtx *>(ctx)->frame_finish(out, out_pitch_bytes);
}

extern "C" int mvcuda_splat_ctx_external(void *ctx)
{
  if (!ctx)
    return -4;
  static_cast<SplatCtx *>(ctx)->set_external();
  return 0;
}

extern "C" int mvcuda_splat_ctx_has_fields(void *ctx, int c)
{
  return ctx && static_cast<SplatCtx *>(ctx)->has_fields(c) ? 1 : 0;
}

extern "C" int mvcuda_splat_ctx_candidates(void *ctx, int c, const float *fwd, int fwd_pitch_bytes, const float *bwd,
  int bwd_pitch_bytes, int n, int bwd_toward, int refine, const float *bias)
{
  if (!ctx || !fwd || !bias)
    return -4;
  if (!mvc::use_device())
    return -1;
  return static_cast<SplatCtx *>(ctx)->candidates(c, fwd, fwd_pitch_bytes, bwd, bwd_pitch_bytes, n, bwd_toward != 0,
    refine != 0, bias);
}
