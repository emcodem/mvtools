// SplatFps device pipeline: the per-filter context behind mvcuda_splat_ctx_* (mvcuda.h). One implementation per variant
// (mvcuda_splat_ctx.cuh, compiled into mvcuda_splat_exact.cu and mvcuda_splat_fast.cu).
#ifndef MVCUDA_SPLAT_CTX_H
#define MVCUDA_SPLAT_CTX_H

#include "mvcuda.h"

struct SplatCtx
{
  virtual ~SplatCtx() {}
  virtual bool ok() const = 0;
  virtual bool has_source(int q) const = 0;
  // source picture q: 10-bit 4:2:2 codes in host planes (Y width x height, U / V width / 2 x height); returns once they
  // are staged (the planes may be released), the device work runs on
  virtual int source(int q, const void *const planes[3], const int pitch_bytes[3]) = 0;
  // the in-between picture at phase d between source pictures p and p + 1 (needs the sources p - 1 .. p + 2, p - 1 only
  // when p > 0): frame_start queues it (d_adj, or null: the phase after the temporal fallback), frame_finish waits for
  // the oldest started frame and writes it as codes into host planes (out null: dropped). source() may be called in
  // between; up to two frames may be started before the first is finished (a third start drops the oldest).
  // external candidates: the fields of a picture come only from candidates() (no phase correlation in the context);
  // call before the first source()
  virtual void set_external() = 0;
  virtual bool has_fields(int c) const = 0;
  // the fields of picture c from host candidate sets of the ME grid: fwd 2n planes (vx, vy per candidate, vectors
  // from c into c+1), bwd 2n planes or null (single side), bwd_toward: they point from c into c-1 (else from c-1 into
  // c); refine: run the refinement on them; bias n + 1 values. Needs the sources c-1 (with bwd) .. c+1.
  virtual int candidates(int c, const float *fwd, int fwd_pitch_bytes, const float *bwd, int bwd_pitch_bytes, int n,
    bool bwd_toward, bool refine, const float *bias) = 0;
  virtual int frame_start(int p, double d, double strength, double *d_adj) = 0;
  virtual int frame_finish(void *const out[3], const int out_pitch_bytes[3]) = 0;
  int frame(int p, double d, double strength, void *const out[3], const int out_pitch_bytes[3], double *d_adj)
  {
    const int rc = frame_start(p, d, strength, d_adj);
    return rc ? rc : frame_finish(out, out_pitch_bytes);
  }
};

namespace splat_exact { SplatCtx *ctx_new(const MvCudaSplatParams &p); }
namespace splat_fast { SplatCtx *ctx_new(const MvCudaSplatParams &p); }

#endif
