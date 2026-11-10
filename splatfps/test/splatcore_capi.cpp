// splatcore_test: a C ABI over splat_core for the stage parity tests (Python ctypes: the exact CPU path, the fast CPU
// path and the CUDA stages are compared on inputs this library derives itself). Not shipped.
// Arrays are float32, planar, rows contiguous: a picture is 3 planes, a field 4 planes.
#include "../core/splat_const.h"
#include "../core/splat_core.h"
#include "../core/splat_me.h"
#include "../core/splat_motion.h"
#include "../core/splat_pcorr.h"
#include "../core/splat_fast.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <memory>

#if defined(_WIN32)
#define SC_API extern "C" __declspec(dllexport)
#else
#define SC_API extern "C" __attribute__((visibility("default")))
#endif

using namespace splat;

static Plane plane_in(const float *p, int w, int h)
{
  Plane o(w, h);
  std::memcpy(o.d.data(), p, sizeof(float) * (size_t)w * h);
  return o;
}

static void plane_out(const Plane &s, float *p) { std::memcpy(p, s.d.data(), sizeof(float) * s.d.size()); }

static Pic3 pic_in(const float *p, int w, int h)
{
  Pic3 o;
  for (int c = 0; c < 3; c++)
    o.c[c] = plane_in(p + (size_t)c * w * h, w, h);
  return o;
}

static void pic_out(const Pic3 &s, float *p)
{
  const size_t n = s.c[0].d.size();
  for (int c = 0; c < 3; c++)
    plane_out(s.c[c], p + c * n);
}

static Field4 field_in(const float *p, int w, int h)
{
  Field4 o;
  for (int c = 0; c < 4; c++)
    o.c[c] = plane_in(p + (size_t)c * w * h, w, h);
  return o;
}

static void field_out(const Field4 &s, float *p)
{
  const size_t n = s.c[0].d.size();
  for (int c = 0; c < 4; c++)
    plane_out(s.c[c], p + c * n);
}

SC_API int sc_version() { return 8; }

// The configuration every entry uses: the defaults of a picture size (sc_set_size; 1920 x 1080 until called).
static Config g_cfg = config_for_size(1920, 1080);

// returns 0, or -1 when the defaults of that size do not pass the checks (g_cfg is set anyway)
SC_API int sc_set_size(int w, int h)
{
  g_cfg = config_for_size(w, h);
  char why[256];
  return config_check(g_cfg, w, h, why, sizeof(why)) ? 0 : -1;
}

// the current configuration as bytes (a splat::Config, for MvCudaSplatParams::config): returns its size; out (or null)
SC_API int sc_config(void *out)
{
  if (out)
    std::memcpy(out, &g_cfg, sizeof(Config));
  return (int)sizeof(Config);
}

// peaks per block of the three phase-correlation stages (block_peaks); returns 0, or -1 when the configuration of the
// current size does not pass the checks (g_cfg is set anyway).
SC_API int sc_set_peaks(int p0, int p1, int p2, int w, int h)
{
  g_cfg.peaks[0] = p0;
  g_cfg.peaks[1] = p1;
  g_cfg.peaks[2] = p2;
  char why[256];
  return config_check(g_cfg, w, h, why, sizeof(why)) ? 0 : -1;
}

// number of phase-correlation candidates (without the zero vector) on the ME grid of the current configuration
SC_API int sc_ncand(int mw, int mh) { return pcorr_ncand(g_cfg, mw, mh); }

// the candidate costs (n + 1 values, zero vector last) of the phase-correlation set; returns n + 1
SC_API int sc_costs(int mw, int mh, float *out) { return pcorr_costs(g_cfg, mw, mh, out); }

// Which path the stage entries below run: 0 = exact (default), 1 = fast path without the AVX2 kernels, 2 = fast path
// (AVX2 kernels when the CPU has AVX2 and FMA). Entries without a fast version always run the exact function.
static std::unique_ptr<FastEngine> g_fast;


// returns 1 when the fast path uses the AVX2 kernels, 0 otherwise
// (mode >> 4: thread count of the fast path, 0 = all hardware threads)
SC_API int sc_set_fast(int mode)
{
  g_fast.reset();
  if ((mode & 15) > 0)
    g_fast = std::make_unique<FastEngine>(mode >> 4, (mode & 15) >= 2);
  return g_fast && g_fast->simd() ? 1 : 0;
}

SC_API void sc_c422to444(const float *in, int w2, int rows, float *out)
{
  for (int r = 0; r < rows; r++)
    chroma_upsample_row(in + (size_t)r * w2, w2, out + (size_t)r * 2 * w2);
}

SC_API void sc_pad(const float *src, int w, int h, float *dst, int gw, int gh, int px, int py)
{
  plane_out(pad_edge(plane_in(src, w, h), gw, gh, px, py), dst);
}

SC_API void sc_pyr_reduce(const float *src, int w, int h, float *dst) { plane_out(pyr_reduce(plane_in(src, w, h), g_cfg.pyr_reduce), dst); }

SC_API void sc_pyr_expand(const float *low, int lw, int lh, float *dst, int ow, int oh)
{
  plane_out(pyr_expand(plane_in(low, lw, lh), ow, oh), dst);
}

SC_API void sc_border_weight(const float *F, int w, int h, float d, int n_side, float *G)
{
  field_out(border_weight(field_in(F, w, h), d, n_side != 0, g_cfg), G);
}

SC_API void sc_vec_up4(const float *G, int w, int h, float *out)
{
  field_out(field_upsample(field_in(G, w, h), g_cfg.scale), out);
}

SC_API void sc_vec_dn2(const float *F, int w, int h, float *out) { field_out(field_reduce2(field_in(F, w, h)), out); }

// P, N: 3 planes w x h; FP, FN: 4 planes fw x fh; acc_init / acc_splat: 4 planes w x h each, or null
SC_API void sc_build_level(const float *P, const float *N, int w, int h, const float *FP, const float *FN, int fw, int fh,
  int stride, float d, float *out, float *acc_init, float *acc_splat)
{
  LevelDebug dbg;
  Pic3 o = build_level(pic_in(P, w, h), pic_in(N, w, h), field_in(FP, fw, fh), field_in(FN, fw, fh), stride, d, g_cfg,
    (acc_init || acc_splat) ? &dbg : nullptr);
  pic_out(o, out);
  for (int c = 0; c < 4; c++)
  {
    if (acc_init)
      plane_out(dbg.acc_init[c], acc_init + (size_t)c * w * h);
    if (acc_splat)
      plane_out(dbg.acc_splat[c], acc_splat + (size_t)c * w * h);
  }
}

// P, N: 3 planes W x H; MEP, MEN: 4 planes mw x mh
SC_API void sc_build(const float *P, const float *N, int W, int H, const float *MEP, const float *MEN, int mw, int mh,
  float d, float *out)
{
  if (g_fast)
    pic_out(g_fast->build(pic_in(P, W, H), pic_in(N, W, H), field_in(MEP, mw, mh), field_in(MEN, mw, mh), d, g_cfg), out);
  else
    pic_out(build(pic_in(P, W, H), pic_in(N, W, H), field_in(MEP, mw, mh), field_in(MEN, mw, mh), d, g_cfg), out);
}

// ---- stages around the motion vectors (splat_me.h)

// fir: 0 = A, 1 = B; out: ((w + 32) / 4) x ((h + 24) / 4)
SC_API void sc_me_picture(const float *y, int w, int h, int fir, float *out)
{
  const Plane yp = plane_in(y, w, h);
  plane_out(g_fast ? g_fast->me_picture(yp, fir ? MeFir::B : MeFir::A, g_cfg) : me_picture(yp, fir ? MeFir::B : MeFir::A, g_cfg),
    out);
}

SC_API void sc_error(const float *cur, const float *other, int w, int h, const float *vx, const float *vy, float *out)
{
  if (g_fast)
  {
    const VecSet v = { plane_in(vx, w, h), plane_in(vy, w, h) };
    plane_out(g_fast->errors(plane_in(cur, w, h), plane_in(other, w, h), v, g_cfg)[0], out);
    return;
  }
  plane_out(error_of(plane_in(cur, w, h), plane_in(other, w, h), plane_in(vx, w, h), plane_in(vy, w, h), g_cfg), out);
}

SC_API void sc_disorder_weight(const float *vx, const float *vy, int w, int h, float *out)
{
  if (g_fast)
    plane_out(g_fast->disorder_weight(plane_in(vx, w, h), plane_in(vy, w, h), nullptr, g_cfg), out);
  else
    plane_out(disorder_weight(plane_in(vx, w, h), plane_in(vy, w, h), nullptr, g_cfg), out);
}

// fP, fN: 4 planes w x h; info: m, p, f, g2, d
SC_API double sc_phase_pull(const float *fP, const float *fN, int w, int h, double d, double strength, double *info)
{
  PullInfo ti;
  const double r = phase_pull_adjust(field_in(fP, w, h), field_in(fN, w, h), d, strength, g_cfg, &ti);
  if (info)
  {
    info[0] = ti.m; info[1] = ti.p; info[2] = ti.f; info[3] = ti.g2; info[4] = ti.d;
  }
  return r;
}

SC_API double sc_phase_pull_strength(long long a, long long b, long long c, long long d)
{
  return phase_pull_strength(a, b, c, d, g_cfg);
}

// returns the frame count; fills up to cap entries: k, p, dnum, dden, kind (0 copyP, 1 copyN, 2 built, 3 eos)
SC_API int sc_plan(int n, long long rin_n, long long rin_d, long long rout_n, long long rout_d, int cap, long long *out5)
{
  const std::vector<PlanEntry> v = plan(n, rin_n, rin_d, rout_n, rout_d, g_cfg.copy_snap);
  for (size_t i = 0; i < v.size() && (int)i < cap; i++)
  {
    long long *o = out5 + 5 * i;
    o[0] = v[i].k; o[1] = v[i].p; o[2] = v[i].dnum; o[3] = v[i].dden; o[4] = (long long)v[i].kind;
  }
  return (int)v.size();
}

// ---- motion estimation (splat_pcorr.h, splat_motion.h)

static VecSet vecs_in(const float *p, int n, int w, int h)
{
  VecSet v(2 * (size_t)n);
  for (int i = 0; i < 2 * n; i++)
    v[i] = plane_in(p + (size_t)i * w * h, w, h);
  return v;
}

static void planes_out(const std::vector<Plane> &v, float *p)
{
  for (size_t i = 0; i < v.size(); i++)
    plane_out(v[i], p + i * v[i].d.size());
}

// phase correlation: candidates of the pair (B_prev, B_cur), sc_ncand(w, h) x (vx, vy) planes (29 at 1080p); hp (w x h) of B_cur and
// glob (dx, dy, height, vx, vy) are optional
SC_API void sc_pcorr(const float *Bprev, const float *Bcur, int w, int h, float *cands, float *hp, float *glob)
{
  const Plane pa = plane_in(Bprev, w, h), pb = plane_in(Bcur, w, h);
  const PcorrPicture a = g_fast ? g_fast->pcorr_ingest(pa, g_cfg) : pcorr_ingest(pa, g_cfg);
  const PcorrPicture b = g_fast ? g_fast->pcorr_ingest(pb, g_cfg) : pcorr_ingest(pb, g_cfg);
  PcorrGlobal g;
  planes_out(g_fast ? g_fast->pcorr_candidates(a, b, g_cfg, &g) : pcorr_candidates(a, b, g_cfg, &g), cands);
  if (hp)
    plane_out(b.hp, hp);
  if (glob)
  {
    glob[0] = g.dx; glob[1] = g.dy; glob[2] = g.height; glob[3] = g.vx; glob[4] = g.vy;
  }
}


// refinement: n candidates (2 planes each) -> n refined
SC_API void sc_refine(const float *cur, const float *other, int w, int h, const float *cands, int n, int neg, float *out)
{
  const Plane c = plane_in(cur, w, h), o = plane_in(other, w, h);
  const VecSet v = vecs_in(cands, n, w, h);
  planes_out(g_fast ? g_fast->refine(c, o, v, neg != 0, g_cfg) : refine(c, o, v, neg != 0, g_cfg), out);
}

// assignment: rf / rb n candidates, ef / eb n + 1 planes, bias n + 1; single: forward side only (rb, eb unused);
// fwd / bwd 4 planes
SC_API void sc_assign(const float *rf, const float *ef, const float *rb, const float *eb, int n, int w, int h,
  const float *bias, int single, float *fwd, float *bwd)
{
  const VecSet vf = vecs_in(rf, n, w, h);
  std::vector<Plane> pf(n + 1), pb;
  for (int i = 0; i <= n; i++)
    pf[i] = plane_in(ef + (size_t)i * w * h, w, h);
  Field4 F, B;
  if (single)
    assign_single(vf, pf, bias, g_cfg.vector_length_cost, F, B);
  else
  {
    pb.resize(n + 1);
    for (int i = 0; i <= n; i++)
      pb[i] = plane_in(eb + (size_t)i * w * h, w, h);
  {
    if (g_fast)
      g_fast->assign(vf, pf, vecs_in(rb, n, w, h), pb, bias, g_cfg.speed_change_cost, g_cfg.occlusion_error, g_cfg.occlusion_error_factor, g_cfg.vector_length_cost, F, B);
    else
      assign(vf, pf, vecs_in(rb, n, w, h), pb, bias, g_cfg.speed_change_cost, g_cfg.occlusion_error, g_cfg.occlusion_error_factor, g_cfg.vector_length_cost, F, B);
  }
  }
  field_out(F, fwd);
  field_out(B, bwd);
}

// refinement to local fallback from candidates: Aprev null = single side; cand_bwd as delivered by the source
// (toward_b: pointing from c); refine: run the refinement on both sides
SC_API void sc_fields(const float *Aprev, const float *Acur, const float *Anext, int w, int h, const float *cand_fwd,
  const float *cand_bwd, int n, int toward_b, int refine_, const float *bias, float *fwd, float *bwd)
{
  const Plane cur = plane_in(Acur, w, h), nxt = plane_in(Anext, w, h);
  Plane prv;
  if (Aprev)
    prv = plane_in(Aprev, w, h);
  const VecSet vf = vecs_in(cand_fwd, n, w, h);
  VecSet vb;
  if (Aprev)
    vb = vecs_in(cand_bwd, n, w, h);
  CandSet cf{ &vf, true, refine_ != 0 }, cb{ &vb, toward_b != 0, refine_ != 0 };
  Field4 F, B;
  if (g_fast)
    g_fast->fields(Aprev ? &prv : nullptr, cur, nxt, cf, cb, bias, true, g_cfg, F, B);
  else
    fields(Aprev ? &prv : nullptr, cur, nxt, cf, cb, bias, true, g_cfg, F, B);
  field_out(F, fwd);
  field_out(B, bwd);
}

// Stage timings of the current path (sc_set_fast) as the plugin runs them, best of reps, in ms: y[0..3] the luma of
// pictures c-1 .. c+2, cb / cr the chroma (4:4:4) of c and c+1 (W x H each). ms: [0] ME picture A, [1] ME picture B,
// [2] phase-correlation ingest of one picture, [3] phase-correlation candidates of one pair, [4] the fields of one
// centre picture (refinement to local fallback, both sides), [5] of these the assignment, [6] of these the local
// fallback, [7] the pair statistic, [8] the build of one frame.
SC_API void sc_profile(const float *const *y, const float *const *cb, const float *const *cr, int W, int H, int reps,
  double *ms)
{
  using clk = std::chrono::steady_clock;
  auto best = [&](int k, const std::function<void()> &fn) {
    double b = 1e30;
    for (int r = 0; r < reps; r++)
    {
      const auto t = clk::now();
      fn();
      b = std::min(b, std::chrono::duration<double, std::milli>(clk::now() - t).count());
    }
    ms[k] = b;
  };
  FastEngine *e = g_fast.get();
  float costs[CFG_MAX_COSTS];
  Plane Y[4], A[4], Bp[4];
  PcorrPicture pc[4];
  for (int i = 0; i < 4; i++)
  {
    Y[i] = plane_in(y[i], W, H);
    A[i] = e ? e->me_picture(Y[i], MeFir::A, g_cfg) : me_picture(Y[i], MeFir::A, g_cfg);
    Bp[i] = e ? e->me_picture(Y[i], MeFir::B, g_cfg) : me_picture(Y[i], MeFir::B, g_cfg);
    pc[i] = e ? e->pcorr_ingest(Bp[i], g_cfg) : pcorr_ingest(Bp[i], g_cfg);
  }
  best(0, [&] { Plane a = e ? e->me_picture(Y[1], MeFir::A, g_cfg) : me_picture(Y[1], MeFir::A, g_cfg); });
  best(1, [&] { Plane a = e ? e->me_picture(Y[1], MeFir::B, g_cfg) : me_picture(Y[1], MeFir::B, g_cfg); });
  best(2, [&] { PcorrPicture p = e ? e->pcorr_ingest(Bp[1], g_cfg) : pcorr_ingest(Bp[1], g_cfg); });
  VecSet cand[3];
  pcorr_costs(g_cfg, A[0].w, A[0].h, costs);
  for (int q = 0; q < 3; q++)
    cand[q] = e ? e->pcorr_candidates(pc[q], pc[q + 1], g_cfg) : pcorr_candidates(pc[q], pc[q + 1], g_cfg);
  best(3, [&] { VecSet v = e ? e->pcorr_candidates(pc[1], pc[2], g_cfg) : pcorr_candidates(pc[1], pc[2], g_cfg); });
  // fields of c (= 1) and c + 1
  Field4 F[2], B[2];
  for (int c = 1; c <= 2; c++)
  {
    CandSet cf{ &cand[c], true, true }, cbk{ &cand[c - 1], false, true };
    if (e)
      e->fields(&A[c - 1], A[c], A[c + 1], cf, cbk, costs, true, g_cfg, F[c - 1], B[c - 1]);
    else
      fields(&A[c - 1], A[c], A[c + 1], cf, cbk, costs, true, g_cfg, F[c - 1], B[c - 1]);
  }
  best(4, [&] {
    CandSet cf{ &cand[1], true, true }, cbk{ &cand[0], false, true };
    Field4 f, b;
    if (e)
      e->fields(&A[0], A[1], A[2], cf, cbk, costs, true, g_cfg, f, b);
    else
      fields(&A[0], A[1], A[2], cf, cbk, costs, true, g_cfg, f, b);
  });
  {
    FieldsDebug dbg;
    CandSet cf{ &cand[1], true, true }, cbk{ &cand[0], false, true };
    Field4 f, b;
    fields(&A[0], A[1], A[2], cf, cbk, costs, true, g_cfg, f, b, &dbg);
    best(5, [&] {
      Field4 ff, bb;
      if (e)
        e->assign(dbg.rf, dbg.ef, dbg.rb, dbg.eb, costs, g_cfg.speed_change_cost, g_cfg.occlusion_error, g_cfg.occlusion_error_factor, g_cfg.vector_length_cost, ff, bb);
      else
        assign(dbg.rf, dbg.ef, dbg.rb, dbg.eb, costs, g_cfg.speed_change_cost, g_cfg.occlusion_error, g_cfg.occlusion_error_factor, g_cfg.vector_length_cost, ff, bb);
    });
    best(6, [&] {
      Plane p = e ? e->disorder_weight(f.c[0], f.c[1], &f.c[2], g_cfg) : disorder_weight(f.c[0], f.c[1], &f.c[2], g_cfg);
      Plane q = e ? e->disorder_weight(b.c[0], b.c[1], &b.c[2], g_cfg) : disorder_weight(b.c[0], b.c[1], &b.c[2], g_cfg);
    });
  }
  double dd = 0.5;
  best(7, [&] { dd = phase_pull_adjust(F[0], B[1], 0.5, 0.5, g_cfg); });
  Pic3 P, N;
  P.c[0] = Y[1];
  P.c[1] = plane_in(cb[0], W, H);
  P.c[2] = plane_in(cr[0], W, H);
  N.c[0] = Y[2];
  N.c[1] = plane_in(cb[1], W, H);
  N.c[2] = plane_in(cr[1], W, H);
  Pic3 o;
  best(8, [&] {
    if (e)
      e->build_into(P, N, F[0], B[1], (float)dd, g_cfg, o);
    else
      o = build(P, N, F[0], B[1], (float)dd, g_cfg);
  });
}

// the whole chain (ME pictures to local fallback) of centre picture c from the luma of c-1 (or null: single side), c, c+1 (W x H): phase
// correlation candidates, refinement, errors, assignment, local fallback; fwd / bwd 4 planes of the ME grid
SC_API void sc_fields_luma(const float *yprev, const float *ycur, const float *ynext, int W, int H, float *fwd, float *bwd)
{
  float costs[CFG_MAX_COSTS];
  if (g_fast)
  {
    FastEngine &e = *g_fast;
    const Plane Yc = plane_in(ycur, W, H), Yn = plane_in(ynext, W, H);
    const Plane Ac = e.me_picture(Yc, MeFir::A, g_cfg), An = e.me_picture(Yn, MeFir::A, g_cfg);
    const PcorrPicture pc = e.pcorr_ingest(e.me_picture(Yc, MeFir::B, g_cfg), g_cfg), pn = e.pcorr_ingest(e.me_picture(Yn, MeFir::B, g_cfg), g_cfg);
    const VecSet vf = e.pcorr_candidates(pc, pn, g_cfg);
    pcorr_costs(g_cfg, Ac.w, Ac.h, costs);
    VecSet vb;
    Plane Ap;
    if (yprev)
    {
      const Plane Yp = plane_in(yprev, W, H);
      Ap = e.me_picture(Yp, MeFir::A, g_cfg);
      vb = e.pcorr_candidates(e.pcorr_ingest(e.me_picture(Yp, MeFir::B, g_cfg), g_cfg), pc, g_cfg);
    }
    CandSet cf{ &vf, true, true }, cb{ &vb, false, true };
    Field4 F, B;
    e.fields(yprev ? &Ap : nullptr, Ac, An, cf, cb, costs, true, g_cfg, F, B);
    field_out(F, fwd);
    field_out(B, bwd);
    return;
  }
  const Plane Yc = plane_in(ycur, W, H), Yn = plane_in(ynext, W, H);
  const Plane Ac = me_picture(Yc, MeFir::A, g_cfg), An = me_picture(Yn, MeFir::A, g_cfg);
  const PcorrPicture pc = pcorr_ingest(me_picture(Yc, MeFir::B, g_cfg), g_cfg), pn = pcorr_ingest(me_picture(Yn, MeFir::B, g_cfg), g_cfg);
  const VecSet vf = pcorr_candidates(pc, pn, g_cfg);
  pcorr_costs(g_cfg, Ac.w, Ac.h, costs);
  VecSet vb;
  Plane Ap;
  if (yprev)
  {
    const Plane Yp = plane_in(yprev, W, H);
    Ap = me_picture(Yp, MeFir::A, g_cfg);
    vb = pcorr_candidates(pcorr_ingest(me_picture(Yp, MeFir::B, g_cfg), g_cfg), pc, g_cfg);
  }
  CandSet cf{ &vf, true, true }, cb{ &vb, false, true };
  Field4 F, B;
  fields(yprev ? &Ap : nullptr, Ac, An, cf, cb, costs, true, g_cfg, F, B);
  field_out(F, fwd);
  field_out(B, bwd);
}
