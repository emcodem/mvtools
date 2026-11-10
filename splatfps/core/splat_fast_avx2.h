// splat_fast_avx2: the 8-wide (float) / 4-wide (double) kernels of the fast path, compiled for AVX2 + FMA in
// splat_fast_avx2.cpp only and called by splat_fast.cpp after the run-time CPU check.
//
// Every kernel evaluates the same float32 operations in the same order as the matching exact function (the lanes are
// neighbouring pixels, or neighbouring lines of an FFT), so its results are those of the exact path, with one
// exception: exp, atan2, sin and cos are double-precision polynomials (Taylor series after an exact range reduction,
// relative error below 1e-15) rounded to float, which equals the float32 rounding of the C library's double result
// except when that double lies within about 1e-15 of a rounding boundary.
//
// Interfaces are plain pointers (planes row-major, rows contiguous) and row ranges, so the callers split the work over
// threads. The file uses no templates or inline functions shared with other files (code compiled for AVX2 must not be
// picked by the linker for the portable files).
#pragma once

#include "../third_party/pocketfft/fftp_pow2.h"

#include <cstddef>

namespace splat
{
  namespace avx2
  {
    // ---- ME pictures: every sc-th output of the symmetric FIR h (centre first, taps per side incl. centre, at most 64),
    // reads clamped. Horizontal pass over the image rows [y0, y1): output j of a row reads around sc j - overscan;
    // out: nout x h
    void down_h(const float *img, int w, int y0, int y1, const float *h, int taps, int overscan, int sc, float *out,
      int nout);
    // vertical pass, output rows [j0, j1) of a plane w wide with n input rows
    void down_v(const float *img, int w, int n, int j0, int j1, const float *h, int taps, int overscan, int sc, float *out);

    // ---- separable symmetric FIR, zero outside (sep_zero of splat_pcorr): one direction, output rows [y0, y1)
    void fir_zero_v(const float *a, int w, int hgt, int y0, int y1, const float *h, int taps, float *out);
    void fir_zero_h(const float *a, int w, int y0, int y1, const float *h, int taps, float *out);

    // Arguments named ring*: the plane they belong to is a ring buffer of that many rows (row y stored at y mod ring;
    // 0: a whole plane), so a stage streamed from top to bottom keeps only the rows in flight.

    // ---- box sums (box of splat_me): vertical pass (optional |.| on the input), horizontal pass (optional / size^2,
    // optional cap when max_val >= 0); zero outside, output rows [y0, y1)
    void box_v(const float *a, int w, int hgt, int y0, int y1, int size, bool take_abs, float *out, int ring_in = 0,
      int ring_out = 0);
    void box_h(const float *v, int w, int y0, int y1, int size, bool average, float max_val, float *out, int ring_in = 0,
      int ring_out = 0);

    // ---- refinement / match errors: displaced difference cur(x) - other(x + s v) with tex_bilinear sampling, rows [y0, y1)
    void dfd(const float *cur, const float *other, int w, int hgt, const float *vx, const float *vy, float s, int y0,
      int y1, float *out, int ring_out = 0);
    // refinement: per-pixel Lucas-Kanade terms T[0..4] (b1 wt, b2 wt, (gx^2 + eps) wt, (gy^2 + eps) wt, gx gy wt) with
    // wt = exp32(-err / wscale), rows [y0, y1); err, d and T in ring buffers of ring rows
    void refine_terms(const float *cur, const float *err, const float *d, const float *vx, const float *vy, bool neg,
      int w, int hgt, int y0, int y1, float eps, float wscale, float *const *T, int ring = 0);
    // refinement: 2 x 2 solve of the box-summed terms B[0..4] (ring buffers of ring rows) for the rows [y0, y1)
    void refine_solve(const float *const *B, int w, int y0, int y1, float *ox, float *oy, int ring = 0);
    // match errors: edge-aware filter (opposite_min_filter of splat_me, neighbours at distance dist), rows [y0, y1)
    void opposite_min(const float *e, int w, int hgt, int y0, int y1, float wq, int dist, int mode, float *out, int ring_in = 0);

    // ---- assignment of the samples [k0, k1): rf / rb 2 (n - 1) vector planes, ef / eb n error planes (zero
    // vector last), bias n values; out: fwd vx, vy, err, bwd vx, vy, err
    void assign(const float *const *rf, const float *const *ef, const float *const *rb, const float *const *eb, int n,
      const float *bias, float lam, float thresh, float ratio, float lenc, size_t k0, size_t k1, float *const *out);

    // ---- phase correlation: lines [g0, g1) of an S x S complex tile (separate re / im planes), 8 lines per group: rows (along x)
    // when rows, else columns; the pocketfft float transform of each line, scaled by fct when fct != 1
    void fft_lines_f(float *re, float *im, int S, bool rows, int g0, int g1, bool fwd, const fftp::Plan &plan,
      const fftp::cpx<float> *tw, int ntw, float fct);

    // ---- picture build
    // one row y of a side's per-sample values of a pyramid level (build_level of splat_core): the projection field G
    // (4 planes gw x gh) at the full-resolution positions (x stride, y stride) through the x (1 << sh) bilinear
    // up-sampling of field_upsample (full_res), or read at (x, y) (the own field of a coarse level); vectors / stride;
    // st = channel 3, plus mc where |v| <= zero_len (zero = 1)
    void level_side(const float *const *G, int gw, int gh, bool full_res, int stride, int sh, int w, int y, float zero_len,
      float *vx, float *vy, float *mc, float *st, unsigned char *zero);
    // one row of the accumulator start: per sample P[c] stP + N[c] stN (c = 0 .. 2), stP + stN
    void level_acc_row(const float *const *P, const float *const *N, const float *stP, const float *stN, int w, float *acc);
    // pyramid reduce, vertical pass (even rows, edge clamp): rows [j0, j1) of the w x ceil(h / 2) result
    void pyr_reduce_v(const float *s, int w, int h, int j0, int j1, const float *taps, float *v);
    // horizontal pass (even columns): rows [y0, y1) of the ceil(w / 2) wide result
    void pyr_reduce_h(const float *v, int w, int y0, int y1, const float *taps, float *o);
    // pyramid expand of the low rows [y0, y1) into the output rows 2 y, 2 y + 1 (cropped to ow x oh), combined with base
    // (ow x oh): mode 0 o = e (base unused), 1 o = base - e, 2 o = base + e, 3 o = base + e clamped to [lo, hi].
    // crop (optional): only the cw x ch window at (cx, cy) of the output is made, written to crop (row pitch cpitch
    // floats) instead of o; low rows that feed no output row of the window are skipped
    struct Crop
    {
      float *dst;
      int pitch, cx, cy, cw, ch;
    };
    void pyr_expand(const float *l, int lw, int lh, int y0, int y1, const float *ka, const float *kb, const float *kc,
      float *o, int ow, int oh, const float *base, int mode, float lo, float hi, const Crop *crop = nullptr);
  }
}
