// splat_me: the exact CPU path of SplatFps' stages around the motion vectors (same exactness rules as splat_core.h):
// ME pictures, match errors, local fallback, schedule, pair statistic (scene cut and temporal fallback).
//
// In own words:
// - ME pictures: the luma filtered by a long symmetric FIR (a Kaiser-windowed sinc, splat_config.h) and kept at every
//   scale-th sample, horizontal pass then vertical pass, the input extended by pad_x / pad_y samples of repeated edge
//   (so the ME grid is the internal grid / scale; at 1080p, scale 4 and border 16 x 20, ME pixel m sits at full-res
//   x = 4 m - 16, y = 4 m - 20). Picture A (sharper) serves the match errors, the refinement and the local fallback;
//   picture B (softer) is the phase-correlation input. Vectors are in ME pixels (scale full-res pixels).
// - Match errors: per candidate vector the displaced frame difference (DFD) cur(x) - other(x + v) on picture A
//   (bilinear sampling), its absolute value averaged over an error_box x error_box box (the spatial low-pass of the
//   error surface that makes the per-pixel choice robust to noise, as in G. A. Thomas, BBC RD 1987/11 section 4.2.1),
//   then an edge-aware filter: out = (1 - error_edge_weight) e + error_edge_weight m, where m is the smallest error
//   at the four diagonal shifts by error_edge_dist (error_edge_mode 1, shiftable windows), or the mean of the smaller
//   of the errors left / right and the smaller of those up / down (mode 0). Next to a moving edge one of these
//   neighbours lies inside the region that owns the vector, so the correct vector's error near the edge rises less
//   than with a plain linear filter, which would let the error of the other region leak across (the background next
//   to an object would be "pulled along" with it).
// - Local fallback: where the field is inconsistent (a vector far from the 3x3 Gaussian average of its neighbours)
//   the fallback weight rises: the box mean over fallback_box of fallback_strength * clamp((|v - G(v)| -
//   fallback_dead_zone) / fallback_range, 0, 1) (plus an optional match-error term), capped at fallback_cap. The
//   picture build uses this weight to blend the two pictures in place (no motion compensation) where the field is
//   too irregular to trust.
// - Pair statistic (splat_phase_pull.h): the mean capped match error m of the two fields that meet at an output
//   frame. A scene cut (m >= scene_cut_error) repeats the nearer picture; the temporal fallback (pull_strength > 0,
//   off by default) moves the output phase d towards the nearer source picture when m is high. At the limit the
//   output is a repeat of the nearer picture, which judders but shows no interpolation artefacts.
// - Schedule: output frame k sits at source time k * rate_in / rate_out (exact fractions); phases within
//   1 / copy_snap of a source picture are copies.
// - The defaults are those of splat::Config (splat_config.h): at 1080p ME scale 4, border 16 x 20, error box 7,
//   fallback box 31, copy_snap 128; other sizes and the filter arguments change them.
#pragma once

#include "splat_core.h"
#include "splat_phase_pull.h"

#include <cstdint>
#include <vector>

namespace splat
{
  enum class MeFir { A, B };
  // luma W x H -> ME picture MW x MH (filter A or B of cfg, decimation by cfg.scale, border cfg.pad_x / pad_y)
  Plane me_picture(const Plane &y, MeFir fir, const Config &cfg);
  // every scale-th output of the symmetric FIR h (centre first, taps per side) along one direction; output j reads
  // around scale j - overscan, reads clamped to the picture
  Plane me_down(const Plane &img, const float *h, int taps, int overscan, int scale, bool horizontal);

  // bilinear sample at texel-centre coordinates (x, y), edge clamp
  float tex_bilinear(const Plane &img, float x, float y);
  // error_box mean |cur - other(x + v)|, edge-aware filtered (error_edge_weight, error_edge_dist)
  Plane error_of(const Plane &cur, const Plane &other, const Plane &vx, const Plane &vy, const Config &cfg);
  // zero-padded size x size box sum (vertical then horizontal), optional |.| on the input, / size^2, cap
  Plane box(const Plane &a, int size, bool take_abs, bool average, float max_val = -1.0f);
  // edge-aware error filter: (1 - wq) e + wq / 2 (min of e dist px left / right + min of e dist px up / down), edge
  // clamp
  Plane opposite_min_filter(const Plane &e, float wq, int dist, int mode);

  // local-fallback weight (channel 3) of one field: box mean of the local vector inconsistency, capped
  // err: the assigned match error of the field (or null: no error term)
  Plane disorder_weight(const Plane &vx, const Plane &vy, const Plane *err, const Config &cfg);

  // sums in float64 in a fixed order: rows left to right, then the rows top to bottom
  double sum64(const Plane &a, int x0 = 0, int y0 = 0, int x1 = -1, int y1 = -1);

  // pair-statistic diagnostics: m mean capped error, p (unused, 0), f smooth step of m, g2 fallback amount, d adjusted phase
  struct PullInfo { double m, p, f, g2, d; };
  // per-row error sums of the two fields (the input of phase_pull_from_rows in splat_phase_pull.h)
  PullSums phase_pull_rows(const Field4 &fP, const Field4 &fN, const Config &cfg);
  // fP: forward field of P (vx, vy, err, ...), fN: backward field of N; returns the adjusted phase
  double phase_pull_adjust(const Field4 &fP, const Field4 &fN, double d, double strength, const Config &cfg,
    PullInfo *info = nullptr);

  // schedule: one output frame k between source pictures p (P) and p + 1 (N). Built needs source p + 2 as well (the
  // backward field of N is assigned together with the forward field of N, which looks at N + 1); frames near the end
  // of the clip that lack it are EosPlaceholder (the caller shows the nearest source picture).
  struct PlanEntry
  {
    int k, p;
    int64_t dnum, dden; // phase d = dnum / dden in [0, 1]
    enum Kind { CopyP, CopyN, Built, EosPlaceholder } kind;
  };
  // schedule of all output frames: ceil(n rout / rin) entries.
  // rates as fractions rin_n / rin_d, rout_n / rout_d; n source pictures
  // copy_snap: phases within 1 / copy_snap of a source picture are copies (0: only exact ones)
  std::vector<PlanEntry> plan(int n, int64_t rin_n, int64_t rin_d, int64_t rout_n, int64_t rout_d, int copy_snap = 128);
  // temporal-fallback strength (cfg.pull_strength; the rates are not used)
  double phase_pull_strength(int64_t rin_n, int64_t rin_d, int64_t rout_n, int64_t rout_d, const Config &cfg);
}
