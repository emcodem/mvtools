// splat_motion: the exact CPU path of SplatFps' refinement and forward / backward assignment, and the chain
// refinement -> match errors -> assignment -> local fallback that turns candidate sets into the two vector fields of
// a source picture (same exactness rules as splat_core.h).
//
// In own words:
// - Candidate sets: per side of the centre picture c, n vector fields on the ME grid; the zero vector is always added
//   as candidate n. Forward candidates point from c into c+1. Backward candidates of the phase correlation come from
//   the pair (c-1, c) and point from c-1 into c; the refinement turns them around (it measures the difference with
//   the other picture sampled at x - v and solves for the vector pointing from c).
// - Refinement, per candidate: the signed difference d = cur - other(x + v) (bilinear); its error_box mean absolute
//   value e; weights wt = exp(-e / refine_err_scale); central gradients gx, gy of the current picture (edge clamp,
//   / 2). Per pixel the weighted Lucas-Kanade terms
//     b1 = (gx^2 vx + gx gy vy + gx d) wt,  b2 = (gx gy vx + gy^2 vy + gy d) wt,
//     a = (gx^2 + refine_reg) wt,  c = (gy^2 + refine_reg) wt,  bxy = gx gy wt,
//   each summed over a refine_box x refine_box box (zero outside); the new vector solves [a bxy; bxy c] v = [b1 b2].
//   This is one Gauss-Newton step of the gradient method of B. D. Lucas, T. Kanade, "An iterative image registration
//   technique with an application to stereo vision", IJCAI 1981: the candidate plus the least-squares correction
//   G^-1 sum(g d) over a window, written as one solve (G v_new = G v + sum(g d)), with the gradient of the current
//   picture. refine_reg keeps the system solvable in flat areas (where it pulls the vector towards zero, since the
//   right-hand side does not contain it), and wt lets pixels the candidate matches badly count less. It turns the
//   integer phase-correlation peaks into sub-pixel vectors that follow the local motion.
// - Match errors: splat_me.h, for every refined candidate and the zero vector.
// - Assignment, per ME pixel. The cost of a candidate on one side is its match error + vector_length_cost
//   (|vx| + |vy|) + its candidate cost (bias: secondary_peak_cost for a block's further peaks, global_cost,
//   zero_cost). With speed_change_cost 0 (the default) each side takes its own lowest cost. Otherwise every pair
//   (forward candidate i, backward candidate p) costs the sum of both sides' costs plus
//   speed_change_cost (|vf_i + vb_p|_x + |vf_i + vb_p|_y) (both vectors point away from c, so a constant motion gives
//   vf = -vb and no penalty), and the cheapest pair wins (strictly lower cost replaces, in the order i outer, p
//   inner). Then the occlusion switch (splat::occlusion_switch): a side whose error is above occlusion_error and above
//   occlusion_error_factor times the other side's takes the other side's vector reversed; the errors stay as
//   measured. With one side only (first picture), the lowest cost of the forward candidates.
//   Named techniques: vector assignment by the minimum of a biased, spatially filtered displaced frame difference over
//   a menu of candidate vectors (G. A. Thomas, BBC RD 1987/11 section 4.2,
//   https://downloads.bbc.co.uk/rd/pubs/reports/1987-11.pdf, and US 4,890,160, BBC, 1986; there the bias grows with
//   the vector length (vector_length_cost), here it is mainly a constant per candidate); a forward and a backward
//   vector field per input picture (US 6,011,596, BBC, 1991); occlusion handling by one-sided matching: covered
//   background is only visible in the earlier picture and uncovered background only in the later one, so where one
//   side matches badly the side that does match supplies the motion for both (the one-sided choice of US 6,011,596,
//   also proposed in the 1987 report).
// - Local fallback: splat_me.h, on the assigned fields.
// - The defaults (refine_err_scale 15, refine_reg 0.001, refine_box 15, error_box 7, occlusion_error 6,
//   occlusion_error_factor 1.5, secondary_peak_cost 3) are those of splat::Config (splat_config.h).
#pragma once

#include "splat_core.h"

#include <vector>

namespace splat
{
  // candidate fields: 2 planes (vx, vy) per candidate, without the zero vector
  using VecSet = std::vector<Plane>;

  // refinement: refine every candidate of cands against other; neg: the candidates point from other into cur (they are
  // sampled at x - v and come back pointing from cur)
  VecSet refine(const Plane &cur, const Plane &other, const VecSet &cands, bool neg, const Config &cfg);
  // match errors of a set: n + 1 error planes (zero vector last)
  std::vector<Plane> errors(const Plane &cur, const Plane &other, const VecSet &vecs, const Config &cfg);

  // assignment: rf / rb n refined candidates of both sides, ef / eb n + 1 errors, bias n + 1 values (zero vector
  // last); lam = speed_change_cost, thresh / ratio = occlusion_error / occlusion_error_factor, lenc =
  // vector_length_cost. Fills channels 0-2 of fwd and bwd; channel 3 = 0.
  void assign(const VecSet &rf, const std::vector<Plane> &ef, const VecSet &rb, const std::vector<Plane> &eb,
    const float *bias, float lam, float thresh, float ratio, float lenc, Field4 &fwd, Field4 &bwd);
  // assignment with the forward side only: biased argmin; bwd = (0, 0, NO_SIDE_ERROR, 0)
  void assign_single(const VecSet &rf, const std::vector<Plane> &ef, const float *bias, float lenc, Field4 &fwd,
    Field4 &bwd);

  struct CandSet
  {
    const VecSet *vecs = nullptr;
    bool toward = true; // vectors point from c into the other picture (false: from the other picture into c)
    bool refine = true; // refine them
  };

  struct FieldsDebug
  {
    VecSet rf, rb;
    std::vector<Plane> ef, eb;
    Field4 assigned_fwd, assigned_bwd;
  };

  // refinement, match errors, assignment and local fallback for centre picture c: prv / cur / nxt the A pictures of
  // c-1, c, c+1 (prv null: single side, cb unused);
  // bias n + 1 values. Returns the forward and backward field (4 channels each).
  void fields(const Plane *prv, const Plane &cur, const Plane &nxt, const CandSet &cf, const CandSet &cb,
    const float *bias, bool local_fb, const Config &cfg, Field4 &fwd, Field4 &bwd, FieldsDebug *dbg = nullptr);
}
