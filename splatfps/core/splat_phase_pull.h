// Error statistic of a picture pair and the temporal fallback: shared by the CPU path (splat_me.cpp) and the GPU
// (mvtools_cuda computes the row sums on the device, this finishes on the host), so both give the same double.
//
// m is the mean match error of the two fields that meet at the output frame (the forward field of P and the backward
// field of N), each error capped at error_window_cap, over the picture without a border of 1 / error_window_div of
// its width and height on each side (the border holds the padding and the parts that enter or leave the picture).
// - Scene cut: m >= scene_cut_error means the two pictures do not belong together; the nearer one is repeated.
// - Temporal fallback (pull_strength > 0): when motion compensation is likely to fail (m high), the output is built at
//   a time nearer to the nearer source picture, trading smooth motion for fewer interpolation artefacts. The amount g
//   is pull_strength times a smooth step of m from pull_error_lo to pull_error_hi; the phase d moves towards the
//   nearer picture by the fraction g.
// All sums are float64 in a defined order: each row left to right, then the rows top to bottom.
#pragma once

#include "splat_config.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace splat
{
  struct PullSums
  {
    int W = 0, H = 0;               // ME grid
    size_t nwin = 0;                // samples in the window
    std::vector<double> winP, winN; // per window row: sum of the capped errors of the forward field of P / backward of N
  };

  // window of the error statistic on a grid of n samples: [n / div, n - n / div)
  inline int error_window_lo(int n, const Config &c) { return n / c.error_window_div; }
  inline int error_window_hi(int n, const Config &c) { return n - n / c.error_window_div; }

  // the adjusted phase d' from the sums; info (or null): m, 0, the smooth step, g
  inline double phase_pull_from_rows(const PullSums &r, double d, double strength, const Config &cfg, double *info)
  {
    double sP = 0.0, sN = 0.0;
    for (size_t i = 0; i < r.winP.size(); i++)
    {
      sP = sP + r.winP[i];
      sN = sN + r.winN[i];
    }
    const double m = r.nwin ? (sP + sN) / (double)(2 * r.nwin) : 0.0;
    // scene cut: the pictures do not belong together; repeat the nearer one instead of building a picture
    if (cfg.scene_cut_error > 0.0f && m >= (double)cfg.scene_cut_error)
    {
      if (info)
      {
        info[0] = m; info[1] = 0.0; info[2] = 1.0; info[3] = 1.0;
      }
      return d <= 0.5 ? 0.0 : 1.0;
    }
    double f = 0.0;
    if (strength > 0.0)
    {
      const double t = std::min(1.0, std::max(0.0, (m - cfg.pull_error_lo) / (cfg.pull_error_hi - cfg.pull_error_lo)));
      f = t * t * (3.0 - 2.0 * t);
    }
    const double g = std::min(1.0, strength * f);
    if (info)
    {
      info[0] = m; info[1] = 0.0; info[2] = f; info[3] = g;
    }
    return d <= 0.5 ? (1 - g) * d : 1 - (1 - g) * (1 - d);
  }
}
