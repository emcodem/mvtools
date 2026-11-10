// splat_me: the exact CPU path of the stages around the motion vectors (ME pictures, match errors, local fallback,
// schedule, pair-statistic sums); see splat_me.h and splat_core.h.
#include "splat_me.h"
#include "splat_const.h"

#include <algorithm>
#include <cmath>

namespace splat
{
  // ------------------------------------------------------------------------------------------------------- ME pictures
  // every scale-th sample of the symmetric FIR h (centre first) along one direction; source sample under output j is
  // scale j - overscan, reads clamped to the picture
  Plane me_down(const Plane &img, const float *h, int taps, int overscan, int scale, bool horizontal)
  {
    const int n = horizontal ? img.w : img.h;
    const int nout = (n + 2 * overscan) / scale;
    Plane o(horizontal ? nout : img.w, horizontal ? img.h : nout);
    auto g = [&](int line, int pos) {
      pos = pos < 0 ? 0 : (pos >= n ? n - 1 : pos);
      return horizontal ? img.at(pos, line) : img.at(line, pos);
    };
    const int lines = horizontal ? img.h : img.w;
    for (int line = 0; line < lines; line++)
      for (int j = 0; j < nout; j++)
      {
        const int pos = j * scale - overscan;
        float acc = h[0] * g(line, pos);
        for (int i = 1; i < taps; i++)
          acc = acc + h[i] * (g(line, pos - i) + g(line, pos + i));
        if (horizontal)
          o.at(j, line) = acc;
        else
          o.at(line, j) = acc;
      }
    return o;
  }

  Plane me_picture(const Plane &y, MeFir fir, const Config &cfg)
  {
    const float *h = fir == MeFir::A ? cfg.fir_a : cfg.fir_b;
    const int taps = fir == MeFir::A ? cfg.fir_a_n : cfg.fir_b_n;
    return me_down(me_down(y, h, taps, cfg.pad_x, cfg.scale, true), h, taps, cfg.pad_y, cfg.scale, false);
  }

  // ------------------------------------------------------------------------------------------------------ match errors
  float tex_bilinear(const Plane &img, float x, float y)
  {
    const float i0f = std::floor(x), j0f = std::floor(y);
    const float a = x - i0f, b = y - j0f;
    const int i0 = (int)i0f, j0 = (int)j0f;
    const float t00 = img.clamped(i0, j0), t10 = img.clamped(i0 + 1, j0);
    const float t01 = img.clamped(i0, j0 + 1), t11 = img.clamped(i0 + 1, j0 + 1);
    float r = ((1.0f - a) * (1.0f - b)) * t00;
    r = r + (a * (1.0f - b)) * t10;
    r = r + ((1.0f - a) * b) * t01;
    r = r + (a * b) * t11;
    return r;
  }

  Plane box(const Plane &a, int size, bool take_abs, bool average, float max_val)
  {
    const int r = (size - 1) / 2, w = a.w, h = a.h;
    auto in = [&](int x, int y) -> float {
      if (y < 0 || y >= h)
        return 0.0f;
      const float v = a.at(x, y);
      return take_abs ? std::fabs(v) : v;
    };
    Plane v(w, h);
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++)
      {
        float s = in(x, y - r);
        for (int i = 1; i < size; i++)
          s = s + in(x, y - r + i);
        v.at(x, y) = s;
      }
    Plane o(w, h);
    const float div = (float)(size * size);
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++)
      {
        auto vv = [&](int xx) { return (xx < 0 || xx >= w) ? 0.0f : v.at(xx, y); };
        float s = vv(x - r);
        for (int i = 1; i < size; i++)
          s = s + vv(x - r + i);
        if (average)
          s = s / div;
        if (max_val >= 0.0f)
          s = std::fmin(max_val, s);
        o.at(x, y) = s;
      }
    return o;
  }

  // Edge-aware error filter: out = (1 - wq) e + wq m. Next to a motion edge the plain box error of the right vector
  // is spoiled by the other side; m looks for a nearby error that is not.
  // mode 0: m = (min(e left, e right) + min(e up, e down)) / 2 at distance dist.
  // mode 1 (shiftable windows, as in area-based stereo matching: the box is moved so the sample sits at one of its
  // corners, and the best placement counts): m = min of e at the four diagonal shifts (+-dist, +-dist).
  Plane opposite_min_filter(const Plane &e, float wq, int dist, int mode)
  {
    const int w = e.w, h = e.h;
    Plane o(w, h);
    const float half = wq * 0.5f, keep = 1.0f - wq;
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++)
      {
        const int xl = std::max(x - dist, 0), xr = std::min(x + dist, w - 1);
        const int yu = std::max(y - dist, 0), yd = std::min(y + dist, h - 1);
        if (mode == 1)
        {
          const float m = std::fmin(std::fmin(e.at(xl, yu), e.at(xr, yu)), std::fmin(e.at(xl, yd), e.at(xr, yd)));
          o.at(x, y) = keep * e.at(x, y) + wq * m;
          continue;
        }
        const float l = e.at(xl, y), r = e.at(xr, y);
        const float u = e.at(x, yu), d = e.at(x, yd);
        o.at(x, y) = keep * e.at(x, y) + half * (std::fmin(l, r) + std::fmin(u, d));
      }
    return o;
  }

  // match error of one candidate field: DFD against other, error_box mean of |DFD|, edge-aware filter
  Plane error_of(const Plane &cur, const Plane &other, const Plane &vx, const Plane &vy, const Config &cfg)
  {
    Plane dfd(cur.w, cur.h);
    for (int y = 0; y < cur.h; y++)
      for (int x = 0; x < cur.w; x++)
        dfd.at(x, y) = cur.at(x, y) - tex_bilinear(other, (float)x + vx.at(x, y), (float)y + vy.at(x, y));
    return opposite_min_filter(box(dfd, cfg.error_box, true, true), cfg.error_edge_weight, cfg.error_edge_dist,
      cfg.error_edge_mode);
  }

  // ---------------------------------------------------------------------------------------------------- local fallback
  double sum64(const Plane &a, int x0, int y0, int x1, int y1)
  {
    if (x1 < 0)
      x1 = a.w;
    if (y1 < 0)
      y1 = a.h;
    double t = 0.0;
    for (int y = y0; y < y1; y++)
    {
      double r = 0.0;
      for (int x = x0; x < x1; x++)
        r = r + (double)a.at(x, y);
      t = t + r;
    }
    return t;
  }

  // 3-tap Gaussian along one direction, edge repeat: centre, then the pair
  static Plane g3(const Plane &a, bool horizontal)
  {
    const float h0 = bits_to_float(DISORDER_SMOOTH_BITS[0]), h1 = bits_to_float(DISORDER_SMOOTH_BITS[1]);
    Plane o(a.w, a.h);
    for (int y = 0; y < a.h; y++)
      for (int x = 0; x < a.w; x++)
      {
        float acc = h0 * a.at(x, y);
        const float pair = horizontal ? a.clamped(x - 1, y) + a.clamped(x + 1, y) : a.clamped(x, y - 1) + a.clamped(x, y + 1);
        o.at(x, y) = acc + h1 * pair;
      }
    return o;
  }

  // Local fallback weight of one field. Where the vectors around a point disagree (a vector far from the
  // Gaussian-smoothed vectors around it), motion compensation is unreliable there, and the in-place blend of the two
  // pictures gets more weight in the build.
  // Per ME pixel: fallback_strength times a ramp of the inconsistency from fallback_dead_zone (0) over fallback_range
  // (1), optionally plus an error term (where even the assigned vectors match badly: water, leaves, non-rigid motion);
  // then the box mean over fallback_box, capped at fallback_cap. Weights are in units of one full splat.
  Plane disorder_weight(const Plane &vx, const Plane &vy, const Plane *err, const Config &cfg)
  {
    const Plane sx = g3(g3(vx, false), true), sy = g3(g3(vy, false), true);
    Plane d(vx.w, vx.h);
    for (size_t i = 0; i < d.d.size(); i++)
    {
      const float ex = vx.d[i] - sx.d[i], ey = vy.d[i] - sy.d[i];
      const float dist = std::sqrt(ex * ex + ey * ey);
      float v = cfg.fallback_strength * std::fmin(std::fmax((dist - cfg.fallback_dead_zone) / cfg.fallback_range, 0.0f), 1.0f);
      if (err && cfg.fallback_err_gain > 0.0f)
        v = v + cfg.fallback_err_gain *
          std::fmin(std::fmax((err->d[i] - cfg.fallback_err_lo) / cfg.fallback_err_range, 0.0f), 1.0f);
      d.d[i] = v;
    }
    return box(d, cfg.fallback_box, false, true, cfg.fallback_cap);
  }

  // ---------------------------------------------------------------------------------------------------- pair statistic
  // per window row: the sums of the capped match errors of both fields (splat_phase_pull.h)
  PullSums phase_pull_rows(const Field4 &fP, const Field4 &fN, const Config &cfg)
  {
    PullSums r;
    const int W = fP.c[0].w, H = fP.c[0].h;
    const int y0 = error_window_lo(H, cfg), y1 = error_window_hi(H, cfg);
    const int x0 = error_window_lo(W, cfg), x1 = error_window_hi(W, cfg);
    const float cap = cfg.error_window_cap;
    r.W = W;
    r.H = H;
    r.nwin = (size_t)(y1 - y0) * (x1 - x0);
    for (int y = y0; y < y1; y++)
    {
      double rP = 0.0, rN = 0.0;
      for (int x = x0; x < x1; x++)
      {
        rP = rP + (double)std::fmin(cap, fP.c[2].at(x, y));
        rN = rN + (double)std::fmin(cap, fN.c[2].at(x, y));
      }
      r.winP.push_back(rP);
      r.winN.push_back(rN);
    }
    return r;
  }

  double phase_pull_adjust(const Field4 &fP, const Field4 &fN, double d, double strength, const Config &cfg,
    PullInfo *info)
  {
    double inf[4];
    const double dd = phase_pull_from_rows(phase_pull_rows(fP, fN, cfg), d, strength, cfg, inf);
    if (info)
      *info = { inf[0], inf[1], inf[2], inf[3], dd };
    return dd;
  }

  // ---------------------------------------------------------------------------------------------------------- schedule
  std::vector<PlanEntry> plan(int n, int64_t rin_n, int64_t rin_d, int64_t rout_n, int64_t rout_d, int copy_snap)
  {
    const int64_t cs = copy_snap;
    std::vector<PlanEntry> out;
    // frame count = ceil(n * rout / rin)
    const int64_t cn = (int64_t)n * rout_n * rin_d, cd = rout_d * rin_n;
    const int64_t count = (cn + cd - 1) / cd;
    const int64_t den = rin_d * rout_n; // t = k * rin_n * rout_d / den
    // per frame: p = floor(t), d = t - p as a fraction; d <= 1/cs copies P, d >= (cs - 1)/cs copies N (cs = 0: only
    // d = 0 copies)
    for (int64_t k = 0; k < count; k++)
    {
      const int64_t num = k * rin_n * rout_d;
      const int64_t p = num / den;
      int64_t dn = num - p * den, dd = den;
      PlanEntry e;
      e.k = (int)k;
      e.p = (int)p;
      if (cs ? cs * dn <= dd : dn == 0)
      {
        e.kind = PlanEntry::CopyP;
        dn = 0;
        dd = 1;
      }
      else if (cs && cs * dn >= (cs - 1) * dd)
      {
        e.kind = PlanEntry::CopyN;
        dn = 1;
        dd = 1;
      }
      else if (p + 2 <= n - 1)
        e.kind = PlanEntry::Built;
      else
        e.kind = PlanEntry::EosPlaceholder;
      if (e.kind == PlanEntry::CopyN && p + 1 > n - 1)
        e.kind = PlanEntry::EosPlaceholder;
      e.dnum = dn;
      e.dden = dd;
      out.push_back(e);
    }
    return out;
  }

  double phase_pull_strength(int64_t, int64_t, int64_t, int64_t, const Config &cfg) { return cfg.pull_strength; }
}
