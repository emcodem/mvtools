// splat_motion: the exact CPU path of the refinement (Lucas-Kanade step), the forward / backward assignment and the
// chain refinement -> local fallback; see splat_motion.h.
#include "splat_motion.h"
#include "splat_const.h"
#include "splat_me.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

namespace splat
{
  // fn(i) for i < n on a few threads (each call writes its own outputs; the results do not depend on the split)
  template <typename F>
  static void par_for(int n, F fn)
  {
    const int nt = std::max(1, std::min(n, (int)std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int t = 0; t < nt; t++)
      th.emplace_back([&, t] {
        for (int i = t; i < n; i += nt)
          fn(i);
      });
    for (auto &x : th)
      x.join();
  }

  // -------------------------------------------------------------------------------------------------------- refinement
  // one weighted Lucas-Kanade step for one candidate field (see splat_motion.h): DFD, its error_box mean |.| for the
  // weight, the 5 weighted terms per pixel, refine_box sums, 2 x 2 solve per pixel
  static void refine_one(const Plane &cur, const Plane &other, const Plane &vx, const Plane &vy, bool neg,
    const Config &cfg, Plane &ox, Plane &oy)
  {
    const int w = cur.w, h = cur.h;
    const float s = neg ? -1.0f : 1.0f;
    Plane d(w, h);
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++)
        d.at(x, y) = cur.at(x, y) - tex_bilinear(other, (float)x + s * vx.at(x, y), (float)y + s * vy.at(x, y));
    const Plane err = box(d, cfg.error_box, true, true);
    Plane T[5];
    for (auto &t : T)
      t = Plane(w, h);
    const float eps = cfg.refine_reg;
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++)
      {
        const float gx = (cur.at(std::min(x + 1, w - 1), y) - cur.at(std::max(x - 1, 0), y)) / 2.0f;
        const float gy = (cur.at(x, std::min(y + 1, h - 1)) - cur.at(x, std::max(y - 1, 0))) / 2.0f;
        const float ixx = gx * gx, iyy = gy * gy, ixy = gx * gy;
        const float wt = exp32(-err.at(x, y) / cfg.refine_err_scale);
        const float vxx = neg ? -vx.at(x, y) : vx.at(x, y), vyy = neg ? -vy.at(x, y) : vy.at(x, y);
        const float dd = d.at(x, y);
        float b1 = ixx * vxx + ixy * vyy;
        b1 = b1 + gx * dd;
        float b2 = ixy * vxx + iyy * vyy;
        b2 = b2 + gy * dd;
        T[0].at(x, y) = b1 * wt;
        T[1].at(x, y) = b2 * wt;
        T[2].at(x, y) = (ixx + eps) * wt;
        T[3].at(x, y) = (iyy + eps) * wt;
        T[4].at(x, y) = ixy * wt;
      }
    Plane B[5];
    for (int k = 0; k < 5; k++)
      B[k] = box(T[k], cfg.refine_box, false, false);
    ox = Plane(w, h);
    oy = Plane(w, h);
    for (size_t i = 0; i < ox.d.size(); i++)
    {
      const float b1 = B[0].d[i], b2 = B[1].d[i], a = B[2].d[i], c = B[3].d[i], bxy = B[4].d[i];
      const float det = a * c - bxy * bxy;
      ox.d[i] = (c * b1 - bxy * b2) / det;
      oy.d[i] = (a * b2 - bxy * b1) / det;
    }
  }

  VecSet refine(const Plane &cur, const Plane &other, const VecSet &cands, bool neg, const Config &cfg)
  {
    const int n = (int)cands.size() / 2;
    VecSet out(cands.size());
    par_for(n, [&](int i) { refine_one(cur, other, cands[2 * i], cands[2 * i + 1], neg, cfg, out[2 * i], out[2 * i + 1]); });
    return out;
  }

  std::vector<Plane> errors(const Plane &cur, const Plane &other, const VecSet &vecs, const Config &cfg)
  {
    const int n = (int)vecs.size() / 2;
    std::vector<Plane> e(n + 1);
    const Plane z(cur.w, cur.h);
    par_for(n + 1, [&](int i) { e[i] = i < n ? error_of(cur, other, vecs[2 * i], vecs[2 * i + 1], cfg) : error_of(cur, other, z, z, cfg); });
    return e;
  }

  // -------------------------------------------------------------------------------------------------------- assignment
  void assign(const VecSet &rf, const std::vector<Plane> &ef, const VecSet &rb, const std::vector<Plane> &eb,
    const float *bias, float lam, float thresh, float ratio, float lenc, Field4 &fwd, Field4 &bwd)
  {
    const int n = (int)ef.size(), w = ef[0].w, h = ef[0].h;
    for (int c = 0; c < 4; c++)
    {
      fwd.c[c] = Plane(w, h);
      bwd.c[c] = Plane(w, h);
    }
    for (size_t k = 0; k < (size_t)w * h; k++)
    {
      float vx = 0, vy = 0, e = 0, ovx = 0, ovy = 0, oe = 0;
      if (lam == 0.0f)
      {
        // without a speed-change cost the two sides are independent: each takes its own lowest cost
        float bf = std::numeric_limits<float>::infinity(), bb = bf;
        for (int i = 0; i < n; i++)
        {
          const float cx = i < n - 1 ? rf[2 * i].d[k] : 0.0f, cy = i < n - 1 ? rf[2 * i + 1].d[k] : 0.0f;
          // vector length bias (G. A. Thomas, BBC RD 1987/11 section 4.2.2): equal matches go to the shorter vector
          const float c = (ef[i].d[k] + lenc * (std::fabs(cx) + std::fabs(cy))) + bias[i];
          if (c < bf)
          {
            bf = c;
            vx = cx; vy = cy; e = ef[i].d[k];
          }
        }
        for (int p = 0; p < n; p++)
        {
          const float ox = p < n - 1 ? rb[2 * p].d[k] : 0.0f, oy = p < n - 1 ? rb[2 * p + 1].d[k] : 0.0f;
          const float c = (eb[p].d[k] + lenc * (std::fabs(ox) + std::fabs(oy))) + bias[p];
          if (c < bb)
          {
            bb = c;
            ovx = ox; ovy = oy; oe = eb[p].d[k];
          }
        }
      }
      else
      {
        float best = std::numeric_limits<float>::infinity();
        for (int i = 0; i < n; i++)
        {
          const float cx = i < n - 1 ? rf[2 * i].d[k] : 0.0f, cy = i < n - 1 ? rf[2 * i + 1].d[k] : 0.0f;
          const float ei = ef[i].d[k];
          const float li = ei + lenc * (std::fabs(cx) + std::fabs(cy));
          for (int p = 0; p < n; p++)
          {
            const float ox = p < n - 1 ? rb[2 * p].d[k] : 0.0f, oy = p < n - 1 ? rb[2 * p + 1].d[k] : 0.0f;
            const float ep = eb[p].d[k];
            const float lp = ep + lenc * (std::fabs(ox) + std::fabs(oy));
            const float t = li + lp;
            const float pen = lam * (std::fabs(cx + ox) + std::fabs(cy + oy));
            float c = t + pen;
            c = c + (bias[p] + bias[i]);
            if (c < best)
            {
              best = c;
              vx = cx; vy = cy; e = ei;
              ovx = ox; ovy = oy; oe = ep;
            }
          }
        }
      }
      occlusion_switch(vx, vy, e, ovx, ovy, oe, thresh, ratio);
      const float rvx = vx, rvy = vy, e2 = e, rox = ovx, roy = ovy, oe2 = oe;
      fwd.c[0].d[k] = rvx; fwd.c[1].d[k] = rvy; fwd.c[2].d[k] = e2;
      bwd.c[0].d[k] = rox; bwd.c[1].d[k] = roy; bwd.c[2].d[k] = oe2;
    }
  }

  void assign_single(const VecSet &rf, const std::vector<Plane> &ef, const float *bias, float lenc, Field4 &fwd,
    Field4 &bwd)
  {
    const int n = (int)ef.size(), w = ef[0].w, h = ef[0].h;
    for (int c = 0; c < 4; c++)
    {
      fwd.c[c] = Plane(w, h);
      bwd.c[c] = Plane(w, h);
    }
    for (size_t k = 0; k < (size_t)w * h; k++)
    {
      float best = std::numeric_limits<float>::infinity(), vx = 0, vy = 0, e = 0;
      for (int i = 0; i < n; i++)
      {
        const float sx = i < n - 1 ? rf[2 * i].d[k] : 0.0f, sy = i < n - 1 ? rf[2 * i + 1].d[k] : 0.0f;
        const float c = (ef[i].d[k] + lenc * (std::fabs(sx) + std::fabs(sy))) + bias[i];
        if (c < best)
        {
          best = c;
          vx = i < n - 1 ? rf[2 * i].d[k] : 0.0f;
          vy = i < n - 1 ? rf[2 * i + 1].d[k] : 0.0f;
          e = ef[i].d[k];
        }
      }
      fwd.c[0].d[k] = vx; fwd.c[1].d[k] = vy; fwd.c[2].d[k] = e;
      bwd.c[2].d[k] = NO_SIDE_ERROR;
    }
  }

  // -------------------------------------------------------------------------------------- refinement to local fallback
  // the candidates of one side as vectors pointing from cur: refined (turned around on the way when they point into
  // cur), or taken as they are / negated when refinement is off
  static VecSet side_vectors(const Plane &cur, const Plane &other, const CandSet &cs, const Config &cfg)
  {
    if (cs.refine)
      return refine(cur, other, *cs.vecs, !cs.toward, cfg);
    if (cs.toward)
      return *cs.vecs;
    VecSet v = *cs.vecs;
    for (auto &p : v)
      for (auto &x : p.d)
        x = -x;
    return v;
  }

  void fields(const Plane *prv, const Plane &cur, const Plane &nxt, const CandSet &cf, const CandSet &cb,
    const float *bias, bool local_fb, const Config &cfg, Field4 &fwd, Field4 &bwd, FieldsDebug *dbg)
  {
    VecSet rf = side_vectors(cur, nxt, cf, cfg), rb;
    std::vector<Plane> ef = errors(cur, nxt, rf, cfg), eb;
    if (!prv)
      assign_single(rf, ef, bias, cfg.vector_length_cost, fwd, bwd);
    else
    {
      rb = side_vectors(cur, *prv, cb, cfg);
      eb = errors(cur, *prv, rb, cfg);
      assign(rf, ef, rb, eb, bias, cfg.speed_change_cost, cfg.occlusion_error, cfg.occlusion_error_factor, cfg.vector_length_cost, fwd, bwd);
    }
    if (dbg)
    {
      dbg->assigned_fwd = fwd;
      dbg->assigned_bwd = bwd;
    }
    if (local_fb)
    {
      fwd.c[3] = disorder_weight(fwd.c[0], fwd.c[1], &fwd.c[2], cfg);
      bwd.c[3] = disorder_weight(bwd.c[0], bwd.c[1], &bwd.c[2], cfg);
    }
    if (dbg)
    {
      dbg->rf = std::move(rf);
      dbg->rb = std::move(rb);
      dbg->ef = std::move(ef);
      dbg->eb = std::move(eb);
    }
  }
}
