// splat_fast: the fast CPU path of SplatFps; see splat_fast.h. This file splits the stages over the thread pool and
// calls the AVX2 kernels (splat_fast_avx2.cpp) after the run-time CPU check; its own arithmetic follows the exact
// functions operation by operation (same build rules as the exact path: no contraction).
#include "splat_fast.h"
#include "splat_const.h"
#include "splat_par.h"
#include "../third_party/pocketfft/fftp_pow2.h"
#if SPLAT_HAVE_AVX2
#include "splat_fast_avx2.h"
#define AVX2_CALL(...) __VA_ARGS__
#else
#define AVX2_CALL(...) ((void)0)
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#define SPLAT_X86 1
#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace splat
{
  namespace
  {
    // ---- portable row kernels of the picture build (the exact pyr_reduce / pyr_expand restricted to rows)
    void pyr_reduce_v_rows(const Plane &s, int j0, int j1, const float *D, Plane &v)
    {
      for (int j = j0; j < j1; j++)
        for (int x = 0; x < s.w; x++)
        {
          const int y = 2 * j;
          float r = D[0] * s.clamped(x, y);
          r = r + D[1] * (s.clamped(x, y - 1) + s.clamped(x, y + 1));
          r = r + D[2] * (s.clamped(x, y - 2) + s.clamped(x, y + 2));
          r = r + D[3] * (s.clamped(x, y - 3) + s.clamped(x, y + 3));
          v.at(x, j) = r;
        }
    }

    void pyr_reduce_h_rows(const Plane &v, int y0, int y1, const float *D, Plane &o)
    {
      for (int y = y0; y < y1; y++)
        for (int i = 0; i < o.w; i++)
        {
          const int x = 2 * i;
          float r = D[0] * v.clamped(x, y);
          r = r + D[1] * (v.clamped(x - 1, y) + v.clamped(x + 1, y));
          r = r + D[2] * (v.clamped(x - 2, y) + v.clamped(x + 2, y));
          r = r + D[3] * (v.clamped(x - 3, y) + v.clamped(x + 3, y));
          o.at(i, y) = r;
        }
    }

    void pyr_expand_rows(const Plane &l, int y0, int y1, Plane &o)
    {
      const float *A = PYR_EXPAND_A, *B = PYR_EXPAND_B, *Ck = PYR_EXPAND_C;
      const int ow = o.w, oh = o.h;
      for (int y = y0; y < y1; y++)
        for (int x = 0; x < l.w; x++)
        {
          auto C = [&](int dy, int dx) { return l.clamped(x + dx, y + dy); };
          const float c = C(0, 0), n = C(-1, 0), wst = C(0, -1), e = C(0, 1), s = C(1, 0);
          const float nw = C(-1, -1), ne = C(-1, 1), sw = C(1, -1), se = C(1, 1);
          const float c02 = C(0, 2), cm12 = C(-1, 2), c12 = C(1, 2), c20 = C(2, 0), c2m1 = C(2, -1), c21 = C(2, 1),
            c22 = C(2, 2);
          const int ox = 2 * x, oy = 2 * y;
          if (oy < oh && ox < ow)
          {
            float r = A[0] * c;
            r = r + A[1] * (((n + wst) + e) + s);
            r = r + A[2] * (((nw + ne) + sw) + se);
            o.at(ox, oy) = r;
          }
          if (oy < oh && ox + 1 < ow)
          {
            float r = B[0] * (c + e);
            r = r + B[1] * (wst + c02);
            r = r + B[2] * (((n + ne) + s) + se);
            r = r + B[3] * (((nw + cm12) + sw) + c12);
            o.at(ox + 1, oy) = r;
          }
          if (oy + 1 < oh && ox < ow)
          {
            float r = B[0] * (c + s);
            r = r + B[1] * (n + c20);
            r = r + B[2] * (((wst + e) + sw) + se);
            r = r + B[3] * (((nw + ne) + c2m1) + c21);
            o.at(ox, oy + 1) = r;
          }
          if (oy + 1 < oh && ox + 1 < ow)
          {
            float r = Ck[0] * (((c + e) + s) + se);
            r = r + Ck[1] * ((((((((n + ne) + c20) + c21) + wst) + c02) + sw) + c12));
            r = r + Ck[2] * (((nw + cm12) + c2m1) + c22);
            o.at(ox + 1, oy + 1) = r;
          }
        }
    }

    // p becomes a w x h plane; its memory is kept when the size does not change (the contents are then undefined)
    void fit(Plane &p, int w, int h)
    {
      if (p.w != w || p.h != h || p.d.size() != (size_t)w * h)
      {
        p.w = w;
        p.h = h;
        p.d.resize((size_t)w * h);
      }
    }

    // scratch buffers of one thread, kept between tasks (no allocation per task)
    struct ThreadScratch
    {
      std::vector<float> b[20];
      float *get(int i, size_t n)
      {
        if (b[i].size() < n)
          b[i].resize(n);
        return b[i].data();
      }
    };
    thread_local ThreadScratch t_scratch;

    // channel A of a projection field (ME resolution) at full-resolution position (X, Y), as field_upsample of
    // splat_core computes it: bilinear between ME pixel (X / s, Y / s) and its right / lower neighbours (edge clamp)
    // at the fractions (X mod s) / s, (Y mod s) / s; vectors (vec) times s (s = 1 << sh, the ME scale)
    inline float up_s(const Plane &A, bool vec, int X, int Y, int sh)
    {
      const int msk = (1 << sh) - 1;
      const float fs = (float)(1 << sh);
      const int x = std::min(X >> sh, A.w - 1), y = std::min(Y >> sh, A.h - 1);
      const float tx = (float)(X & msk) / fs, ty = (float)(Y & msk) / fs;
      float a = A.clamped(x, y), b = A.clamped(x + 1, y), cc = A.clamped(x, y + 1), e = A.clamped(x + 1, y + 1);
      if (vec)
      {
        a = a * fs;
        b = b * fs;
        cc = cc * fs;
        e = e * fs;
      }
      const float l6 = a * (1.0f - tx) + b * tx;
      const float l18 = cc * (1.0f - tx) + e * tx;
      return l6 * (1.0f - ty) + l18 * ty;
    }
  }

  // float FFT plan and twiddles of one block size
  struct FftSize
  {
    int S = 0;
    fftp::Plan pf;
    std::vector<fftp::cpx<float>> twf;
  };

  struct FastEngine::Impl
  {
    Pool pool;
    bool simd;
    std::map<int, FftSize> ffts; // by block size (made on first use, from the calling thread only)

    Impl(int threads, bool use_simd) : pool(threads), simd(use_simd) {}

    // plan and twiddles of block size S; call outside the parallel sections (the map may grow)
    const FftSize &fft_of(int S)
    {
      auto it = ffts.find(S);
      if (it != ffts.end())
        return it->second;
      FftSize &f = ffts[S];
      f.S = S;
      f.pf = fftp::make_plan<float>(f.S, f.twf);
      return f;
    }

    // the phase-correlation stage's pieces on this engine (pool, AVX2 FIR and FFT; bit-identical to the exact ones)
    PcorrOps pcorr_ops()
    {
      PcorrOps o;
      o.par = [this](int n, const std::function<void(int)> &fn) { pool.run(n, fn); };
      if (!simd)
      {
        o.fir = [](const Plane &a, const float *h, int taps) { return splat::sep_zero(a, h, taps); };
        o.prepare = [](int) {};
        o.fft2 = pcorr_ops_exact().fft2;
        return o;
      }
      o.fir = [this](const Plane &a, const float *h, int taps) { return sep_zero(a, h, taps); };
      o.prepare = [this](int S) { fft_of(S); };
      o.fft2 = [this](float *re, float *im, int S, bool fwd, float fct) {
        const FftSize &f = ffts.find(S)->second;
        AVX2_CALL(avx2::fft_lines_f(re, im, S, true, 0, S / 8, fwd, f.pf, f.twf.data(), (int)f.twf.size(), fct));
        AVX2_CALL(avx2::fft_lines_f(re, im, S, false, 0, S / 8, fwd, f.pf, f.twf.data(), (int)f.twf.size(), fct));
        (void)f; (void)re; (void)im; (void)fwd; (void)fct;
      };
      return o;
    }

    // ---- generic pieces
    // separable symmetric FIR, zero outside, vertical pass then horizontal (sep_zero)
    Plane sep_zero(const Plane &a, const float *h, int taps)
    {
      Plane v(a.w, a.h), o(a.w, a.h);
      pool.rows(a.h, [&](int y0, int y1) { AVX2_CALL(avx2::fir_zero_v(a.d.data(), a.w, a.h, y0, y1, h, taps, v.d.data())); });
      pool.rows(a.h, [&](int y0, int y1) { AVX2_CALL(avx2::fir_zero_h(v.d.data(), a.w, y0, y1, h, taps, o.d.data())); });
      return o;
    }

    // box of splat_me on one thread (w x h planes): tmp is scratch of the plane's size
    static void box1(const float *a, int w, int h, int size, bool take_abs, bool average, float max_val, float *tmp,
      float *out)
    {
      AVX2_CALL(avx2::box_v(a, w, h, 0, h, size, take_abs, tmp));
      AVX2_CALL(avx2::box_h(tmp, w, 0, h, size, average, max_val, out));
      (void)a; (void)w; (void)h; (void)size; (void)take_abs; (void)average; (void)max_val; (void)tmp; (void)out;
    }

    // box of splat_me over all threads
    Plane box(const Plane &a, int size, bool take_abs, bool average, float max_val = -1.0f)
    {
      Plane v(a.w, a.h), o(a.w, a.h);
      pool.rows(a.h, [&](int y0, int y1) { AVX2_CALL(avx2::box_v(a.d.data(), a.w, a.h, y0, y1, size, take_abs, v.d.data())); });
      pool.rows(a.h, [&](int y0, int y1) { AVX2_CALL(avx2::box_h(v.d.data(), a.w, y0, y1, size, average, max_val, o.d.data())); });
      return o;
    }

    // Streaming of the refinement / match errors on one thread: the rows of every intermediate are computed from top
    // to bottom in strips of STRIP rows, each as soon as the rows it reads exist, and kept in ring buffers of a few
    // rows (the working set stays in the core's cache; whole-plane intermediates would be written back to memory).
    // Ring sizes cover the rows a strip reads around it (half the error box, half the refinement box, the edge-filter
    // distance) plus the strip itself; at the default sizes (7, 15, 3) 24 and 32 rows. The results do not depend on
    // the ring sizes.
    static constexpr int STRIP = 8;
    static int ring_e(const Config &c)
    {
      return std::max(24, STRIP + 2 * std::max(c.error_box / 2, c.error_edge_dist) + 4);
    }
    static int ring_r(const Config &c) { return std::max(32, STRIP + 2 * std::max(c.refine_box / 2, c.error_box / 2) + 4); }

    // match error of one vector field (error_of: DFD, box mean |.|, edge-aware filter), streamed; scratch buffers
    // b0 .. b0 + 2 of the thread
    struct ErrStream
    {
      const Plane &cur, &other;
      const float *vx, *vy;
      float *out, *d, *t, *e;
      int w, h, nd = 0, nt = 0, ne = 0, no = 0;
      const Config &cfg;
      const int RING_E;
      ErrStream(const Plane &c, const Plane &o, const float *vx_, const float *vy_, float *out_, int b0, const Config &cf)
        : cur(c), other(o), vx(vx_), vy(vy_), out(out_), w(c.w), h(c.h), cfg(cf), RING_E(ring_e(cf))
      {
        const size_t n = (size_t)w * RING_E;
        d = t_scratch.get(b0, n);
        t = t_scratch.get(b0 + 1, n);
        e = t_scratch.get(b0 + 2, n);
      }
      // the vector rows below avail are available (h: all), at most STRIP more than at the previous call
      void step(int avail)
      {
        if (avail > nd)
        {
          AVX2_CALL(avx2::dfd(cur.d.data(), other.d.data(), w, h, vx, vy, 1.0f, nd, avail, d, RING_E));
          nd = avail;
        }
        const int lt = nd == h ? h : std::max(0, nd - cfg.error_box / 2);
        if (lt > nt)
        {
          AVX2_CALL(avx2::box_v(d, w, h, nt, lt, cfg.error_box, true, t, RING_E, RING_E));
          AVX2_CALL(avx2::box_h(t, w, nt, lt, cfg.error_box, true, -1.0f, e, RING_E, RING_E));
          nt = ne = lt;
        }
        const int lo = ne == h ? h : std::max(0, ne - cfg.error_edge_dist);
        if (lo > no)
        {
          AVX2_CALL(avx2::opposite_min(e, w, h, no, lo, cfg.error_edge_weight, cfg.error_edge_dist, cfg.error_edge_mode, out,
            RING_E));
          no = lo;
        }
      }
    };

    static void error1(const Plane &cur, const Plane &other, const Plane &vx, const Plane &vy, const Config &cfg,
      Plane &out)
    {
      fit(out, cur.w, cur.h);
      ErrStream es(cur, other, vx.d.data(), vy.d.data(), out.d.data(), 0, cfg);
      for (int y = 0; y < cur.h;)
      {
        y = std::min(cur.h, y + STRIP);
        es.step(y);
      }
    }

    // refinement of one candidate on one thread (refine_one of splat_motion), streamed; with err also its match error
    // (the error of the refined vectors), streamed behind it
    static void refine1(const Plane &cur, const Plane &other, const Plane &vx, const Plane &vy, bool neg,
      const Config &cfg, Plane &ox, Plane &oy, Plane *err = nullptr)
    {
      const int w = cur.w, h = cur.h, r21 = cfg.refine_box / 2, r7 = cfg.error_box / 2, RING_R = ring_r(cfg);
      const size_t nr = (size_t)w * RING_R, ns = (size_t)w * STRIP;
      float *d = t_scratch.get(0, nr), *t7 = t_scratch.get(1, nr), *e7 = t_scratch.get(2, nr), *v21 = t_scratch.get(3, ns);
      float *T[5];
      const float *Bc[5];
      float *B[5];
      for (int k = 0; k < 5; k++)
      {
        T[k] = t_scratch.get(4 + k, nr);
        B[k] = t_scratch.get(9 + k, ns);
        Bc[k] = B[k];
      }
      fit(ox, w, h);
      fit(oy, w, h);
      if (err)
        fit(*err, w, h);
      ErrStream es(cur, other, ox.d.data(), oy.d.data(), err ? err->d.data() : nullptr, 14, cfg);
      int nd = 0, nT = 0, nb = 0;
      while (nb < h)
      {
        const int Y1 = std::min(h, nb + STRIP), X = std::min(h, Y1 + r21), D = std::min(h, X + r7);
        if (D > nd)
        {
          AVX2_CALL(avx2::dfd(cur.d.data(), other.d.data(), w, h, vx.d.data(), vy.d.data(), neg ? -1.0f : 1.0f, nd, D, d, RING_R));
          nd = D;
        }
        if (X > nT)
        {
          AVX2_CALL(avx2::box_v(d, w, h, nT, X, cfg.error_box, true, t7, RING_R, RING_R));
          AVX2_CALL(avx2::box_h(t7, w, nT, X, cfg.error_box, true, -1.0f, e7, RING_R, RING_R));
          AVX2_CALL(avx2::refine_terms(cur.d.data(), e7, d, vx.d.data(), vy.d.data(), neg, w, h, nT, X, cfg.refine_reg,
            cfg.refine_err_scale, T, RING_R));
          nT = X;
        }
        for (int k = 0; k < 5; k++)
        {
          AVX2_CALL(avx2::box_v(T[k], w, h, nb, Y1, cfg.refine_box, false, v21, RING_R, STRIP));
          AVX2_CALL(avx2::box_h(v21, w, nb, Y1, cfg.refine_box, false, -1.0f, B[k], STRIP, STRIP));
        }
        AVX2_CALL(avx2::refine_solve(Bc, w, nb, Y1, ox.d.data(), oy.d.data(), STRIP));
        nb = Y1;
        if (err)
          es.step(nb);
      }
      (void)other; (void)neg; (void)Bc; (void)B; (void)T; (void)e7; (void)t7; (void)v21; (void)d; (void)vx; (void)vy;
      (void)r7; (void)RING_R;
    }

    // ---- picture build pieces
    // fn(j, y0, y1) over row chunks of several planes at once (rows(j) rows for plane j): one parallel phase
    void multi(int njobs, const std::function<int(int)> &rows, const std::function<void(int, int, int)> &fn)
    {
      struct Task { int j, y0, y1; };
      std::vector<Task> tasks;
      const int CHUNK = 16;
      for (int j = 0; j < njobs; j++)
        for (int y = 0, n = rows(j); y < n; y += CHUNK)
          tasks.push_back({ j, y, std::min(n, y + CHUNK) });
      pool.run((int)tasks.size(), [&](int i) { fn(tasks[(size_t)i].j, tasks[(size_t)i].y0, tasks[(size_t)i].y1); });
    }

    // pyr_reduce of splat_core for n planes: o[j] = pyr_reduce(s[j]) (v[j]: scratch)
    void reduce_many(int n, const Plane *const *s, Plane *const *v, Plane *const *o, const float *D)
    {
      for (int j = 0; j < n; j++)
      {
        fit(*v[j], s[j]->w, (s[j]->h + 1) / 2);
        fit(*o[j], (s[j]->w + 1) / 2, (s[j]->h + 1) / 2);
      }
      auto nrows = [&](int j) { return o[j]->h; };
      multi(n, nrows, [&](int j, int y0, int y1) {
        if (simd)
          AVX2_CALL(avx2::pyr_reduce_v(s[j]->d.data(), s[j]->w, s[j]->h, y0, y1, D, v[j]->d.data()));
        else
          pyr_reduce_v_rows(*s[j], y0, y1, D, *v[j]);
      });
      multi(n, nrows, [&](int j, int y0, int y1) {
        if (simd)
          AVX2_CALL(avx2::pyr_reduce_h(v[j]->d.data(), s[j]->w, y0, y1, D, o[j]->d.data()));
        else
          pyr_reduce_h_rows(*v[j], y0, y1, D, *o[j]);
      });
    }

    // pyr_expand of splat_core for n planes, combined with their bases: o[j] (sized like base[j]) = base[j] - e (mode 1),
    // base[j] + e (2), base[j] + e clamped to the sample range (3), e = pyr_expand(l[j])
    void expand_many(int n, const Plane *const *l, const Plane *const *base, Plane *const *o, const int *mode)
    {
      for (int j = 0; j < n; j++)
        fit(*o[j], base[j]->w, base[j]->h);
      if (simd)
      {
        multi(n, [&](int j) { return l[j]->h; }, [&](int j, int y0, int y1) {
          AVX2_CALL(avx2::pyr_expand(l[j]->d.data(), l[j]->w, l[j]->h, y0, y1, PYR_EXPAND_A, PYR_EXPAND_B, PYR_EXPAND_C,
            o[j]->d.data(), o[j]->w, o[j]->h, base[j]->d.data(), mode[j], SAMPLE_MIN, SAMPLE_MAX));
          (void)j; (void)y0; (void)y1;
        });
        return;
      }
      multi(n, [&](int j) { return l[j]->h; }, [&](int j, int y0, int y1) { pyr_expand_rows(*l[j], y0, y1, *o[j]); });
      multi(n, [&](int j) { return o[j]->h; }, [&](int j, int y0, int y1) {
        const int ow = o[j]->w;
        for (size_t k = (size_t)y0 * ow; k < (size_t)y1 * ow; k++)
        {
          float v = mode[j] == 1 ? base[j]->d[k] - o[j]->d[k] : base[j]->d[k] + o[j]->d[k];
          if (mode[j] == 3)
            v = std::fmin(std::fmax(v, SAMPLE_MIN), SAMPLE_MAX);
          o[j]->d[k] = v;
        }
      });
    }

    // levels with fewer samples than this run on the calling thread (waking the workers costs more than they save)
    static constexpr size_t SMALL_LEVEL = 65536;
    void maybe_serial(size_t samples, const std::function<void()> &fn)
    {
      if (samples < SMALL_LEVEL)
        pool.serial(fn);
      else
        fn();
    }

    // field_reduce2 of splat_core (the same arithmetic) over rows on all threads, into o (memory reused)
    void field_reduce2_into(const Field4 &F, Field4 &o)
    {
      const int w = F.c[0].w, h = F.c[0].h, ow = w / 2, oh = h / 2;
      for (int c = 0; c < 4; c++)
        fit(o.c[c], ow, oh);
      maybe_serial((size_t)ow * oh, [&] {
        pool.rows(oh, [&](int j0, int j1) {
          for (int c = 0; c < 4; c++)
          {
            const Plane &s = F.c[c];
            // edge clamp, as field_reduce2
            auto g = [&](int y, int x) { return s.at(std::min(std::max(x, 0), w - 1), std::min(std::max(y, 0), h - 1)); };
            for (int j = j0; j < j1; j++)
              for (int i = 0; i < ow; i++)
              {
                const int y = 2 * j, x = 2 * i;
                float r = 0.0625f * (((g(y - 1, x - 1) + g(y - 1, x + 1)) + g(y + 1, x - 1)) + g(y + 1, x + 1));
                r = r + 0.125f * (((g(y - 1, x) + g(y, x - 1)) + g(y, x + 1)) + g(y + 1, x));
                r = r + 0.25f * g(y, x);
                if (c < 2)
                  r = r / 2.0f;
                o.c[c].at(i, j) = r;
              }
          }
        });
      });
    }

    // border_weight of splat_core over all threads
    Field4 border_weight(const Field4 &F, float d, bool n_side, const Config &cfg)
    {
      const float g = n_side ? 1.0f - d : d;
      const float pr = 1.0f - g;
      const int w = F.c[0].w, h = F.c[0].h, XB = cfg.pad_x / cfg.scale, YB = cfg.pad_y / cfg.scale;
      Field4 G;
      for (int c = 0; c < 4; c++)
        G.c[c] = Plane(w, h);
      pool.rows(h, [&](int y0, int y1) {
        for (int y = y0; y < y1; y++)
          for (int x = 0; x < w; x++)
          {
            const int sx = std::min(std::max(x, XB), w - 1 - XB), sy = std::min(std::max(y, YB), h - 1 - YB);
            G.c[0].at(x, y) = F.c[0].at(sx, sy) * g;
            G.c[1].at(x, y) = F.c[1].at(sx, sy) * g;
            const float ex = cfg.splat_err_scale > 0.0f ? exp32(-F.c[2].at(sx, sy) / cfg.splat_err_scale) : 1.0f;
            const float m = std::fmax(cfg.splat_min_weight, ex);
            G.c[2].at(x, y) = (cfg.splat_gain * m) * pr;
            G.c[3].at(x, y) = F.c[3].at(sx, sy) * pr;
          }
      });
      return G;
    }

    // persistent buffers of the picture build, one set per pyramid level, reused from frame to frame (no allocation and
    // no fresh pages per frame)
    struct SideBuf
    {
      Plane vx, vy, mc, st;
      std::vector<uint8_t> zero;
      std::vector<int> idx, start, cnt, off;
    };
    struct LevelBuf
    {
      Pic3 built, recon;
      std::vector<float> acc;
      SideBuf sd[2];
      Field4 coarse[2];
    };
    std::vector<LevelBuf> lv;
    // pyramid_into scratch: the low-pass levels and the vertical pass; build_into's pyramids of P and N
    std::vector<Pic3> plow;
    Plane ptmp[3];
    Pyramid pyr_tmp[2];
    // fields of the coarse pyramid levels (per side, ping-pong between [0] / [1], current: coarse_cur), kept between
    // frames
    Field4 coarse[2][2];
    int coarse_cur = 0;
    // the refined candidates and their errors of fields(), per side, kept between calls
    VecSet fvec[2];
    std::vector<Plane> ferr[2];

    void build_level(LevelBuf &B, const Pic3 &P, const Pic3 &N, const Field4 *G[2], bool full_res, int stride, float d,
      const Config &cfg);
  };

  // One pyramid level, as splat::build_level, on the bands P and N of the level: the side planes (for the full-resolution
  // levels sampled from the up-sampled projection fields G, computed on the fly as field_upsample does; for the
  // coarse levels read from their own fields), the in-place start of the accumulator, the splats by bands of output
  // rows (every source sample filed under the row it lands on, in raster order, so every cell gets its terms in the
  // exact path's order), the normalisation into B.built.
  void FastEngine::Impl::build_level(LevelBuf &B, const Pic3 &P, const Pic3 &N, const Field4 *G[2], bool full_res,
    int stride, float d, const Config &cfg)
  {
    const int sh = ilog2(cfg.scale);
    const int w = P.c[0].w, h = P.c[0].h;
    const float s = (float)stride;
    const int aw = w + 1, ah = h + 1;
    B.acc.resize((size_t)aw * ah * 4);
    float *acc = B.acc.data();
    if (simd)
    {
      // per row: both sides' values (the in-place weights st only for this row) and the accumulator start
      for (int k = 0; k < 2; k++)
      {
        SideBuf &t = B.sd[k];
        fit(t.vx, w, h); fit(t.vy, w, h); fit(t.mc, w, h);
        t.zero.resize((size_t)w * h);
      }
      const float *g[2][4], *pp[3], *pn[3];
      for (int k = 0; k < 2; k++)
        for (int c = 0; c < 4; c++)
          g[k][c] = G[k]->c[c].d.data();
      for (int c = 0; c < 3; c++)
      {
        pp[c] = P.c[c].d.data();
        pn[c] = N.c[c].d.data();
      }
      const int gw = G[0]->c[0].w, gh = G[0]->c[0].h;
      pool.rows(ah, [&](int y0, int y1) {
        float *st[2] = { t_scratch.get(0, (size_t)w), t_scratch.get(1, (size_t)w) };
        for (int y = y0; y < y1; y++)
        {
          float *row = acc + (size_t)y * aw * 4;
          if (y >= h)
          {
            std::memset(row, 0, sizeof(float) * (size_t)aw * 4);
            continue;
          }
          const size_t o = (size_t)y * w;
          for (int k = 0; k < 2; k++)
          {
            SideBuf &t = B.sd[k];
            AVX2_CALL(avx2::level_side(g[k], gw, gh, full_res, stride, sh, w, y, cfg.still_vector, t.vx.d.data() + o,
              t.vy.d.data() + o, t.mc.d.data() + o, st[k], t.zero.data() + o));
          }
          const float *pr[3] = { pp[0] + o, pp[1] + o, pp[2] + o }, *nr[3] = { pn[0] + o, pn[1] + o, pn[2] + o };
          AVX2_CALL(avx2::level_acc_row(pr, nr, st[0], st[1], w, row));
          std::memset(row + (size_t)w * 4, 0, sizeof(float) * 4);
          (void)pr; (void)nr; (void)o;
        }
      });
      (void)gw; (void)gh;
    }
    else
    {
    for (int k = 0; k < 2; k++)
    {
      SideBuf &t = B.sd[k];
      fit(t.vx, w, h); fit(t.vy, w, h); fit(t.mc, w, h); fit(t.st, w, h);
      t.zero.resize((size_t)w * h);
    }
    pool.run(2 * h, [&](int i) {
      const int k = i / h, y = i % h;
      SideBuf &t = B.sd[k];
      const Field4 &F = *G[k];
      for (int x = 0; x < w; x++)
      {
        float vx, vy, mc, st;
        if (full_res)
        {
          const int X = x * stride, Y = y * stride;
          vx = up_s(F.c[0], true, X, Y, sh) / s;
          vy = up_s(F.c[1], true, X, Y, sh) / s;
          mc = up_s(F.c[2], false, X, Y, sh);
          st = up_s(F.c[3], false, X, Y, sh);
        }
        else
        {
          vx = F.c[0].clamped(x, y) / s;
          vy = F.c[1].clamped(x, y) / s;
          mc = F.c[2].clamped(x, y);
          st = F.c[3].clamped(x, y);
        }
        const bool z = std::sqrt(vx * vx + vy * vy) <= cfg.still_vector;
        if (z)
          st = st + mc;
        t.vx.at(x, y) = vx; t.vy.at(x, y) = vy; t.mc.at(x, y) = mc; t.st.at(x, y) = st;
        t.zero[(size_t)y * w + x] = z;
      }
    });
    pool.rows(ah, [&](int y0, int y1) {
      for (int y = y0; y < y1; y++)
      {
        float *row = acc + (size_t)y * aw * 4;
        if (y >= h)
        {
          std::memset(row, 0, sizeof(float) * (size_t)aw * 4);
          continue;
        }
        for (int x = 0; x < w; x++)
        {
          float *a = row + (size_t)x * 4;
          const float sP = B.sd[0].st.at(x, y), sN = B.sd[1].st.at(x, y);
          for (int c = 0; c < 3; c++)
            a[c] = P.c[c].at(x, y) * sP + N.c[c].at(x, y) * sN;
          a[3] = sP + sN;
        }
        std::memset(row + (size_t)w * 4, 0, sizeof(float) * 4);
      }
    });
    }
    // per side: the samples that splat, grouped by the row they land on (stable counting sort over row chunks)
    const float wl = (float)(w - 1), hl = (float)(h - 1);
    const int CH = 16, nch = (h + CH - 1) / CH;
    for (int k = 0; k < 2; k++)
    {
      SideBuf &t = B.sd[k];
      auto target = [&](int x, int y) -> int {
        if (t.zero[(size_t)y * w + x])
          return -1;
        const float px = (float)x + t.vx.at(x, y), py = (float)y + t.vy.at(x, y);
        if (!(px >= 0.0f && px <= wl && py >= 0.0f && py <= hl))
          return -1;
        return (int)std::floor(py);
      };
      t.cnt.assign((size_t)nch * h, 0);
      pool.run(nch, [&](int c) {
        int *cc = &t.cnt[(size_t)c * h];
        for (int y = c * CH; y < std::min(h, (c + 1) * CH); y++)
          for (int x = 0; x < w; x++)
          {
            const int r = target(x, y);
            if (r >= 0)
              cc[r]++;
          }
      });
      // offsets: row major, chunk minor (raster order within a row's group)
      t.off.resize((size_t)nch * h);
      t.start.resize((size_t)h + 1);
      int run = 0;
      for (int r = 0; r < h; r++)
      {
        t.start[r] = run;
        for (int c = 0; c < nch; c++)
        {
          t.off[(size_t)c * h + r] = run;
          run += t.cnt[(size_t)c * h + r];
        }
      }
      t.start[h] = run;
      t.idx.resize((size_t)run);
      pool.run(nch, [&](int c) {
        int *oo = &t.off[(size_t)c * h];
        for (int y = c * CH; y < std::min(h, (c + 1) * CH); y++)
          for (int x = 0; x < w; x++)
          {
            const int r = target(x, y);
            if (r >= 0)
              t.idx[(size_t)oo[r]++] = (y << 16) | x; // position packed as row, column
          }
      });
    }
    // splats by bands of accumulator rows: side by side, corner by corner, rows of the band, samples in raster order
    const Pic3 *pic[2] = { &P, &N };
    const int BAND = 8, nb = (ah + BAND - 1) / BAND;
    pool.run(nb, [&](int b) {
      const int R0 = b * BAND, R1 = std::min(ah, R0 + BAND);
      for (int k = 0; k < 2; k++)
      {
        const SideBuf &t = B.sd[k];
        const float *p0 = pic[k]->c[0].d.data(), *p1 = pic[k]->c[1].d.data(), *p2 = pic[k]->c[2].d.data();
        for (int corner = 0; corner < 4; corner++)
        {
          const int oy = corner >> 1, ox = corner & 1;
          for (int R = R0; R < R1; R++)
          {
            const int iyr = R - oy;
            if (iyr < 0 || iyr >= h)
              continue;
            for (int j = t.start[iyr]; j < t.start[iyr + 1]; j++)
            {
              const int x = t.idx[(size_t)j] & 0xffff, y = t.idx[(size_t)j] >> 16, i = y * w + x;
              const float px = (float)x + t.vx.d[(size_t)i], py = (float)y + t.vy.d[(size_t)i];
              const float ix = std::floor(px), iy = std::floor(py);
              const float fx = px - ix, fy = py - iy;
              float a;
              switch (corner)
              {
              case 0: a = (1.0f - fx) * (1.0f - fy); break;
              case 1: a = fx * (1.0f - fy); break;
              case 2: a = (1.0f - fx) * fy; break;
              default: a = fx * fy; break;
              }
              const float wt = t.mc.d[(size_t)i];
              float *dst = acc + ((size_t)R * aw + (size_t)ix + ox) * 4;
              dst[0] = dst[0] + (p0[i] * wt) * a;
              dst[1] = dst[1] + (p1[i] * wt) * a;
              dst[2] = dst[2] + (p2[i] * wt) * a;
              dst[3] = dst[3] + a * wt;
            }
          }
        }
      }
    });
    for (int c = 0; c < 3; c++)
      fit(B.built.c[c], w, h);
    pool.rows(h, [&](int y0, int y1) {
      for (int y = y0; y < y1; y++)
        for (int x = 0; x < w; x++)
        {
          const float *a = acc + ((size_t)y * aw + x) * 4;
          const float sw = a[3];
          for (int c = 0; c < 3; c++)
            B.built.c[c].at(x, y) = sw < cfg.hole_weight ? P.c[c].at(x, y) * (1.0f - d) + N.c[c].at(x, y) * d : a[c] / sw;
        }
    });
  }

  // ------------------------------------------------------------------------------------------------------------ engine
  bool FastEngine::cpu_has_avx2_fma()
  {
#if SPLAT_X86 && SPLAT_HAVE_AVX2
    unsigned a, b, c, d;
#if defined(_MSC_VER)
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7)
      return false;
    __cpuid(r, 1);
    c = (unsigned)r[2];
#else
    if (__get_cpuid_max(0, nullptr) < 7)
      return false;
    __get_cpuid(1, &a, &b, &c, &d);
#endif
    const bool fma = (c >> 12) & 1, osxsave = (c >> 27) & 1, avx = (c >> 28) & 1;
    if (!fma || !osxsave || !avx)
      return false;
#if defined(_MSC_VER)
    const unsigned long long xcr0 = _xgetbv(0);
    __cpuidex(r, 7, 0);
    b = (unsigned)r[1];
#else
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    const unsigned long long xcr0 = ((unsigned long long)hi << 32) | lo;
    __cpuid_count(7, 0, a, b, c, d);
#endif
    if ((xcr0 & 6) != 6)
      return false;
    (void)a; (void)d;
    return (b >> 5) & 1;
#else
    return false;
#endif
  }

  FastEngine::FastEngine(int threads, bool simd) : impl_(new Impl(threads, simd && cpu_has_avx2_fma())) {}
  FastEngine::~FastEngine() = default;
  bool FastEngine::simd() const { return impl_->simd; }
  int FastEngine::threads() const { return impl_->pool.threads(); }
  void FastEngine::rows(int n, const std::function<void(int, int)> &fn) { impl_->pool.rows(n, fn); }

  // ------------------------------------------------------------------------------------------------------- ME pictures
  Plane FastEngine::me_picture(const Plane &y, MeFir fir, const Config &cfg)
  {
    Impl &m = *impl_;
    if (!m.simd)
      return splat::me_picture(y, fir, cfg);
    const float *h = fir == MeFir::A ? cfg.fir_a : cfg.fir_b;
    const int taps = fir == MeFir::A ? cfg.fir_a_n : cfg.fir_b_n, sc = cfg.scale;
    const int nh = (y.w + 2 * cfg.pad_x) / sc, nv = (y.h + 2 * cfg.pad_y) / sc;
    Plane t(nh, y.h), o(nh, nv);
    m.pool.rows(y.h, [&](int y0, int y1) {
      AVX2_CALL(avx2::down_h(y.d.data(), y.w, y0, y1, h, taps, cfg.pad_x, sc, t.d.data(), nh));
    });
    m.pool.rows(nv, [&](int j0, int j1) {
      AVX2_CALL(avx2::down_v(t.d.data(), nh, y.h, j0, j1, h, taps, cfg.pad_y, sc, o.d.data()));
    });
    (void)h; (void)taps; (void)sc;
    return o;
  }

  // ------------------------------------------------------------------------------------------------- phase correlation
  PcorrPicture FastEngine::pcorr_ingest(const Plane &B, const Config &cfg)
  {
    PcorrPicture o;
    pcorr_ingest_into(B, cfg, o);
    return o;
  }

  void FastEngine::pcorr_ingest_into(const Plane &B, const Config &cfg, PcorrPicture &o)
  {
    o = splat::pcorr_ingest(B, cfg, impl_->pcorr_ops());
  }

  std::vector<Plane> FastEngine::pcorr_candidates(const PcorrPicture &prev, const PcorrPicture &cur, const Config &cfg,
    PcorrGlobal *glob)
  {
    std::vector<Plane> out;
    pcorr_candidates_into(prev, cur, cfg, out, glob);
    return out;
  }

  void FastEngine::pcorr_candidates_into(const PcorrPicture &prev, const PcorrPicture &cur, const Config &cfg,
    std::vector<Plane> &out, PcorrGlobal *glob)
  {
    splat::pcorr_candidates(prev, cur, cfg, impl_->pcorr_ops(), out, glob);
  }

  // -------------------------------------------------------------- refinement, match errors, assignment, local fallback
  VecSet FastEngine::refine(const Plane &cur, const Plane &other, const VecSet &cands, bool neg, const Config &cfg)
  {
    Impl &m = *impl_;
    if (!m.simd)
      return splat::refine(cur, other, cands, neg, cfg);
    const int n = (int)cands.size() / 2;
    VecSet out(cands.size());
    m.pool.run(n, [&](int i) {
      Impl::refine1(cur, other, cands[2 * i], cands[2 * i + 1], neg, cfg, out[2 * i], out[2 * i + 1]);
    });
    return out;
  }

  std::vector<Plane> FastEngine::errors(const Plane &cur, const Plane &other, const VecSet &vecs, const Config &cfg)
  {
    Impl &m = *impl_;
    if (!m.simd)
      return splat::errors(cur, other, vecs, cfg);
    const int n = (int)vecs.size() / 2;
    std::vector<Plane> e(n + 1);
    const Plane z(cur.w, cur.h);
    m.pool.run(n + 1, [&](int i) {
      Impl::error1(cur, other, i < n ? vecs[2 * i] : z, i < n ? vecs[2 * i + 1] : z, cfg, e[i]);
    });
    return e;
  }

  void FastEngine::assign(const VecSet &rf, const std::vector<Plane> &ef, const VecSet &rb, const std::vector<Plane> &eb,
    const float *bias, float lam, float thresh, float ratio, float lenc, Field4 &fwd, Field4 &bwd)
  {
    Impl &m = *impl_;
    const int n = (int)ef.size();
    if (!m.simd || n > 64)
    {
      splat::assign(rf, ef, rb, eb, bias, lam, thresh, ratio, lenc, fwd, bwd);
      return;
    }
    const int w = ef[0].w, h = ef[0].h;
    for (int c = 0; c < 4; c++)
    {
      fwd.c[c] = Plane(w, h);
      bwd.c[c] = Plane(w, h);
    }
    std::vector<const float *> prf, pef, prb, peb;
    for (const Plane &p : rf)
      prf.push_back(p.d.data());
    for (const Plane &p : rb)
      prb.push_back(p.d.data());
    for (const Plane &p : ef)
      pef.push_back(p.d.data());
    for (const Plane &p : eb)
      peb.push_back(p.d.data());
    float *out[6] = { fwd.c[0].d.data(), fwd.c[1].d.data(), fwd.c[2].d.data(), bwd.c[0].d.data(), bwd.c[1].d.data(),
      bwd.c[2].d.data() };
    const size_t total = (size_t)w * h, chunk = 1024;
    m.pool.run((int)((total + chunk - 1) / chunk), [&](int i) {
      const size_t k0 = (size_t)i * chunk, k1 = std::min(total, k0 + chunk);
      AVX2_CALL(avx2::assign(prf.data(), pef.data(), prb.data(), peb.data(), n, bias, lam, thresh, ratio, lenc, k0, k1, out));
      (void)k0; (void)k1;
    });
    (void)out;
  }

  Plane FastEngine::disorder_weight(const Plane &vx, const Plane &vy, const Plane *err, const Config &cfg)
  {
    Impl &m = *impl_;
    if (!m.simd)
      return splat::disorder_weight(vx, vy, err, cfg);
    const int w = vx.w, h = vx.h;
    const float h0 = bits_to_float(DISORDER_SMOOTH_BITS[0]), h1 = bits_to_float(DISORDER_SMOOTH_BITS[1]);
    // 3-tap Gaussian, vertical then horizontal, edge repeat (g3 of splat_me)
    auto g3 = [&](const Plane &a, bool horizontal) {
      Plane o(w, h);
      m.pool.rows(h, [&](int y0, int y1) {
        for (int y = y0; y < y1; y++)
          for (int x = 0; x < w; x++)
          {
            float acc = h0 * a.at(x, y);
            const float pair = horizontal ? a.clamped(x - 1, y) + a.clamped(x + 1, y) : a.clamped(x, y - 1) + a.clamped(x, y + 1);
            o.at(x, y) = acc + h1 * pair;
          }
      });
      return o;
    };
    const Plane sx = g3(g3(vx, false), true), sy = g3(g3(vy, false), true);
    Plane d(w, h);
    m.pool.rows(h, [&](int y0, int y1) {
      for (size_t i = (size_t)y0 * w; i < (size_t)y1 * w; i++)
      {
        const float ex = vx.d[i] - sx.d[i], ey = vy.d[i] - sy.d[i];
        const float dist = std::sqrt(ex * ex + ey * ey);
        float v = cfg.fallback_strength * std::fmin(std::fmax((dist - cfg.fallback_dead_zone) / cfg.fallback_range, 0.0f), 1.0f);
        if (err && cfg.fallback_err_gain > 0.0f)
          v = v + cfg.fallback_err_gain *
            std::fmin(std::fmax((err->d[i] - cfg.fallback_err_lo) / cfg.fallback_err_range, 0.0f), 1.0f);
        d.d[i] = v;
      }
    });
    return m.box(d, cfg.fallback_box, false, true, cfg.fallback_cap);
  }

  void FastEngine::fields(const Plane *prv, const Plane &cur, const Plane &nxt, const CandSet &cf, const CandSet &cb,
    const float *bias, bool local_fb, const Config &cfg, Field4 &fwd, Field4 &bwd)
  {
    Impl &m = *impl_;
    const int nsides = prv ? 2 : 1;
    const CandSet *cs[2] = { &cf, &cb };
    const Plane *oth[2] = { &nxt, prv };
    VecSet *vec = m.fvec;
    std::vector<Plane> *err = m.ferr;
    // the candidates as vectors pointing from cur: refined (turned around when they point into cur), or as they are
    int nc[2] = { 0, 0 };
    for (int k = 0; k < nsides; k++)
    {
      nc[k] = (int)cs[k]->vecs->size() / 2;
      vec[k].resize(cs[k]->vecs->size());
      err[k].resize((size_t)nc[k] + 1);
      if (!cs[k]->refine)
      {
        vec[k] = *cs[k]->vecs;
        if (!cs[k]->toward)
          for (auto &p : vec[k])
            for (auto &x : p.d)
              x = -x;
      }
    }
    if (!m.simd)
    {
      for (int k = 0; k < nsides; k++)
      {
        if (cs[k]->refine)
          vec[k] = splat::refine(cur, *oth[k], *cs[k]->vecs, !cs[k]->toward, cfg);
        err[k] = splat::errors(cur, *oth[k], vec[k], cfg);
      }
    }
    else
    {
      // one task per candidate (refinement and its error) and one per zero vector, both sides at once
      struct Job { int k, i; };
      std::vector<Job> jobs;
      for (int k = 0; k < nsides; k++)
        for (int i = 0; i <= nc[k]; i++)
          jobs.push_back({ k, i });
      const Plane z(cur.w, cur.h);
      m.pool.run((int)jobs.size(), [&](int j) {
        const int k = jobs[(size_t)j].k, i = jobs[(size_t)j].i;
        if (i == nc[k])
        {
          Impl::error1(cur, *oth[k], z, z, cfg, err[k][(size_t)i]);
          return;
        }
        if (cs[k]->refine)
          Impl::refine1(cur, *oth[k], (*cs[k]->vecs)[2 * i], (*cs[k]->vecs)[2 * i + 1], !cs[k]->toward, cfg,
            vec[k][2 * i], vec[k][2 * i + 1], &err[k][(size_t)i]);
        else
          Impl::error1(cur, *oth[k], vec[k][2 * i], vec[k][2 * i + 1], cfg, err[k][(size_t)i]);
      });
    }
    if (!prv)
      assign_single(vec[0], err[0], bias, cfg.vector_length_cost, fwd, bwd);
    else
      assign(vec[0], err[0], vec[1], err[1], bias, cfg.speed_change_cost, cfg.occlusion_error, cfg.occlusion_error_factor, cfg.vector_length_cost, fwd, bwd);
    if (local_fb)
    {
      fwd.c[3] = disorder_weight(fwd.c[0], fwd.c[1], &fwd.c[2], cfg);
      bwd.c[3] = disorder_weight(bwd.c[0], bwd.c[1], &bwd.c[2], cfg);
    }
  }

  // ----------------------------------------------------------------------------------------------------- picture build
  Pic3 FastEngine::build(const Pic3 &Pin, const Pic3 &Nin, const Field4 &MEP, const Field4 &MEN, float d, const Config &cfg)
  {
    Pic3 out;
    build_into(Pin, Nin, MEP, MEN, d, cfg, out);
    return out;
  }

  void FastEngine::build_into(const Pic3 &Pin, const Pic3 &Nin, const Field4 &MEP, const Field4 &MEN, float d,
    const Config &cfg, Pic3 &res)
  {
    pyramid_into(Pin, cfg, impl_->pyr_tmp[0]);
    pyramid_into(Nin, cfg, impl_->pyr_tmp[1]);
    build_pyr_into(impl_->pyr_tmp[0], impl_->pyr_tmp[1], MEP, MEN, d, cfg, res);
  }

  void FastEngine::pyramid_into(const Pic3 &pic, const Config &cfg, Pyramid &out)
  {
    Impl &m = *impl_;
    const Geometry g(pic.c[0].w, pic.c[0].h, cfg);
    const int L = cfg.levels, PADX = cfg.pad_x, PADY = cfg.pad_y;
    if ((int)m.plow.size() < L + 1)
      m.plow.resize((size_t)L + 1);
    std::vector<Pic3> &low = m.plow;
    out.band.resize((size_t)L + 1);
    // pad (edge repeat) into the finest low-pass level
    for (int c = 0; c < 3; c++)
      fit(low[0].c[c], g.GW, g.GH);
    m.pool.run(3 * g.GH, [&](int t) {
      const int c = t / g.GH, y = t % g.GH;
      const Plane &s = pic.c[c];
      float *o = &low[0].c[c].at(0, y);
      const float *row = s.d.data() + (size_t)std::min(std::max(y - PADY, 0), s.h - 1) * s.w;
      for (int x = 0; x < PADX; x++)
        o[x] = row[0];
      std::memcpy(o + PADX, row, sizeof(float) * (size_t)s.w);
      for (int x = PADX + s.w; x < g.GW; x++)
        o[x] = row[s.w - 1];
    });
    // Laplacian pyramid: low[i + 1] = reduce(low[i]), band[i] = low[i] - expand(low[i + 1]), band[L] = low[L]
    for (int i = 0; i < L; i++)
    {
      const Plane *s[3];
      Plane *v[3], *o[3];
      for (int j = 0; j < 3; j++)
      {
        s[j] = &low[i].c[j];
        v[j] = &m.ptmp[j];
        o[j] = &low[i + 1].c[j];
      }
      m.maybe_serial(s[0]->d.size(), [&] { m.reduce_many(3, s, v, o, cfg.pyr_reduce); });
    }
    for (int i = 0; i < L; i++)
    {
      const Plane *l[3], *base[3];
      Plane *o[3];
      const int mode[3] = { 1, 1, 1 };
      for (int j = 0; j < 3; j++)
      {
        l[j] = &low[i + 1].c[j];
        base[j] = &low[i].c[j];
        o[j] = &out.band[i].c[j];
      }
      m.maybe_serial(base[0]->d.size(), [&] { m.expand_many(3, l, base, o, mode); });
    }
    for (int c = 0; c < 3; c++)
    {
      const Plane &lo = low[L].c[c];
      fit(out.band[L].c[c], lo.w, lo.h);
      std::memcpy(out.band[L].c[c].d.data(), lo.d.data(), sizeof(float) * lo.d.size());
    }
    out.valid = true;
  }

  void FastEngine::build_pyr_into(const Pyramid &PP, const Pyramid &NP, const Field4 &MEP, const Field4 &MEN, float d,
    const Config &cfg, Pic3 &res)
  {
    Impl &m = *impl_;
    const int L = cfg.levels, stride_max = cfg.stride_max, PADX = cfg.pad_x, PADY = cfg.pad_y;
    const int GW = PP.band[0].c[0].w, GH = PP.band[0].c[0].h;
    const Geometry g(GW - 2 * PADX, GH - 2 * PADY, cfg);
    if ((int)m.lv.size() < L + 1)
      m.lv.resize((size_t)L + 1);
    std::vector<Impl::LevelBuf> &lv = m.lv;
    // fields per level: levels with stride 2^i <= stride_max sample the up-sampled projection field; the first
    // coarser level reduces the ME-res field (resolution 1 / scale) i - log2(scale) times, each further level once more
    const Field4 gP = m.border_weight(MEP, d, false, cfg), gN = m.border_weight(MEN, d, true, cfg);
    const int log2max = ilog2(cfg.scale);
    bool have_coarse = false;
    for (int i = 0; i <= L; i++)
    {
      const int st = 1 << i;
      const Field4 *G[2];
      if (st <= stride_max)
      {
        G[0] = &gP;
        G[1] = &gN;
        m.maybe_serial(PP.band[i].c[0].d.size(), [&] { m.build_level(lv[i], PP.band[i], NP.band[i], G, true, st, d, cfg); });
        continue;
      }
      // reduced from the previous coarse fields (from gP / gN the first time, i - log2max times), ping-pong
      const int nred = have_coarse ? 1 : i - log2max;
      for (int r = 0; r < nred; r++)
      {
        const Field4 &fp = (have_coarse || r > 0) ? m.coarse[0][m.coarse_cur] : gP;
        const Field4 &fn = (have_coarse || r > 0) ? m.coarse[1][m.coarse_cur] : gN;
        m.field_reduce2_into(fp, m.coarse[0][1 - m.coarse_cur]);
        m.field_reduce2_into(fn, m.coarse[1][1 - m.coarse_cur]);
        m.coarse_cur = 1 - m.coarse_cur;
      }
      if (nred <= 0) // (stride_max above the ME scale: the coarse levels start from the projection fields)
      {
        m.coarse[0][m.coarse_cur] = gP;
        m.coarse[1][m.coarse_cur] = gN;
      }
      have_coarse = true;
      G[0] = &m.coarse[0][m.coarse_cur];
      G[1] = &m.coarse[1][m.coarse_cur];
      m.maybe_serial(PP.band[i].c[0].d.size(), [&] { m.build_level(lv[i], PP.band[i], NP.band[i], G, false, 1, d, cfg); });
    }
    // Laplacian reconstruction, coarse to fine: out[i] = built[i] + expand(out[i + 1]); clamp at the finest level.
    // With the AVX2 kernels the finest level is made only inside the border, straight into res
    for (int c = 0; c < 3; c++)
      fit(res.c[c], g.W, g.H);
    const Pic3 *out = &lv[L].built;
    for (int i = L - 1; i >= 0; i--)
    {
#if SPLAT_HAVE_AVX2
      if (i == 0 && m.simd)
      {
        const Pic3 &lo = *out;
        m.multi(3, [&](int c) { return lo.c[c].h; }, [&](int c, int y0, int y1) {
          const avx2::Crop cr = { res.c[c].d.data(), g.W, PADX, PADY, g.W, g.H };
          const Plane &b = lv[0].built.c[c];
          avx2::pyr_expand(lo.c[c].d.data(), lo.c[c].w, lo.c[c].h, y0, y1, PYR_EXPAND_A, PYR_EXPAND_B, PYR_EXPAND_C,
            nullptr, b.w, b.h, b.d.data(), 3, SAMPLE_MIN, SAMPLE_MAX, &cr);
        });
        return;
      }
#endif
      const Plane *l[3], *base[3];
      Plane *o[3];
      const int md = i == 0 ? 3 : 2, mode[3] = { md, md, md };
      for (int c = 0; c < 3; c++)
      {
        l[c] = &out->c[c];
        base[c] = &lv[i].built.c[c];
        o[c] = &lv[i].recon.c[c];
      }
      m.maybe_serial(base[0]->d.size(), [&] { m.expand_many(3, l, base, o, mode); });
      out = &lv[i].recon;
    }
    m.pool.run(3 * g.H, [&](int t) {
      const int c = t / g.H, y = t % g.H;
      std::memcpy(&res.c[c].at(0, y), out->c[c].d.data() + (size_t)(y + PADY) * out->c[c].w + PADX, sizeof(float) * (size_t)g.W);
    });
  }
}
