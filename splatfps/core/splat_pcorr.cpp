// splat_pcorr: the exact CPU path of the phase-correlation vector source; see splat_pcorr.h.
#include "splat_pcorr.h"
#include "../third_party/pocketfft/fftp_pow2.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace splat
{

  // one direction of sep_zero
  static Plane sep_fir_zeropad(const Plane &a, const float *h, int taps, bool vertical)
  {
    Plane o(a.w, a.h);
    auto g = [&](int x, int y) -> float {
      if (x < 0 || x >= a.w || y < 0 || y >= a.h)
        return 0.0f;
      return a.at(x, y);
    };
    for (int y = 0; y < a.h; y++)
      for (int x = 0; x < a.w; x++)
      {
        float acc = h[0] * a.at(x, y);
        for (int i = 1; i < taps; i++)
          acc = acc + h[i] * (vertical ? g(x, y - i) + g(x, y + i) : g(x - i, y) + g(x + i, y));
        o.at(x, y) = acc;
      }
    return o;
  }

  Plane sep_zero(const Plane &a, const float *h, int taps)
  {
    return sep_fir_zeropad(sep_fir_zeropad(a, h, taps, true), h, taps, false);
  }

  // 2-D FFT of an S x S tile: every row first, then every column, each a 1-D pocketfft transform scaled by fct
  template <bool fwd, typename T>
  static void fft2(fftp::cpx<T> *x, int S, const fftp::Plan &p, const std::vector<fftp::cpx<T>> &tw, T fct,
    std::vector<fftp::cpx<T>> &line, std::vector<fftp::cpx<T>> &scratch)
  {
    line.resize(S);
    scratch.resize(S);
    for (int y = 0; y < S; y++)
      fftp::exec<fwd>(p, x + (size_t)y * S, scratch.data(), tw.data(), fct);
    for (int c = 0; c < S; c++)
    {
      for (int y = 0; y < S; y++)
        line[y] = x[(size_t)y * S + c];
      fftp::exec<fwd>(p, line.data(), scratch.data(), tw.data(), fct);
      for (int y = 0; y < S; y++)
        x[(size_t)y * S + c] = line[y];
    }
  }

  // ------------------------------------------------------------------------------------------------- phase correlation
  // The design and the arithmetic: splat_pcorr.h.

  static void serial_par(int n, const std::function<void(int)> &fn)
  {
    for (int i = 0; i < n; i++)
      fn(i);
  }

  // float plans of the powers of two 16 .. 512 (made once)
  struct FloatPlans
  {
    fftp::Plan p[6];
    std::vector<fftp::cpx<float>> tw[6];
    FloatPlans()
    {
      for (int i = 0; i < 6; i++)
        p[i] = fftp::make_plan<float>(16 << i, tw[i]);
    }
    int index(int S) const
    {
      int i = 0;
      while ((16 << i) < S)
        i++;
      return i;
    }
  };

  const PcorrOps &pcorr_ops_exact()
  {
    static const PcorrOps ops = [] {
      PcorrOps o;
      o.par = serial_par;
      o.fir = [](const Plane &a, const float *h, int taps) { return sep_zero(a, h, taps); };
      o.prepare = [](int) {};
      o.fft2 = [](float *re, float *im, int S, bool fwd, float fct) {
        static const FloatPlans plans;
        const int k = plans.index(S);
        const size_t SS = (size_t)S * S;
        std::vector<fftp::cpx<float>> x(SS), line, scratch;
        for (size_t i = 0; i < SS; i++)
          x[i] = { re[i], im[i] };
        if (fwd)
          fft2<true>(x.data(), S, plans.p[k], plans.tw[k], fct, line, scratch);
        else
          fft2<false>(x.data(), S, plans.p[k], plans.tw[k], fct, line, scratch);
        for (size_t i = 0; i < SS; i++)
        {
          re[i] = x[i].r;
          im[i] = x[i].i;
        }
      };
      return o;
    }();
    return ops;
  }

  // windowed block of pic at (x0, y0), its windowed mean removed, into re (S x S); sw = pcorr_window_sum(S, w). The mean
  // is summed per row, then over the rows (the order a GPU reproduces: one thread per row)
  static void block_tile(const Plane &pic, int x0, int y0, int S, const float *w, float sw, float *re)
  {
    float swx = 0.0f;
    for (int yy = 0; yy < S; yy++)
    {
      float r = 0.0f;
      for (int xx = 0; xx < S; xx++)
        r = r + (w[yy] * w[xx]) * pic.at(x0 + xx, y0 + yy);
      swx = swx + r;
    }
    const float m = sw > 0.0f ? swx / sw : 0.0f;
    for (int yy = 0; yy < S; yy++)
      for (int xx = 0; xx < S; xx++)
        re[(size_t)yy * S + xx] = (pic.at(x0 + xx, y0 + yy) - m) * w[yy] * w[xx];
  }

  PcorrPicture pcorr_ingest(const Plane &B, const Config &cfg, const PcorrOps &ops)
  {
    PcorrPicture o;
    if (cfg.highpass_sigma > 0.0f)
    {
      float g[CFG_MAX_FIR];
      const int n = gauss_taps(cfg.highpass_sigma, g);
      const Plane lp = ops.fir(B, g, n);
      o.hp = Plane(B.w, B.h);
      for (size_t i = 0; i < o.hp.d.size(); i++)
        o.hp.d[i] = B.d[i] - lp.d[i];
    }
    else
      o.hp = B;
    // the stages side by side: one task per pair of blocks (2 t, 2 t + 1) of every stage, the largest blocks first
    PcorrGrid gs[PCORR_NSTAGE];
    std::vector<float> win[PCORR_NSTAGE];
    float sws[PCORR_NSTAGE];
    struct Task { int s, t; };
    std::vector<Task> tasks;
    for (int s = PCORR_NSTAGE - 1; s >= 0; s--)
    {
      if (!cfg.peaks[s])
        continue;
      const PcorrGrid g = gs[s] = pcorr_grid(cfg, s, B.w, B.h);
      const int S = g.S, nb = g.nx * g.ny;
      const size_t SS = (size_t)S * S;
      PcorrSpectrum &sp = o.st[s];
      sp.S = S;
      sp.nblocks = nb;
      sp.R.resize(SS * nb);
      sp.I.resize(SS * nb);
      sp.N.resize(nb);
      win[s].resize((size_t)S);
      pcorr_window(S, cfg.block_taper, win[s].data());
      sws[s] = pcorr_window_sum(S, win[s].data());
      ops.prepare(S);
      for (int t = 0; t < (nb + 1) / 2; t++)
        tasks.push_back({ s, t });
    }
    const Plane &pic = o.hp;
    ops.par((int)tasks.size(), [&](int k) {
      const int s = tasks[(size_t)k].s, t = tasks[(size_t)k].t;
      const PcorrGrid &g = gs[s];
      const int S = g.S, hs = S / 2, nb = g.nx * g.ny;
      const size_t SS = (size_t)S * S;
      PcorrSpectrum &sp = o.st[s];
      const float *w = win[s].data();
      const int b0 = 2 * t, b1 = b0 + 1 < nb ? b0 + 1 : -1;
      std::vector<float> zr(SS), zi(SS, 0.0f), m2(SS - 1);
      block_tile(pic, g.fx + (b0 % g.nx) * hs, g.fy + (b0 / g.nx) * hs, S, w, sws[s], zr.data());
      if (b1 >= 0)
        block_tile(pic, g.fx + (b1 % g.nx) * hs, g.fy + (b1 / g.nx) * hs, S, w, sws[s], zi.data());
      ops.fft2(zr.data(), zi.data(), S, true, 1.0f);
      for (int q = 0; q < 2; q++)
      {
        const int b = q ? b1 : b0;
        if (b < 0)
          break;
        float *R = sp.R.data() + SS * b, *I = sp.I.data() + SS * b;
        for (int ky = 0; ky < S; ky++)
        {
          const int my = (S - ky) & (S - 1);
          for (int kx = 0; kx < S; kx++)
          {
            const size_t i = (size_t)ky * S + kx, j = (size_t)my * S + ((S - kx) & (S - 1));
            if (q == 0)
            {
              R[i] = 0.5f * (zr[i] + zr[j]);
              I[i] = 0.5f * (zi[i] - zi[j]);
            }
            else
            {
              R[i] = 0.5f * (zi[i] + zi[j]);
              I[i] = 0.5f * (zr[j] - zr[i]);
            }
          }
        }
        // noise floor: the median amplitude (DC left out), as the root of the median squared amplitude
        for (size_t i = 1; i < SS; i++)
          m2[i - 1] = R[i] * R[i] + I[i] * I[i];
        const size_t mid = m2.size() / 2;
        std::nth_element(m2.begin(), m2.begin() + (std::ptrdiff_t)mid, m2.end());
        sp.N[b] = std::sqrt(m2[mid]);
      }
    });
    return o;
  }

  // the n highest positive local maxima of an S x S surface (strictly above the 8 neighbours earlier in raster order,
  // at least equal to the later ones; the surface border excluded), ties to the earlier index; returns how many
  static int surface_peaks(const float *C, int S, int n, int *px, int *py, float *ph)
  {
    int cnt = 0;
    for (int y = 1; y < S - 1; y++)
      for (int x = 1; x < S - 1; x++)
      {
        const size_t i = (size_t)y * S + x;
        const float c = C[i];
        if (!(c > 0.0f) || (cnt == n && !(c > ph[n - 1])))
          continue;
        if (!(c > C[i - S - 1] && c > C[i - S] && c > C[i - S + 1] && c > C[i - 1] && c >= C[i + 1] &&
          c >= C[i + S - 1] && c >= C[i + S] && c >= C[i + S + 1]))
          continue;
        int j = cnt < n ? cnt++ : n - 1;
        while (j > 0 && c > ph[j - 1])
        {
          ph[j] = ph[j - 1];
          px[j] = px[j - 1];
          py[j] = py[j - 1];
          j--;
        }
        ph[j] = c;
        px[j] = x;
        py[j] = y;
      }
    return cnt;
  }

  void pcorr_candidates(const PcorrPicture &prev, const PcorrPicture &cur, const Config &cfg, const PcorrOps &ops,
    std::vector<Plane> &out, PcorrGlobal *glob)
  {
    const int W = cur.hp.w, H = cur.hp.h;
    const float k2 = cfg.noise_weight * cfg.noise_weight;
    std::vector<int> vx[PCORR_NSTAGE], vy[PCORR_NSTAGE];
    std::vector<float> h0[PCORR_NSTAGE];
    // the stages side by side: one task per pair of blocks of every stage (X_a + i X_b through one inverse FFT, the
    // surfaces in its real and imaginary part), the largest blocks first
    PcorrGrid gs[PCORR_NSTAGE];
    struct Task { int s, t; };
    std::vector<Task> tasks;
    for (int s = PCORR_NSTAGE - 1; s >= 0; s--)
    {
      if (!cfg.peaks[s])
        continue;
      const PcorrGrid g = gs[s] = pcorr_grid(cfg, s, W, H);
      const int nb = g.nx * g.ny;
      vx[s].resize((size_t)nb * g.npk);
      vy[s].resize((size_t)nb * g.npk);
      h0[s].resize(nb);
      ops.prepare(g.S);
      for (int t = 0; t < (nb + 1) / 2; t++)
        tasks.push_back({ s, t });
    }
    ops.par((int)tasks.size(), [&](int k) {
      const int s = tasks[(size_t)k].s, t = tasks[(size_t)k].t;
      const PcorrGrid &g = gs[s];
      const int S = g.S, nb = g.nx * g.ny, npk = g.npk;
      const size_t SS = (size_t)S * S;
      const PcorrSpectrum &sc = cur.st[s], &sp = prev.st[s];
      const int bb[2] = { 2 * t, 2 * t + 1 < nb ? 2 * t + 1 : -1 };
      std::vector<float> xr(SS, 0.0f), xi(SS, 0.0f);
      for (int q = 0; q < 2; q++)
      {
        const int b = bb[q];
        if (b < 0)
          break;
        const size_t off = SS * b;
        const float *Rc = sc.R.data() + off, *Ic = sc.I.data() + off, *Rp = sp.R.data() + off, *Ip = sp.I.data() + off;
        const float kn = k2 * (sc.N[b] * sp.N[b]);
        for (int ky = 0; ky < S; ky++)
          for (int kx = 0; kx < S; kx++)
          {
            const size_t i = (size_t)ky * S + kx;
            const float rc = Rc[i], ic = Ic[i], rp = Rp[i], ip = Ip[i];
            const float aa = std::sqrt((rc * rc + ic * ic) * (rp * rp + ip * ip));
            // (-1)^(kx + ky) moves the zero shift to the surface centre (shift theorem)
            const float gn = aa > 0.0f ? (((kx + ky) & 1) ? -1.0f : 1.0f) / (aa + kn) : 0.0f;
            const float cr = (rc * rp + ic * ip) * gn, ci = (ic * rp - rc * ip) * gn;
            // X_a + i X_b: a in the real, b in the imaginary part of the result
            if (q == 0)
            {
              xr[i] = cr;
              xi[i] = ci;
            }
            else
            {
              xr[i] = xr[i] - ci;
              xi[i] = xi[i] + cr;
            }
          }
      }
      ops.fft2(xr.data(), xi.data(), S, false, 1.0f / (float)S);
      for (int q = 0; q < 2; q++)
      {
        const int b = bb[q];
        if (b < 0)
          break;
        int px[CFG_MAX_PEAKS], py[CFG_MAX_PEAKS];
        float hh[CFG_MAX_PEAKS];
        const int n = surface_peaks(q ? xi.data() : xr.data(), S, npk, px, py, hh);
        if (n == 0)
        {
          px[0] = S / 2; // no peak at all (a flat block): the zero vector
          py[0] = S / 2;
          hh[0] = 0.0f;
        }
        for (int p = 0; p < npk; p++)
        {
          const bool ok = p == 0 || (p < n && hh[p] >= cfg.peak_ratio * hh[0]);
          vx[s][(size_t)p * nb + b] = (ok ? px[p] : px[0]) - S / 2;
          vy[s][(size_t)p * nb + b] = (ok ? py[p] : py[0]) - S / 2;
        }
        h0[s][b] = hh[0];
      }
    });
    // candidate fields: per stage peak, the blocks containing the pixel (near / far along each axis), nearest first
    struct Job { int s, p, c; };
    std::vector<Job> jobs;
    for (int s = 0; s < PCORR_NSTAGE; s++)
    {
      if (!cfg.peaks[s])
        continue;
      const PcorrGrid g = pcorr_grid(cfg, s, W, H);
      for (int p = 0; p < g.npk; p++)
        for (int c = 0; c < pcorr_fields(g); c++)
          jobs.push_back({ s, p, c });
    }
    const int nglob = pcorr_has_global(cfg);
    out.resize(2 * (jobs.size() + nglob));
    // global: the weighted mode of the blocks' strongest vectors
    float gx = 0.0f, gy = 0.0f, gh = 0.0f;
    if (nglob)
    {
      std::vector<std::pair<long long, long long>> votes;
      for (int s = 0; s < PCORR_NSTAGE; s++)
        for (size_t b = 0; b < h0[s].size(); b++)
        {
          const long long wq = pcorr_vote_weight(h0[s][b]);
          if (wq > 0)
            votes.push_back({ ((long long)vy[s][b] << 32) ^ (long long)(unsigned)vx[s][b], wq });
        }
      std::sort(votes.begin(), votes.end());
      long long best = 0, bw = 0;
      int blen = 0;
      for (size_t i = 0; i < votes.size();)
      {
        size_t j = i;
        long long sum = 0;
        while (j < votes.size() && votes[j].first == votes[i].first)
          sum += votes[j++].second;
        const int x = (int)(unsigned)(votes[i].first & 0xffffffffll), y = (int)(votes[i].first >> 32);
        const int len = std::abs(x) + std::abs(y);
        if (sum > bw || (sum == bw && len < blen))
        {
          bw = sum;
          best = votes[i].first;
          blen = len;
        }
        i = j;
      }
      if (bw > 0)
      {
        gx = (float)(int)(unsigned)(best & 0xffffffffll);
        gy = (float)(int)(best >> 32);
        gh = (float)((double)bw / PCORR_VOTE_ONE);
      }
    }
    if (glob)
      *glob = PcorrGlobal{ gx, gy, gh, gx, gy };
    ops.par((int)jobs.size() + nglob, [&](int j) {
      Plane &ox = out[2 * (size_t)j], &oy = out[2 * (size_t)j + 1];
      if (ox.w != W || ox.h != H)
        ox = Plane(W, H);
      if (oy.w != W || oy.h != H)
        oy = Plane(W, H);
      if (j == (int)jobs.size())
      {
        std::fill(ox.d.begin(), ox.d.end(), gx);
        std::fill(oy.d.begin(), oy.d.end(), gy);
        return;
      }
      const Job &jb = jobs[(size_t)j];
      const PcorrGrid g = pcorr_grid(cfg, jb.s, W, H);
      const int nb = g.nx * g.ny, hs = g.S / 2;
      std::vector<int> bxs(W);
      for (int x = 0; x < W; x++)
      {
        int n, f;
        block_cover(x, g.fx, g.nx, hs, n, f);
        bxs[x] = (jb.c & 1) ? f : n;
      }
      for (int y = 0; y < H; y++)
      {
        int n, f;
        block_cover(y, g.fy, g.ny, hs, n, f);
        const int by = jb.c >= 2 ? f : n;
        const int *rx = vx[jb.s].data() + (size_t)jb.p * nb + (size_t)by * g.nx;
        const int *ry = vy[jb.s].data() + (size_t)jb.p * nb + (size_t)by * g.nx;
        float *dx = ox.d.data() + (size_t)y * W, *dy = oy.d.data() + (size_t)y * W;
        for (int x = 0; x < W; x++)
        {
          dx[x] = (float)rx[bxs[x]];
          dy[x] = (float)ry[bxs[x]];
        }
      }
    });
  }

  std::vector<Plane> pcorr_candidates(const PcorrPicture &prev, const PcorrPicture &cur, const Config &cfg,
    PcorrGlobal *glob)
  {
    std::vector<Plane> out;
    pcorr_candidates(prev, cur, cfg, pcorr_ops_exact(), out, glob);
    return out;
  }
}
