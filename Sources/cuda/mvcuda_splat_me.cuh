// SplatFps stages around the motion vectors on the GPU: ME pictures, match errors (filtered displaced frame
// difference), local fallback (vector-field inconsistency) and the sums of the pair statistic (scene cut, temporal
// fallback). Included after mvcuda_splat.cuh by both variants, inside namespace SPLAT_NS (same exactness rules). Exact
// CPU path and description: splatfps/core/splat_me.cpp.

#include "../../splatfps/core/splat_const.h"
#include "../../splatfps/core/splat_phase_pull.h"

namespace SPLAT_NS
{
  struct Taps { float h[splat::CFG_MAX_FIR]; int n; };

  // ---- ME pictures: every sc-th sample of the symmetric FIR, horizontal pass (overscan px) then vertical (py):
  // luma W x H -> ME picture mw x mh (filter A for errors / refinement / local fallback, B for phase correlation)
  __global__ void k_me_h(const float *y, int W, int H, Taps t, float *out, int mw, int px, int sc)
  {
    const int j = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= mw || row >= H)
      return;
    const float *s = y + (size_t)row * W;
    const int pos = j * sc - px;
    auto g = [&](int x) { return s[clampi(x, 0, W - 1)]; };
    float acc = t.h[0] * g(pos);
    for (int i = 1; i < t.n; i++)
      acc = acc + t.h[i] * (g(pos - i) + g(pos + i));
    out[(size_t)row * mw + j] = acc;
  }

  __global__ void k_me_v(const float *in, int mw, int H, Taps t, float *out, int mh, int py, int sc)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= mw || j >= mh)
      return;
    const int pos = j * sc - py;
    auto g = [&](int y) { return in[(size_t)clampi(y, 0, H - 1) * mw + x]; };
    float acc = t.h[0] * g(pos);
    for (int i = 1; i < t.n; i++)
      acc = acc + t.h[i] * (g(pos - i) + g(pos + i));
    out[(size_t)j * mw + x] = acc;
  }

  // ---- match errors: per candidate the box mean (error_box) of the absolute displaced frame difference
  // |cur(x) - other(x + v)|, then the edge-aware filter
  // bilinear sample at (x, y) with edge clamp
  __device__ __forceinline__ float tex_bilinear(const float *img, int w, int h, float x, float y)
  {
    const float i0f = floorf(x), j0f = floorf(y);
    const float a = x - i0f, b = y - j0f;
    const int i0 = (int)i0f, j0 = (int)j0f;
    const float t00 = ldc(img, w, h, i0, j0), t10 = ldc(img, w, h, i0 + 1, j0);
    const float t01 = ldc(img, w, h, i0, j0 + 1), t11 = ldc(img, w, h, i0 + 1, j0 + 1);
    float r = ((1.0f - a) * (1.0f - b)) * t00;
    r = r + (a * (1.0f - b)) * t10;
    r = r + ((1.0f - a) * b) * t01;
    r = r + (a * b) * t11;
    return r;
  }

  // |dfd| of candidate blockIdx.z (vecs: n x (vx, vy) planes; z == n: the zero vector)
  __global__ void k_absdfd(const float *cur, const float *other, int w, int h, const float *vecs, int n, float *out)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.z;
    if (x >= w || y >= h)
      return;
    const size_t np = (size_t)w * h, i = (size_t)y * w + x;
    const float vx = c < n ? vecs[(size_t)2 * c * np + i] : 0.0f, vy = c < n ? vecs[(size_t)(2 * c + 1) * np + i] : 0.0f;
    out[c * np + i] = fabsf(cur[i] - tex_bilinear(other, w, h, (float)x + vx, (float)y + vy));
  }

  // zero-padded box: vertical sums (rows y - r .. y + r in order), then horizontal sums; planes along blockIdx.z
  __global__ void k_box_v(const float *in, int w, int h, int size, float *out)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
      return;
    const size_t off = (size_t)blockIdx.z * w * h;
    const int r = (size - 1) / 2;
    auto g = [&](int yy) { return (yy < 0 || yy >= h) ? 0.0f : in[off + (size_t)yy * w + x]; };
    float s = g(y - r);
    for (int i = 1; i < size; i++)
      s = s + g(y - r + i);
    out[off + (size_t)y * w + x] = s;
  }

  // max_val < 0: no cap
  __global__ void k_box_h(const float *in, int w, int h, int size, int average, float max_val, float *out)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
      return;
    const size_t off = (size_t)blockIdx.z * w * h + (size_t)y * w;
    const int r = (size - 1) / 2;
    auto g = [&](int xx) { return (xx < 0 || xx >= w) ? 0.0f : in[off + xx]; };
    float s = g(x - r);
    for (int i = 1; i < size; i++)
      s = s + g(x - r + i);
    if (average)
      s = s / (float)(size * size);
    if (max_val >= 0.0f)
      s = fminf(max_val, s);
    out[off + x] = s;
  }

  // edge-aware error filter, mode 0: (1 - wq) e + wq / 2 (min of e D px left / right + min of e D px up / down; D =
  // error_edge_dist, 3 by default); next to a motion edge one of each opposite pair lies on the side that owns the vector
  // mode 1: shiftable windows, (1 - wq) e + wq min of e at the four diagonal shifts (+-D, +-D)
  __global__ void k_oppmin(const float *e, int w, int h, float wq, int D, int mode, float *out)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
      return;
    const float *p = e + (size_t)blockIdx.z * w * h;
    const float l = p[(size_t)y * w + max(x - D, 0)], r = p[(size_t)y * w + min(x + D, w - 1)];
    const float u = p[(size_t)max(y - D, 0) * w + x], d = p[(size_t)min(y + D, h - 1) * w + x];
    const float half = wq * 0.5f, keep = 1.0f - wq;
    if (mode == 1)
    {
      const int xl = max(x - D, 0), xr = min(x + D, w - 1);
      const size_t ru = (size_t)max(y - D, 0) * w, rd = (size_t)min(y + D, h - 1) * w;
      const float m = fminf(fminf(p[ru + xl], p[ru + xr]), fminf(p[rd + xl], p[rd + xr]));
      out[(size_t)blockIdx.z * w * h + (size_t)y * w + x] = keep * p[(size_t)y * w + x] + wq * m;
      return;
    }
    out[(size_t)blockIdx.z * w * h + (size_t)y * w + x] = keep * p[(size_t)y * w + x] + half * (fminf(l, r) + fminf(u, d));
  }

  // ---- local fallback (splat::disorder_weight): per field the inconsistency of the vectors against their 3x3
  // Gaussian smoothing, a ramp, spread over a box and capped
  // 3-tap Gaussian, edge repeat, along one direction; planes along blockIdx.z
  __global__ void k_g3(const float *in, int w, int h, int horizontal, float h0, float h1, float *out)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
      return;
    const float *p = in + (size_t)blockIdx.z * w * h;
    const float pair = horizontal ? ldc(p, w, h, x - 1, y) + ldc(p, w, h, x + 1, y) : ldc(p, w, h, x, y - 1) + ldc(p, w, h, x, y + 1);
    const float acc = h0 * p[(size_t)y * w + x];
    out[(size_t)blockIdx.z * w * h + (size_t)y * w + x] = acc + h1 * pair;
  }

  // inconsistency weight of fields blockIdx.z: v = 2 planes per field, sm = their smoothed planes;
  // out = strength * clamp((|v - sm| - dead) / range, 0, 1)
  // plus the error term of splat::disorder_weight: e0 / e1 the assigned error planes of fields 0 / 1 (or null)
  struct LfArgs { float strength, dead, range, elo, erange, egain; };
  __global__ void k_lf_d(const float *v, const float *sm, int w, int h, LfArgs a, const float *e0, const float *e1,
    float *out)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
      return;
    const size_t np = (size_t)w * h, i = (size_t)y * w + x, f = blockIdx.z;
    const float ex = v[2 * f * np + i] - sm[2 * f * np + i], ey = v[(2 * f + 1) * np + i] - sm[(2 * f + 1) * np + i];
    const float dist = sqrtf(ex * ex + ey * ey);
    float r = a.strength * fminf(fmaxf((dist - a.dead) / a.range, 0.0f), 1.0f);
    const float *e = f == 0 ? e0 : e1;
    if (e && a.egain > 0.0f)
      r = r + a.egain * fminf(fmaxf((e[i] - a.elo) / a.erange, 0.0f), 1.0f);
    out[f * np + i] = r;
  }
  static LfArgs lf_args(const splat::Config &c)
  {
    return { c.fallback_strength, c.fallback_dead_zone, c.fallback_range, c.fallback_err_lo, c.fallback_err_range,
      c.fallback_err_gain };
  }

  // ---- pair-statistic sums: per row of the window [y0, y1) x [x0, x1), the capped match errors of the forward field
  // of P and of the backward field of N; rows: 2 x (y1 - y0) doubles. The host finishes with
  // splat::phase_pull_from_rows (splat_phase_pull.h), as the CPU does.
  __global__ void k_tf_rows(const float *fP, const float *fN, int W, int H, int x0, int x1, int y0, int y1, float cap,
    double *rows)
  {
    const int r = blockIdx.x * blockDim.x + threadIdx.x, n_rows = y1 - y0;
    if (r >= n_rows)
      return;
    const size_t n = (size_t)W * H, row = (size_t)(y0 + r) * W;
    double rP = 0, rN = 0;
    for (int x = x0; x < x1; x++)
    {
      rP = rP + (double)fminf(cap, fP[2 * n + row + x]);
      rN = rN + (double)fminf(cap, fN[2 * n + row + x]);
    }
    rows[r] = rP;
    rows[n_rows + r] = rN;
  }

  // the window of the pair-statistic sums on a W x H grid
  struct TfWin
  {
    int x0, x1, y0, y1;
  };
  static TfWin tf_win(const splat::Config &c, int W, int H)
  {
    return { splat::error_window_lo(W, c), splat::error_window_hi(W, c), splat::error_window_lo(H, c),
      splat::error_window_hi(H, c) };
  }
  static void tf_launch(cudaStream_t st, const splat::Config &c, const float *fP, const float *fN, int W, int H,
    double *rows)
  {
    const TfWin t = tf_win(c, W, H);
    const int n = t.y1 - t.y0;
    if (n > 0)
      k_tf_rows<<<(n + 63) / 64, 64, 0, st>>>(fP, fN, W, H, t.x0, t.x1, t.y0, t.y1, c.error_window_cap, rows);
  }

  // ---- host-memory entries (tests)
  static int finish(cudaStream_t st)
  {
    return (cudaStreamSynchronize(st) != cudaSuccess || cudaGetLastError() != cudaSuccess) ? -2 : 0;
  }

  static void me_dims(const splat::Config &c, int width, int height, int &mw, int &mh)
  {
    mw = splat::me_width(c, width);
    mh = splat::me_height(c, height);
  }

  // the pair-statistic sums of a pair of fields into the host struct (rows: 2 x window rows doubles, as k_tf_rows
  // writes them)
  static splat::PullSums pull_sums(const splat::Config &c, int W, int H, const double *hr)
  {
    const TfWin t = tf_win(c, W, H);
    const int n = std::max(0, t.y1 - t.y0);
    splat::PullSums r;
    r.W = W;
    r.H = H;
    r.nwin = (size_t)n * std::max(0, t.x1 - t.x0);
    r.winP.assign(hr, hr + n);
    r.winN.assign(hr + n, hr + 2 * n);
    return r;
  }

  static Taps taps_of_cfg(const splat::Config &c, int fir)
  {
    Taps t;
    t.n = fir ? c.fir_b_n : c.fir_a_n;
    for (int i = 0; i < t.n; i++)
      t.h[i] = fir ? c.fir_b[i] : c.fir_a[i];
    return t;
  }

  int me_picture_host(const MvCudaSplatParams &p, const float *y, int fir, float *out)
  {
    const splat::Config cfg = cfg_of(p);
    int mw, mh;
    me_dims(cfg, p.width, p.height, mw, mh);
    const int W = p.width, H = p.height;
    const Taps t = taps_of_cfg(cfg, fir);
    mvc::StreamLease lease;
    cudaStream_t st = lease.s;
    Bufs B;
    float *dy = B.get((size_t)W * H), *tmp = B.get((size_t)mw * H), *dout = B.get((size_t)mw * mh);
    if (B.failed)
      return -2;
    const dim3 blk(32, 8);
    cudaMemcpyAsync(dy, y, sizeof(float) * W * H, cudaMemcpyHostToDevice, st);
    k_me_h<<<grid(mw, H, 1, blk), blk, 0, st>>>(dy, W, H, t, tmp, mw, cfg.pad_x, cfg.scale);
    k_me_v<<<grid(mw, mh, 1, blk), blk, 0, st>>>(tmp, mw, H, t, dout, mh, cfg.pad_y, cfg.scale);
    cudaMemcpyAsync(out, dout, sizeof(float) * mw * mh, cudaMemcpyDeviceToHost, st);
    return finish(st);
  }

  int errors_host(const MvCudaSplatParams &p, const float *cur, const float *other, const float *vecs, int n, float *errs)
  {
    const splat::Config cfg = cfg_of(p);
    int w, h;
    me_dims(cfg, p.width, p.height, w, h);
    const size_t np = (size_t)w * h;
    mvc::StreamLease lease;
    cudaStream_t st = lease.s;
    Bufs B;
    float *dc = B.get(np), *dn = B.get(np), *dv = B.get(2 * np * (n > 0 ? n : 1)), *t0 = B.get(np * (n + 1)),
      *t1 = B.get(np * (n + 1));
    if (B.failed)
      return -2;
    const dim3 blk(32, 8);
    cudaMemcpyAsync(dc, cur, sizeof(float) * np, cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(dn, other, sizeof(float) * np, cudaMemcpyHostToDevice, st);
    if (n > 0)
      cudaMemcpyAsync(dv, vecs, sizeof(float) * 2 * np * n, cudaMemcpyHostToDevice, st);
    k_absdfd<<<grid(w, h, n + 1, blk), blk, 0, st>>>(dc, dn, w, h, dv, n, t0);
    k_box_v<<<grid(w, h, n + 1, blk), blk, 0, st>>>(t0, w, h, cfg.error_box, t1);
    k_box_h<<<grid(w, h, n + 1, blk), blk, 0, st>>>(t1, w, h, cfg.error_box, 1, -1.0f, t0);
    k_oppmin<<<grid(w, h, n + 1, blk), blk, 0, st>>>(t0, w, h, cfg.error_edge_weight, cfg.error_edge_dist, cfg.error_edge_mode, t1);
    cudaMemcpyAsync(errs, t1, sizeof(float) * np * (n + 1), cudaMemcpyDeviceToHost, st);
    return finish(st);
  }

  int disorder_weight_host(const MvCudaSplatParams &p, const float *fv, int count, float *ch3)
  {
    const splat::Config cfg = cfg_of(p);
    int w, h;
    me_dims(cfg, p.width, p.height, w, h);
    const size_t np = (size_t)w * h;
    mvc::StreamLease lease;
    cudaStream_t st = lease.s;
    Bufs B;
    float *dv = B.get(2 * np * count), *sm0 = B.get(2 * np * count), *sm1 = B.get(2 * np * count), *dd = B.get(np * count),
      *dch = B.get(np * count);
    if (B.failed)
      return -2;
    const dim3 blk(32, 8);
    cudaMemcpyAsync(dv, fv, sizeof(float) * 2 * np * count, cudaMemcpyHostToDevice, st);
    // channel 3 of each field
    const float h0 = splat::bits_to_float(splat::DISORDER_SMOOTH_BITS[0]), h1 = splat::bits_to_float(splat::DISORDER_SMOOTH_BITS[1]);
    k_g3<<<grid(w, h, 2 * count, blk), blk, 0, st>>>(dv, w, h, 0, h0, h1, sm0);
    k_g3<<<grid(w, h, 2 * count, blk), blk, 0, st>>>(sm0, w, h, 1, h0, h1, sm1);
    k_lf_d<<<grid(w, h, count, blk), blk, 0, st>>>(dv, sm1, w, h, lf_args(cfg), nullptr, nullptr, dd);
    k_box_v<<<grid(w, h, count, blk), blk, 0, st>>>(dd, w, h, cfg.fallback_box, dch);
    k_box_h<<<grid(w, h, count, blk), blk, 0, st>>>(dch, w, h, cfg.fallback_box, 1, cfg.fallback_cap, dd);
    cudaMemcpyAsync(ch3, dd, sizeof(float) * np * count, cudaMemcpyDeviceToHost, st);
    return finish(st);
  }

  int phase_pull_host(const MvCudaSplatParams &p, const float *fP, const float *fN, double d, double strength, double *d_adj,
    double *info)
  {
    const splat::Config cfg = cfg_of(p);
    int W, H;
    me_dims(cfg, p.width, p.height, W, H);
    const size_t np = (size_t)W * H;
    mvc::StreamLease lease;
    cudaStream_t st = lease.s;
    Bufs B;
    float *dP = B.get(4 * np), *dN = B.get(4 * np);
    double *rows = (double *)B.get(2 * 2 * (size_t)H);
    if (B.failed)
      return -2;
    cudaMemcpyAsync(dP, fP, sizeof(float) * 4 * np, cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(dN, fN, sizeof(float) * 4 * np, cudaMemcpyHostToDevice, st);
    tf_launch(st, cfg, dP, dN, W, H, rows);
    std::vector<double> hr(2 * (size_t)H);
    cudaMemcpyAsync(hr.data(), rows, sizeof(double) * 2 * H, cudaMemcpyDeviceToHost, st);
    if (finish(st))
      return -2;
    *d_adj = splat::phase_pull_from_rows(pull_sums(cfg, W, H, hr.data()), d, strength, cfg, info);
    return 0;
  }
}
