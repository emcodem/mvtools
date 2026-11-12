// SplatFps motion estimation on the GPU: the phase-correlation vector source, the refinement, the forward / backward
// assignment and the chain from the ME pictures to the local fallback of a source picture. Included after
// mvcuda_splat_me.cuh by both variants, inside namespace SPLAT_NS (same exactness rules). Exact CPU path and
// description: splatfps/core/splat_pcorr.h (phase correlation) and splatfps/core/splat_motion.h (refinement,
// assignment).
//
// Techniques: the phase correlation is block phase correlation (Fourier-domain normalised cross-power spectrum, inverse
// transform, search for the highest peaks; G. A. Thomas, BBC RD 1987/11,
// https://downloads.bbc.co.uk/rd/pubs/reports/1987-11.pdf, and US 4,890,160, BBC, 1986) on overlapping windowed blocks
// of 32, 64 and 128; the peaks form a menu of candidate vectors per pixel. The refinement is a weighted Lucas-Kanade
// step per candidate (sub-pixel refinement). The assignment picks per pixel the forward and the backward candidate with
// the lowest biased, filtered displaced frame difference (each side on its own, or as pairs with a speed-change penalty
// when speed_change_cost > 0), and handles occlusions by letting the side that matches supply the vector of the side
// that does not (forward / backward fields per input picture as in US 6,011,596, BBC, 1991).
//
// The phase correlation exists only in the exact variant (compiled without contraction) and both variants use it (state
// API below): it computes the CPU's bits (float and IEEE sqrt / division, the float FFT as pocketfft's mixed-radix
// passes, splatfps/third_party/pocketfft/fftp_pow2.h, one line per thread block with the pass work items spread over
// the threads, rows first). Its candidates are integer peak positions, so the variants share them.

#include "../../splatfps/third_party/pocketfft/fftp_pow2.h"

#include <algorithm>
#include <climits>

// phase-correlation per-picture state (exact variant): spectra of up to 3 pictures in slots, candidates of a pair of
// slots
namespace splat_exact
{
  struct PcorrState;
  PcorrState *pcorr_state_new(cudaStream_t st, int w, int h, const splat::Config &cfg);
  void pcorr_state_free(PcorrState *s);
  // spectra of ME picture B (device, w x h) into slot
  void pcorr_ingest(PcorrState *s, cudaStream_t st, const float *B, int slot);
  // the splat::pcorr_ncand candidate fields (2 planes each, device) of the pair (slot a, slot b); glob: 5 floats
  // (device)
  void pcorr_pair(PcorrState *s, cudaStream_t st, int a, int b, float *cands, float *glob);
  // the prefiltered picture of a slot (device, w x h)
  const float *pcorr_hp(const PcorrState *s, int slot);
}

namespace SPLAT_NS
{
  constexpr int MAX_NC = splat::CFG_MAX_COSTS - 1; // candidates of a set without the zero vector, at most

  static unsigned nblk(size_t n) { return (unsigned)((n + 255) / 256); }

#ifdef SPLAT_EXACT
  // ---- phase correlation -------------------------------------------------------------------------------------------
  // block grids of the stages: splat::pcorr_grid (shared with the CPU)
  using splat::PcorrGrid;

  // symmetric FIR (centre first, then pairs in order), zero outside, along one direction; planes along blockIdx.z
  __global__ void k_sepfir_zp(const float *in, int w, int h, Taps t, int vertical, float *out)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
      return;
    const float *p = in + (size_t)blockIdx.z * w * h;
    auto g = [&](int xx, int yy) { return (xx < 0 || xx >= w || yy < 0 || yy >= h) ? 0.0f : p[(size_t)yy * w + xx]; };
    float acc = t.h[0] * p[(size_t)y * w + x];
    for (int i = 1; i < t.n; i++)
      acc = acc + t.h[i] * (vertical ? g(x, y - i) + g(x, y + i) : g(x - i, y) + g(x + i, y));
    out[(size_t)blockIdx.z * w * h + (size_t)y * w + x] = acc;
  }

  __global__ void k_sub(const float *a, const float *b, size_t n, float *out)
  {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
      out[i] = a[i] - b[i];
  }

  // 1-D transforms of every row (cols = 0) or column (cols = 1) of the S x S tiles: line blockIdx.x of tile blockIdx.y.
  // The line is loaded into shared memory and run through the plan's mixed-radix passes (ping-pong between a and b),
  // the work items of each pass spread over the threads; fct scales the result (1 / S per axis for the inverse)
  template <bool fwd, typename T>
  __global__ void k_fft(fftp::cpx<T> *x, int S, int cols, fftp::Plan p, const fftp::cpx<T> *tw, T fct)
  {
    extern __shared__ unsigned char smem_fft[];
    fftp::cpx<T> *a = (fftp::cpx<T> *)smem_fft, *b = a + S;
    fftp::cpx<T> *base = x + (size_t)blockIdx.y * S * S;
    const int line = blockIdx.x;
    const size_t st = cols ? (size_t)S : 1, off = cols ? (size_t)line : (size_t)line * S;
    for (int j = threadIdx.x; j < S; j += blockDim.x)
      a[j] = base[off + j * st];
    __syncthreads();
    fftp::cpx<T> *p1 = a, *p2 = b;
    for (int s = 0; s < p.npass; s++)
    {
      const int items = p.l1[s] * p.ido[s];
      for (int it = threadIdx.x; it < items; it += blockDim.x)
        fftp::pass_item<fwd>(p, s, it, p1, p2, tw);
      __syncthreads();
      fftp::cpx<T> *t = p1;
      p1 = p2;
      p2 = t;
    }
    for (int j = threadIdx.x; j < S; j += blockDim.x)
    {
      fftp::cpx<T> v = p1[j];
      if (fct != T(1))
        v = { v.r * fct, v.i * fct };
      base[off + j * st] = v;
    }
  }

  constexpr int PEAK_THREADS = 256, PEAK_CHUNK = 4096; // samples per chunk (16 per thread)
  struct PeakEnt
  {
    float v;
    int i;
  };
  __device__ inline bool peak_better(const PeakEnt &a, const PeakEnt &b)
  {
    return a.v > b.v || (a.v == b.v && a.i < b.i);
  }
  // inserts e into the sorted list l of K entries (best first)
  template <int K> __device__ inline void peak_insert(PeakEnt (&l)[K], PeakEnt e)
  {
#pragma unroll
    for (int k = 0; k < K; k++)
      if (peak_better(e, l[k]))
      {
        const PeakEnt t = l[k];
        l[k] = e;
        e = t;
      }
  }
  static unsigned peak_chunks(int S) { return (unsigned)(((size_t)S * S + PEAK_CHUNK - 1) / PEAK_CHUNK); }

  // part: peak_part_size(S, nb) entries
  static size_t peak_part_size(int S, int nb) { return (size_t)peak_chunks(S) * nb * splat::CFG_MAX_PEAKS; }

  // block vectors -> per-pixel candidate fields: candidate c of the block stages belongs to group cg[c] (a stage peak,
  // in candidate order) and takes field cc[c] of it (splat::block_cover: bit 0 the far block column, bit 1 the far
  // row; one field, 0, in nearest-block mode); bv per group: [nb][2] at off
  struct B2P
  {
    int S[splat::CFG_MAX_GROUPS], fx[splat::CFG_MAX_GROUPS], fy[splat::CFG_MAX_GROUPS], nx[splat::CFG_MAX_GROUPS],
      ny[splat::CFG_MAX_GROUPS], off[splat::CFG_MAX_GROUPS];
    signed char cg[splat::CFG_MAX_COSTS], cc[splat::CFG_MAX_COSTS];
    int nc;
  };
  // device constants of the phase correlation per block stage: window (splat::pcorr_window) and its sum, float FFT plan
  // and twiddles
  struct PcorrConst
  {
    int S[3];
    float *win[3];
    float wsum[3];
    fftp::Plan pf[3];
    fftp::cpx<float> *twf[3];
    bool init(Bufs &B, cudaStream_t st, std::vector<std::vector<unsigned char>> &keep, const splat::Config &cfg)
    {
      for (int k = 0; k < 3; k++)
      {
        const int s = cfg.block_size << k;
        S[k] = s;
        std::vector<float> w((size_t)s);
        splat::pcorr_window(s, cfg.block_taper, w.data());
        wsum[k] = splat::pcorr_window_sum(s, w.data());
        std::vector<fftp::cpx<float>> tf;
        pf[k] = fftp::make_plan<float>(s, tf);
        win[k] = B.get(s);
        twf[k] = (fftp::cpx<float> *)B.get(2 * tf.size());
        if (B.failed)
          return false;
        // host copies stay alive until the stream is synchronised
        keep.emplace_back((unsigned char *)w.data(), (unsigned char *)(w.data() + w.size()));
        keep.emplace_back((unsigned char *)tf.data(), (unsigned char *)(tf.data() + tf.size()));
        const size_t n = keep.size();
        cudaMemcpyAsync(win[k], keep[n - 2].data(), keep[n - 2].size(), cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(twf[k], keep[n - 1].data(), keep[n - 1].size(), cudaMemcpyHostToDevice, st);
      }
      return true;
    }
  };

  // per-picture phase-correlation state: the prefiltered picture, per stage the spectrum (real, imaginary part) and per
  // block its noise floor
  struct PcorrPic
  {
    float *hp, *R[3], *I[3], *N[3];
  };

  static size_t pcorr_bins(const splat::Config &c, int k, int w, int h)
  {
    const PcorrGrid g = splat::pcorr_grid(c, k, w, h);
    return (size_t)g.nx * g.ny * g.S * g.S;
  }

  static bool pcorr_alloc(Bufs &B, const splat::Config &c, int w, int h, PcorrPic &p)
  {
    p.hp = B.get((size_t)w * h);
    for (int k = 0; k < 3; k++)
    {
      const PcorrGrid g = splat::pcorr_grid(c, k, w, h);
      const size_t n = c.peaks[k] ? pcorr_bins(c, k, w, h) : 1;
      p.R[k] = B.get(n);
      p.I[k] = B.get(n);
      p.N[k] = B.get((size_t)g.nx * g.ny);
    }
    return !B.failed;
  }

  // scratch of the phase correlation: the complex tiles and the surfaces of the largest stage, 2 planes of the ME
  // picture, the block vectors of all stages, the peak search's chunk lists, the block means of a stage, and per block
  // of all stages the strongest peak height and the global vote (key, weight, summed weight)
  struct PcorrTmp
  {
    fftp::cpx<float> *X;
    float *surf, *s0, *s1, *bv, *mean, *h0;
    PeakEnt *part;
    long long *vkey, *vw, *vsum;
    bool alloc(Bufs &B, const splat::Config &c, int w, int h)
    {
      size_t mx = 1, nv = 0, np = 1, nbt = 0, nbm = 1;
      for (int k = 0; k < 3; k++)
      {
        const PcorrGrid g = splat::pcorr_grid(c, k, w, h);
        mx = std::max(mx, pcorr_bins(c, k, w, h));
        nv += 2 * (size_t)g.npk * g.nx * g.ny;
        np = std::max(np, peak_part_size(g.S, g.nx * g.ny));
        nbt += (size_t)g.nx * g.ny;
        nbm = std::max(nbm, (size_t)g.nx * g.ny);
      }
      X = (fftp::cpx<float> *)B.get(2 * mx);
      surf = B.get(mx);
      s0 = B.get((size_t)w * h);
      s1 = B.get((size_t)w * h);
      bv = B.get(std::max(nv, (size_t)1));
      part = (PeakEnt *)B.get(2 * np);
      mean = B.get(nbm);
      h0 = B.get(nbt);
      vkey = (long long *)B.get(2 * nbt);
      vw = (long long *)B.get(2 * nbt);
      vsum = (long long *)B.get(2 * nbt);
      return !B.failed;
    }
  };

  template <bool fwd, typename T>
  static void fft2_dev(cudaStream_t st, fftp::cpx<T> *x, int S, int tiles, const fftp::Plan &p, const fftp::cpx<T> *tw, T fct)
  {
    const int thr = std::min(S / 2, 128);
    const size_t sm = 2 * (size_t)S * sizeof(fftp::cpx<T>);
    k_fft<fwd, T><<<dim3(S, tiles), thr, sm, st>>>(x, S, 0, p, tw, fct);
    k_fft<fwd, T><<<dim3(S, tiles), thr, sm, st>>>(x, S, 1, p, tw, fct);
  }


  // splat::pcorr_ingest / pcorr_candidates (splatfps/core/splat_pcorr.cpp): the same operations in the same order.
  // Two real blocks share one complex tile (tile t = blocks 2 t and 2 t + 1) in both directions.

  // windowed mean of block b (blockIdx.x): one thread per row sums its row, thread 0 the rows (the CPU's order)
  __global__ void k_pc_mean(const float *pic, int pw, int S, int fx, int fy, int nx, const float *win, float sw,
    float *mean)
  {
    extern __shared__ float pc_rs[];
    const int b = blockIdx.x, x0 = fx + (b % nx) * (S / 2), y0 = fy + (b / nx) * (S / 2);
    for (int yy = threadIdx.x; yy < S; yy += blockDim.x)
    {
      const float *row = pic + (size_t)(y0 + yy) * pw + x0;
      float r = 0.0f;
      for (int xx = 0; xx < S; xx++)
        r = r + (win[yy] * win[xx]) * row[xx];
      pc_rs[yy] = r;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
      float s = 0.0f;
      for (int yy = 0; yy < S; yy++)
        s = s + pc_rs[yy];
      mean[b] = sw > 0.0f ? s / sw : 0.0f;
    }
  }

  // tile blockIdx.z: block 2 t windowed (mean removed) in the real part, block 2 t + 1 (if any) in the imaginary part
  __global__ void k_pc_fill(const float *pic, int pw, int S, int fx, int fy, int nx, int nb, const float *win,
    const float *mean, fftp::cpx<float> *out)
  {
    const int xx = blockIdx.x * blockDim.x + threadIdx.x, yy = blockIdx.y * blockDim.y + threadIdx.y, t = blockIdx.z;
    if (xx >= S || yy >= S)
      return;
    float v[2] = { 0.0f, 0.0f };
    for (int q = 0; q < 2; q++)
    {
      const int b = 2 * t + q;
      if (b >= nb)
        break;
      const int x0 = fx + (b % nx) * (S / 2), y0 = fy + (b / nx) * (S / 2);
      v[q] = (pic[(size_t)(y0 + yy) * pw + x0 + xx] - mean[b]) * win[yy] * win[xx];
    }
    out[(size_t)t * S * S + (size_t)yy * S + xx] = { v[0], v[1] };
  }

  // the two spectra of every tile, separated by their symmetry: R / I of block 2 t from (Z(k) + Z*(-k)) / 2, of block
  // 2 t + 1 from (Z(k) - Z*(-k)) / 2i; n = tiles x S x S
  __global__ void k_pc_unpack(const fftp::cpx<float> *Z, int S, int nb, size_t n, float *R, float *I)
  {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x, SS = (size_t)S * S;
    if (idx >= n)
      return;
    const int t = (int)(idx / SS), i = (int)(idx % SS), ky = i / S, kx = i % S;
    const int j = ((S - ky) & (S - 1)) * S + ((S - kx) & (S - 1));
    const fftp::cpx<float> a = Z[(size_t)t * SS + i], c = Z[(size_t)t * SS + j];
    const size_t o0 = (size_t)(2 * t) * SS + i;
    R[o0] = 0.5f * (a.r + c.r);
    I[o0] = 0.5f * (a.i - c.i);
    if (2 * t + 1 < nb)
    {
      const size_t o1 = o0 + SS;
      R[o1] = 0.5f * (a.i + c.i);
      I[o1] = 0.5f * (c.r - a.r);
    }
  }

  // noise floor of block blockIdx.x: the root of the median squared amplitude (DC left out; the element at index
  // (S S - 1) / 2 of the sorted values, as std::nth_element picks it), by radix selection over the float bits (the
  // values are >= 0, so their bit patterns sort as the values); 256 threads or more
  __global__ void k_pc_median(const float *R, const float *I, int S, float *N)
  {
    __shared__ unsigned hist[256];
    __shared__ unsigned s_pre, s_k;
    const size_t SS = (size_t)S * S, off = (size_t)blockIdx.x * SS;
    if (threadIdx.x == 0)
    {
      s_pre = 0;
      s_k = (unsigned)((SS - 1) / 2);
    }
    unsigned mask = 0;
    for (int shift = 24; shift >= 0; shift -= 8)
    {
      if (threadIdx.x < 256)
        hist[threadIdx.x] = 0;
      __syncthreads();
      const unsigned pre = s_pre;
      for (size_t i = 1 + threadIdx.x; i < SS; i += blockDim.x)
      {
        const float r = R[off + i], q = I[off + i];
        const unsigned u = __float_as_uint(r * r + q * q);
        if ((u & mask) == pre)
          atomicAdd(&hist[(u >> shift) & 255u], 1u);
      }
      __syncthreads();
      // inclusive prefix sums of the histogram (threads 0 .. 255), then the bucket holding the k-th value
      const unsigned k = s_k, d = threadIdx.x;
      const bool act = d < 256;
      unsigned v = act ? hist[d] : 0u;
      for (int o = 1; o < 256; o <<= 1)
      {
        const unsigned add = act && d >= (unsigned)o ? hist[d - o] : 0u;
        __syncthreads();
        v += add;
        if (act)
          hist[d] = v;
        __syncthreads();
      }
      const unsigned below = act && d ? hist[d - 1] : 0u;
      __syncthreads();
      if (act && below <= k && k < v)
      {
        s_k = k - below;
        s_pre = pre | (d << shift);
      }
      mask |= 255u << shift;
      __syncthreads();
    }
    if (threadIdx.x == 0)
      N[blockIdx.x] = sqrtf(__uint_as_float(s_pre));
  }

  // weighted cross power of the tile's two blocks, X_a + i X_b: G G'* / (A A' + k^2 N N'), (-1)^(kx + ky);
  // p = previous, c = current picture; n = tiles x S x S
  __global__ void k_pc_cross(const float *Rp, const float *Ip, const float *Np, const float *Rc, const float *Ic,
    const float *Nc, int S, int nb, size_t n, float k2, fftp::cpx<float> *X)
  {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x, SS = (size_t)S * S;
    if (idx >= n)
      return;
    const int t = (int)(idx / SS), i = (int)(idx % SS), ky = i / S, kx = i % S;
    float xr = 0.0f, xi = 0.0f;
    for (int q = 0; q < 2; q++)
    {
      const int b = 2 * t + q;
      if (b >= nb)
        break;
      const size_t o = (size_t)b * SS + i;
      const float rc = Rc[o], ic = Ic[o], rp = Rp[o], ip = Ip[o];
      const float kn = k2 * (Nc[b] * Np[b]);
      const float aa = sqrtf((rc * rc + ic * ic) * (rp * rp + ip * ip));
      const float gn = aa > 0.0f ? (((kx + ky) & 1) ? -1.0f : 1.0f) / (aa + kn) : 0.0f;
      const float cr = (rc * rp + ic * ip) * gn, ci = (ic * rp - rc * ip) * gn;
      if (q == 0)
      {
        xr = cr;
        xi = ci;
      }
      else
      {
        xr = xr - ci;
        xi = xi + cr;
      }
    }
    X[idx] = { xr, xi };
  }

  // the surfaces of the tiles: block 2 t from the real, 2 t + 1 from the imaginary part
  __global__ void k_pc_split(const fftp::cpx<float> *X, int S, int nb, size_t n, float *surf)
  {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x, SS = (size_t)S * S;
    if (idx >= n)
      return;
    const int t = (int)(idx / SS);
    const size_t i = idx % SS;
    surf[(size_t)(2 * t) * SS + i] = X[idx].r;
    if (2 * t + 1 < nb)
      surf[(size_t)(2 * t + 1) * SS + i] = X[idx].i;
  }

  // the K best positive local maxima of one chunk of a surface (own_peaks of the CPU: > the neighbours earlier in
  // raster order, >= the later ones, border excluded; value, then index)
  template <int K> __global__ void k_pc_peaks_part(const float *C0, int S, PeakEnt *part)
  {
    const int b = blockIdx.y, nchunk = gridDim.x;
    const float *C = C0 + (size_t)b * S * S;
    const int SS = S * S, i0 = blockIdx.x * PEAK_CHUNK, i1 = min(SS, i0 + PEAK_CHUNK);
    PeakEnt l[K];
#pragma unroll
    for (int k = 0; k < K; k++)
      l[k] = { -INFINITY, INT_MAX };
    for (int i = i0 + threadIdx.x; i < i1; i += PEAK_THREADS)
    {
      const float c = C[i];
      // a later sample of this thread loses ties
      if (!(c > 0.0f) || !(c > l[K - 1].v))
        continue;
      const int x = i % S, y = i / S;
      if (x > 0 && x < S - 1 && y > 0 && y < S - 1 && c > C[i - S - 1] && c > C[i - S] && c > C[i - S + 1] &&
        c > C[i - 1] && c >= C[i + 1] && c >= C[i + S - 1] && c >= C[i + S] && c >= C[i + S + 1])
        peak_insert<K>(l, { c, i });
    }
    __shared__ PeakEnt sh[PEAK_THREADS][K];
#pragma unroll
    for (int k = 0; k < K; k++)
      sh[threadIdx.x][k] = l[k];
    __syncthreads();
    for (int s = PEAK_THREADS / 2; s > 0; s >>= 1)
    {
      if (threadIdx.x < s)
      {
#pragma unroll
        for (int k = 0; k < K; k++)
          peak_insert<K>(l, sh[threadIdx.x + s][k]);
#pragma unroll
        for (int k = 0; k < K; k++)
          sh[threadIdx.x][k] = l[k];
      }
      __syncthreads();
    }
    if (threadIdx.x < K)
      part[((size_t)b * nchunk + blockIdx.x) * K + threadIdx.x] = sh[0][threadIdx.x];
  }

  // one thread per surface: the npk best over the chunks, then the CPU's rules (no peak: the zero vector; a further
  // peak below ratio x the strongest: the strongest instead); vec[(p nb + b) 2 + 0/1], h0[b] = the strongest height
  template <int K> __global__ void k_pc_peaks_merge(const PeakEnt *part, int nchunk, int S, int npk, int nb, float ratio,
    float *vec, float *h0)
  {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb)
      return;
    PeakEnt l[K];
#pragma unroll
    for (int k = 0; k < K; k++)
      l[k] = { -INFINITY, INT_MAX };
    for (int c = 0; c < nchunk * K; c++)
      peak_insert<K>(l, part[(size_t)b * nchunk * K + c]);
    int n = 0;
    while (n < npk && l[n].v > 0.0f)
      n++;
    int px0 = S / 2, py0 = S / 2;
    float hh0 = 0.0f;
    if (n > 0)
    {
      px0 = l[0].i % S;
      py0 = l[0].i / S;
      hh0 = l[0].v;
    }
    for (int p = 0; p < npk; p++)
    {
      const bool ok = p == 0 || (p < n && l[p].v >= ratio * hh0);
      const int px = p == 0 || !ok ? px0 : l[p].i % S, py = p == 0 || !ok ? py0 : l[p].i / S;
      vec[((size_t)p * nb + b) * 2] = (float)(px - S / 2);
      vec[((size_t)p * nb + b) * 2 + 1] = (float)(py - S / 2);
    }
    h0[b] = hh0;
  }

  static void pc_peaks_dev(cudaStream_t st, const float *C, int S, int npk, int nb, float ratio, PeakEnt *part, float *vec,
    float *h0)
  {
    const unsigned nc = peak_chunks(S);
    switch (npk)
    {
    case 1:
      k_pc_peaks_part<1><<<dim3(nc, nb), PEAK_THREADS, 0, st>>>(C, S, part);
      k_pc_peaks_merge<1><<<(nb + 63) / 64, 64, 0, st>>>(part, (int)nc, S, npk, nb, ratio, vec, h0);
      break;
    case 2:
      k_pc_peaks_part<2><<<dim3(nc, nb), PEAK_THREADS, 0, st>>>(C, S, part);
      k_pc_peaks_merge<2><<<(nb + 63) / 64, 64, 0, st>>>(part, (int)nc, S, npk, nb, ratio, vec, h0);
      break;
    default:
      k_pc_peaks_part<splat::CFG_MAX_PEAKS><<<dim3(nc, nb), PEAK_THREADS, 0, st>>>(C, S, part);
      k_pc_peaks_merge<splat::CFG_MAX_PEAKS><<<(nb + 63) / 64, 64, 0, st>>>(part, (int)nc, S, npk, nb, ratio, vec, h0);
      break;
    }
  }

  // global candidate: the vector with the largest summed weight over the blocks' strongest peaks
  // (splat::pcorr_vote_weight, fixed point), ties to the shorter |vx| + |vy|, then the smaller key
  // (vy << 32) ^ (unsigned) vx, as the CPU's sorted walk. Votes v of stage s: v0[s] <= v < v0[s + 1], block
  // v - v0[s], vector bv + vo[s] + 2 (v - v0[s])
  struct PcVotes
  {
    int v0[splat::CFG_NSTAGE + 1], vo[splat::CFG_NSTAGE];
  };
  __device__ inline void pc_vote(const float *bv, const float *h0, const PcVotes &V, int v, long long &key, float &h)
  {
    int s = 0;
    while (v >= V.v0[s + 1])
      s++;
    const int b = v - V.v0[s];
    const int vx = (int)bv[V.vo[s] + 2 * b], vy = (int)bv[V.vo[s] + 2 * b + 1];
    key = ((long long)vy << 32) ^ (long long)(unsigned)vx;
    h = h0[v];
  }
  // vote v: its key and weight (splat::pcorr_vote_weight; 0: no vote) into key / wq
  __global__ void k_pc_votes(const float *bv, const float *h0, PcVotes V, long long *key, long long *wq)
  {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V.v0[splat::CFG_NSTAGE])
      return;
    long long k;
    float h;
    pc_vote(bv, h0, V, v, k, h);
    key[v] = k;
    wq[v] = splat::pcorr_vote_weight(h);
  }
  // per vote (blockIdx.x): the summed weight of its vector over all votes (integers: any order gives the same sum)
  constexpr int PC_SUM_THREADS = 128;
  __global__ void k_pc_vote_sums(const long long *key, const long long *wq, int nv, long long *sum)
  {
    __shared__ long long part[PC_SUM_THREADS];
    const int i = blockIdx.x;
    const long long ki = key[i];
    long long s = 0;
    if (wq[i] > 0)
      for (int j = threadIdx.x; j < nv; j += PC_SUM_THREADS)
        if (key[j] == ki)
          s += wq[j];
    part[threadIdx.x] = s;
    __syncthreads();
    for (int o = PC_SUM_THREADS / 2; o > 0; o >>= 1)
    {
      if (threadIdx.x < o)
        part[threadIdx.x] += part[threadIdx.x + o];
      __syncthreads();
    }
    if (threadIdx.x == 0)
      sum[i] = part[0];
  }
  // the best vote (largest sum, then shorter length, then smaller key) -> glob; one thread block
  constexpr int PC_GLOB_THREADS = 512;
  __global__ void k_pc_glob(const long long *key, const long long *sum, int nv, float *glob)
  {
    __shared__ long long s_sum[PC_GLOB_THREADS];
    __shared__ long long s_key[PC_GLOB_THREADS];
    __shared__ int s_len[PC_GLOB_THREADS];
    long long bs = 0, bk = 0;
    int bl = 0;
    for (int i = threadIdx.x; i < nv; i += blockDim.x)
    {
      const long long si = sum[i];
      if (si <= 0)
        continue;
      const long long ki = key[i];
      const int x = (int)(unsigned)(ki & 0xffffffffll), y = (int)(ki >> 32);
      const int len = (x < 0 ? -x : x) + (y < 0 ? -y : y);
      if (si > bs || (si == bs && (len < bl || (len == bl && ki < bk))))
      {
        bs = si;
        bk = ki;
        bl = len;
      }
    }
    s_sum[threadIdx.x] = bs;
    s_key[threadIdx.x] = bk;
    s_len[threadIdx.x] = bl;
    __syncthreads();
    for (int s = PC_GLOB_THREADS / 2; s > 0; s >>= 1)
    {
      if (threadIdx.x < s)
      {
        const long long o = s_sum[threadIdx.x + s], m = s_sum[threadIdx.x];
        const int ol = s_len[threadIdx.x + s], ml = s_len[threadIdx.x];
        const long long ok = s_key[threadIdx.x + s], mk = s_key[threadIdx.x];
        if (o > m || (o == m && o > 0 && (ol < ml || (ol == ml && ok < mk))))
        {
          s_sum[threadIdx.x] = o;
          s_len[threadIdx.x] = ol;
          s_key[threadIdx.x] = ok;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0)
    {
      float gx = 0.0f, gy = 0.0f, gh = 0.0f;
      if (s_sum[0] > 0)
      {
        gx = (float)(int)(unsigned)(s_key[0] & 0xffffffffll);
        gy = (float)(int)(s_key[0] >> 32);
        gh = (float)((double)s_sum[0] / splat::PCORR_VOTE_ONE);
      }
      glob[0] = gx;
      glob[1] = gy;
      glob[2] = gh;
      glob[3] = gx;
      glob[4] = gy;
    }
  }

  // block vectors -> per-pixel candidate fields (block_cover: c & 1 the far block column, c >= 2 the far row); the
  // candidates from nbc on are the global vector (blockIdx.z = candidate)
  // one thread per pixel, all g.nc candidates; the covering blocks are worked out once per block grid (the groups of a
  // stage share it: they differ only in the peak, g.off)
  __global__ void k_pc_cands(const float *bv, B2P g, int nbc, const float *glob, int w, int h, float *cands)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
      return;
    const size_t np = (size_t)w * h;
    float *o = cands + (size_t)y * w + x;
    int last = -1, nxb = 0, fxb = 0, nyb = 0, fyb = 0;
    for (int c = 0; c < g.nc; c++, o += 2 * np)
    {
      float vx, vy;
      if (c >= nbc)
      {
        vx = glob[3];
        vy = glob[4];
      }
      else
      {
        const int gi = g.cg[c], combo = g.cc[c];
        if (last < 0 || g.S[gi] != g.S[last] || g.fx[gi] != g.fx[last] || g.nx[gi] != g.nx[last] ||
            g.fy[gi] != g.fy[last] || g.ny[gi] != g.ny[last])
        {
          const int hs = g.S[gi] / 2;
          splat::block_cover(x, g.fx[gi], g.nx[gi], hs, nxb, fxb);
          splat::block_cover(y, g.fy[gi], g.ny[gi], hs, nyb, fyb);
        }
        last = gi;
        const int bx = (combo & 1) ? fxb : nxb, by = combo >= 2 ? fyb : nyb;
        const float *v = bv + g.off[gi] + 2 * (by * g.nx[gi] + bx);
        vx = v[0];
        vy = v[1];
      }
      o[0] = vx;
      o[np] = vy;
    }
  }

  static void pcorr_ingest_dev(cudaStream_t st, const splat::Config &cfg, const PcorrConst &K, PcorrTmp &T, const float *B,
    int w, int h, PcorrPic &P)
  {
    const dim3 blk(32, 8);
    if (cfg.highpass_sigma > 0.0f)
    {
      Taps g;
      g.n = splat::gauss_taps(cfg.highpass_sigma, g.h);
      k_sepfir_zp<<<grid(w, h, 1, blk), blk, 0, st>>>(B, w, h, g, 1, T.s0);
      k_sepfir_zp<<<grid(w, h, 1, blk), blk, 0, st>>>(T.s0, w, h, g, 0, T.s1);
      k_sub<<<nblk((size_t)w * h), 256, 0, st>>>(B, T.s1, (size_t)w * h, P.hp);
    }
    else
      cudaMemcpyAsync(P.hp, B, sizeof(float) * w * h, cudaMemcpyDeviceToDevice, st);
    for (int k = 0; k < splat::CFG_NSTAGE; k++)
    {
      if (!cfg.peaks[k])
        continue;
      const PcorrGrid g = splat::pcorr_grid(cfg, k, w, h);
      const int S = g.S, nb = g.nx * g.ny, nt = (nb + 1) / 2;
      const size_t n = (size_t)nt * S * S;
      k_pc_mean<<<nb, std::min(S, 128), sizeof(float) * S, st>>>(P.hp, w, S, g.fx, g.fy, g.nx, K.win[k], K.wsum[k], T.mean);
      k_pc_fill<<<grid(S, S, nt, blk), blk, 0, st>>>(P.hp, w, S, g.fx, g.fy, g.nx, nb, K.win[k], T.mean, T.X);
      fft2_dev<true, float>(st, T.X, S, nt, K.pf[k], K.twf[k], 1.0f);
      k_pc_unpack<<<nblk(n), 256, 0, st>>>(T.X, S, nb, n, P.R[k], P.I[k]);
      k_pc_median<<<nb, S >= 64 ? 1024 : 256, 0, st>>>(P.R[k], P.I[k], S, P.N[k]);
    }
  }

  static void pcorr_pair_dev(cudaStream_t st, const splat::Config &cfg, const PcorrConst &K, PcorrTmp &T, const PcorrPic &a,
    const PcorrPic &b, int w, int h, float *cands, float *glob)
  {
    const dim3 blk(32, 8);
    const float k2 = cfg.noise_weight * cfg.noise_weight;
    B2P g;
    PcVotes V;
    int gi = 0, off = 0, c = 0, nv = 0;
    for (int k = 0; k < splat::CFG_NSTAGE; k++)
    {
      V.v0[k] = nv;
      V.vo[k] = off;
      if (!cfg.peaks[k])
        continue;
      const PcorrGrid pg = splat::pcorr_grid(cfg, k, w, h);
      const int S = pg.S, nb = pg.nx * pg.ny, nt = (nb + 1) / 2, npk = pg.npk;
      const size_t n = (size_t)nt * S * S;
      k_pc_cross<<<nblk(n), 256, 0, st>>>(a.R[k], a.I[k], a.N[k], b.R[k], b.I[k], b.N[k], S, nb, n, k2, T.X);
      fft2_dev<false, float>(st, T.X, S, nt, K.pf[k], K.twf[k], 1.0f / (float)S);
      k_pc_split<<<nblk(n), 256, 0, st>>>(T.X, S, nb, n, T.surf);
      pc_peaks_dev(st, T.surf, S, npk, nb, cfg.peak_ratio, T.part, T.bv + off, T.h0 + nv);
      for (int p = 0; p < npk; p++, gi++)
      {
        g.S[gi] = S;
        g.fx[gi] = pg.fx;
        g.fy[gi] = pg.fy;
        g.nx[gi] = pg.nx;
        g.ny[gi] = pg.ny;
        g.off[gi] = off + 2 * p * nb;
        for (int f = 0; f < splat::pcorr_fields(pg); f++, c++)
        {
          g.cg[c] = (signed char)gi;
          g.cc[c] = (signed char)f;
        }
      }
      off += 2 * npk * nb;
      nv += nb;
    }
    V.v0[splat::CFG_NSTAGE] = nv;
    if (splat::pcorr_has_global(cfg))
    {
      k_pc_votes<<<nblk(nv), 256, 0, st>>>(T.bv, T.h0, V, T.vkey, T.vw);
      k_pc_vote_sums<<<nv, PC_SUM_THREADS, 0, st>>>(T.vkey, T.vw, nv, T.vsum);
      k_pc_glob<<<1, PC_GLOB_THREADS, 0, st>>>(T.vkey, T.vsum, nv, glob);
    }
    g.nc = c + splat::pcorr_has_global(cfg);
    k_pc_cands<<<grid(w, h, 1, blk), blk, 0, st>>>(T.bv, g, c, glob, w, h, cands);
  }

  struct PcorrState
  {
    Bufs B;
    std::vector<std::vector<unsigned char>> keep;
    splat::Config cfg;
    PcorrConst K;
    PcorrTmp T;
    PcorrPic pic[3];
    int w, h;
  };

  PcorrState *pcorr_state_new(cudaStream_t st, int w, int h, const splat::Config &cfg)
  {
    PcorrState *s = new PcorrState;
    s->w = w;
    s->h = h;
    s->cfg = cfg;
    bool ok = s->K.init(s->B, st, s->keep, cfg) && s->T.alloc(s->B, cfg, w, h);
    for (int i = 0; i < 3 && ok; i++)
      ok = pcorr_alloc(s->B, cfg, w, h, s->pic[i]);
    if (!ok)
    {
      delete s;
      return nullptr;
    }
    return s;
  }

  void pcorr_state_free(PcorrState *s) { delete s; }

  void pcorr_ingest(PcorrState *s, cudaStream_t st, const float *B, int slot)
  {
    pcorr_ingest_dev(st, s->cfg, s->K, s->T, B, s->w, s->h, s->pic[slot]);
  }

  void pcorr_pair(PcorrState *s, cudaStream_t st, int a, int b, float *cands, float *glob)
  {
    pcorr_pair_dev(st, s->cfg, s->K, s->T, s->pic[a], s->pic[b], s->w, s->h, cands, glob);
  }

  const float *pcorr_hp(const PcorrState *s, int slot) { return s->pic[slot].hp; }
#endif

  // ---- refinement --------------------------------------------------------------------------------------------------
  // Refinement: one weighted Lucas-Kanade (Gauss-Newton) step per candidate, B. D. Lucas, T. Kanade, IJCAI 1981.
  // signed difference d = cur - other(x + s v) of candidate blockIdx.z (s = -1: the candidate points into cur)
  // one thread per pixel, all n candidates; ad = |d|
  __global__ void k_dfd(const float *__restrict__ cur, const float *__restrict__ other, int w, int h,
    const float *__restrict__ vecs, int n, float s, float *__restrict__ d, float *__restrict__ ad)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
      return;
    const size_t np = (size_t)w * h, i = (size_t)y * w + x;
    const float cv = cur[i], fx = (float)x, fy = (float)y;
    const float *v = vecs + i;
#pragma unroll 4
    for (int c = 0; c < n; c++, v += 2 * np)
    {
      const float r = cv - tex_bilinear(other, w, h, fx + s * v[0], fy + s * v[np]);
      d[c * np + i] = r;
      ad[c * np + i] = fabsf(r);
    }
  }

  // weighted Lucas-Kanade terms of candidate blockIdx.z: T[(5 c + k) planes] = b1, b2, a, c, bxy, from the central
  // gradients of cur, the DFD d and the weight exp(-err / wscale) (err: box mean |d| of the unrefined candidate);
  // eps regularises the squared gradients (refine_err_scale 15, error_box 7, refine_reg 0.001 by default)
  __global__ void k_terms(const float *cur, const float *vecs, const float *d, const float *err, int w, int h, int neg,
    float eps, float wscale, float *T)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.z;
    if (x >= w || y >= h)
      return;
    const size_t np = (size_t)w * h, i = (size_t)y * w + x;
    const float gx = (cur[(size_t)y * w + min(x + 1, w - 1)] - cur[(size_t)y * w + max(x - 1, 0)]) / 2.0f;
    const float gy = (cur[(size_t)min(y + 1, h - 1) * w + x] - cur[(size_t)max(y - 1, 0) * w + x]) / 2.0f;
    const float ixx = gx * gx, iyy = gy * gy, ixy = gx * gy;
    const float wt = exp_ref(-err[c * np + i] / wscale);
    float vx = vecs[2 * c * np + i], vy = vecs[(2 * c + 1) * np + i];
    if (neg)
    {
      vx = -vx;
      vy = -vy;
    }
    const float dd = d[c * np + i];
    float b1 = ixx * vx + ixy * vy;
    b1 = b1 + gx * dd;
    float b2 = ixy * vx + iyy * vy;
    b2 = b2 + gy * dd;
    float *o = T + 5 * c * np + i;
    o[0] = b1 * wt;
    o[np] = b2 * wt;
    o[2 * np] = (ixx + eps) * wt;
    o[3 * np] = (iyy + eps) * wt;
    o[4 * np] = ixy * wt;
  }

  // per pixel: solve [a bxy; bxy c] v = [b1 b2] (Cramer's rule) on the box sums (refine_box) -> refined vector
  __global__ void k_solve(const float *Bs, int w, int h, float *out)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.z;
    if (x >= w || y >= h)
      return;
    const size_t np = (size_t)w * h, i = (size_t)y * w + x;
    const float *s = Bs + 5 * c * np + i;
    const float b1 = s[0], b2 = s[np], a = s[2 * np], cc = s[3 * np], bxy = s[4 * np];
    const float det = a * cc - bxy * bxy;
    out[2 * c * np + i] = (cc * b1 - bxy * b2) / det;
    out[(2 * c + 1) * np + i] = (a * b2 - bxy * b1) / det;
  }

#ifndef SPLAT_EXACT
  // k_terms + refine_box sums + k_solve in one pass per TX x TY tile of candidate blockIdx.z (fast variant): the five
  // weighted terms of the tile and its box apron are computed into shared memory, box-summed there (sliding sums along
  // rows, then along columns) and solved, so the 5 term planes never travel through device memory. Same zero padding
  // outside the picture as box_sum.
  // Tile 16 x 16: about 28 KB of shared memory at the default box, so at least two blocks fit an SM on every supported
  // GPU (64 KB per SM on Turing; three per SM on 100 KB parts, measured faster there than 32 x 16 tiles).
  constexpr int REFINE_R_FIXED = 7; // radius of the default refine_box (15): k_refine_strip<7>
  template <int TX, int TY> static size_t refine_fused_smem(int r)
  {
    const size_t hh = TY + 2 * r, hw = TX + 2 * r;
    return sizeof(float) * 5 * hh * (hw + TX);
  }

  // Block 32 x 8 (refine_dev). Plane offsets in int: a plane has fewer than 2^31 samples.
  template <int REF_TX, int REF_TY>
  __global__ void __launch_bounds__(256) k_refine_fused(const float *cur, const float *vecs, const float *d, const float *err,
    int w, int h, int neg, float eps, float wscale, int r, float *out)
  {
    extern __shared__ float sm[];
    const int HH = REF_TY + 2 * r, HW = REF_TX + 2 * r, c = blockIdx.z;
    constexpr int nt = 256;
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int x0 = blockIdx.x * REF_TX - r, y0 = blockIdx.y * REF_TY - r;
    const size_t np = (size_t)w * h;
    float *T = sm;                         // [5][HH][HW] terms; later [5][REF_TY][REF_TX] column sums
    float *Hs = sm + 5 * HH * HW;          // [5][HH][REF_TX] row sums
    const int tp = HH * HW, hp = HH * REF_TX, vp = REF_TY * REF_TX;
    const float *errc = err + c * np, *dc = d + c * np, *vxc = vecs + 2 * c * np, *vyc = vxc + np;
    const float nis = __fdividef(-1.0f, wscale);
    // 1. terms (as k_terms), zero outside the picture
    for (int k = tid; k < tp; k += nt)
    {
      const int ly = k / HW, lx = k - ly * HW, x = x0 + lx, y = y0 + ly;
      float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f, t4 = 0.0f;
      if (x >= 0 && x < w && y >= 0 && y < h)
      {
        const int i = y * w + x;
        const float gx = (cur[i - x + min(x + 1, w - 1)] - cur[i - x + max(x - 1, 0)]) * 0.5f;
        const float gy = (cur[min(y + 1, h - 1) * w + x] - cur[max(y - 1, 0) * w + x]) * 0.5f;
        const float ixx = gx * gx, iyy = gy * gy, ixy = gx * gy;
        const float wt = __expf(errc[i] * nis);
        float vx = vxc[i], vy = vyc[i];
        if (neg)
        {
          vx = -vx;
          vy = -vy;
        }
        const float dd = dc[i];
        float b1 = ixx * vx + ixy * vy;
        b1 = b1 + gx * dd;
        float b2 = ixy * vx + iyy * vy;
        b2 = b2 + gy * dd;
        t0 = b1 * wt;
        t1 = b2 * wt;
        t2 = (ixx + eps) * wt;
        t3 = (iyy + eps) * wt;
        t4 = ixy * wt;
      }
      T[k] = t0;
      T[tp + k] = t1;
      T[2 * tp + k] = t2;
      T[3 * tp + k] = t3;
      T[4 * tp + k] = t4;
    }
    __syncthreads();
    // 2. row sums: item = (term, apron row, segment of 8 columns)
    const int size = 2 * r + 1;
    constexpr int SEG = 8, NSX = REF_TX / SEG, NSY = REF_TY / SEG;
    for (int it = tid; it < 5 * HH * NSX; it += nt)
    {
      const int sg = it % NSX, ly = (it / NSX) % HH, t = it / (NSX * HH), xs = sg * SEG;
      const float *row = T + t * tp + ly * HW + xs;
      float *o = Hs + t * hp + ly * REF_TX + xs;
      float s = 0.0f;
      for (int i = 0; i < size; i++)
        s += row[i];
      o[0] = s;
#pragma unroll
      for (int x = 1; x < SEG; x++)
      {
        s += row[x + 2 * r] - row[x - 1];
        o[x] = s;
      }
    }
    __syncthreads();
    // 3. column sums: item = (term, column, segment of 8 rows) -> T reused as [5][REF_TY][REF_TX]
    for (int it = tid; it < 5 * REF_TX * NSY; it += nt)
    {
      const int lx = it % REF_TX, sg = (it / REF_TX) % NSY, t = it / (REF_TX * NSY), ys = sg * SEG;
      const float *col = Hs + t * hp + ys * REF_TX + lx;
      float *o = T + t * vp + ys * REF_TX + lx;
      float s = 0.0f;
      for (int i = 0; i < size; i++)
        s += col[i * REF_TX];
      o[0] = s;
#pragma unroll
      for (int y = 1; y < SEG; y++)
      {
        s += col[(y + 2 * r) * REF_TX] - col[(y - 1) * REF_TX];
        o[y * REF_TX] = s;
      }
    }
    __syncthreads();
    // 4. solve (as k_solve, one reciprocal of the determinant)
    float *oxc = out + 2 * c * np, *oyc = oxc + np;
    for (int k = tid; k < vp; k += nt)
    {
      const int x = x0 + r + k % REF_TX, y = y0 + r + k / REF_TX;
      if (x >= w || y >= h)
        continue;
      const float b1 = T[k], b2 = T[vp + k], a = T[2 * vp + k], cc = T[3 * vp + k], bxy = T[4 * vp + k];
      const float id = 1.0f / (a * cc - bxy * bxy);
      const int i = y * w + x;
      oxc[i] = (cc * b1 - bxy * b2) * id;
      oyc[i] = (a * b2 - bxy * b1) * id;
    }
  }

  // The same refinement for a box radius R fixed at compile time, one block per strip of RS_W x RS_H output pixels
  // (block 32 x 8): the block walks down the strip's rows plus apron in chunks of RS_CH rows. Per chunk: the five terms
  // of the chunk's rows over the strip's columns plus apron (zero outside the picture), their row sums into a ring of
  // the last RS_RING rows, the column sums of the output rows the chunk completes, and the solve of those rows. Each
  // term is computed (RS_W + 2 R) / RS_W x (RS_H + 2 R) / RS_H times per output instead of the square of the
  // tile's apron ratio. Row and column sums slide over at most RS_CH (or 8) outputs from a fresh sum, as the tile kernel.
  constexpr int RS_W = 32, RS_H = 64, RS_CH = 8, RS_RING = 32;

  template <int R>
  __global__ void __launch_bounds__(256) k_refine_strip(const float *__restrict__ cur, const float *__restrict__ vecs,
    const float *__restrict__ d, const float *__restrict__ err, int w, int h, int neg, float eps, float wscale,
    float *__restrict__ out)
  {
    constexpr int AW = RS_W + 2 * R, AH = RS_H + 2 * R, NCH = (AH + RS_CH - 1) / RS_CH, SEG = 8, NSEG = RS_W / SEG;
    constexpr int BOX = 2 * R + 1;
    static_assert(RS_RING >= BOX + RS_CH - 1 && (RS_RING & (RS_RING - 1)) == 0, "ring too short or not a power of two");
    static_assert(RS_CH * AW <= 512 && 5 * RS_CH * RS_W <= 5 * RS_CH * AW, "term items per thread / V in the term buffer");
    static_assert(5 * RS_CH * NSEG <= 256 && 5 * RS_W <= 256 && RS_CH * 32 == 256, "work items per phase exceed the block");
    __shared__ float Tt[5][RS_CH][AW];     // terms of the chunk's rows; then V
    __shared__ float Hr[5][RS_RING][RS_W]; // row sums, apron row a in slot a % RS_RING
    float (*V)[RS_CH][RS_W] = reinterpret_cast<float (*)[RS_CH][RS_W]>(&Tt[0][0][0]); // box sums of the completed rows
    const int tid = threadIdx.y * 32 + threadIdx.x, c = blockIdx.z;
    const int x0 = blockIdx.x * RS_W, y0 = blockIdx.y * RS_H; // first output pixel; apron row a = picture row y0 - R + a
    const size_t np = (size_t)w * h;
    const float *errc = err + c * np, *dc = d + c * np, *vxc = vecs + 2 * c * np, *vyc = vxc + np;
    const float nis = __fdividef(-1.0f, wscale);
    float *oxc = out + 2 * c * np, *oyc = oxc + np;
    // this thread's term items (chunk row, apron column): q = tid and tid + 256, the same in every chunk
    int qj[2], qx[2];
    for (int u = 0; u < 2; u++)
    {
      const int q = tid + 256 * u;
      qj[u] = q / AW;
      qx[u] = q - qj[u] * AW;
    }
    for (int k = 0; k < NCH; k++)
    {
      const int a0 = k * RS_CH;
      // 1. terms (as k_terms) of rows a0 .. a0 + RS_CH - 1
#pragma unroll
      for (int u = 0; u < 2; u++)
      {
        if (tid + 256 * u >= RS_CH * AW)
          break;
        const int j = qj[u], lx = qx[u], x = x0 - R + lx, y = y0 - R + a0 + j;
        float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f, t4 = 0.0f;
        if (x >= 0 && x < w && y >= 0 && y < h)
        {
          const int i = y * w + x;
          const float gx = (cur[i - x + min(x + 1, w - 1)] - cur[i - x + max(x - 1, 0)]) * 0.5f;
          const float gy = (cur[min(y + 1, h - 1) * w + x] - cur[max(y - 1, 0) * w + x]) * 0.5f;
          const float ixx = gx * gx, iyy = gy * gy, ixy = gx * gy;
          const float wt = __expf(errc[i] * nis);
          float vx = vxc[i], vy = vyc[i];
          if (neg)
          {
            vx = -vx;
            vy = -vy;
          }
          const float dd = dc[i];
          t0 = (ixx * vx + ixy * vy + gx * dd) * wt;
          t1 = (ixy * vx + iyy * vy + gy * dd) * wt;
          t2 = (ixx + eps) * wt;
          t3 = (iyy + eps) * wt;
          t4 = ixy * wt;
        }
        Tt[0][j][lx] = t0;
        Tt[1][j][lx] = t1;
        Tt[2][j][lx] = t2;
        Tt[3][j][lx] = t3;
        Tt[4][j][lx] = t4;
      }
      __syncthreads();
      // 2. row sums: item = (term, chunk row, segment of SEG columns)
      if (tid < 5 * RS_CH * NSEG)
      {
        const int sg = tid % NSEG, j = (tid / NSEG) % RS_CH, t = tid / (NSEG * RS_CH);
        const float *row = &Tt[t][j][sg * SEG];
        float *o = &Hr[t][(a0 + j) & (RS_RING - 1)][sg * SEG];
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < BOX; i++)
          s += row[i];
        o[0] = s;
#pragma unroll
        for (int x = 1; x < SEG; x++)
        {
          s += row[x + 2 * R] - row[x - 1];
          o[x] = s;
        }
      }
      __syncthreads();
      // 3. column sums: item = (term, column); apron row a completes output row a - 2 R
      if (tid < 5 * RS_W && a0 + RS_CH > 2 * R)
      {
        const int t = tid / RS_W, col = tid % RS_W;
        const int j0 = a0 >= 2 * R ? 0 : 2 * R - a0;
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < BOX; i++)
          s += Hr[t][(a0 + j0 - 2 * R + i) & (RS_RING - 1)][col];
        V[t][j0][col] = s;
        for (int j = j0 + 1; j < RS_CH; j++)
        {
          s += Hr[t][(a0 + j) & (RS_RING - 1)][col] - Hr[t][(a0 + j - BOX) & (RS_RING - 1)][col];
          V[t][j][col] = s;
        }
      }
      __syncthreads();
      // 4. solve (as k_solve) the completed output rows: item = (chunk row, column)
      {
        const int col = tid % 32, j = tid / 32, o = a0 + j - 2 * R, x = x0 + col, y = y0 + o;
        if (o >= 0 && o < RS_H && x < w && y < h)
        {
          const float b1 = V[0][j][col], b2 = V[1][j][col], a = V[2][j][col], cc = V[3][j][col], bxy = V[4][j][col];
          const float id = 1.0f / (a * cc - bxy * bxy);
          const int i = y * w + x;
          oxc[i] = (cc * b1 - bxy * b2) * id;
          oyc[i] = (a * b2 - bxy * b1) * id;
        }
      }
      __syncthreads(); // V lives in the term buffer of the next chunk
    }
  }

  // true when the fused refinement (16 x 16 tiles) runs on this device for box radius r: its block must fit twice into
  // an SM's shared memory (one block per SM leaves it latency-bound); else the plane-wise path runs
  static bool refine_fused_ready(int r)
  {
    static int optin = -1, per_sm = 0, reserved = 0;
    if (optin < 0)
    {
      int dev = 0, v = 0;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
      cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
      cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, dev);
      // the kernel is latency-bound: prefer the largest shared-memory carve-out so more tiles fit per SM
      cudaFuncSetAttribute(k_refine_fused<16, 16>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
      optin = v;
    }
    auto fits = [&](size_t need) { return need <= (size_t)optin && 2 * (need + reserved) <= (size_t)per_sm; };
    auto allow = [&](const void *f, size_t need) {
      return need <= 48 * 1024 || cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)need) == cudaSuccess;
    };
    return fits(refine_fused_smem<16, 16>(r)) && allow((const void *)k_refine_fused<16, 16>, refine_fused_smem<16, 16>(r));
  }
#endif

  struct RefTmp
  {
    float *d, *t0, *t1, *T, *T2;
    bool alloc(Bufs &B, size_t np, int n)
    {
      d = B.get(np * n);
      t0 = B.get(np * n);
      t1 = B.get(np * n);
      T = B.get(5 * np * n);
      T2 = B.get(5 * np * n);
      return !B.failed;
    }
  };

  static void refine_dev(cudaStream_t st, const splat::Config &cfg, RefTmp &R, const float *cur, const float *other, int w,
    int h, const float *vecs, int n, bool neg, float *out)
  {
    const dim3 blk(32, 8);
    k_dfd<<<grid(w, h, 1, blk), blk, 0, st>>>(cur, other, w, h, vecs, n, neg ? -1.0f : 1.0f, R.d, R.t0);
    box_sum(st, R.t0, R.t1, R.t0, w, h, n, cfg.error_box, 1, -1.0f);
#ifndef SPLAT_EXACT
    const int r = (cfg.refine_box - 1) / 2;
    if (r == REFINE_R_FIXED)
    {
      k_refine_strip<REFINE_R_FIXED><<<dim3((w + RS_W - 1) / RS_W, (h + RS_H - 1) / RS_H, n), blk, 0, st>>>(cur, vecs, R.d,
        R.t0, w, h, neg ? 1 : 0, cfg.refine_reg, cfg.refine_err_scale, out);
      return;
    }
    if (refine_fused_ready(r))
    {
      k_refine_fused<16, 16><<<dim3((w + 15) / 16, (h + 15) / 16, n), blk, refine_fused_smem<16, 16>(r), st>>>(cur, vecs,
        R.d, R.t0, w, h, neg ? 1 : 0, cfg.refine_reg, cfg.refine_err_scale, r, out);
      return;
    }
#endif
    k_terms<<<grid(w, h, n, blk), blk, 0, st>>>(cur, vecs, R.d, R.t0, w, h, neg ? 1 : 0, cfg.refine_reg,
      cfg.refine_err_scale, R.T);
    box_sum(st, R.T, R.T2, R.T, w, h, 5 * n, cfg.refine_box, 0, -1.0f);
    k_solve<<<grid(w, h, n, blk), blk, 0, st>>>(R.T, w, h, out);
  }

  // ---- match errors for a set (n candidates + the zero vector) -----------------------------------------------------
  static void errors_dev(cudaStream_t st, const splat::Config &cfg, RefTmp &R, const float *cur, const float *other, int w,
    int h, const float *vecs, int n, float *errs)
  {
    const dim3 blk(32, 8);
    k_absdfd<<<grid(w, h, 1, blk), blk, 0, st>>>(cur, other, w, h, vecs, n, R.T);
    box_sum(st, R.T, R.T2, R.T, w, h, n + 1, cfg.error_box, 1, -1.0f);
    k_oppmin<<<grid(w, h, 1, blk), blk, 0, st>>>(R.T, w, h, n + 1, cfg.error_edge_weight, cfg.error_edge_dist, cfg.error_edge_mode,
      errs);
  }

  // ---- assignment --------------------------------------------------------------------------------------------------
  struct Bias { float b[splat::CFG_MAX_COSTS]; };

  // Assignment per ME pixel, as splat::assign, with li = ef[i] + lenc (|vf_i,x| + |vf_i,y|) and lp likewise for the
  // backward side: lam == 0: each side its own lowest li + bias[i] (lp + bias[p]); lam > 0: over all pairs
  // (forward i, backward p) the cost
  //   li + lp + lam (|vf_i,x + vb_p,x| + |vf_i,y + vb_p,y|) + bias[i] + bias[p]
  // (both vectors point away from the centre picture, so constant motion costs nothing); the first strict minimum
  // wins. Then splat::occlusion_switch.
  // n = number of errors per side (candidates + zero); rb == null: forward side only (biased argmin)
  __global__ void k_assign(const float *rf, const float *ef, const float *rb, const float *eb, int n, int w, int h, Bias bias,
    float lam, float thresh, float ratio, float lenc, float *fwd, float *bwd)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
      return;
    const size_t np = (size_t)w * h, k = (size_t)y * w + x;
    float best = INFINITY, vx = 0.0f, vy = 0.0f, e = 0.0f, ovx = 0.0f, ovy = 0.0f, oe = 0.0f;
    if (!rb)
    {
      for (int i = 0; i < n; i++)
      {
        const float sx = i < n - 1 ? rf[2 * i * np + k] : 0.0f, sy = i < n - 1 ? rf[(2 * i + 1) * np + k] : 0.0f;
        const float c = (ef[i * np + k] + lenc * (fabsf(sx) + fabsf(sy))) + bias.b[i];
        if (c < best)
        {
          best = c;
          vx = i < n - 1 ? rf[2 * i * np + k] : 0.0f;
          vy = i < n - 1 ? rf[(2 * i + 1) * np + k] : 0.0f;
          e = ef[i * np + k];
        }
      }
      fwd[k] = vx; fwd[np + k] = vy; fwd[2 * np + k] = e; fwd[3 * np + k] = 0.0f;
      bwd[k] = 0.0f; bwd[np + k] = 0.0f; bwd[2 * np + k] = splat::NO_SIDE_ERROR; bwd[3 * np + k] = 0.0f;
      return;
    }
    if (lam == 0.0f)
    {
      float bb = INFINITY;
      for (int i = 0; i < n; i++)
      {
        const float cx = i < n - 1 ? rf[2 * i * np + k] : 0.0f, cy = i < n - 1 ? rf[(2 * i + 1) * np + k] : 0.0f;
        const float c = (ef[i * np + k] + lenc * (fabsf(cx) + fabsf(cy))) + bias.b[i];
        if (c < best)
        {
          best = c;
          vx = cx; vy = cy; e = ef[i * np + k];
        }
      }
      for (int p = 0; p < n; p++)
      {
        const float ox = p < n - 1 ? rb[2 * p * np + k] : 0.0f, oy = p < n - 1 ? rb[(2 * p + 1) * np + k] : 0.0f;
        const float c = (eb[p * np + k] + lenc * (fabsf(ox) + fabsf(oy))) + bias.b[p];
        if (c < bb)
        {
          bb = c;
          ovx = ox; ovy = oy; oe = eb[p * np + k];
        }
      }
    }
    else
      for (int i = 0; i < n; i++)
      {
        const float cx = i < n - 1 ? rf[2 * i * np + k] : 0.0f, cy = i < n - 1 ? rf[(2 * i + 1) * np + k] : 0.0f;
        const float ei = ef[i * np + k];
        const float li = ei + lenc * (fabsf(cx) + fabsf(cy));
        for (int p = 0; p < n; p++)
        {
          const float ox = p < n - 1 ? rb[2 * p * np + k] : 0.0f, oy = p < n - 1 ? rb[(2 * p + 1) * np + k] : 0.0f;
          const float ep = eb[p * np + k];
          const float lp = ep + lenc * (fabsf(ox) + fabsf(oy));
          const float t = li + lp;
          const float pen = lam * (fabsf(cx + ox) + fabsf(cy + oy));
          float c = t + pen;
          c = c + (bias.b[p] + bias.b[i]);
          if (c < best)
          {
            best = c;
            vx = cx; vy = cy; e = ei;
            ovx = ox; ovy = oy; oe = ep;
          }
        }
      }
    splat::occlusion_switch(vx, vy, e, ovx, ovy, oe, thresh, ratio);
    const float rvx = vx, rvy = vy, e2 = e, rox = ovx, roy = ovy, oe2 = oe;
    fwd[k] = rvx; fwd[np + k] = rvy; fwd[2 * np + k] = e2; fwd[3 * np + k] = 0.0f;
    bwd[k] = rox; bwd[np + k] = roy; bwd[2 * np + k] = oe2; bwd[3 * np + k] = 0.0f;
  }

  __global__ void k_neg(const float *in, size_t n, float *out)
  {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
      out[i] = -in[i];
  }

  // ---- local fallback on both fields (4 planes each, channel 3 written) --------------------------------------------
  // the vector inconsistency weight of both fields, box mean, cap (splat::disorder_weight)
  static void lf_dev(cudaStream_t st, const splat::Config &cfg, int w, int h, float *fwd, float *bwd,
    float *tmp /* 12 np */)
  {
    const dim3 blk(32, 8);
    const size_t np = (size_t)w * h;
    // dv, sm0, sm1: the 4 vector planes (fwd vx, vy, bwd vx, vy) and their smoothings
    float *dv = tmp, *sm0 = tmp + 4 * np, *sm1 = tmp + 8 * np;
    cudaMemcpyAsync(dv, fwd, sizeof(float) * 2 * np, cudaMemcpyDeviceToDevice, st);
    cudaMemcpyAsync(dv + 2 * np, bwd, sizeof(float) * 2 * np, cudaMemcpyDeviceToDevice, st);
    const float h0 = splat::bits_to_float(splat::DISORDER_SMOOTH_BITS[0]), h1 = splat::bits_to_float(splat::DISORDER_SMOOTH_BITS[1]);
    k_g3<<<grid(w, h, 4, blk), blk, 0, st>>>(dv, w, h, 0, h0, h1, sm0);
    k_g3<<<grid(w, h, 4, blk), blk, 0, st>>>(sm0, w, h, 1, h0, h1, sm1);
    // disorder and its box sums (2 planes each) in the space of the first smoothing, no longer needed
    float *dd = sm0, *dch = sm0 + 2 * np;
    k_lf_d<<<grid(w, h, 2, blk), blk, 0, st>>>(dv, sm1, w, h, lf_args(cfg), fwd + 2 * np, bwd + 2 * np, dd);
    box_sum(st, dd, dch, dd, w, h, 2, cfg.fallback_box, 1, cfg.fallback_cap);
    cudaMemcpyAsync(fwd + 3 * np, dd, sizeof(float) * np, cudaMemcpyDeviceToDevice, st);
    cudaMemcpyAsync(bwd + 3 * np, dd + np, sizeof(float) * np, cudaMemcpyDeviceToDevice, st);
  }

  // ---- refinement to local fallback of centre picture c on device buffers -------------------------------------------
  struct FieldsTmp
  {
    RefTmp R;
    float *rf, *rb, *ef, *eb, *lf;
    bool alloc(Bufs &B, int w, int h, int n)
    {
      const size_t np = (size_t)w * h;
      rf = B.get(2 * np * n);
      rb = B.get(2 * np * n);
      ef = B.get(np * (n + 1));
      eb = B.get(np * (n + 1));
      lf = B.get(12 * np);
      return !B.failed && R.alloc(B, np, n + 1);
    }
  };

  // forward side: refine against nxt, errors; backward side (when prv): refine against prv (turning the candidates
  // around unless toward_b), errors; assignment; local fallback.
  // prv null: single side; cf / cb as delivered by the source; fwd / bwd: 4 planes each
  static void fields_dev(cudaStream_t st, const splat::Config &cfg, FieldsTmp &F, const float *prv, const float *cur,
    const float *nxt, int w, int h, const float *cf, const float *cb, int n, bool toward_b, bool refine, const Bias &bias,
    float *fwd, float *bwd)
  {
    const dim3 blk(32, 8);
    const size_t np = (size_t)w * h;
    if (refine)
      refine_dev(st, cfg, F.R, cur, nxt, w, h, cf, n, false, F.rf);
    else
      cudaMemcpyAsync(F.rf, cf, sizeof(float) * 2 * np * n, cudaMemcpyDeviceToDevice, st);
    errors_dev(st, cfg, F.R, cur, nxt, w, h, F.rf, n, F.ef);
    if (prv)
    {
      if (refine)
        refine_dev(st, cfg, F.R, cur, prv, w, h, cb, n, !toward_b, F.rb);
      else if (toward_b)
        cudaMemcpyAsync(F.rb, cb, sizeof(float) * 2 * np * n, cudaMemcpyDeviceToDevice, st);
      else
        k_neg<<<nblk(2 * np * n), 256, 0, st>>>(cb, 2 * np * n, F.rb);
      errors_dev(st, cfg, F.R, cur, prv, w, h, F.rb, n, F.eb);
    }
    k_assign<<<grid(w, h, 1, blk), blk, 0, st>>>(F.rf, F.ef, prv ? F.rb : nullptr, F.eb, n + 1, w, h, bias,
      cfg.speed_change_cost, cfg.occlusion_error, cfg.occlusion_error_factor, cfg.vector_length_cost, fwd, bwd);
    lf_dev(st, cfg, w, h, fwd, bwd, F.lf);
  }

  static Bias bias_of(const float *b, int n1)
  {
    Bias r = {};
    for (int i = 0; i < n1 && i < splat::CFG_MAX_COSTS; i++)
      r.b[i] = b[i];
    return r;
  }

  // the costs of the phase-correlation set of the configuration (n + 1 values)
  static Bias pcorr_bias(const splat::Config &cfg, int w, int h)
  {
    float b[splat::CFG_MAX_COSTS];
    return bias_of(b, splat::pcorr_costs(cfg, w, h, b));
  }

  // ---- host-memory entries (tests) ---------------------------------------------------------------------------------
  int pcorr_host(const MvCudaSplatParams &p, const float *Bp, const float *Bc, float *cands, float *hp, float *glob)
  {
    const splat::Config cfg = cfg_of(p);
    int w, h;
    me_dims(cfg, p.width, p.height, w, h);
    const size_t np = (size_t)w * h;
    const int nc = splat::pcorr_ncand(cfg, w, h);
    mvc::StreamLease lease;
    cudaStream_t st = lease.s;
    Bufs B;
    float *dB0 = B.get(np), *dB1 = B.get(np), *dc = B.get(2 * np * nc), *dg = B.get(8);
    if (B.failed)
      return -2;
    splat_exact::PcorrState *S = splat_exact::pcorr_state_new(st, w, h, cfg);
    if (!S)
      return -2;
    cudaMemcpyAsync(dB0, Bp, sizeof(float) * np, cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(dB1, Bc, sizeof(float) * np, cudaMemcpyHostToDevice, st);
    splat_exact::pcorr_ingest(S, st, dB0, 0);
    splat_exact::pcorr_ingest(S, st, dB1, 1);
    splat_exact::pcorr_pair(S, st, 0, 1, dc, dg);
    cudaMemcpyAsync(cands, dc, sizeof(float) * 2 * np * nc, cudaMemcpyDeviceToHost, st);
    if (hp)
      cudaMemcpyAsync(hp, splat_exact::pcorr_hp(S, 1), sizeof(float) * np, cudaMemcpyDeviceToHost, st);
    if (glob)
      cudaMemcpyAsync(glob, dg, sizeof(float) * 5, cudaMemcpyDeviceToHost, st);
    const int rc = finish(st);
    splat_exact::pcorr_state_free(S);
    return rc;
  }

  int refine_host(const MvCudaSplatParams &p, const float *cur, const float *other, const float *cands, int n, int neg, float *out)
  {
    const splat::Config cfg = cfg_of(p);
    int w, h;
    me_dims(cfg, p.width, p.height, w, h);
    const size_t np = (size_t)w * h;
    mvc::StreamLease lease;
    cudaStream_t st = lease.s;
    Bufs B;
    RefTmp R;
    float *dc = B.get(np), *dn = B.get(np), *dv = B.get(2 * np * n), *dout = B.get(2 * np * n);
    if (B.failed || !R.alloc(B, np, n))
      return -2;
    cudaMemcpyAsync(dc, cur, sizeof(float) * np, cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(dn, other, sizeof(float) * np, cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(dv, cands, sizeof(float) * 2 * np * n, cudaMemcpyHostToDevice, st);
    refine_dev(st, cfg, R, dc, dn, w, h, dv, n, neg != 0, dout);
    cudaMemcpyAsync(out, dout, sizeof(float) * 2 * np * n, cudaMemcpyDeviceToHost, st);
    return finish(st);
  }

  int assign_host(const MvCudaSplatParams &p, const float *rf, const float *ef, const float *rb, const float *eb, int n,
    const float *bias, float *fwd, float *bwd)
  {
    const splat::Config cfg = cfg_of(p);
    int w, h;
    me_dims(cfg, p.width, p.height, w, h);
    const size_t np = (size_t)w * h;
    if (n + 1 > splat::CFG_MAX_COSTS)
      return -4;
    mvc::StreamLease lease;
    cudaStream_t st = lease.s;
    Bufs B;
    float *drf = B.get(2 * np * n), *def = B.get(np * (n + 1)), *drb = B.get(2 * np * n), *deb = B.get(np * (n + 1)),
      *df = B.get(4 * np), *db = B.get(4 * np);
    if (B.failed)
      return -2;
    cudaMemcpyAsync(drf, rf, sizeof(float) * 2 * np * n, cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(def, ef, sizeof(float) * np * (n + 1), cudaMemcpyHostToDevice, st);
    if (rb)
    {
      cudaMemcpyAsync(drb, rb, sizeof(float) * 2 * np * n, cudaMemcpyHostToDevice, st);
      cudaMemcpyAsync(deb, eb, sizeof(float) * np * (n + 1), cudaMemcpyHostToDevice, st);
    }
    const dim3 blk(32, 8);
    k_assign<<<grid(w, h, 1, blk), blk, 0, st>>>(drf, def, rb ? drb : nullptr, deb, n + 1, w, h, bias_of(bias, n + 1),
      cfg.speed_change_cost, cfg.occlusion_error, cfg.occlusion_error_factor, cfg.vector_length_cost, df, db);
    cudaMemcpyAsync(fwd, df, sizeof(float) * 4 * np, cudaMemcpyDeviceToHost, st);
    cudaMemcpyAsync(bwd, db, sizeof(float) * 4 * np, cudaMemcpyDeviceToHost, st);
    return finish(st);
  }

  int fields_host(const MvCudaSplatParams &p, const float *Ap, const float *Ac, const float *An, const float *cand_f,
    const float *cand_b, int n, int toward_b, int refine, const float *bias, float *fwd, float *bwd)
  {
    const splat::Config cfg = cfg_of(p);
    int w, h;
    me_dims(cfg, p.width, p.height, w, h);
    const size_t np = (size_t)w * h;
    if (n + 1 > splat::CFG_MAX_COSTS)
      return -4;
    mvc::StreamLease lease;
    cudaStream_t st = lease.s;
    Bufs B;
    FieldsTmp F;
    float *dp = B.get(np), *dc = B.get(np), *dn = B.get(np), *vf = B.get(2 * np * n), *vb = B.get(2 * np * n),
      *df = B.get(4 * np), *db = B.get(4 * np);
    if (B.failed || !F.alloc(B, w, h, n))
      return -2;
    if (Ap)
    {
      cudaMemcpyAsync(dp, Ap, sizeof(float) * np, cudaMemcpyHostToDevice, st);
      cudaMemcpyAsync(vb, cand_b, sizeof(float) * 2 * np * n, cudaMemcpyHostToDevice, st);
    }
    cudaMemcpyAsync(dc, Ac, sizeof(float) * np, cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(dn, An, sizeof(float) * np, cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(vf, cand_f, sizeof(float) * 2 * np * n, cudaMemcpyHostToDevice, st);
    fields_dev(st, cfg, F, Ap ? dp : nullptr, dc, dn, w, h, vf, vb, n, toward_b != 0, refine != 0, bias_of(bias, n + 1), df,
      db);
    cudaMemcpyAsync(fwd, df, sizeof(float) * 4 * np, cudaMemcpyDeviceToHost, st);
    cudaMemcpyAsync(bwd, db, sizeof(float) * 4 * np, cudaMemcpyDeviceToHost, st);
    return finish(st);
  }

  // ME pictures A and B of a luma plane (device), into a / b
  static void me_pics_dev(cudaStream_t st, const splat::Config &cfg, const float *y, int W, int H, int mw, int mh, float *tmp,
    float *a, float *b)
  {
    const dim3 blk(32, 8);
    for (int fir = 0; fir < 2; fir++)
    {
      const Taps t = taps_of_cfg(cfg, fir);
      k_me_h<<<grid(mw, H, 1, blk), blk, 0, st>>>(y, W, H, t, tmp, mw, cfg.pad_x, cfg.scale);
      k_me_v<<<grid(mw, mh, 1, blk), blk, 0, st>>>(tmp, mw, H, t, fir ? b : a, mh, cfg.pad_y, cfg.scale);
    }
  }

  int fields_luma_host(const MvCudaSplatParams &p, const float *yp, const float *yc, const float *yn, float *fwd, float *bwd,
    float *ms)
  {
    const splat::Config cfg = cfg_of(p);
    int w, h;
    me_dims(cfg, p.width, p.height, w, h);
    const int NC = splat::pcorr_ncand(cfg, w, h);
    const int W = p.width, H = p.height;
    const size_t np = (size_t)w * h, NP = (size_t)W * H;
    mvc::StreamLease lease;
    cudaStream_t st = lease.s;
    Bufs B;
    FieldsTmp F;
    float *dy[3], *A[3], *Bm[3];
    for (int i = 0; i < 3; i++)
    {
      dy[i] = B.get(NP);
      A[i] = B.get(np);
      Bm[i] = B.get(np);
    }
    float *tmp = B.get((size_t)w * H), *vf = B.get(2 * np * NC), *vb = B.get(2 * np * NC), *dg = B.get(8),
      *df = B.get(4 * np), *db = B.get(4 * np);
    if (B.failed || !F.alloc(B, w, h, NC))
      return -2;
    splat_exact::PcorrState *S = splat_exact::pcorr_state_new(st, w, h, cfg);
    if (!S)
      return -2;
    const float *src[3] = { yp, yc, yn };
    for (int i = 0; i < 3; i++)
      if (src[i])
        cudaMemcpyAsync(dy[i], src[i], sizeof(float) * NP, cudaMemcpyHostToDevice, st);
    cudaEvent_t ev[5];
    for (auto &e : ev)
      cudaEventCreate(&e);
    cudaEventRecord(ev[0], st);
    // one source picture's ingest (timed: the centre picture's), then the others
    me_pics_dev(st, cfg, dy[1], W, H, w, h, tmp, A[1], Bm[1]);
    splat_exact::pcorr_ingest(S, st, Bm[1], 1);
    cudaEventRecord(ev[1], st);
    me_pics_dev(st, cfg, dy[2], W, H, w, h, tmp, A[2], Bm[2]);
    splat_exact::pcorr_ingest(S, st, Bm[2], 2);
    if (yp)
    {
      me_pics_dev(st, cfg, dy[0], W, H, w, h, tmp, A[0], Bm[0]);
      splat_exact::pcorr_ingest(S, st, Bm[0], 0);
    }
    cudaEventRecord(ev[2], st);
    // one pair's candidates (timed), then the other
    splat_exact::pcorr_pair(S, st, 1, 2, vf, dg);
    cudaEventRecord(ev[3], st);
    if (yp)
      splat_exact::pcorr_pair(S, st, 0, 1, vb, dg);
    cudaEventRecord(ev[4], st);
    cudaEvent_t e5, e6;
    cudaEventCreate(&e5);
    cudaEventCreate(&e6);
    cudaEventRecord(e5, st);
    fields_dev(st, cfg, F, yp ? A[0] : nullptr, A[1], A[2], w, h, vf, vb, NC, false, true, pcorr_bias(cfg, w, h), df, db);
    cudaEventRecord(e6, st);
    cudaMemcpyAsync(fwd, df, sizeof(float) * 4 * np, cudaMemcpyDeviceToHost, st);
    cudaMemcpyAsync(bwd, db, sizeof(float) * 4 * np, cudaMemcpyDeviceToHost, st);
    const int rc = finish(st);
    splat_exact::pcorr_state_free(S);
    if (!rc && ms)
    {
      // ms: ingest of one picture (ME pictures + spectra), candidates of one pair, the fields of one centre picture,
      // their sum
      cudaEventElapsedTime(&ms[0], ev[0], ev[1]);
      cudaEventElapsedTime(&ms[1], ev[2], ev[3]);
      cudaEventElapsedTime(&ms[2], e5, e6);
      ms[3] = ms[0] + ms[1] + ms[2];
    }
    for (auto &e : ev)
      cudaEventDestroy(e);
    cudaEventDestroy(e5);
    cudaEventDestroy(e6);
    return rc;
  }
}
