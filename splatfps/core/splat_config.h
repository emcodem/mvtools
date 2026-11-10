// splat_config: the run-time configuration of SplatFps (geometry of the motion measurement and every tuned scalar of
// the stages), its defaults per frame size, its checks, and the small block-grid helpers shared by the CPU code and the
// CUDA kernels. Header only; Config is plain data (copied by value into CUDA kernels).
//
// Life cycle: config_defaults() gives the tuned scalars and marks the size-dependent fields "auto" (0); the caller may
// override any field (filter arguments); config_resolve(cfg, w, h) fills every field still auto from the rules below
// and generates the ME filter taps; config_check() validates the result. config_for_size(w, h) is defaults + resolve.
#pragma once

#include "splat_const.h"

#include <cmath>
#include <cstdio>
#include <cstring>

#if defined(__CUDACC__)
#define SPLAT_HD __host__ __device__
#else
#define SPLAT_HD
#endif

namespace splat
{
  constexpr int CFG_NSTAGE = 3;                             // phase-correlation block stages
  constexpr int CFG_MAX_PEAKS = 4;                          // peaks per block stage
  constexpr int CFG_MAX_GROUPS = CFG_NSTAGE * CFG_MAX_PEAKS; // (stage, peak) candidate groups
  constexpr int CFG_MAX_COSTS = 32;                         // candidates incl. global and zero vector
  constexpr int CFG_MAX_FIR = 64;                           // ME filter taps per side incl. centre

  // stop-band attenuation of the ME filters and the matching Kaiser window shape (J. F. Kaiser, "Nonrecursive digital
  // filter design using the I0-sinh window function", 1974: beta = 0.1102 (A - 8.7) for A > 50 dB)
  constexpr double kFirAttenDb = 60.0;
  constexpr float kFirBeta = (float)(0.1102 * (kFirAttenDb - 8.7));
  // taps per side incl. the centre from Kaiser's length estimate N = (A - 7.95) / (14.36 df), df = fc
  inline int kaiser_half_taps(float fc, double atten_db)
  {
    const double n = (atten_db - 7.95) / (14.36 * (double)fc);
    return (int)std::ceil(n / 2.0) + 1;
  }
  constexpr int CFG_MAX_LEVELS = 7;                         // pyramid levels below the full-resolution band

  struct Config
  {
    // ---- geometry
    int scale;       // ME scale (both axes): ME pixel = scale x scale picture pixels; 2, 4 or 8
    int pad_x, pad_y; // internal-grid border (edge repeat), picture pixels
    int levels;      // Laplacian-pyramid levels below the full-resolution band (bands = levels + 1)
    int stride_max;  // pyramid levels with stride 2^i <= stride_max read the up-sampled full-resolution field
    // ---- ME picture filters: symmetric windowed-sinc low-pass (Kaiser window), decimation by scale
    int fir_generated;                 // must be 1 (the generated design; the setting is kept for scripts)
    float fir_a_cutoff, fir_b_cutoff;  // -6 dB point, cycles per picture pixel (A: errors / refinement, B: phase corr.)
    int fir_a_taps, fir_b_taps;        // taps per side incl. the centre
    float fir_beta;                    // Kaiser window beta
    int fir_a_n, fir_b_n;              // resolved tap counts
    float fir_a[CFG_MAX_FIR], fir_b[CFG_MAX_FIR]; // resolved taps (centre first)
    // ---- phase correlation
    int block_size;                    // stage 0 block size (stage k: block_size << k)
    int peaks[CFG_NSTAGE];             // peaks per block per stage (0: the stage is off)
    int nearest[CFG_NSTAGE];           // 1: one candidate per peak from the nearest block; 0: four (the even / odd
                                       // blocks covering the pixel); -1 auto: nearest when the grid has one block on
                                       // an axis
    float block_taper;                 // block window: tapered part of a block, both edges together (0 .. 1; 1: Hann)
    float highpass_sigma;              // prefilter: the ME picture minus its Gaussian low-pass of this sigma (0: none)
    float noise_weight;                // spectral weighting q / (q + k^2), q = A A' / (N N'), k = noise_weight (0: phase
                                       // only)
    float peak_ratio;                  // a block's further peak counts when it reaches this fraction of its strongest
    int global_candidate;              // 1: a global candidate (the vector most blocks agree on); 0: none
    float secondary_peak_cost;         // candidate cost of a block's further peaks (the strongest costs nothing)
    // ---- refinement
    float refine_err_scale;            // weight exp(-err / refine_err_scale)
    float refine_reg;                  // added to the squared gradients
    int refine_box;                    // box of the Lucas-Kanade sums (ME pixels, odd)
    // ---- match errors
    int error_box;                     // box of the mean |DFD| (ME pixels, odd)
    float error_edge_weight;           // weight of the edge-aware filter (min of opposite neighbours)
    int error_edge_dist;               // distance of those neighbours (ME pixels)
    int error_edge_mode;               // 0: min of each opposite pair (left / right, up / down); 1: shiftable windows
                                       // (min over the four diagonal shifts by error_edge_dist)
    // ---- assignment
    float speed_change_cost;           // cost per ME pixel of |vf + vb| (L1)
    float vector_length_cost;          // cost per ME pixel of |vf| + |vb| (L1): equal matches go to shorter vectors
    float occlusion_error;             // a side whose error is above this ...
    float occlusion_error_factor;      // ... and above this x the other side's takes the other side's vector reversed
    float global_cost, zero_cost;      // cost of the global and of the zero-vector candidate
    // ---- local fallback
    int fallback_box;                  // spreading box of the inconsistency weight (ME pixels, odd)
    float fallback_dead_zone;          // inconsistency (distance from the smoothed vectors, ME pixels) below this: 0
    float fallback_range;              // ... rising linearly to 1 over this much above it
    float fallback_strength;           // weight of a fully inconsistent vector (units of one full splat)
    float fallback_cap;                // cap of the spread weight
    // ... plus an error term: fallback_err_gain * clamp((e - fallback_err_lo) / fallback_err_range, 0, 1) of the
    // assigned match error e (where even the best vectors match badly, the in-place blend gains weight); gain 0 = off
    float fallback_err_lo, fallback_err_range, fallback_err_gain;
    // ---- schedule, pair statistic, scene cut and temporal fallback
    int copy_snap;                     // phases within 1 / copy_snap of a source picture are copies (0: exact only)
    // the error statistic m of a picture pair (splat_phase_pull.h): mean match error of both fields, each capped at
    // error_window_cap, over the picture without a border of 1 / error_window_div on each side
    int error_window_div;
    float error_window_cap;
    // temporal fallback (doubles: the formula runs in double precision): the phase moves towards the nearer picture by
    // pull_strength * smoothstep(m, pull_error_lo, pull_error_hi); 0 = off
    double pull_strength;
    double pull_error_lo, pull_error_hi;
    // scene cut: when m reaches this, the pictures are taken as unrelated and the output is a copy of the nearer one
    // (0: off)
    float scene_cut_error;
    // ---- picture build
    float splat_err_scale;             // splat weight gain * max(min_weight, exp(-err / splat_err_scale)); 0: no error weighting
    float splat_min_weight, splat_gain;
    float still_vector;                // vectors up to this long (level pixels) stay in place
    float hole_weight;                 // total weight below this: hole (cross-fade)
    // pyramid reduce filter: a sampled Gaussian of this sigma (in samples of the finer level), 7 taps, normalised
    float pyr_sigma;
    float pyr_reduce[4];               // resolved taps: centre, distance 1, 2, 3
  };

  // ------------------------------------------------------------------------------------------------ grid helpers
  // phase-correlation block grid of a stage on an mw x mh ME picture: as many half-overlapping blocks of S as fit,
  // centred (first corner = half the leftover). npk = peaks per block. nearest: the stage offers one candidate field
  // per peak (the block whose centre is nearest to the pixel) instead of four (the near and the far block covering the
  // pixel along each axis; Config::nearest, auto: when the grid has a single block along an axis, where the four would
  // repeat the same block).
  struct PcorrGrid { int S, npk, fx, fy, nx, ny, nearest; };
  SPLAT_HD inline PcorrGrid pcorr_grid(const Config &c, int stage, int mw, int mh)
  {
    const int S = c.block_size << stage, hs = S / 2;
    const int nx = (mw - S) / hs + 1, ny = (mh - S) / hs + 1;
    const int nr = c.nearest[stage] < 0 ? ((nx == 1 || ny == 1) ? 1 : 0) : c.nearest[stage];
    return { S, c.peaks[stage], (mw - (nx - 1) * hs - S) / 2, (mh - (ny - 1) * hs - S) / 2, nx, ny, nr };
  }
  // candidate fields per peak of a stage
  SPLAT_HD inline int pcorr_fields(const PcorrGrid &g) { return g.nearest ? 1 : 4; }

  // the stage offers a global candidate
  SPLAT_HD inline int pcorr_has_global(const Config &c) { return c.global_candidate ? 1 : 0; }

  // Occlusion: where one side matches badly (its error above thresh and above ratio times the other side's error), the
  // area is taken as covered or uncovered on that side, and the output there is interpolated from the other side
  // alone: the side takes the other side's vector reversed. The errors stay as measured.
  SPLAT_HD inline void occlusion_switch(float &vx, float &vy, float e, float &ox, float &oy, float oe, float thresh,
    float ratio)
  {
    if (e > oe)
    {
      if (e > thresh && e > ratio * oe)
      {
        vx = -ox;
        vy = -oy;
      }
    }
    else if (oe > thresh && oe > ratio * e)
    {
      ox = -vx;
      oy = -vy;
    }
  }

  // ---- phase correlation: pieces shared by the CPU and the GPU (splat_pcorr.cpp has the description)
  // the two blocks of a half-overlapping grid (first corner `first`, n blocks, step hs, size 2 hs) that contain ME
  // pixel x, clamped to the grid: nr = the one whose centre is nearer (ties: the lower index), fr = the other
  SPLAT_HD inline void block_cover(int x, int first, int n, int hs, int &nr, int &fr)
  {
    const int d = x - first, j = d >= 0 ? d / hs : -((-d + hs - 1) / hs);
    int a = j < 0 ? 0 : (j > n - 1 ? n - 1 : j), b = j - 1 < 0 ? 0 : (j - 1 > n - 1 ? n - 1 : j - 1);
    // distances to the centres first + i hs + hs, doubled (pixel centre x + 1/2)
    int da = 2 * x + 1 - 2 * (first + a * hs + hs), db = 2 * x + 1 - 2 * (first + b * hs + hs);
    da = da < 0 ? -da : da;
    db = db < 0 ? -db : db;
    if (db < da || (db == da && b < a))
    {
      const int t = a;
      a = b;
      b = t;
    }
    nr = a;
    fr = b;
  }

  // global vote weight of a peak height: fixed point (exact: a power-of-two scale, then truncation), so sums of them
  // are exact in any order; 0 for no vote
  constexpr double PCORR_VOTE_ONE = 16777216.0;
  SPLAT_HD inline long long pcorr_vote_weight(float h)
  {
    return h > 0.0f ? (long long)((h < 1.0e9f ? h : 1.0e9f) * 16777216.0f) : 0;
  }

  // Tukey window of S samples: 1 in the middle, sin^2 ramps over taper * S / 2 samples at each edge (taper 1: Hann)
  inline void pcorr_window(int S, float taper, float *w)
  {
    for (int t = 0; t < S; t++)
      w[t] = 1.0f;
    const int r = (int)std::lround(0.5 * (double)taper * (double)S);
    for (int t = 0; t < r; t++)
    {
      const double s = std::sin(1.5707963267948966 * ((double)t + 0.5) / (double)r);
      w[t] = (float)(s * s);
      w[S - 1 - t] = w[t];
    }
  }

  // sum of the 2-D window weights w[y] w[x] (as the block mean sums: per row, then the rows; float)
  inline float pcorr_window_sum(int S, const float *w)
  {
    float sw = 0.0f;
    for (int y = 0; y < S; y++)
    {
      float r = 0.0f;
      for (int x = 0; x < S; x++)
        r = r + w[y] * w[x];
      sw = sw + r;
    }
    return sw;
  }

  // sampled Gaussian, radius ceil(3 sigma), normalised to a sum of 1 (half filter, centre first, at most CFG_MAX_FIR
  // taps); returns the taps
  inline int gauss_taps(float sigma, float *h)
  {
    const int r = (int)std::ceil(3.0 * (double)sigma);
    double g[CFG_MAX_FIR], sum = 0.0;
    for (int k = 0; k <= r; k++)
    {
      g[k] = std::exp(-(double)(k * k) / (2.0 * (double)sigma * (double)sigma));
      sum += k ? 2.0 * g[k] : g[k];
    }
    for (int k = 0; k <= r; k++)
      h[k] = (float)(g[k] / sum);
    return r + 1;
  }

  // number of phase-correlation candidates (block stages + global, without the zero vector) on an mw x mh ME grid
  inline int pcorr_ncand(const Config &c, int mw, int mh)
  {
    int n = pcorr_has_global(c);
    for (int s = 0; s < CFG_NSTAGE; s++)
    {
      const PcorrGrid g = pcorr_grid(c, s, mw, mh);
      n += g.npk * pcorr_fields(g);
    }
    return n;
  }
  // per-candidate costs of the phase-correlation set (a block's strongest peak 0, its further peaks
  // secondary_peak_cost, each over its fields; then global, zero): n + 1
  // values into out (CFG_MAX_COSTS); returns n + 1
  inline int pcorr_costs(const Config &c, int mw, int mh, float *out)
  {
    int k = 0;
    for (int s = 0; s < CFG_NSTAGE; s++)
    {
      const PcorrGrid pg = pcorr_grid(c, s, mw, mh);
      for (int p = 0; p < pg.npk; p++)
        for (int f = 0; f < pcorr_fields(pg); f++)
          out[k++] = p ? c.secondary_peak_cost : 0.0f;
    }
    if (pcorr_has_global(c))
      out[k++] = c.global_cost;
    out[k++] = c.zero_cost;
    return k;
  }

  inline int ilog2(int v)
  {
    int k = 0;
    while ((1 << (k + 1)) <= v)
      k++;
    return k;
  }
  inline int me_size(const Config &c, int size, int pad) { return (size + 2 * pad) / c.scale; }
  inline int me_width(const Config &c, int w) { return me_size(c, w, c.pad_x); }
  inline int me_height(const Config &c, int h) { return me_size(c, h, c.pad_y); }

  // ------------------------------------------------------------------------------------------------ defaults
  inline Config config_defaults()
  {
    Config c;
    std::memset(&c, 0, sizeof(c));
    // size-dependent: 0 = auto (config_resolve); the borders -1 (0 is a valid border)
    c.pad_x = c.pad_y = -1;
    c.fir_generated = 1;
    c.fir_beta = kFirBeta;
    c.peaks[0] = 2; c.peaks[1] = 2; c.peaks[2] = 3;
    c.nearest[0] = c.nearest[1] = c.nearest[2] = -1;
    c.block_taper = 0.5f; c.highpass_sigma = 8.0f; c.noise_weight = 1.0f; c.peak_ratio = 0.0f;
    c.global_candidate = 1;
    // The tuned values below were fitted on a ground-truth test corpus (high-frame-rate clips of news, sports, movie,
    // nature, low-quality and problem content, frames dropped and interpolated back), judged by local block errors,
    // artefact counts and flicker against the real frames, balanced over all content classes.
    c.refine_err_scale = 15.0f; c.refine_reg = 0.001f; c.refine_box = 15;
    c.error_box = 7; c.error_edge_weight = 0.75f; c.error_edge_dist = 3; c.error_edge_mode = 1;
    c.speed_change_cost = 0.0f; c.vector_length_cost = 0.0f; c.occlusion_error = 6.0f; c.occlusion_error_factor = 1.5f;
    // candidate costs: only a block's further peaks pay (3). A peak that is not a block's strongest is often an alias
    // of the strongest one (a periodic structure such as a row of windows shows peaks one period apart), so it must
    // match clearly better to win; every other candidate competes on its match error alone.
    c.secondary_peak_cost = 3.0f; c.global_cost = 0.0f; c.zero_cost = 0.0f;
    c.fallback_box = 31; c.fallback_dead_zone = 3.0f; c.fallback_range = 2.0f; c.fallback_strength = 1.5f;
    c.fallback_cap = 1.25f;
    c.fallback_err_lo = 8.0f; c.fallback_err_range = 16.0f; c.fallback_err_gain = 0.0f;
    c.copy_snap = 128;
    // the temporal fallback (output time pulled towards the nearer picture) is off by default: on the test corpus it
    // cost more (judder, misplaced detail) than it saved; scene cuts are handled by scene_cut_error
    c.error_window_div = 8; c.error_window_cap = 64.0f;
    c.pull_strength = 0.0; c.pull_error_lo = 6.0; c.pull_error_hi = 18.0;
    // 15: catches most cuts between similar shots (sports: grass, crowd), where the motion search still finds partial
    // matches; 10 already fires on water and splashes
    c.scene_cut_error = 15.0f;
    // weights in units of one full splat: a vector shorter than 1/128 level pixel counts as still (in-place), a total
    // weight below 1e-4 as a hole (cross-fade)
    c.splat_err_scale = 0.0f; c.splat_min_weight = 0.02f; c.splat_gain = 1.0f;
    c.still_vector = 1.0f / 128.0f; c.hole_weight = 1.0e-4f;
    c.pyr_sigma = 4.0f;
    return c;
  }

  // ---- the per-size rules (the auto defaults): ME scale 2 below 900 rows, else 4; the pyramid has log2(scale) + 2
  // levels; the border is at least 16 pixels; change them here.
  // ME scale by the height only (the width does not matter): 2 below 900 rows (SD, 720p), 4 from 900 rows on (1080p,
  // UHD). Measured on a ground-truth corpus: at UHD scale 4 instead of 8 cut the error composite by about 12 % for
  // about 7 % of speed; at 1080p scale 2 instead of 4 was mixed (better on average, worse on low-quality sources and
  // in flicker). Scale 8 stays available as a setting.
  inline int auto_scale(int w, int h)
  {
    (void)w;
    return h < 900 ? 2 : 4;
  }
  // pyramid levels below the full-resolution band: log2(scale) + 2 (3 / 4 / 5 for scale 2 / 4 / 8)
  inline int auto_levels(int scale, int w, int h)
  {
    (void)w;
    (void)h;
    return ilog2(scale) + 2;
  }
  // border per axis: the smallest integer >= 16 picture pixels (room for the motion carried into the padding) that
  // makes size + 2 border a multiple of 2^levels (16 when none does: an odd size)
  inline int auto_pad(int size, int levels)
  {
    const int m = 1 << levels;
    for (int p = 16; p < 16 + m; p++)
      if ((size + 2 * p) % m == 0)
        return p;
    return 16;
  }
  // stage 0 block: 32 (blocks 32 / 64 / 128) at every size
  inline int auto_block(int scale, int w, int h)
  {
    (void)scale;
    (void)w;
    (void)h;
    return 32;
  }

  // Kaiser-windowed sinc low-pass, taps per side incl. centre, -6 dB at fc cycles/sample, normalised to DC gain 1
  inline int fir_design(float fc, int half, float beta, float *h)
  {
    auto i0 = [](double x) {
      double s = 1.0, t = 1.0;
      for (int k = 1; k < 200; k++)
      {
        t *= (x / (2.0 * k)) * (x / (2.0 * k));
        s += t;
        if (t < 1e-17 * s)
          break;
      }
      return s;
    };
    const double pi = 3.14159265358979323846;
    double d[CFG_MAX_FIR], sum = 0.0;
    for (int n = 0; n < half; n++)
    {
      const double x = 2.0 * fc * n, sinc = n == 0 ? 1.0 : std::sin(pi * x) / (pi * x);
      const double r = (double)n / half;
      d[n] = 2.0 * fc * sinc * i0(beta * std::sqrt(1.0 - r * r)) / i0(beta);
      sum += n == 0 ? d[n] : 2.0 * d[n];
    }
    for (int n = 0; n < half; n++)
      h[n] = (float)(d[n] / sum);
    return half;
  }

  // fills every field left auto (0, or -1 for the borders pad_x / pad_y) from the rules above; generates the ME filter taps
  inline void config_resolve(Config &c, int w, int h)
  {
    if (!c.scale)
      c.scale = auto_scale(w, h);
    if (!c.levels)
      c.levels = auto_levels(c.scale, w, h);
    const int lv = c.levels >= 1 && c.levels <= CFG_MAX_LEVELS ? c.levels : 1; // (a bad value fails config_check)
    if (c.pad_x < 0)
      c.pad_x = auto_pad(w, lv);
    if (c.pad_y < 0)
      c.pad_y = auto_pad(h, lv);
    if (!c.stride_max)
      c.stride_max = c.scale;
    if (!c.block_size)
      c.block_size = auto_block(c.scale, w, h);
    // ME filters: -6 dB at a fraction of the ME grid's Nyquist frequency (0.5 / scale cycles per picture pixel), A
    // (match errors, refinement) 0.75, B (phase correlation, a smoother picture) 0.6; taps per side from Kaiser's
    // length estimate for the stop-band attenuation fir_atten_db with a transition width equal to the cutoff
    const float nyq = 0.5f / (float)c.scale;
    if (!c.fir_a_cutoff)
      c.fir_a_cutoff = 0.75f * nyq;
    if (!c.fir_b_cutoff)
      c.fir_b_cutoff = 0.6f * nyq;
    if (!c.fir_a_taps)
      c.fir_a_taps = kaiser_half_taps(c.fir_a_cutoff, kFirAttenDb);
    if (!c.fir_b_taps)
      c.fir_b_taps = kaiser_half_taps(c.fir_b_cutoff, kFirAttenDb);
    if (c.fir_a_taps >= 1 && c.fir_a_taps <= CFG_MAX_FIR && c.fir_b_taps >= 1 && c.fir_b_taps <= CFG_MAX_FIR)
    {
      c.fir_a_n = fir_design(c.fir_a_cutoff, c.fir_a_taps, c.fir_beta, c.fir_a);
      c.fir_b_n = fir_design(c.fir_b_cutoff, c.fir_b_taps, c.fir_beta, c.fir_b);
    }
    // pyramid reduce taps: g(k) = exp(-k^2 / (2 sigma^2)), k = 0 .. 3, normalised to a sum of 1 over the 7 taps
    if (c.pyr_sigma > 0.0f)
    {
      double g[4], sum = 0.0;
      for (int k = 0; k < 4; k++)
      {
        g[k] = std::exp(-(double)(k * k) / (2.0 * (double)c.pyr_sigma * (double)c.pyr_sigma));
        sum += k ? 2.0 * g[k] : g[k];
      }
      for (int k = 0; k < 4; k++)
        c.pyr_reduce[k] = (float)(g[k] / sum);
    }
  }

  // The defaults for a w x h picture: the tuned scalars, then the per-size rules above. For 1920 x 1080: scale 4,
  // border 16 x 20, 4 levels, stride_max 4, blocks 32 / 64 / 128, the generated 4:1 ME filters.
  inline Config config_for_size(int w, int h)
  {
    Config c = config_defaults();
    config_resolve(c, w, h);
    return c;
  }

  // false (and the reason in why) when c cannot run on a w x h picture
  inline bool config_check(const Config &c, int w, int h, char *why, size_t n)
  {
#define SPLAT_CFG_FAIL(...)                                                                                              \
  do                                                                                                                   \
  {                                                                                                                    \
    std::snprintf(why, n, __VA_ARGS__);                                                                                \
    return false;                                                                                                      \
  } while (0)
    if (c.scale != 2 && c.scale != 4 && c.scale != 8)
      SPLAT_CFG_FAIL("me_scale must be 2, 4 or 8");
    if (c.fir_generated != 1)
      SPLAT_CFG_FAIL("me_filter_generated must be true (the tabled ME filters were removed)");
    if (!(c.fallback_err_range > 0.0f) || !(c.fallback_err_gain >= 0.0f) || !(c.fallback_range > 0.0f) ||
        !(c.fallback_strength >= 0.0f) || !(c.fallback_cap > 0.0f))
      SPLAT_CFG_FAIL("fallback_range, fallback_err_range and fallback_cap must be > 0, fallback_strength and "
        "fallback_err_gain >= 0");
    if (c.error_edge_mode < 0 || c.error_edge_mode > 1)
      SPLAT_CFG_FAIL("error_edge_mode must be 0 or 1");
    if (!(c.scene_cut_error >= 0.0f))
      SPLAT_CFG_FAIL("scene_cut_error must be >= 0");
    if (!(c.pyr_sigma >= 0.3f && c.pyr_sigma <= 16.0f))
      SPLAT_CFG_FAIL("pyramid_sigma must be 0.3 .. 16");
    if (c.levels < 1 || c.levels > CFG_MAX_LEVELS)
      SPLAT_CFG_FAIL("pyramid_levels must be 1 .. %d", CFG_MAX_LEVELS);
    if (c.pad_x < 0 || c.pad_x > 256 || c.pad_y < 0 || c.pad_y > 256)
      SPLAT_CFG_FAIL("pad_x and pad_y must be 0 .. 256");
    const int m = 1 << c.levels;
    // an even size can always be padded to a multiple of 2^levels (and of the scale); an odd one cannot
    if (w % 2 == 0 && ((w + 2 * c.pad_x) % m || (w + 2 * c.pad_x) % c.scale))
      SPLAT_CFG_FAIL("width + 2 pad_x (%d) must be a multiple of 2^pyramid_levels (%d) and of me_scale", w + 2 * c.pad_x, m);
    if (h % 2 == 0 && ((h + 2 * c.pad_y) % m || (h + 2 * c.pad_y) % c.scale))
      SPLAT_CFG_FAIL("height + 2 pad_y (%d) must be a multiple of 2^pyramid_levels (%d) and of me_scale", h + 2 * c.pad_y, m);
    if (c.stride_max < c.scale / 2 || c.stride_max > (1 << c.levels))
      SPLAT_CFG_FAIL("field_stride_max must be me_scale / 2 .. 2^pyramid_levels");
    if (c.block_size != 16 && c.block_size != 32 && c.block_size != 64 && c.block_size != 128)
      SPLAT_CFG_FAIL("block_size must be 16, 32, 64 or 128");
    const int mw = me_width(c, w), mh = me_height(c, h), smax = c.block_size << (CFG_NSTAGE - 1);
    if (mw < smax || mh < smax)
      SPLAT_CFG_FAIL("the ME picture (%d x %d) must hold the largest block (%d): a larger picture, a smaller me_scale or "
        "block_size", mw, mh, smax);
    for (int s = 0; s < CFG_NSTAGE; s++)
      if (c.peaks[s] < 0 || c.peaks[s] > CFG_MAX_PEAKS)
        SPLAT_CFG_FAIL("block_peaks must be 0 .. %d", CFG_MAX_PEAKS);
    if (!(c.block_taper >= 0.0f && c.block_taper <= 1.0f) || !(c.highpass_sigma >= 0.0f && c.highpass_sigma <= 16.0f) ||
        !(c.noise_weight >= 0.0f) || !(c.secondary_peak_cost >= 0.0f) || !(c.peak_ratio >= 0.0f && c.peak_ratio <= 1.0f) ||
        c.global_candidate < 0 || c.global_candidate > 1)
      SPLAT_CFG_FAIL("block_taper and peak_ratio must be 0 .. 1, highpass_sigma 0 .. 16, noise_weight and "
        "secondary_peak_cost >= 0, global_candidate 0 or 1");
    if (pcorr_ncand(c, mw, mh) < 1)
      SPLAT_CFG_FAIL("no candidates: block_peaks all 0 and global_candidate 0");
    if (pcorr_ncand(c, mw, mh) + 1 > CFG_MAX_COSTS)
      SPLAT_CFG_FAIL("too many candidates (%d, at most %d)", pcorr_ncand(c, mw, mh), CFG_MAX_COSTS - 1);
    if (c.fir_generated && (c.fir_a_taps < 1 || c.fir_a_taps > CFG_MAX_FIR || c.fir_b_taps < 1 || c.fir_b_taps > CFG_MAX_FIR))
      SPLAT_CFG_FAIL("me_filter_a_taps / me_filter_b_taps must be 1 .. %d", CFG_MAX_FIR);
    if (!(c.fir_a_cutoff > 0.0f && c.fir_a_cutoff < 0.5f && c.fir_b_cutoff > 0.0f && c.fir_b_cutoff < 0.5f) ||
        !(c.fir_beta >= 0.0f))
      SPLAT_CFG_FAIL("me_filter cutoffs must be in (0, 0.5) cycles per pixel, me_filter_beta >= 0");
    auto odd = [](int v, int lo, int hi) { return v >= lo && v <= hi && v % 2 == 1; };
    if (!odd(c.refine_box, 1, 63) || !odd(c.error_box, 1, 31) || !odd(c.fallback_box, 1, 255))
      SPLAT_CFG_FAIL("boxes must be odd: refine_box 1 .. 63, error_box 1 .. 31, fallback_box 1 .. 255");
    if (c.error_edge_dist < 0 || c.error_edge_dist > 16)
      SPLAT_CFG_FAIL("error_edge_dist must be 0 .. 16");
    if (c.copy_snap != 0 && c.copy_snap < 2)
      SPLAT_CFG_FAIL("copy_snap must be 0 or >= 2");
    if (c.error_window_div < 3)
      SPLAT_CFG_FAIL("error_window_div must be >= 3");
    if (!(c.error_window_cap > 0.0f) || !(c.pull_strength >= 0.0) || !(c.pull_error_hi > c.pull_error_lo))
      SPLAT_CFG_FAIL("error_window_cap must be positive, pull_strength >= 0 and pull_error_hi > pull_error_lo");
    if (!(c.refine_err_scale > 0.0f) || !(c.splat_err_scale >= 0.0f) || !(c.splat_gain > 0.0f) || !(c.hole_weight >= 0.0f))
      SPLAT_CFG_FAIL("refine_err_scale and splat_gain must be positive, splat_err_scale and hole_weight >= 0");
    for (int s = 0; s < CFG_NSTAGE; s++)
      if (c.nearest[s] < -1 || c.nearest[s] > 1)
        SPLAT_CFG_FAIL("nearest_block values must be -1 (auto), 0 or 1");
#undef SPLAT_CFG_FAIL
    return true;
  }
}
