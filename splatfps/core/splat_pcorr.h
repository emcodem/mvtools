// splat_pcorr: the exact CPU path of SplatFps' phase-correlation vector source.
//
// Technique: phase correlation motion measurement (G. A. Thomas, "Television motion measurement for DATV and other
// applications", BBC RD 1987/11, https://downloads.bbc.co.uk/rd/pubs/reports/1987-11.pdf; US 4,890,160, BBC, 1986,
// https://patents.google.com/patent/US4890160). Two pictures are Fourier-transformed block by block; the cross-power
// spectrum G1 G2* is normalised to (roughly) unit amplitude so only the phase difference remains, and its inverse
// transform is a correlation surface with a sharp peak at each displacement present in the block, almost
// independent of brightness and contrast. The several highest peaks of a block are not the answer for any pixel yet;
// they form a menu of candidate vectors from which the assignment picks per pixel (the two-stage measurement /
// assignment scheme of the report and the patent). A block measures motions up to half its size, hence several block
// sizes. Peaks are taken at integer positions (no sub-pixel peak interpolation); the refinement adds the
// fractions.
//
// The stage, as built here (the free choices are run-time settings, splat::Config, chosen by measurement on a
// ground-truth test corpus):
// - Input: the soft ME picture B. Prefilter (highpass_sigma > 0): B minus its Gaussian low-pass, so slow shading does
//   not dominate the spectra.
// - Block stages of size S = block_size << k (k = 0, 1, 2; 32, 64, 128 by default): blocks overlap by S / 2 on a
//   centred grid (splat::pcorr_grid). Each block: its windowed mean removed (the DC bin carries no motion), a
//   separable Tukey window (block_taper; the raised-cosine edge taper of report section 4.1.2), 2-D FFT.
// - Cross power of a block pair: G G'* / (A A' + k^2 N N') with A, A' the amplitudes, N, N' the median amplitudes of
//   the two block spectra (their noise floor) and k = noise_weight: the unit-amplitude (phase-only) spectrum weighted
//   by q / (q + k^2), q = A A' / (N N'), so components far above the noise keep full weight and components in the
//   noise are attenuated (the signal-dependent weighting of report section 4.1.3, in Wiener-filter form); times
//   (-1)^(kx + ky), which puts zero shift at the surface centre. Inverse FFT -> real surface.
// - Peaks: the block_peaks highest positive local maxima over the 8 neighbours (greater than the earlier ones in
//   raster order, greater or equal the later ones; the surface border excluded; ties to the earlier index); vector =
//   position - S / 2. A further peak counts when it reaches peak_ratio of the strongest (peaks well above the noise,
//   report section 4.2.1), else the strongest stands in for it; a block without a positive maximum gives the zero
//   vector.
// - Candidate fields on the ME grid, per stage peak: the vectors of the blocks containing the pixel, the near and the
//   far block along each axis (splat::block_cover; near x near, far x near, near x far, far x far): an object that is
//   only a small part of one block is often the main content of a neighbouring one (the report offers each pixel the
//   vectors of its own and the adjacent blocks). A stage in nearest-block mode (Config::nearest) gives one field per
//   peak: the nearest block.
// - Global candidate (global_candidate): the vector most blocks agree on, the strongest peaks weighted by their height
//   (fixed-point sums, splat::pcorr_vote_weight; ties to the shorter vector): constant over the picture (a pan).
// - The zero vector is added later by the assignment, so a still area never depends on a peak.
// Candidate order: per stage (32, 64, 128) per peak its fields, then the global. They point from the previous picture
// into the current one.
//
// Arithmetic: float throughout (the block means summed per row, then over the rows), the FFTs pocketfft's float
// transform (third_party/pocketfft/fftp_pow2.h; rows first, the inverse scaled by 1 / S per axis). Two real blocks
// share one complex FFT in both directions: forward, a + i b is transformed and the two spectra are separated by
// their symmetry (G(k) = (Z(k) + Z*(-k)) / 2 for a, (Z(k) - Z*(-k)) / 2i for b); inverse, X_a + i X_b gives the two
// real surfaces in the real and the imaginary part. The fast CPU path and the CUDA kernels compute the same bits.
#pragma once

#include "splat_core.h"

#include <functional>
#include <vector>

namespace splat
{
  constexpr int PCORR_NSTAGE = CFG_NSTAGE;

  struct PcorrSpectrum
  {
    int S = 0, nblocks = 0;
    std::vector<float> R, I; // nblocks x S x S each: real and imaginary part, blocks in row-major grid order
    std::vector<float> N;    // per block: the median amplitude (DC left out), the spectrum's noise floor
  };

  struct PcorrPicture
  {
    Plane hp;                         // B after the prefilter
    PcorrSpectrum st[PCORR_NSTAGE];
  };

  // symmetric FIR (centre first, then pairs accumulated in order), zero outside, vertical pass then horizontal
  Plane sep_zero(const Plane &a, const float *h, int taps);

  // The pieces a path supplies; every implementation gives the same bits as pcorr_ops_exact (the paths share all other
  // arithmetic).
  struct PcorrOps
  {
    // par(n, fn): fn(0) .. fn(n - 1) in any order, possibly in parallel (independent tasks)
    std::function<void(int, const std::function<void(int)> &)> par;
    // sep_zero
    std::function<Plane(const Plane &, const float *, int)> fir;
    // prepare(S): called outside par for every block size before fft2 uses it
    std::function<void(int)> prepare;
    // 2-D FFT of an S x S complex tile in split re / im planes (every row, then every column, pocketfft's float
    // transform scaled by fct); thread-safe
    std::function<void(float *, float *, int, bool, float)> fft2;
  };
  // serial, scalar
  const PcorrOps &pcorr_ops_exact();

  struct PcorrGlobal { float dx, dy, height, vx, vy; };

  // per source picture, once: the prefiltered picture and the spectra of all stages (each picture's spectra serve both
  // pairs it belongs to, so a pair costs only the cross power, the inverse FFTs and the peaks)
  PcorrPicture pcorr_ingest(const Plane &B, const Config &cfg, const PcorrOps &ops = pcorr_ops_exact());
  // the candidate fields of the pair (prev, cur), pcorr_ncand(cfg, w, h) of them (29 at 1080p): 2 planes (vx, vy) per
  // candidate, ME grid; glob (or null): the global vector (dx, dy = vx, vy; height = its summed peak height)
  void pcorr_candidates(const PcorrPicture &prev, const PcorrPicture &cur, const Config &cfg, const PcorrOps &ops,
    std::vector<Plane> &out, PcorrGlobal *glob = nullptr);
  std::vector<Plane> pcorr_candidates(const PcorrPicture &prev, const PcorrPicture &cur, const Config &cfg,
    PcorrGlobal *glob = nullptr);
}
