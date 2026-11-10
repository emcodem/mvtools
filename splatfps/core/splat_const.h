// Numeric constants of the stages (half filters, centre first; exact float32 bit patterns where computed values could
// round differently): the local-fallback smoothing, the chroma filters (input, output) and the pyramid kernels
// (build). Shared by the CPU code (splatfps/core) and the CUDA kernels, so both use the same bits.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace splat
{
  // The ME picture down-converters are generated at run time (splat_config.h: a Kaiser-windowed sinc low-pass per
  // decimation factor, Config::fir_a / fir_b).

  // Sampled Gaussians, normalised to a sum of 1 (half filters, centre first, float32 bit patterns so every path uses
  // the same bits): g(k) = exp(-k^2 / (2 sigma^2)) / sum.
  // sigma 1, 3 taps: the smoothed vector field the local fallback compares each vector with
  static const uint32_t DISORDER_SMOOTH_BITS[2] = {
    0x3ee75a8eu, 0x3e8c52b9u };

  // The tuned scalars of the stages (weights, thresholds, candidate costs, box sizes, fallback constants) are run-time
  // settings: splat::Config in splat_config.h.

  // chroma 4:2:2 -> 4:4:4 (input): half-band interpolator, a Lanczos-windowed sinc (a = 7) at the half-sample offsets
  // k + 1/2: h(k) = sinc(k + 1/2) sinc((k + 1/2) / 7), normalised so each side sums to 1/2 (DC gain 1); taps per side,
  // nearest first. The odd output 2 j + 1 sits half way between inputs j and j + 1; the pairs (j - k, j + 1 + k) are
  // symmetric around it.
  constexpr int CHROMA_HB_TAPS = 7;
  constexpr float CHROMA_HB[CHROMA_HB_TAPS] = { 0.631581008f, -0.19662784f, 0.102288768f, -0.0579244718f,
    0.0315706059f, -0.0146252112f, 0.00373716559f };
  // The same pairs, read down a column of a 4:4:4 picture at the half-way point between rows 2 j and 2 j + 1, make
  // the 4:2:0 output chroma (output): the round trip 4:2:0 -> 4:4:4 -> 4:2:0 returns almost every code unchanged.

  // chroma 4:2:0 -> 4:2:2 (input), vertical, MPEG-2 siting (chroma row j half way between luma rows 2 j and 2 j + 1, so
  // luma row 2 j lies a quarter of a chroma row above it and 2 j + 1 a quarter below): 3 taps in 1/128 units, the
  // chroma row j, its neighbour on the side of the output row, the neighbour on the other side: quadratic Lagrange
  // interpolation through the three rows at -1, 0, +1 evaluated at -1/4: (1 - x^2, x (x - 1) / 2, x (x + 1) / 2).
  constexpr float CHROMA_420[3] = { 0.9375f, 0.15625f, -0.09375f };

  // picture build: the Laplacian pyramid of P. J. Burt and E. H. Adelson ("The Laplacian Pyramid as a Compact
  // Image Code", IEEE Trans. Communications 31(4), 1983). The expand step uses their generating kernel a = 3/8:
  // w = (1, 4, 6, 4, 1) / 16.
  // The pyramid reduce filter is a run-time setting (Config::pyr_sigma, a sampled Gaussian, 7 taps).
  // pyramid expand (2x zero insertion, then 2 w per axis), one 2-D kernel per output phase; per axis the even output
  // takes (1, 6, 1) / 8 of the low samples x - 1, x, x + 1 and the odd output (4, 4) / 8 of x, x + 1:
  // A (even row, even column): 3x3 around low sample (x, y): centre A0, the 4 edge neighbours A1, the 4 corners A2;
  // B (one odd coordinate): 4 x 3 samples symmetric about the half-way point, B0 / B1 on the centre line, B2 / B3 on
  //   the two lines beside it;
  // C (both odd): 4 x 4 samples, C0 for the inner 2 x 2, C1 for the 8 edge samples, C2 for the 4 corners.
  constexpr float PYR_EXPAND_A[3] = { 0.5625f, 0.09375f, 0.015625f };
  constexpr float PYR_EXPAND_B[4] = { 0.375f, 0.0f, 0.0625f, 0.0f };
  constexpr float PYR_EXPAND_C[3] = { 0.25f, 0.0f, 0.0f };
  // match error recorded for a side that does not exist (the backward side of the first picture): far above every
  // real error (errors are mean absolute sample differences, at most 255), so it never passes for a match (with
  // splat_err_scale > 0 that side would get the minimum splat weight; the first picture is never the later picture N
  // of an output frame, so no build reads that field)
  constexpr float NO_SIDE_ERROR = 1.0e4f;
  // the built picture is clamped to the sample range [SAMPLE_MIN, SAMPLE_MAX]
  constexpr float SAMPLE_MIN = -128.0f, SAMPLE_MAX = 127.0f;

  inline float bits_to_float(uint32_t b)
  {
    float f;
    std::memcpy(&f, &b, 4);
    return f;
  }
}
