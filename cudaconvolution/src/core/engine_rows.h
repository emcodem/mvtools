#pragma once
// Per-row kernels of the CPU engine, scalar reference and AVX2/FMA3 (engine_avx2.cpp, compiled with /arch:AVX2).
// Both use a float32 FMA per tap in ascending tap order, so they are bit-identical.
#include <algorithm>
#include <cmath>
#include <cstdint>

#include "core/preset.h"

namespace cudaconvolution {

inline uint16_t output_rule(float o, const Preset& p) {
  const double r = static_cast<double>(o) + 0.5;
  const int q = static_cast<int>(p.arith == Arith::F32FmaTrunc ? std::trunc(r) : std::floor(r));
  return static_cast<uint16_t>(std::clamp(p.offset + q, p.output_lo, p.output_hi));
}

namespace rows {

// out[x] = clip(sum_t taps[t] * row[x * factor + t]), x < dw. work: >= padded row length + 16 * factor floats.
void h_scalar(const float* row, float* out, int dw, int factor, const float* taps, int nh, float ilo, float ihi,
              float* work);
void h_avx2(const float* row, float* out, int dw, int factor, const float* taps, int nh, float ilo, float ihi,
            float* work);

// d[x] = output_rule(sum_t taps[t] * src[t][x]), x < dw.
void v_scalar(const float* const* src, int nv, const float* taps, uint16_t* d, int dw, const Preset& p);
void v_avx2(const float* const* src, int nv, const float* taps, uint16_t* d, int dw, const Preset& p);

}  // namespace rows
}  // namespace cudaconvolution
