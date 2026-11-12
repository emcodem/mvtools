// AVX2/FMA3 row kernels: 8 output samples per step, _mm256_fmadd_ps per tap in ascending tap order (each lane is the
// IEEE fused multiply-add of the scalar path), tails via the scalar code. Compiled with /arch:AVX2.
#include <immintrin.h>

#include "core/engine_rows.h"

namespace cudaconvolution {
namespace rows {

void h_avx2(const float* row, float* out, int dw, int factor, const float* taps, int nh, float ilo, float ihi,
            float* work) {
  // split the row into `factor` phases so that row[x * factor + t] = phase[t % factor][x + t / factor] is contiguous in x
  const int vec_end = dw / 8 * 8;
  const int phase_len = vec_end + (nh - 1) / factor;  // indices x + t / factor used by the vector loop
  const float* phase[kMaxFactor];
  if (factor == 1) {
    phase[0] = row;
  } else {
    for (int r = 0; r < factor; ++r) {
      float* ph = work + static_cast<std::size_t>(r) * phase_len;
      for (int k = 0; k < phase_len; ++k) ph[k] = row[k * factor + r];
      phase[r] = ph;
    }
  }
  const __m256 lo = _mm256_set1_ps(ilo), hi = _mm256_set1_ps(ihi);
  for (int x = 0; x < vec_end; x += 8) {
    __m256 acc = _mm256_setzero_ps();
    for (int t = 0; t < nh; ++t) {
      const float* s = phase[t % factor] + x + t / factor;
      acc = _mm256_fmadd_ps(_mm256_set1_ps(taps[t]), _mm256_loadu_ps(s), acc);
    }
    _mm256_storeu_ps(out + x, _mm256_max_ps(_mm256_min_ps(acc, hi), lo));
  }
  h_scalar(row + vec_end * factor, out + vec_end, dw - vec_end, factor, taps, nh, ilo, ihi, nullptr);
}

void v_avx2(const float* const* src, int nv, const float* taps, uint16_t* d, int dw, const Preset& p) {
  const int vec_end = dw / 8 * 8;
  const bool trunc_rule = p.arith == Arith::F32FmaTrunc;
  const __m256i off = _mm256_set1_epi32(p.offset), olo = _mm256_set1_epi32(p.output_lo),
                ohi = _mm256_set1_epi32(p.output_hi);
  const __m256 half = _mm256_set1_ps(0.5f);
  for (int x = 0; x < vec_end; x += 8) {
    __m256 acc = _mm256_setzero_ps();
    for (int t = 0; t < nv; ++t) acc = _mm256_fmadd_ps(_mm256_set1_ps(taps[t]), _mm256_loadu_ps(src[t] + x), acc);
    // o + 0.5 is exact in float32 for |o| < 2^22 (levels are < 2^11), so this equals the double rule of output_rule()
    const __m256 r = _mm256_add_ps(acc, half);
    const __m256i q = trunc_rule ? _mm256_cvttps_epi32(r) : _mm256_cvttps_epi32(_mm256_floor_ps(r));
    const __m256i v = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(q, off), olo), ohi);
    const __m128i packed = _mm_packus_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(d + x), packed);
  }
  if (vec_end < dw) {
    const float* tail[kMaxTaps];
    for (int t = 0; t < nv; ++t) tail[t] = src[t] + vec_end;
    v_scalar(tail, nv, taps, d + vec_end, dw - vec_end, p);
  }
}

}  // namespace rows
}  // namespace cudaconvolution
