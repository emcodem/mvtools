// Resampler arithmetic of AviSynth+ (resize_v_c_planar / resize_h_c_planar), shared by rs_pass and rs_bb
// (port of rs_sum in Sources/cuda/mvcuda_resample.cu). Include after common.glsl in an x.pix.comp file.
// 8-bit: 14-bit coefficients, rounder 1 << 13; 10..15-bit: 13-bit coefficients, rounder 1 << 12; 16-bit: samples shifted to
// signed first and back after the sum.
// s: plane address, i0: index of the first tap's sample, step: samples between taps; c: coefficient address (int16),
// ci: index of the first coefficient
int rs_sum(uint64_t s, uint i0, uint step, uint64_t c, uint ci, int taps, int bits)
{
  int result;
#ifdef MVC_PIX8
  result = 1 << 13;
  for (int i = 0; i < taps; i++)
    result += ldpix(s, i0 + uint(i) * step) * int(PI16(c).v[ci + uint(i)]);
  result >>= 14;
  result = clamp(result, 0, 255);
#else
  const int limit = (1 << bits) - 1;
  result = 1 << 12;
  for (int i = 0; i < taps; i++)
  {
    int v = ldpix(s, i0 + uint(i) * step);
    if (bits == 16)
      v -= 32768;
    result += v * int(PI16(c).v[ci + uint(i)]);
  }
  if (bits == 16)
    result += 32768 << 13;
  result >>= 13;
  result = clamp(result, 0, limit);
#endif
  return result;
}
