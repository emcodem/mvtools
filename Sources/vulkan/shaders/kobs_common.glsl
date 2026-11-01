// MaskTools / RgTools building blocks of Sources/cuda/mvcuda_kobs.cu (see the header there for the rules), shared by
// kobs, kobs_rg, mt_ops and qtg_sharp. Include after common.glsl in an x.pix.comp file.

// read-only views (lets the compiler use the read-only data path for the input planes)
layout(buffer_reference, scalar, buffer_reference_align = 1) readonly restrict buffer RO_PU8 { uint8_t v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 2) readonly restrict buffer RO_PU16 { uint16_t v[]; };
#ifdef MVC_PIX8
int ldro(uint64_t base, uint i) { return int(uint(RO_PU8(base).v[i])); }
#else
int ldro(uint64_t base, uint i) { return int(uint(RO_PU16(base).v[i])); }
#endif

// sample (x, y) of the plane at base with pitch (in samples)
int ldq(uint64_t base, int pitch, int x, int y) { return ldro(base, uint(y) * uint(pitch) + uint(x)); }
void stq(uint64_t base, int pitch, int x, int y, int v) { stpix(base, uint(y) * uint(pitch) + uint(x), v); }

// samples above the bit depth's max are read as the max: the table stays in bounds. Entries have the sample size
// (8-bit tables for 8-bit samples, 16-bit above).
int lut_at(uint64_t lut, int n, int a, int b)
{
  a = min(a, n - 1);
  b = min(b, n - 1);
  return ldro(lut, uint(a) * uint(n) + uint(b));
}

int clipv(int v, int maxv) { return clamp(v, 0, maxv); }
int avg2(int a, int b) { return (a + b + 1) >> 1; }

// RemoveGrain 20 of a 3 x 3 sum that already holds the + 4
int rg20(uint sum)
{
#ifdef MVC_PIX8
  return int((sum * 7282u) >> 16);
#else
  return min(int((sum * 3641u) >> 15), 65535);
#endif
}

int rg12(int a1, int a2, int a3, int a4, int c, int a5, int a6, int a7, int a8)
{
  const int a123 = avg2(a2, avg2(a1, a3));
  const int a678 = avg2(a7, avg2(a6, a8));
  const int a4c5 = avg2(c, avg2(a4, a5));
  int a123678 = avg2(a123, a678);
  a123678 = a123678 > 0 ? a123678 - 1 : 0;
  return avg2(a4c5, a123678);
}
