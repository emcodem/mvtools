// calc_nlambda of the MAnalyse shaders (included by ana_common.glsl and ana_interp.comp)
// ---- calc_nlambda: lambda * (lsad / (lsad + pred_sad / 2))^2 in IEEE double, exactly, with integers ----------------
// (shaderFloat64 is optional; this is the double arithmetic of the CPU code bit for bit: RN(a / b), two RN products,
// truncation; checked against hardware doubles on 20M random inputs)
int sd_blen64(uint64_t v)
{
  const uint hi = uint(v >> 32), lo = uint(v);
  return hi != 0u ? 33 + findMSB(hi) : (lo != 0u ? 1 + findMSB(lo) : 0);
}

void sd_mul64(uint64_t a, uint64_t b, out uint64_t hi, out uint64_t lo)
{
  const uint64_t M = 0xffffffffUL;
  const uint64_t a0 = a & M, a1 = a >> 32, b0 = b & M, b1 = b >> 32;
  const uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
  const uint64_t mid = (p00 >> 32) + (p01 & M) + (p10 & M);
  lo = (mid << 32) | (p00 & M);
  hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

// (hi, lo) * 2^e (non-zero) rounded to 53 bits (nearest even): mantissa m, returns the exponent
int sd_round128(uint64_t hi, uint64_t lo, int e, out uint64_t m)
{
  const int L = hi != 0UL ? 64 + sd_blen64(hi) : sd_blen64(lo);
  const int sh = L - 53;
  if (sh <= 0)
  {
    m = lo << uint(-sh);
    return e + sh;
  }
  uint64_t q, rest_hi, rest_lo;
  if (sh >= 64)
  {
    q = hi >> uint(sh - 64);
    rest_hi = sh == 64 ? 0UL : (hi & ((1UL << uint(sh - 64)) - 1UL));
    rest_lo = lo;
  }
  else
  {
    q = (lo >> uint(sh)) | (hi << uint(64 - sh));
    rest_hi = 0UL;
    rest_lo = lo & ((1UL << uint(sh)) - 1UL);
  }
  bool guard, sticky;
  if (sh - 1 >= 64)
  {
    guard = ((rest_hi >> uint(sh - 1 - 64)) & 1UL) != 0UL;
    sticky = (rest_hi & ((1UL << uint(sh - 1 - 64)) - 1UL)) != 0UL || rest_lo != 0UL;
  }
  else
  {
    guard = ((rest_lo >> uint(sh - 1)) & 1UL) != 0UL;
    sticky = (sh - 1 > 0 ? (rest_lo & ((1UL << uint(sh - 1)) - 1UL)) != 0UL : false) || rest_hi != 0UL;
  }
  if (guard && (sticky || (q & 1UL) != 0UL))
    q++;
  e += sh;
  if ((q >> 53) != 0UL)
  {
    q >>= 1;
    e++;
  }
  m = q;
  return e;
}

// RN(a / b), 0 < a, b < 2^32
int sd_div(uint64_t a, uint64_t b, out uint64_t m)
{
  uint64_t r = a, d = b;
  int e = 0;
  while (r < d)
  {
    r <<= 1;
    e--;
  }
  while (r >= 2UL * d)
  {
    d <<= 1;
    e++;
  }
  uint64_t q = 0UL;
  for (int i = 0; i < 54; i++)
  {
    q <<= 1;
    if (r >= d)
    {
      r -= d;
      q |= 1UL;
    }
    r <<= 1;
  }
  const bool guard = (q & 1UL) != 0UL, sticky = r != 0UL;
  q >>= 1;
  e -= 52;
  if (guard && (sticky || (q & 1UL) != 0UL))
    q++;
  if ((q >> 53) != 0UL)
  {
    q >>= 1;
    e++;
  }
  m = q;
  return e;
}

// PlaneOfBlocks::search_mv_slice: nLambda of a block whose predictor SAD is pred_sad. 0 on the first row of a slice.
// The host only allows lsad >= 0 and lambda >= 0 (other signs: CPU fallback), so the operands are non-negative here.
int calc_nlambda(int lambda_level, int lsad, int pred_sad, bool first_row)
{
  const int nl = first_row ? 0 : lambda_level;
  if (lsad == 0 || nl == 0)
    return 0;
  const int64_t den = int64_t(lsad) + int64_t(pred_sad >> 1);
  if (den <= 0L)
    return 0; // not reached for SADs >= 0
  uint64_t mf, hi, lo, m1, m2;
  const int ef = sd_div(uint64_t(lsad), uint64_t(den), mf);
  sd_mul64(uint64_t(nl), mf, hi, lo);
  const int e1 = sd_round128(hi, lo, ef, m1);
  sd_mul64(m1, mf, hi, lo);
  const int e2 = sd_round128(hi, lo, e1 + ef, m2);
  if (e2 >= 0)
  {
    if (e2 > 10 || (m2 << uint(e2)) > 0x7fffffffUL)
      return 0x7fffffff;
    return int(m2 << uint(e2));
  }
  if (-e2 >= 64)
    return 0;
  const uint64_t v = m2 >> uint(-e2);
  return v > 0x7fffffffUL ? 0x7fffffff : int(v);
}

