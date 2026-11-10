// splat_fast_avx2: AVX2 + FMA kernels of SplatFps' fast path; see splat_fast_avx2.h. Compiled with /arch:AVX2 (MSVC)
// or -mavx2 -mfma (GCC / Clang); called only after the run-time CPU check. Plain additions and multiplications are
// written as separate operations (no fused multiply-add), so each lane rounds like the exact scalar code; the
// polynomials of exp, atan2, sin and cos use FMA (their double results are rounded to float afterwards).
#include "splat_fast_avx2.h"
#include "splat_config.h"

#include <immintrin.h>

#include <cstdlib>
#include <cstring>
#include <limits>

namespace splat
{
  namespace avx2
  {
    namespace
    {
      inline int imin(int a, int b) { return a < b ? a : b; }
      inline int imax(int a, int b) { return a > b ? a : b; }
      inline int iclamp(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
      // row y of a plane w wide, or of a ring buffer of ring rows (row y at y mod ring) when ring > 0
      inline size_t ro(int y, int w, int ring) { return (size_t)(ring > 0 ? y % ring : y) * w; }

      // scratch memory, 64-byte aligned
      struct Scratch
      {
        void *p;
        explicit Scratch(size_t bytes) : p(_mm_malloc(bytes < 64 ? 64 : bytes, 64)) {}
        ~Scratch() { _mm_free(p); }
        Scratch(const Scratch &) = delete;
        Scratch &operator=(const Scratch &) = delete;
        float *f() const { return (float *)p; }
      };

      inline __m256 ld(const float *p) { return _mm256_loadu_ps(p); }
      inline void st(float *p, __m256 v) { _mm256_storeu_ps(p, v); }
      inline __m256 bc(float v) { return _mm256_set1_ps(v); }
      inline __m256 add(__m256 a, __m256 b) { return _mm256_add_ps(a, b); }
      inline __m256 sub(__m256 a, __m256 b) { return _mm256_sub_ps(a, b); }
      inline __m256 mul(__m256 a, __m256 b) { return _mm256_mul_ps(a, b); }
      inline __m256 vabs(__m256 a) { return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a); }
      inline __m256 vneg(__m256 a) { return _mm256_xor_ps(_mm256_set1_ps(-0.0f), a); }
      const __m256 LANE = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);

      // ---- double-precision polynomials (FMA), arguments as described per function
      inline __m256d bcd(double v) { return _mm256_set1_pd(v); }

      // exp(x), x clamped to [-708, 708]: x = k ln2 + r (two-part ln2, |r| <= ln2 / 2), Taylor series of degree 13
      // (remainder below 1e-17), times 2^k
      __m256d exp_pd(__m256d x)
      {
        x = _mm256_max_pd(_mm256_min_pd(x, bcd(708.0)), bcd(-708.0));
        const __m256d k = _mm256_round_pd(_mm256_mul_pd(x, bcd(1.4426950408889634)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256d r = _mm256_fnmadd_pd(k, bcd(6.93147180369123816490e-01), x);
        r = _mm256_fnmadd_pd(k, bcd(1.90821492927058770002e-10), r);
        __m256d p = bcd(1.0 / 6227020800.0);
        p = _mm256_fmadd_pd(p, r, bcd(1.0 / 479001600.0));
        p = _mm256_fmadd_pd(p, r, bcd(1.0 / 39916800.0));
        p = _mm256_fmadd_pd(p, r, bcd(1.0 / 3628800.0));
        p = _mm256_fmadd_pd(p, r, bcd(1.0 / 362880.0));
        p = _mm256_fmadd_pd(p, r, bcd(1.0 / 40320.0));
        p = _mm256_fmadd_pd(p, r, bcd(1.0 / 5040.0));
        p = _mm256_fmadd_pd(p, r, bcd(1.0 / 720.0));
        p = _mm256_fmadd_pd(p, r, bcd(1.0 / 120.0));
        p = _mm256_fmadd_pd(p, r, bcd(1.0 / 24.0));
        p = _mm256_fmadd_pd(p, r, bcd(1.0 / 6.0));
        p = _mm256_fmadd_pd(p, r, bcd(0.5));
        p = _mm256_fmadd_pd(p, r, bcd(1.0));
        p = _mm256_fmadd_pd(p, r, bcd(1.0));
        const __m256i k64 = _mm256_cvtepi32_epi64(_mm256_cvtpd_epi32(k));
        const __m256i bits = _mm256_slli_epi64(_mm256_add_epi64(k64, _mm256_set1_epi64x(1023)), 52);
        return _mm256_mul_pd(p, _mm256_castsi256_pd(bits));
      }

      // float32 rounding of exp((double)x) for 8 floats (exp32 of splat_core)
      __m256 exp32_ps(__m256 x)
      {
        const __m128 lo = _mm256_cvtpd_ps(exp_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(x))));
        const __m128 hi = _mm256_cvtpd_ps(exp_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(x, 1))));
        return _mm256_set_m128(hi, lo);
      }

      // ---- 8 x 8 float and 4 x 4 double transposes
      inline void tr8(__m256 *r)
      {
        const __m256 t0 = _mm256_unpacklo_ps(r[0], r[1]), t1 = _mm256_unpackhi_ps(r[0], r[1]);
        const __m256 t2 = _mm256_unpacklo_ps(r[2], r[3]), t3 = _mm256_unpackhi_ps(r[2], r[3]);
        const __m256 t4 = _mm256_unpacklo_ps(r[4], r[5]), t5 = _mm256_unpackhi_ps(r[4], r[5]);
        const __m256 t6 = _mm256_unpacklo_ps(r[6], r[7]), t7 = _mm256_unpackhi_ps(r[6], r[7]);
        const __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44), s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
        const __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44), s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
        const __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44), s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
        const __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44), s7 = _mm256_shuffle_ps(t5, t7, 0xEE);
        r[0] = _mm256_permute2f128_ps(s0, s4, 0x20);
        r[1] = _mm256_permute2f128_ps(s1, s5, 0x20);
        r[2] = _mm256_permute2f128_ps(s2, s6, 0x20);
        r[3] = _mm256_permute2f128_ps(s3, s7, 0x20);
        r[4] = _mm256_permute2f128_ps(s0, s4, 0x31);
        r[5] = _mm256_permute2f128_ps(s1, s5, 0x31);
        r[6] = _mm256_permute2f128_ps(s2, s6, 0x31);
        r[7] = _mm256_permute2f128_ps(s3, s7, 0x31);
      }

      inline void tr4(__m256d *r)
      {
        const __m256d t0 = _mm256_unpacklo_pd(r[0], r[1]), t1 = _mm256_unpackhi_pd(r[0], r[1]);
        const __m256d t2 = _mm256_unpacklo_pd(r[2], r[3]), t3 = _mm256_unpackhi_pd(r[2], r[3]);
        r[0] = _mm256_permute2f128_pd(t0, t2, 0x20);
        r[1] = _mm256_permute2f128_pd(t1, t3, 0x20);
        r[2] = _mm256_permute2f128_pd(t0, t2, 0x31);
        r[3] = _mm256_permute2f128_pd(t1, t3, 0x31);
      }

      // ---- vector element types for the pocketfft passes (one lane per line); each operator is one rounding
      struct VF
      {
        __m256 v;
        VF() = default;
        VF(__m256 x) : v(x) {}
        explicit VF(double d) : v(_mm256_set1_ps((float)d)) {}
        static VF load(const float *p) { return VF(_mm256_loadu_ps(p)); }
        void store(float *p) const { _mm256_storeu_ps(p, v); }
        static VF bcast(float f) { return VF(_mm256_set1_ps(f)); }
      };
      inline VF operator+(VF a, VF b) { return _mm256_add_ps(a.v, b.v); }
      inline VF operator-(VF a, VF b) { return _mm256_sub_ps(a.v, b.v); }
      inline VF operator*(VF a, VF b) { return _mm256_mul_ps(a.v, b.v); }
      inline VF operator-(VF a) { return _mm256_xor_ps(_mm256_set1_ps(-0.0f), a.v); }
      inline VF &operator+=(VF &a, VF b) { a.v = _mm256_add_ps(a.v, b.v); return a; }

      struct VD
      {
        __m256d v;
        VD() = default;
        VD(__m256d x) : v(x) {}
        explicit VD(double d) : v(_mm256_set1_pd(d)) {}
        static VD load(const double *p) { return VD(_mm256_loadu_pd(p)); }
        void store(double *p) const { _mm256_storeu_pd(p, v); }
        static VD bcast(double f) { return VD(_mm256_set1_pd(f)); }
      };
      inline VD operator+(VD a, VD b) { return _mm256_add_pd(a.v, b.v); }
      inline VD operator-(VD a, VD b) { return _mm256_sub_pd(a.v, b.v); }
      inline VD operator*(VD a, VD b) { return _mm256_mul_pd(a.v, b.v); }
      inline VD operator-(VD a) { return _mm256_xor_pd(_mm256_set1_pd(-0.0), a.v); }
      inline VD &operator+=(VD &a, VD b) { a.v = _mm256_add_pd(a.v, b.v); return a; }

      inline void transpose(VF *r)
      {
        __m256 t[8];
        for (int j = 0; j < 8; j++)
          t[j] = r[j].v;
        tr8(t);
        for (int j = 0; j < 8; j++)
          r[j].v = t[j];
      }
      inline void transpose(VD *r)
      {
        __m256d t[4];
        for (int j = 0; j < 4; j++)
          t[j] = r[j].v;
        tr4(t);
        for (int j = 0; j < 4; j++)
          r[j].v = t[j];
      }

      // the pocketfft transform of L lines at once (L = lanes of V): exec of fftp_pow2.h with vector elements
      template <bool fwd, class V, class S, int L>
      void lines(S *re, S *im, int N, bool rows, int g0, int g1, const fftp::Plan &plan, const fftp::cpx<S> *tw, int ntw,
        S fct)
      {
        Scratch twm(sizeof(fftp::cpx<V>) * (size_t)ntw), bufm(sizeof(fftp::cpx<V>) * 2 * (size_t)N);
        fftp::cpx<V> *twv = (fftp::cpx<V> *)twm.p, *buf = (fftp::cpx<V> *)bufm.p;
        for (int i = 0; i < ntw; i++)
          twv[i] = { V::bcast(tw[i].r), V::bcast(tw[i].i) };
        for (int g = g0; g < g1; g++)
        {
          const int b = g * L;
          fftp::cpx<V> *p1 = buf, *p2 = buf + N;
          if (!rows)
            for (int y = 0; y < N; y++)
              p1[y] = { V::load(re + (size_t)y * N + b), V::load(im + (size_t)y * N + b) };
          else
            for (int x = 0; x < N; x += L)
            {
              V r[L], q[L];
              for (int j = 0; j < L; j++)
              {
                r[j] = V::load(re + (size_t)(b + j) * N + x);
                q[j] = V::load(im + (size_t)(b + j) * N + x);
              }
              transpose(r);
              transpose(q);
              for (int j = 0; j < L; j++)
                p1[x + j] = { r[j], q[j] };
            }
          for (int s = 0; s < plan.npass; s++)
          {
            const int items = plan.l1[s] * plan.ido[s];
            for (int it = 0; it < items; it++)
              fftp::pass_item<fwd>(plan, s, it, p1, p2, twv);
            fftp::cpx<V> *t = p1;
            p1 = p2;
            p2 = t;
          }
          if (fct != (S)1)
          {
            const V f = V::bcast(fct);
            for (int i = 0; i < N; i++)
              p1[i] = { p1[i].r * f, p1[i].i * f };
          }
          if (!rows)
            for (int y = 0; y < N; y++)
            {
              p1[y].r.store(re + (size_t)y * N + b);
              p1[y].i.store(im + (size_t)y * N + b);
            }
          else
            for (int x = 0; x < N; x += L)
            {
              V r[L], q[L];
              for (int j = 0; j < L; j++)
              {
                r[j] = p1[x + j].r;
                q[j] = p1[x + j].i;
              }
              transpose(r);
              transpose(q);
              for (int j = 0; j < L; j++)
              {
                r[j].store(re + (size_t)(b + j) * N + x);
                q[j].store(im + (size_t)(b + j) * N + x);
              }
            }
        }
      }

      // scalar tex_bilinear (edges of the vector loops)
      float tex1(const float *img, int w, int h, float x, float y)
      {
        const float i0f = (float)_mm_cvtss_f32(_mm_round_ss(_mm_setzero_ps(), _mm_set_ss(x), _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
        const float j0f = (float)_mm_cvtss_f32(_mm_round_ss(_mm_setzero_ps(), _mm_set_ss(y), _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
        const float a = x - i0f, b = y - j0f;
        const int i0 = _mm_cvttss_si32(_mm_set_ss(i0f)), j0 = _mm_cvttss_si32(_mm_set_ss(j0f));
        const int xa = iclamp(i0, 0, w - 1), xb = iclamp(i0 + 1, 0, w - 1);
        const int ya = iclamp(j0, 0, h - 1), yb = iclamp(j0 + 1, 0, h - 1);
        const float t00 = img[(size_t)ya * w + xa], t10 = img[(size_t)ya * w + xb];
        const float t01 = img[(size_t)yb * w + xa], t11 = img[(size_t)yb * w + xb];
        float r = ((1.0f - a) * (1.0f - b)) * t00;
        r = r + (a * (1.0f - b)) * t10;
        r = r + ((1.0f - a) * b) * t01;
        r = r + (a * b) * t11;
        return r;
      }

      float exp32_1(float x)
      {
        float o[8];
        _mm256_storeu_ps(o, exp32_ps(_mm256_set1_ps(x)));
        return o[0];
      }
    }

    // ----------------------------------------------------------------------------------------------------- ME pictures
    // The sums below are sequential chains (each addition waits for the previous one), so the loops run 4 independent
    // groups of 8 outputs side by side.
    void down_h(const float *img, int w, int y0, int y1, const float *h, int taps, int overscan, int sc, float *out,
      int nout)
    {
      // padded row q -> img[clamp(q - off)], split into the sc phases q mod sc, so that the 8 outputs j .. j + 7 read
      // 8 consecutive values of one phase for every tap
      const int PT = (taps + sc - 1) / sc * sc, off = overscan + PT, nv = (nout + 31) & ~31;
      const int LP = nv + 2 * PT / sc + 2;
      Scratch m(sizeof(float) * (size_t)(sc * LP + nv));
      float *ph[8];
      for (int k = 0; k < sc; k++)
        ph[k] = m.f() + (size_t)k * LP;
      float *tmp = m.f() + (size_t)sc * LP;
      for (int y = y0; y < y1; y++)
      {
        const float *row = img + (size_t)y * w;
        for (int q = 0; q < sc * LP; q++)
          ph[q % sc][q / sc] = row[iclamp(q - off, 0, w - 1)];
        for (int j = 0; j < nv; j += 32)
        {
          const float *c = ph[PT % sc] + PT / sc + j;
          const __m256 h0 = bc(h[0]);
          __m256 a0 = mul(h0, ld(c)), a1 = mul(h0, ld(c + 8)), a2 = mul(h0, ld(c + 16)), a3 = mul(h0, ld(c + 24));
          for (int i = 1; i < taps; i++)
          {
            const float *m_ = ph[(PT - i) % sc] + (PT - i) / sc + j, *p_ = ph[(PT + i) % sc] + (PT + i) / sc + j;
            const __m256 hi = bc(h[i]);
            a0 = add(a0, mul(hi, add(ld(m_), ld(p_))));
            a1 = add(a1, mul(hi, add(ld(m_ + 8), ld(p_ + 8))));
            a2 = add(a2, mul(hi, add(ld(m_ + 16), ld(p_ + 16))));
            a3 = add(a3, mul(hi, add(ld(m_ + 24), ld(p_ + 24))));
          }
          st(tmp + j, a0);
          st(tmp + j + 8, a1);
          st(tmp + j + 16, a2);
          st(tmp + j + 24, a3);
        }
        std::memcpy(out + (size_t)y * nout, tmp, sizeof(float) * (size_t)nout);
      }
    }

    // symmetric vertical FIR over rows: rows[0] the centre row, rows[2 i - 1] / rows[2 i] the rows at -i / +i
    // (pointers to a zero row outside the picture); out = h0 c + sum h[i] (up + down), in order
    static void vfir_row(const float *const *rows, int w, const float *h, int taps, float *o)
    {
      int x = 0;
      const __m256 h0 = bc(h[0]);
      for (; x + 32 <= w; x += 32)
      {
        __m256 a0 = mul(h0, ld(rows[0] + x)), a1 = mul(h0, ld(rows[0] + x + 8));
        __m256 a2 = mul(h0, ld(rows[0] + x + 16)), a3 = mul(h0, ld(rows[0] + x + 24));
        for (int i = 1; i < taps; i++)
        {
          const float *u = rows[2 * i - 1] + x, *dn = rows[2 * i] + x;
          const __m256 hi = bc(h[i]);
          a0 = add(a0, mul(hi, add(ld(u), ld(dn))));
          a1 = add(a1, mul(hi, add(ld(u + 8), ld(dn + 8))));
          a2 = add(a2, mul(hi, add(ld(u + 16), ld(dn + 16))));
          a3 = add(a3, mul(hi, add(ld(u + 24), ld(dn + 24))));
        }
        st(o + x, a0);
        st(o + x + 8, a1);
        st(o + x + 16, a2);
        st(o + x + 24, a3);
      }
      for (; x + 8 <= w; x += 8)
      {
        __m256 acc = mul(h0, ld(rows[0] + x));
        for (int i = 1; i < taps; i++)
          acc = add(acc, mul(bc(h[i]), add(ld(rows[2 * i - 1] + x), ld(rows[2 * i] + x))));
        st(o + x, acc);
      }
      for (; x < w; x++)
      {
        float acc = h[0] * rows[0][x];
        for (int i = 1; i < taps; i++)
          acc = acc + h[i] * (rows[2 * i - 1][x] + rows[2 * i][x]);
        o[x] = acc;
      }
    }

    void down_v(const float *img, int w, int n, int j0, int j1, const float *h, int taps, int overscan, int sc, float *out)
    {
      const float *rows[128];
      for (int j = j0; j < j1; j++)
      {
        const int pos = j * sc - overscan;
        rows[0] = img + (size_t)iclamp(pos, 0, n - 1) * w;
        for (int i = 1; i < taps; i++)
        {
          rows[2 * i - 1] = img + (size_t)iclamp(pos - i, 0, n - 1) * w;
          rows[2 * i] = img + (size_t)iclamp(pos + i, 0, n - 1) * w;
        }
        vfir_row(rows, w, h, taps, out + (size_t)j * w);
      }
    }

    // -------------------------------------------------------------------------------------------------------- FIR, box
    void fir_zero_v(const float *a, int w, int hgt, int y0, int y1, const float *h, int taps, float *out)
    {
      Scratch zm(sizeof(float) * (size_t)(w + 32));
      std::memset(zm.p, 0, sizeof(float) * (size_t)(w + 32));
      const float *rows[128];
      for (int y = y0; y < y1; y++)
      {
        rows[0] = a + (size_t)y * w;
        for (int i = 1; i < taps; i++)
        {
          rows[2 * i - 1] = y - i >= 0 ? a + (size_t)(y - i) * w : zm.f();
          rows[2 * i] = y + i < hgt ? a + (size_t)(y + i) * w : zm.f();
        }
        vfir_row(rows, w, h, taps, out + (size_t)y * w);
      }
    }

    void fir_zero_h(const float *a, int w, int y0, int y1, const float *h, int taps, float *out)
    {
      const int P = taps, wv = (w + 31) & ~31;
      Scratch m(sizeof(float) * (size_t)(wv + 2 * P + 8 + wv));
      float *pad = m.f(), *tmp = m.f() + wv + 2 * P + 8;
      const __m256 h0 = bc(h[0]);
      for (int y = y0; y < y1; y++)
      {
        std::memset(pad, 0, sizeof(float) * (size_t)(wv + 2 * P + 8));
        std::memcpy(pad + P, a + (size_t)y * w, sizeof(float) * (size_t)w);
        for (int x = 0; x < wv; x += 32)
        {
          const float *c = pad + P + x;
          __m256 a0 = mul(h0, ld(c)), a1 = mul(h0, ld(c + 8)), a2 = mul(h0, ld(c + 16)), a3 = mul(h0, ld(c + 24));
          for (int i = 1; i < taps; i++)
          {
            const __m256 hi = bc(h[i]);
            a0 = add(a0, mul(hi, add(ld(c - i), ld(c + i))));
            a1 = add(a1, mul(hi, add(ld(c + 8 - i), ld(c + 8 + i))));
            a2 = add(a2, mul(hi, add(ld(c + 16 - i), ld(c + 16 + i))));
            a3 = add(a3, mul(hi, add(ld(c + 24 - i), ld(c + 24 + i))));
          }
          st(tmp + x, a0);
          st(tmp + x + 8, a1);
          st(tmp + x + 16, a2);
          st(tmp + x + 24, a3);
        }
        std::memcpy(out + (size_t)y * w, tmp, sizeof(float) * (size_t)w);
      }
    }

    void box_v(const float *a, int w, int hgt, int y0, int y1, int size, bool take_abs, float *out, int ring_in, int ring_out)
    {
      const int r = (size - 1) / 2;
      Scratch zm(sizeof(float) * (size_t)(w + 32));
      std::memset(zm.p, 0, sizeof(float) * (size_t)(w + 32));
      const float *rows[128];
      const __m256 sgn = _mm256_set1_ps(take_abs ? -0.0f : 0.0f);
      for (int y = y0; y < y1; y++)
      {
        for (int i = 0; i < size; i++)
        {
          const int yy = y - r + i;
          rows[i] = (yy >= 0 && yy < hgt) ? a + ro(yy, w, ring_in) : zm.f();
        }
        float *o = out + ro(y, w, ring_out);
        int x = 0;
        for (; x + 32 <= w; x += 32)
        {
          __m256 s0 = _mm256_andnot_ps(sgn, ld(rows[0] + x)), s1 = _mm256_andnot_ps(sgn, ld(rows[0] + x + 8));
          __m256 s2 = _mm256_andnot_ps(sgn, ld(rows[0] + x + 16)), s3 = _mm256_andnot_ps(sgn, ld(rows[0] + x + 24));
          for (int i = 1; i < size; i++)
          {
            const float *p = rows[i] + x;
            s0 = add(s0, _mm256_andnot_ps(sgn, ld(p)));
            s1 = add(s1, _mm256_andnot_ps(sgn, ld(p + 8)));
            s2 = add(s2, _mm256_andnot_ps(sgn, ld(p + 16)));
            s3 = add(s3, _mm256_andnot_ps(sgn, ld(p + 24)));
          }
          st(o + x, s0);
          st(o + x + 8, s1);
          st(o + x + 16, s2);
          st(o + x + 24, s3);
        }
        for (; x + 8 <= w; x += 8)
        {
          __m256 s = _mm256_andnot_ps(sgn, ld(rows[0] + x));
          for (int i = 1; i < size; i++)
            s = add(s, _mm256_andnot_ps(sgn, ld(rows[i] + x)));
          st(o + x, s);
        }
        for (; x < w; x++)
        {
          float s = 0.0f;
          for (int i = 0; i < size; i++)
          {
            float v = rows[i][x];
            if (take_abs)
              v = v < 0.0f ? -v : (v == 0.0f ? 0.0f : v);
            s = i == 0 ? v : s + v;
          }
          o[x] = s;
        }
      }
    }

    void box_h(const float *v, int w, int y0, int y1, int size, bool average, float max_val, float *out, int ring_in,
      int ring_out)
    {
      const int r = (size - 1) / 2, wv = (w + 31) & ~31;
      const __m256 div = bc((float)(size * size)), mx = bc(max_val);
      Scratch m(sizeof(float) * (size_t)(wv + 2 * r + 8 + wv));
      float *pad = m.f(), *tmp = m.f() + wv + 2 * r + 8;
      for (int y = y0; y < y1; y++)
      {
        std::memset(pad, 0, sizeof(float) * (size_t)(wv + 2 * r + 8));
        std::memcpy(pad + r, v + ro(y, w, ring_in), sizeof(float) * (size_t)w);
        for (int x = 0; x < wv; x += 32)
        {
          const float *p = pad + x;
          __m256 s0 = ld(p), s1 = ld(p + 8), s2 = ld(p + 16), s3 = ld(p + 24);
          for (int i = 1; i < size; i++)
          {
            s0 = add(s0, ld(p + i));
            s1 = add(s1, ld(p + 8 + i));
            s2 = add(s2, ld(p + 16 + i));
            s3 = add(s3, ld(p + 24 + i));
          }
          if (average)
          {
            s0 = _mm256_div_ps(s0, div);
            s1 = _mm256_div_ps(s1, div);
            s2 = _mm256_div_ps(s2, div);
            s3 = _mm256_div_ps(s3, div);
          }
          if (max_val >= 0.0f)
          {
            s0 = _mm256_min_ps(mx, s0);
            s1 = _mm256_min_ps(mx, s1);
            s2 = _mm256_min_ps(mx, s2);
            s3 = _mm256_min_ps(mx, s3);
          }
          st(tmp + x, s0);
          st(tmp + x + 8, s1);
          st(tmp + x + 16, s2);
          st(tmp + x + 24, s3);
        }
        std::memcpy(out + ro(y, w, ring_out), tmp, sizeof(float) * (size_t)w);
      }
    }

    // --------------------------------------------------------------------------------------- refinement / match errors
    void dfd(const float *cur, const float *other, int w, int hgt, const float *vx, const float *vy, float s, int y0,
      int y1, float *out, int ring_out)
    {
      const __m256 vs = bc(s), one = bc(1.0f);
      const __m256i wm1 = _mm256_set1_epi32(w - 1), hm1 = _mm256_set1_epi32(hgt - 1), zero = _mm256_setzero_si256();
      const __m256i vw = _mm256_set1_epi32(w), i1 = _mm256_set1_epi32(1);
      for (int y = y0; y < y1; y++)
      {
        const size_t rb = (size_t)y * w, ob = ro(y, w, ring_out);
        const __m256 fy = bc((float)y);
        int x = 0;
        for (; x + 8 <= w; x += 8)
        {
          const __m256 px = add(add(bc((float)x), LANE), mul(vs, ld(vx + rb + x)));
          const __m256 py = add(fy, mul(vs, ld(vy + rb + x)));
          const __m256 i0f = _mm256_floor_ps(px), j0f = _mm256_floor_ps(py);
          const __m256 a = sub(px, i0f);
          const __m256 b = sub(py, j0f);
          const __m256i i0 = _mm256_cvttps_epi32(i0f), j0 = _mm256_cvttps_epi32(j0f);
          const __m256i xa = _mm256_min_epi32(_mm256_max_epi32(i0, zero), wm1);
          const __m256i xb = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(i0, i1), zero), wm1);
          const __m256i ya = _mm256_mullo_epi32(_mm256_min_epi32(_mm256_max_epi32(j0, zero), hm1), vw);
          const __m256i yb = _mm256_mullo_epi32(_mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(j0, i1), zero), hm1), vw);
          const __m256 t00 = _mm256_i32gather_ps(other, _mm256_add_epi32(ya, xa), 4);
          const __m256 t10 = _mm256_i32gather_ps(other, _mm256_add_epi32(ya, xb), 4);
          const __m256 t01 = _mm256_i32gather_ps(other, _mm256_add_epi32(yb, xa), 4);
          const __m256 t11 = _mm256_i32gather_ps(other, _mm256_add_epi32(yb, xb), 4);
          const __m256 na = sub(one, a), nb = sub(one, b);
          __m256 r = mul(mul(na, nb), t00);
          r = add(r, mul(mul(a, nb), t10));
          r = add(r, mul(mul(na, b), t01));
          r = add(r, mul(mul(a, b), t11));
          st(out + ob + x, sub(ld(cur + rb + x), r));
        }
        for (; x < w; x++)
          out[ob + x] = cur[rb + x] - tex1(other, w, hgt, (float)x + s * vx[rb + x], (float)y + s * vy[rb + x]);
      }
    }

    void refine_terms(const float *cur, const float *err, const float *d, const float *vx, const float *vy, bool neg,
      int w, int hgt, int y0, int y1, float eps, float wscale, float *const *T, int ring)
    {
      const __m256 two = bc(2.0f), ve = bc(eps), vws = bc(wscale);
      for (int y = y0; y < y1; y++)
      {
        const size_t rb = (size_t)y * w, qb = ro(y, w, ring);
        const float *up = cur + (size_t)imax(y - 1, 0) * w, *dn = cur + (size_t)imin(y + 1, hgt - 1) * w;
        auto one = [&](int x) {
          const float gx = (cur[rb + imin(x + 1, w - 1)] - cur[rb + imax(x - 1, 0)]) / 2.0f;
          const float gy = (dn[x] - up[x]) / 2.0f;
          const float ixx = gx * gx, iyy = gy * gy, ixy = gx * gy;
          const float wt = exp32_1(-err[qb + x] / wscale);
          const float vxx = neg ? -vx[rb + x] : vx[rb + x], vyy = neg ? -vy[rb + x] : vy[rb + x];
          const float dd = d[qb + x];
          float b1 = ixx * vxx + ixy * vyy;
          b1 = b1 + gx * dd;
          float b2 = ixy * vxx + iyy * vyy;
          b2 = b2 + gy * dd;
          T[0][qb + x] = b1 * wt;
          T[1][qb + x] = b2 * wt;
          T[2][qb + x] = (ixx + eps) * wt;
          T[3][qb + x] = (iyy + eps) * wt;
          T[4][qb + x] = ixy * wt;
        };
        one(0);
        int x = 1;
        for (; x + 8 <= w - 1; x += 8)
        {
          const __m256 gx = _mm256_div_ps(sub(ld(cur + rb + x + 1), ld(cur + rb + x - 1)), two);
          const __m256 gy = _mm256_div_ps(sub(ld(dn + x), ld(up + x)), two);
          const __m256 ixx = mul(gx, gx), iyy = mul(gy, gy), ixy = mul(gx, gy);
          const __m256 wt = exp32_ps(_mm256_div_ps(vneg(ld(err + qb + x)), vws));
          __m256 vxx = ld(vx + rb + x), vyy = ld(vy + rb + x);
          if (neg)
          {
            vxx = vneg(vxx);
            vyy = vneg(vyy);
          }
          const __m256 dd = ld(d + qb + x);
          __m256 b1 = add(mul(ixx, vxx), mul(ixy, vyy));
          b1 = add(b1, mul(gx, dd));
          __m256 b2 = add(mul(ixy, vxx), mul(iyy, vyy));
          b2 = add(b2, mul(gy, dd));
          st(T[0] + qb + x, mul(b1, wt));
          st(T[1] + qb + x, mul(b2, wt));
          st(T[2] + qb + x, mul(add(ixx, ve), wt));
          st(T[3] + qb + x, mul(add(iyy, ve), wt));
          st(T[4] + qb + x, mul(ixy, wt));
        }
        for (; x < w; x++)
          one(x);
      }
    }

    void refine_solve(const float *const *B, int w, int y0, int y1, float *ox, float *oy, int ring)
    {
      for (int y = y0; y < y1; y++)
      {
        const size_t qb = ro(y, w, ring), ob = (size_t)y * w;
        int x = 0;
        for (; x + 8 <= w; x += 8)
        {
          const size_t i = qb + x;
          const __m256 b1 = ld(B[0] + i), b2 = ld(B[1] + i), a = ld(B[2] + i), c = ld(B[3] + i), bxy = ld(B[4] + i);
          const __m256 det = sub(mul(a, c), mul(bxy, bxy));
          st(ox + ob + x, _mm256_div_ps(sub(mul(c, b1), mul(bxy, b2)), det));
          st(oy + ob + x, _mm256_div_ps(sub(mul(a, b2), mul(bxy, b1)), det));
        }
        for (; x < w; x++)
        {
          const size_t i = qb + x;
          const float b1 = B[0][i], b2 = B[1][i], a = B[2][i], c = B[3][i], bxy = B[4][i];
          const float det = a * c - bxy * bxy;
          ox[ob + x] = (c * b1 - bxy * b2) / det;
          oy[ob + x] = (a * b2 - bxy * b1) / det;
        }
      }
    }

    void opposite_min(const float *e, int w, int hgt, int y0, int y1, float wq, int dist, int mode, float *out,
      int ring_in)
    {
      const float half = wq * 0.5f, keep = 1.0f - wq;
      const __m256 vh = bc(half), vk = bc(keep), vw = bc(wq);
      const int D = dist;
      for (int y = y0; y < y1; y++)
      {
        const float *row = e + ro(y, w, ring_in), *u = e + ro(imax(y - D, 0), w, ring_in);
        const float *dn = e + ro(imin(y + D, hgt - 1), w, ring_in);
        float *o = out + (size_t)y * w;
        auto one = [&](int x) {
          if (mode == 1)
          {
            const int xl = imax(x - D, 0), xr = imin(x + D, w - 1);
            const float p = u[xl] < u[xr] ? u[xl] : u[xr], q = dn[xl] < dn[xr] ? dn[xl] : dn[xr];
            o[x] = keep * row[x] + wq * (p < q ? p : q);
            return;
          }
          const float l = row[imax(x - D, 0)], r = row[imin(x + D, w - 1)];
          const float a = l < r ? l : r, b = u[x] < dn[x] ? u[x] : dn[x];
          o[x] = keep * row[x] + half * (a + b);
        };
        int x = 0;
        for (; x < D && x < w; x++)
          one(x);
        for (; mode == 1 && x + 8 <= w - D; x += 8)
        {
          const __m256 m = _mm256_min_ps(_mm256_min_ps(ld(u + x - D), ld(u + x + D)),
            _mm256_min_ps(ld(dn + x - D), ld(dn + x + D)));
          st(o + x, add(mul(vk, ld(row + x)), mul(vw, m)));
        }
        for (; mode == 0 && x + 8 <= w - D; x += 8)
        {
          const __m256 a = _mm256_min_ps(ld(row + x - D), ld(row + x + D));
          const __m256 b = _mm256_min_ps(ld(u + x), ld(dn + x));
          st(o + x, add(mul(vk, ld(row + x)), mul(vh, add(a, b))));
        }
        for (; x < w; x++)
          one(x);
      }
    }

    // ------------------------------------------------------------------------------------------------------ assignment
    // lam == 0: per sample and side the candidate of the lowest cost (the first strict minimum, as in the exact loop).
    // lam > 0: the pair (i, p) of the lowest cost, the first one in the order i outer, p inner. Only the costs and the
    // indices are carried through the loops, two groups of 8 samples side by side (two independent chains), and the
    // vectors and errors of the winners are read afterwards.
    void assign(const float *const *rf, const float *const *ef, const float *const *rb, const float *const *eb, int n,
      const float *bias, float lam, float thresh, float ratio, float lenc, size_t k0, size_t k1, float *const *out)
    {
      const __m256 sgn = bc(-0.0f), vl = bc(lam), vlc = bc(lenc);
      Scratch bm(sizeof(float) * (size_t)n * n);
      float *bsum = bm.f();
      for (int i = 0; i < n; i++)
        for (int p = 0; p < n; p++)
          bsum[i * n + p] = bias[p] + bias[i];
      __m256 ox[2][64], oy[2][64], ep[2][64], lp[2][64];
      for (size_t k = k0; k < k1; k += 16)
      {
        const int cnt = (int)(k1 - k < 16 ? k1 - k : 16);
        // the 16 samples' values of a plane (partial groups through a zero-padded copy)
        auto load2 = [&](const float *p, __m256 &a, __m256 &b) {
          if (cnt == 16)
          {
            a = ld(p + k);
            b = ld(p + k + 8);
            return;
          }
          float t[16] = { 0 };
          for (int j = 0; j < cnt; j++)
            t[j] = p[k + j];
          a = ld(t);
          b = ld(t + 8);
        };
        for (int p = 0; p < n; p++)
        {
          if (p < n - 1)
          {
            load2(rb[2 * p], ox[0][p], ox[1][p]);
            load2(rb[2 * p + 1], oy[0][p], oy[1][p]);
          }
          else
            ox[0][p] = ox[1][p] = oy[0][p] = oy[1][p] = _mm256_setzero_ps();
          load2(eb[p], ep[0][p], ep[1][p]);
          for (int g = 0; g < 2; g++) // error plus the vector length bias of the backward side
            lp[g][p] = add(ep[g][p], mul(vlc, add(_mm256_andnot_ps(sgn, ox[g][p]), _mm256_andnot_ps(sgn, oy[g][p]))));
        }
        __m256 best0 = bc(std::numeric_limits<float>::infinity()), best1 = best0;
        __m256i idx0 = _mm256_set1_epi32(-1), idx1 = idx0;
        int ix[16], ixb[16];
        if (lam == 0.0f)
        {
          // forward side: lowest le + bias[i]; backward side: lowest lp + bias[p]
          __m256 bb0 = best0, bb1 = best0;
          __m256i ib0 = idx0, ib1 = idx0;
          for (int i = 0; i < n; i++)
          {
            __m256 cx0, cx1, cy0, cy1, e0, e1;
            if (i < n - 1)
            {
              load2(rf[2 * i], cx0, cx1);
              load2(rf[2 * i + 1], cy0, cy1);
            }
            else
              cx0 = cx1 = cy0 = cy1 = _mm256_setzero_ps();
            load2(ef[i], e0, e1);
            const __m256 bi = bc(bias[i]);
            const __m256i id = _mm256_set1_epi32(i);
            const __m256 c0 = add(add(e0, mul(vlc, add(_mm256_andnot_ps(sgn, cx0), _mm256_andnot_ps(sgn, cy0)))), bi);
            const __m256 c1 = add(add(e1, mul(vlc, add(_mm256_andnot_ps(sgn, cx1), _mm256_andnot_ps(sgn, cy1)))), bi);
            const __m256 m0 = _mm256_cmp_ps(c0, best0, _CMP_LT_OQ), m1 = _mm256_cmp_ps(c1, best1, _CMP_LT_OQ);
            best0 = _mm256_blendv_ps(best0, c0, m0);
            best1 = _mm256_blendv_ps(best1, c1, m1);
            idx0 = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(idx0), _mm256_castsi256_ps(id), m0));
            idx1 = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(idx1), _mm256_castsi256_ps(id), m1));
            const __m256 d0 = add(lp[0][i], bi), d1 = add(lp[1][i], bi);
            const __m256 q0 = _mm256_cmp_ps(d0, bb0, _CMP_LT_OQ), q1 = _mm256_cmp_ps(d1, bb1, _CMP_LT_OQ);
            bb0 = _mm256_blendv_ps(bb0, d0, q0);
            bb1 = _mm256_blendv_ps(bb1, d1, q1);
            ib0 = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(ib0), _mm256_castsi256_ps(id), q0));
            ib1 = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(ib1), _mm256_castsi256_ps(id), q1));
          }
          _mm256_storeu_si256((__m256i *)ix, idx0);
          _mm256_storeu_si256((__m256i *)(ix + 8), idx1);
          _mm256_storeu_si256((__m256i *)ixb, ib0);
          _mm256_storeu_si256((__m256i *)(ixb + 8), ib1);
        }
        else
        {
        for (int i = 0; i < n; i++)
        {
          __m256 cx0, cx1, cy0, cy1, e0, e1;
          if (i < n - 1)
          {
            load2(rf[2 * i], cx0, cx1);
            load2(rf[2 * i + 1], cy0, cy1);
          }
          else
            cx0 = cx1 = cy0 = cy1 = _mm256_setzero_ps();
          load2(ef[i], e0, e1);
          const __m256 le0 = add(e0, mul(vlc, add(_mm256_andnot_ps(sgn, cx0), _mm256_andnot_ps(sgn, cy0))));
          const __m256 le1 = add(e1, mul(vlc, add(_mm256_andnot_ps(sgn, cx1), _mm256_andnot_ps(sgn, cy1))));
          for (int p = 0; p < n; p++)
          {
            const __m256 bs = bc(bsum[i * n + p]);
            const __m256i id = _mm256_set1_epi32(i * n + p);
            __m256 c0 = add(add(le0, lp[0][p]), mul(vl, add(_mm256_andnot_ps(sgn, add(cx0, ox[0][p])), _mm256_andnot_ps(sgn, add(cy0, oy[0][p])))));
            __m256 c1 = add(add(le1, lp[1][p]), mul(vl, add(_mm256_andnot_ps(sgn, add(cx1, ox[1][p])), _mm256_andnot_ps(sgn, add(cy1, oy[1][p])))));
            c0 = add(c0, bs);
            c1 = add(c1, bs);
            const __m256 m0 = _mm256_cmp_ps(c0, best0, _CMP_LT_OQ), m1 = _mm256_cmp_ps(c1, best1, _CMP_LT_OQ);
            best0 = _mm256_blendv_ps(best0, c0, m0);
            best1 = _mm256_blendv_ps(best1, c1, m1);
            idx0 = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(idx0), _mm256_castsi256_ps(id), m0));
            idx1 = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(idx1), _mm256_castsi256_ps(id), m1));
          }
        }
        _mm256_storeu_si256((__m256i *)ix, idx0);
        _mm256_storeu_si256((__m256i *)(ix + 8), idx1);
        for (int j = 0; j < 16; j++)
          if (ix[j] >= 0)
          {
            ixb[j] = ix[j] % n;
            ix[j] = ix[j] / n;
          }
          else
            ixb[j] = -1;
        }
        for (int j = 0; j < cnt; j++)
        {
          const size_t s = k + j;
          float vx = 0, vy = 0, e = 0, ovx = 0, ovy = 0, oe = 0;
          if (ix[j] >= 0)
          {
            const int i = ix[j];
            vx = i < n - 1 ? rf[2 * i][s] : 0.0f;
            vy = i < n - 1 ? rf[2 * i + 1][s] : 0.0f;
            e = ef[i][s];
          }
          if (ixb[j] >= 0)
          {
            const int p = ixb[j];
            ovx = p < n - 1 ? rb[2 * p][s] : 0.0f;
            ovy = p < n - 1 ? rb[2 * p + 1][s] : 0.0f;
            oe = eb[p][s];
          }
          splat::occlusion_switch(vx, vy, e, ovx, ovy, oe, thresh, ratio);
          const float rvx = vx, rvy = vy, e2 = e, rox = ovx, roy = ovy, oe2 = oe;
          out[0][s] = rvx; out[1][s] = rvy; out[2][s] = e2;
          out[3][s] = rox; out[4][s] = roy; out[5][s] = oe2;
        }
      }
    }

    void fft_lines_f(float *re, float *im, int S, bool rows, int g0, int g1, bool fwd, const fftp::Plan &plan,
      const fftp::cpx<float> *tw, int ntw, float fct)
    {
      if (fwd)
        lines<true, VF, float, 8>(re, im, S, rows, g0, g1, plan, tw, ntw, fct);
      else
        lines<false, VF, float, 8>(re, im, S, rows, g0, g1, plan, tw, ntw, fct);
    }

    // --------------------------------------------------------------------------------------------------- picture build
    void level_side(const float *const *G, int gw, int gh, bool full_res, int stride, int sh, int w, int y, float zero_len,
      float *vx, float *vy, float *mc, float *st, unsigned char *zero)
    {
      const float s = (float)stride, fsc = (float)(1 << sh);
      const int msk = (1 << sh) - 1;
      const __m256 vs = bc(s), vz = bc(zero_len), one = bc(1.0f), four = bc(fsc);
      auto finish = [&](int x, __m256 v0, __m256 v1, __m256 m, __m256 t) {
        v0 = _mm256_div_ps(v0, vs);
        v1 = _mm256_div_ps(v1, vs);
        const __m256 z = _mm256_cmp_ps(_mm256_sqrt_ps(add(mul(v0, v0), mul(v1, v1))), vz, _CMP_LE_OQ);
        t = _mm256_blendv_ps(t, add(t, m), z);
        _mm256_storeu_ps(vx + x, v0);
        _mm256_storeu_ps(vy + x, v1);
        _mm256_storeu_ps(mc + x, m);
        _mm256_storeu_ps(st + x, t);
        const int bits = _mm256_movemask_ps(z);
        for (int j = 0; j < 8; j++)
          zero[x + j] = (unsigned char)((bits >> j) & 1);
      };
      // one sample, scalar (the same operations)
      auto one1 = [&](int x) {
        float v[4];
        for (int c = 0; c < 4; c++)
        {
          if (!full_res)
          {
            v[c] = G[c][(size_t)imin(y, gh - 1) * gw + imin(x, gw - 1)];
            continue;
          }
          const int X = x * stride, Y = y * stride, xa = imin(X >> sh, gw - 1), ya = imin(Y >> sh, gh - 1);
          const int xb = imin(xa + 1, gw - 1), yb = imin(ya + 1, gh - 1);
          const float tx = (float)(X & msk) / fsc, ty = (float)(Y & msk) / fsc;
          float a = G[c][(size_t)ya * gw + xa], b = G[c][(size_t)ya * gw + xb];
          float cc = G[c][(size_t)yb * gw + xa], e = G[c][(size_t)yb * gw + xb];
          if (c < 2)
          {
            a = a * fsc;
            b = b * fsc;
            cc = cc * fsc;
            e = e * fsc;
          }
          const float l6 = a * (1.0f - tx) + b * tx;
          const float l18 = cc * (1.0f - tx) + e * tx;
          v[c] = l6 * (1.0f - ty) + l18 * ty;
        }
        const float v0 = v[0] / s, v1 = v[1] / s;
        float t = v[3];
        const float r = _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(v0 * v0 + v1 * v1)));
        const bool z = r <= zero_len;
        if (z)
          t = t + v[2];
        vx[x] = v0;
        vy[x] = v1;
        mc[x] = v[2];
        st[x] = t;
        zero[x] = z;
      };
      int x = 0;
      if (!full_res)
      {
        // edge clamp past the field (a coarse level can be one sample larger): the last row, scalar past the last column
        const size_t ro = (size_t)imin(y, gh - 1) * gw;
        const float *r[4] = { G[0] + ro, G[1] + ro, G[2] + ro, G[3] + ro };
        for (; x + 8 <= imin(w, gw); x += 8)
          finish(x, ld(r[0] + x), ld(r[1] + x), ld(r[2] + x), ld(r[3] + x));
      }
      else
      {
        const int Y = y * stride, ya = imin(Y >> sh, gh - 1), yb = imin(ya + 1, gh - 1);
        const __m256 ty = bc((float)(Y & msk) / fsc), nty = sub(one, ty);
        const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), gm1 = _mm256_set1_epi32(gw - 1);
        const __m256i three = _mm256_set1_epi32(msk), i1 = _mm256_set1_epi32(1), vst = _mm256_set1_epi32(stride);
        const __m128i vsh = _mm_cvtsi32_si128(sh);
        for (; x + 8 <= w; x += 8)
        {
          const __m256i X = _mm256_mullo_epi32(_mm256_add_epi32(_mm256_set1_epi32(x), lane), vst);
          const __m256i xa = _mm256_min_epi32(_mm256_sra_epi32(X, vsh), gm1), xb = _mm256_min_epi32(_mm256_add_epi32(xa, i1), gm1);
          const __m256 tx = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_and_si256(X, three)), four), ntx = sub(one, tx);
          __m256 v[4];
          for (int c = 0; c < 4; c++)
          {
            const float *ra = G[c] + (size_t)ya * gw, *rb = G[c] + (size_t)yb * gw;
            __m256 a = _mm256_i32gather_ps(ra, xa, 4), b = _mm256_i32gather_ps(ra, xb, 4);
            __m256 cc = _mm256_i32gather_ps(rb, xa, 4), e = _mm256_i32gather_ps(rb, xb, 4);
            if (c < 2)
            {
              a = mul(a, four);
              b = mul(b, four);
              cc = mul(cc, four);
              e = mul(e, four);
            }
            const __m256 l6 = add(mul(a, ntx), mul(b, tx));
            const __m256 l18 = add(mul(cc, ntx), mul(e, tx));
            v[c] = add(mul(l6, nty), mul(l18, ty));
          }
          finish(x, v[0], v[1], v[2], v[3]);
        }
      }
      for (; x < w; x++)
        one1(x);
    }

    void level_acc_row(const float *const *P, const float *const *N, const float *stP, const float *stN, int w, float *acc)
    {
      int x = 0;
      for (; x + 8 <= w; x += 8)
      {
        const __m256 sP = ld(stP + x), sN = ld(stN + x);
        const __m256 a0 = add(mul(ld(P[0] + x), sP), mul(ld(N[0] + x), sN));
        const __m256 a1 = add(mul(ld(P[1] + x), sP), mul(ld(N[1] + x), sN));
        const __m256 a2 = add(mul(ld(P[2] + x), sP), mul(ld(N[2] + x), sN));
        const __m256 a3 = add(sP, sN);
        // 4 x 8 -> 8 samples of 4 interleaved values
        const __m256 t0 = _mm256_unpacklo_ps(a0, a1), t1 = _mm256_unpackhi_ps(a0, a1);
        const __m256 t2 = _mm256_unpacklo_ps(a2, a3), t3 = _mm256_unpackhi_ps(a2, a3);
        const __m256 u0 = _mm256_shuffle_ps(t0, t2, 0x44), u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
        const __m256 u2 = _mm256_shuffle_ps(t1, t3, 0x44), u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
        float *o = acc + (size_t)x * 4;
        st(o, _mm256_permute2f128_ps(u0, u1, 0x20));
        st(o + 8, _mm256_permute2f128_ps(u2, u3, 0x20));
        st(o + 16, _mm256_permute2f128_ps(u0, u1, 0x31));
        st(o + 24, _mm256_permute2f128_ps(u2, u3, 0x31));
      }
      for (; x < w; x++)
      {
        float *a = acc + (size_t)x * 4;
        for (int c = 0; c < 3; c++)
          a[c] = P[c][x] * stP[x] + N[c][x] * stN[x];
        a[3] = stP[x] + stN[x];
      }
    }

    void pyr_reduce_v(const float *s, int w, int h, int j0, int j1, const float *taps, float *v)
    {
      const __m256 d0 = bc(taps[0]), d1 = bc(taps[1]), d2 = bc(taps[2]), d3 = bc(taps[3]);
      for (int j = j0; j < j1; j++)
      {
        const int y = 2 * j;
        const float *r0 = s + (size_t)y * w;
        const float *rm1 = s + (size_t)iclamp(y - 1, 0, h - 1) * w, *rp1 = s + (size_t)iclamp(y + 1, 0, h - 1) * w;
        const float *rm2 = s + (size_t)iclamp(y - 2, 0, h - 1) * w, *rp2 = s + (size_t)iclamp(y + 2, 0, h - 1) * w;
        const float *rm3 = s + (size_t)iclamp(y - 3, 0, h - 1) * w, *rp3 = s + (size_t)iclamp(y + 3, 0, h - 1) * w;
        float *o = v + (size_t)j * w;
        int x = 0;
        for (; x + 8 <= w; x += 8)
        {
          __m256 r = mul(d0, ld(r0 + x));
          r = add(r, mul(d1, add(ld(rm1 + x), ld(rp1 + x))));
          r = add(r, mul(d2, add(ld(rm2 + x), ld(rp2 + x))));
          r = add(r, mul(d3, add(ld(rm3 + x), ld(rp3 + x))));
          st(o + x, r);
        }
        for (; x < w; x++)
        {
          float r = taps[0] * r0[x];
          r = r + taps[1] * (rm1[x] + rp1[x]);
          r = r + taps[2] * (rm2[x] + rp2[x]);
          r = r + taps[3] * (rm3[x] + rp3[x]);
          o[x] = r;
        }
      }
    }

    void pyr_reduce_h(const float *v, int w, int y0, int y1, const float *taps, float *o)
    {
      // padded row q -> v[clamp(q - 4)] split into even and odd q: output i (x = 2 i) reads E[i + 2] (centre),
      // O[i + 1], O[i + 2] (distance 1), E[i + 1], E[i + 3] (2), O[i], O[i + 3] (3)
      const int ow = (w + 1) / 2, owv = (ow + 7) & ~7, half = owv + 8;
      const __m256 d0 = bc(taps[0]), d1 = bc(taps[1]), d2 = bc(taps[2]), d3 = bc(taps[3]);
      Scratch m(sizeof(float) * (size_t)(2 * half + owv));
      float *E = m.f(), *O = m.f() + half, *tmp = m.f() + 2 * half;
      for (int y = y0; y < y1; y++)
      {
        const float *row = v + (size_t)y * w;
        for (int q = 0; q < 2 * half; q++)
          (q & 1 ? O : E)[q >> 1] = row[iclamp(q - 4, 0, w - 1)];
        for (int i = 0; i < owv; i += 8)
        {
          __m256 r = mul(d0, ld(E + i + 2));
          r = add(r, mul(d1, add(ld(O + i + 1), ld(O + i + 2))));
          r = add(r, mul(d2, add(ld(E + i + 1), ld(E + i + 3))));
          r = add(r, mul(d3, add(ld(O + i), ld(O + i + 3))));
          st(tmp + i, r);
        }
        std::memcpy(o + (size_t)y * ow, tmp, sizeof(float) * (size_t)ow);
      }
    }

    void pyr_expand(const float *l, int lw, int lh, int y0, int y1, const float *ka, const float *kb, const float *kc,
      float *o, int ow, int oh, const float *base, int mode, float lo, float hi, const Crop *crop)
    {
      // rows y - 1 .. y + 2 of the low plane padded by edge repeat: R[k][x + 1] = l(clamp(x), clamp(y - 1 + k))
      const int lwv = (lw + 7) & ~7, pw = lwv + 4;
      const __m256 A0 = bc(ka[0]), A1 = bc(ka[1]), A2 = bc(ka[2]);
      const __m256 B0 = bc(kb[0]), B1 = bc(kb[1]), B2 = bc(kb[2]), B3 = bc(kb[3]);
      const __m256 C0 = bc(kc[0]), C1 = bc(kc[1]), C2 = bc(kc[2]);
      Scratch m(sizeof(float) * (size_t)(4 * pw + 4 * lwv));
      float *R[4] = { m.f(), m.f() + pw, m.f() + 2 * pw, m.f() + 3 * pw };
      float *T0 = m.f() + 4 * pw, *T1 = T0 + 2 * lwv;
      for (int y = y0; y < y1; y++)
      {
        if (crop && (2 * y + 1 < crop->cy || 2 * y >= crop->cy + crop->ch))
          continue;
        for (int k = 0; k < 4; k++)
        {
          const float *src = l + (size_t)iclamp(y - 1 + k, 0, lh - 1) * lw;
          for (int q = 0; q < pw; q++)
            R[k][q] = src[iclamp(q - 1, 0, lw - 1)];
        }
        for (int x = 0; x < lwv; x += 8)
        {
          const __m256 nw = ld(R[0] + x), n = ld(R[0] + x + 1), ne = ld(R[0] + x + 2), cm12 = ld(R[0] + x + 3);
          const __m256 wst = ld(R[1] + x), c = ld(R[1] + x + 1), e = ld(R[1] + x + 2), c02 = ld(R[1] + x + 3);
          const __m256 sw = ld(R[2] + x), s = ld(R[2] + x + 1), se = ld(R[2] + x + 2), c12 = ld(R[2] + x + 3);
          const __m256 c2m1 = ld(R[3] + x), c20 = ld(R[3] + x + 1), c21 = ld(R[3] + x + 2), c22 = ld(R[3] + x + 3);
          __m256 ee = mul(A0, c);
          ee = add(ee, mul(A1, add(add(add(n, wst), e), s)));
          ee = add(ee, mul(A2, add(add(add(nw, ne), sw), se)));
          __m256 eo = mul(B0, add(c, e));
          eo = add(eo, mul(B1, add(wst, c02)));
          eo = add(eo, mul(B2, add(add(add(n, ne), s), se)));
          eo = add(eo, mul(B3, add(add(add(nw, cm12), sw), c12)));
          __m256 oe = mul(B0, add(c, s));
          oe = add(oe, mul(B1, add(n, c20)));
          oe = add(oe, mul(B2, add(add(add(wst, e), sw), se)));
          oe = add(oe, mul(B3, add(add(add(nw, ne), c2m1), c21)));
          __m256 oo = mul(C0, add(add(add(c, e), s), se));
          oo = add(oo, mul(C1, add(add(add(add(add(add(add(n, ne), c20), c21), wst), c02), sw), c12)));
          oo = add(oo, mul(C2, add(add(add(nw, cm12), c2m1), c22)));
          // interleave even / odd columns: [e0 o0 e1 o1 .. e7 o7]
          const __m256 l0 = _mm256_unpacklo_ps(ee, eo), h0 = _mm256_unpackhi_ps(ee, eo);
          st(T0 + 2 * x, _mm256_permute2f128_ps(l0, h0, 0x20));
          st(T0 + 2 * x + 8, _mm256_permute2f128_ps(l0, h0, 0x31));
          const __m256 l1 = _mm256_unpacklo_ps(oe, oo), h1 = _mm256_unpackhi_ps(oe, oo);
          st(T1 + 2 * x, _mm256_permute2f128_ps(l1, h1, 0x20));
          st(T1 + 2 * x + 8, _mm256_permute2f128_ps(l1, h1, 0x31));
        }
        int n0 = imin(ow, 2 * lw);
        for (int r = 0; r < 2; r++)
        {
          const int oy = 2 * y + r;
          if (oy >= oh)
            break;
          const float *e = r ? T1 : T0;
          float *dst = o + (size_t)oy * ow;
          const float *b = base ? base + (size_t)oy * ow : nullptr;
          int x = 0;
          if (crop)
          {
            // output columns [cx, cx + cw) of row oy, written at crop row oy - cy (dst indexed in output columns)
            if (oy < crop->cy || oy >= crop->cy + crop->ch)
              continue;
            dst = crop->dst + (size_t)(oy - crop->cy) * crop->pitch - crop->cx;
            x = crop->cx;
            n0 = imin(n0, crop->cx + crop->cw);
          }
          if (mode == 0)
          {
            std::memcpy(dst + x, e + x, sizeof(float) * (size_t)(n0 - x));
            continue;
          }
          for (; x + 8 <= n0; x += 8)
          {
            __m256 v = mode == 1 ? sub(ld(b + x), ld(e + x)) : add(ld(b + x), ld(e + x));
            if (mode == 3)
              v = _mm256_min_ps(_mm256_max_ps(v, bc(lo)), bc(hi));
            st(dst + x, v);
          }
          for (; x < n0; x++)
          {
            float v = mode == 1 ? b[x] - e[x] : b[x] + e[x];
            if (mode == 3)
              v = (v > lo ? v : lo) < hi ? (v > lo ? v : lo) : hi;
            dst[x] = v;
          }
        }
      }
    }
  }
}
