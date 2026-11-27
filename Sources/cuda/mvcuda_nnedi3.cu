// NNEDI3 field interpolation on device frames, bit for bit with plugins_JPSDR 4.1.0 nnedi3 (nnedi3.cpp evalFunc_1 /
// evalFunc_2) for 8-bit data as it runs on AVX2 + FMA3 CPUs without AVX-512 (opt 6): new prescreener with int16 weights
// (computeNetwork0new_AVX2, processLine0_AVX2_ASM) and the int16 predictor (extract_m8_i16_AVX2,
// dotProd_m32/48_m16_i16_AVX2, e0_m16_FMA3, weightedAvgElliottMul5_m16_FMA3, castScale_AVX).
// 9 / 10-bit data (16-bit containers) take the plugin's _16 functions (evalFunc_1_16 / evalFunc_2_16): uc2s64_C_16,
// processLine0_AVX2_ASM_16, extract_m8_i16_AVX2_16, castScale_AVX_16. Their float steps are those of the 8-bit path; the
// prescreener and predictor read the samples as int16 (pmaddwd), the cubic kernel as uint16; the prescreener weights are
// scaled by the bit depth on the host.
//
// Every floating point step follows the order of the asm (explicit __f*_rn intrinsics: no contraction into FMA unless the
// asm uses one). The asm uses the CPU's approximate rcpps / rsqrtps, whose results are implementation specific: the host
// passes tables measured on its own CPU (on the CPUs checked the result depends only on the top 11 / 10 mantissa bits
// and the exponent, see cudafilters\CudaNnedi3.cpp, which verifies that before using them).
//
// The plugin pads the kept field by mirroring (32 columns, 3 field lines on each side, the edge not repeated); the
// kernels read the field through the same reflection. Work per frame: k_nn_pre makes every missing pixel by the cubic
// kernel and queues the ones the prescreener sends to the predictor; k_nn_pred runs the predictor on the queue
// (k_nn_pred_mma on sm_80 and newer: the same results with the integer dot products on the tensor cores).

#include "mvcuda_internal.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace
{
  struct NnNet
  {
    int xdia, ydia, nns, qual, asize;
    int16_t *pre_w;        // [4][64]
    float *pre_f;          // [28]
    int16_t *pred_w[2];    // [2 * nns][asize]
    float *pred_scale[2];  // [2 * nns]
    float *pred_bias[2];
    unsigned *rcp_tab;     // [2048]
    unsigned *rsq_tab;     // [2048]
    float r_asize;         // rcpss((float)asize) as the CPU computes it
    // tensor core predictor (sm_80 and newer): weights as B fragments of mma m16n8k32, per quality
    // [neuron group jt][softmax / elliott][k step][lane] uint4 (high bytes b0 b1, low bytes b0 b1)
    uint4 *pred_mma[2];
    int ks;                // k steps of 32 taps (asize rounded up to 32, the extra taps have weight 0)
    bool mma;
  };

  template <typename T>
  struct NnPlanes
  {
    const T *src[3];
    T *dst[3];
    size_t pitch_s[3], pitch_d[3]; // in samples
    int width[3], height[3];
    int interp[3];          // 1: made by NNEDI3, 0: copied
    int vmin[3], vmax[3];
  };

  // the CPU's rcpps for x >= tiny normal positive (all the inputs here: |t| + 1, sums of exponentials, 1 / sqrt of a
  // variance, the window size): the mantissa result comes from the table, the exponent is the input's mirrored
  __device__ __forceinline__ float rcp_hw(const unsigned *tab, float x)
  {
    const unsigned u = __float_as_uint(x);
    const int e = (int)(u >> 23) - 127;
    const unsigned r = __ldg(tab + ((u >> 12) & 2047u));
    return __uint_as_float(r - ((unsigned)e << 23));
  }

  // rsqrtps for positive normal x: x = y * 4^k with y in [1, 4), result = rsqrt(y) * 2^-k
  __device__ __forceinline__ float rsq_hw(const unsigned *tab, float x)
  {
    const unsigned u = __float_as_uint(x);
    const int e = (int)(u >> 23) - 127;
    const int par = e & 1;
    const int k = (e - par) >> 1;
    const unsigned r = __ldg(tab + par * 1024 + ((u >> 13) & 1023u));
    return __uint_as_float(r - ((unsigned)k << 23));
  }

  __device__ __forceinline__ int reflect(int i, int n)
  {
    if (i < 0)
      i = -i;
    if (i >= n)
      i = 2 * (n - 1) - i;
    return i < 0 ? 0 : i >= n ? n - 1 : i;
  }

  // sample of field line L (the kept lines: frame row 2 * L + off), column c, both reflected as the plugin's padding
  template <typename T>
  __device__ __forceinline__ int fld(const NnPlanes<T> &pl, int p, int off, int L, int c)
  {
    const int nl = pl.height[p] >> 1;
    L = reflect(L, nl);
    c = reflect(c, pl.width[p]);
    return pl.src[p][(size_t)(2 * L + off) * pl.pitch_s[p] + c];
  }

  // the sample as the network's int16 dot products (pmaddwd) see it: 8-bit samples zero-extended, 16-bit ones as int16
  template <typename T>
  __device__ __forceinline__ int net_in(int v)
  {
    return sizeof(T) == 1 ? v : (int)(int16_t)v;
  }

  // integer dot product of a word of two int16 weights with two samples of a word of samples (all products and sums in
  // int32, wrapping as pmaddwd / paddd): 8-bit samples (4 per word, lo = bytes 0 and 1, hi = bytes 2 and 3) zero-extended,
  // 16-bit samples (2 per word) as int16
  __device__ __forceinline__ int dot_u8_lo(unsigned w, unsigned v, int c)
  {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    int d;
    asm("dp2a.lo.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(w), "r"(v), "r"(c));
    return d;
#else
    return c + (int)(int16_t)(w & 0xffffu) * (int)(v & 0xffu) + (int)(int16_t)(w >> 16) * (int)((v >> 8) & 0xffu);
#endif
  }
  __device__ __forceinline__ int dot_u8_hi(unsigned w, unsigned v, int c)
  {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    int d;
    asm("dp2a.hi.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(w), "r"(v), "r"(c));
    return d;
#else
    return c + (int)(int16_t)(w & 0xffffu) * (int)((v >> 16) & 0xffu) + (int)(int16_t)(w >> 16) * (int)(v >> 24);
#endif
  }
  __device__ __forceinline__ int dot_i16(unsigned w, unsigned v, int c)
  {
    return c + (int)(int16_t)(w & 0xffffu) * (int)(int16_t)(v & 0xffffu) + (int)(int16_t)(w >> 16) * (int)(int16_t)(v >> 16);
  }

  constexpr int PRE_NT = 128;            // output columns per CTA
  constexpr int PRE_TW = PRE_NT + 16;    // field columns base - 6 .. base + 137 (the last group needs up to base + 133)
  constexpr int PRE_WS = 72;             // prescreener weight row stride in the shared copy (16-byte rows, 4 rows on 4 banks)

  // one CTA per 128 output samples of a row: kept lines copied, missing lines made by the cubic kernel; prescreened
  // pixels queued. The 4 field lines around the row are read once into shared memory; in each group of 4 pixels lane j
  // computes neuron j of the prescreener and the lanes exchange the results (integer sums: the same values as the
  // plugin's in any order; the float steps per lane as before). count_next (the queue counter of the next call, used by
  // the previous one) is cleared here.
  template <typename T>
  __global__ void __launch_bounds__(PRE_NT) k_nn_pre(NnNet net, NnPlanes<T> pl, int off, unsigned *queue, unsigned *count, unsigned *count_next)
  {
    __shared__ __align__(16) T tile[4][PRE_TW];
    __shared__ __align__(16) int16_t sw[4][PRE_WS];
    const int base = blockIdx.x * PRE_NT, c = base + threadIdx.x, r = blockIdx.y, p = blockIdx.z;
    if ((c | r | p) == 0)
      *count_next = 0;
    if (base >= pl.width[p] || r >= pl.height[p]) // whole CTA
      return;
    const bool in = c < pl.width[p];
    T *drow = pl.dst[p] + (size_t)r * pl.pitch_d[p];
    if (!pl.interp[p] || (r & 1) == off) // whole CTA
    {
      if (in)
        drow[c] = pl.src[p][(size_t)r * pl.pitch_s[p] + c];
      return;
    }
    const int L0 = (r - off - 3) >> 1; // field line of row r - 3 (r - off is odd)
    for (int i = threadIdx.x; i < 4 * PRE_TW; i += PRE_NT)
    {
      const int y = i / PRE_TW, x = i - y * PRE_TW;
      tile[y][x] = (T)fld(pl, p, off, L0 + y, base - 6 + x);
    }
    for (int i = threadIdx.x; i < 4 * 64; i += PRE_NT)
      sw[i >> 6][i & 63] = net.pre_w[i];
    __syncthreads();
    const int tc = c - base + 6; // tile column of c

    // processLine0_AVX2_ASM: (19 * (b + c) -us 3 * (a + d) +us 16) >> 5, clamped, written for every missing pixel;
    // processLine0_AVX2_ASM_16: (19 * (b + c) - 3 * (a + d) + 16) >>a 5 in int32, packusdw, min max, max min: the same
    // result for every input (a negative sum stays below 0 after the shift and saturates to 0)
    if (in)
    {
      const int a = tile[0][tc], b = tile[1][tc], cc = tile[2][tc], d = tile[3][tc];
      int t = 19 * (b + cc) - 3 * (a + d);
      t = t < 0 ? 0 : t;
      t = (t + 16) >> 5;
      const int smax = sizeof(T) == 1 ? 255 : 65535;
      t = t > smax ? smax : t;
      t = t > pl.vmax[p] ? pl.vmax[p] : t;
      t = t < pl.vmin[p] ? pl.vmin[p] : t;
      drow[c] = (T)(t < 0 ? 0 : t > smax ? smax : t);
    }

    // computeNetwork0new_AVX2 for the group of 4 pixels this one belongs to: 4 field lines x 16 columns from gx - 6
    // (tile column gx - base, a multiple of 4); every lane of the group runs, also past the plane's width
    const int gx = c & ~3, lane = c & 3;
    int sj = 0;
    {
      const unsigned *wr = reinterpret_cast<const unsigned *>(sw[lane]); // weights k, k + 1 of neuron lane per word
      #pragma unroll
      for (int y = 0; y < 4; y++)
      {
        const unsigned *tr = reinterpret_cast<const unsigned *>(&tile[y][gx - base]);
        if (sizeof(T) == 1)
        {
          #pragma unroll
          for (int q = 0; q < 4; q++)
          {
            const unsigned v = tr[q];
            sj = dot_u8_lo(wr[y * 8 + 2 * q], v, sj);
            sj = dot_u8_hi(wr[y * 8 + 2 * q + 1], v, sj);
          }
        }
        else
        {
          #pragma unroll
          for (int q = 0; q < 8; q++)
            sj = dot_i16(wr[y * 8 + q], tr[q], sj);
        }
      }
    }
    float hj;
    {
      const float t = __fadd_rn(__fmul_rn(__int2float_rn(sj), __ldg(net.pre_f + lane)), __ldg(net.pre_f + 4 + lane));
      hj = __fmul_rn(rcp_hw(net.rcp_tab, __fadd_rn(fabsf(t), 1.0f)), t);
    }
    const int gl = threadIdx.x & 28; // first lane of the group in the warp
    float h[4];
    #pragma unroll
    for (int j = 0; j < 4; j++)
      h[j] = __shfl_sync(0xffffffffu, hj, gl + j);
    if (!in)
      return;
    const float *w2 = net.pre_f + 8;
    const float o = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(h[0], __ldg(w2 + lane)), __fmul_rn(h[1], __ldg(w2 + 4 + lane))),
      __fadd_rn(__fmul_rn(h[2], __ldg(w2 + 8 + lane)), __fmul_rn(h[3], __ldg(w2 + 12 + lane)))), __ldg(net.pre_f + 24 + lane));
    if (o < 0.0f) // vcmpps lt: these go through the predictor
    {
      const unsigned i = atomicAdd(count, 1u);
      queue[i] = ((unsigned)p << 30) | ((unsigned)r << 16) | (unsigned)c;
    }
  }

  // one thread per queued pixel: extract_m8_i16 + qual x (dotProd_i16, e0 exp, weightedAvgElliottMul5) + castScale
  template <typename T>
  __global__ void k_nn_pred(NnNet net, NnPlanes<T> pl, int off, const unsigned *queue, const unsigned *count)
  {
    extern __shared__ __align__(16) unsigned sh[]; // window as words of 4 / 2 samples (taps k in order), [word][thread]
    constexpr int SPW = 4 / (int)sizeof(T);
    const unsigned n = *count;
    const int tid = threadIdx.x, nt = blockDim.x;
    for (unsigned qi = blockIdx.x * nt + tid; qi - tid < n; qi += gridDim.x * nt) // whole blocks loop together
    {
      const bool act = qi < n;
      int p = 0, r = 0, c = 0;
      if (act)
      {
        const unsigned e = queue[qi];
        p = (int)(e >> 30);
        r = (int)((e >> 16) & 0x3fff);
        c = (int)(e & 0xffff);
      }
      const int L0 = (r - off - (net.ydia - 1)) >> 1;
      const int x0 = c - ((net.xdia >> 1) - 1);
      int sum = 0, sumsq = 0;
      if (act)
        for (int y = 0; y < net.ydia; y++)
          for (int xw = 0; xw < net.xdia / SPW; xw++)
          {
            unsigned word = 0;
            #pragma unroll
            for (int e = 0; e < SPW; e++)
            {
              // extract_m8_i16_AVX2(_16): sum and sum of squares by pmaddwd / psadbw, int32
              const int raw = fld(pl, p, off, L0 + y, x0 + xw * SPW + e);
              const int v = net_in<T>(raw);
              word |= (unsigned)raw << (8 * (int)sizeof(T) * e);
              sum += v;
              sumsq += v * v;
            }
            sh[(y * (net.xdia / SPW) + xw) * nt + tid] = word;
          }
      __syncthreads();
      if (act)
      {
        // mean, variance with the CPU's rcpss(n); 1 / stddev by rsqrtss, stddev by rcpss of that
        const float mean = __fmul_rn(__int2float_rn(sum), net.r_asize);
        const float msq = __fmul_rn(__int2float_rn(sumsq), net.r_asize);
        const float var = __fsub_rn(msq, __fmul_rn(mean, mean));
        float istd = 0.0f, sd = 0.0f;
        if (var > 1.192092896e-07f) // vcomiss / jbe FLT_EPSILON
        {
          istd = rsq_hw(net.rsq_tab, var);
          sd = rcp_hw(net.rcp_tab, istd);
        }
        float acc = 0.0f; // mstd[3]
        const int nns = net.nns, asize = net.asize;
        for (int q = 0; q < net.qual; q++)
        {
          const int16_t *w = net.pred_w[q];
          const float *sc = net.pred_scale[q], *bi = net.pred_bias[q];
          float lw[8], lv[8]; // the 8 lanes of the ymm accumulators
          for (int i = 0; i < 8; i++)
          {
            lw[i] = 0.0f;
            lv[i] = 0.0f;
          }
          for (int j = 0; j < nns; j++)
          {
            // softmax neuron j and elliott neuron nns + j: int16 dot products, then (float * scale) * istd + bias
            // (asize is a multiple of 8: 8 weights per 16-byte load, the same load in every lane)
            int s0 = 0, s1 = 0;
            const uint4 *w0 = reinterpret_cast<const uint4 *>(w + (size_t)j * asize);
            const uint4 *w1 = reinterpret_cast<const uint4 *>(w + (size_t)(nns + j) * asize);
            for (int k8 = 0; k8 < asize / 8; k8++)
            {
              const uint4 a = __ldg(w0 + k8), b = __ldg(w1 + k8);
              if (sizeof(T) == 1)
              {
                const unsigned v0 = sh[(2 * k8) * nt + tid], v1 = sh[(2 * k8 + 1) * nt + tid];
                s0 = dot_u8_lo(a.x, v0, s0); s0 = dot_u8_hi(a.y, v0, s0); s0 = dot_u8_lo(a.z, v1, s0); s0 = dot_u8_hi(a.w, v1, s0);
                s1 = dot_u8_lo(b.x, v0, s1); s1 = dot_u8_hi(b.y, v0, s1); s1 = dot_u8_lo(b.z, v1, s1); s1 = dot_u8_hi(b.w, v1, s1);
              }
              else
              {
                const unsigned v0 = sh[(4 * k8) * nt + tid], v1 = sh[(4 * k8 + 1) * nt + tid];
                const unsigned v2 = sh[(4 * k8 + 2) * nt + tid], v3 = sh[(4 * k8 + 3) * nt + tid];
                s0 = dot_i16(a.x, v0, s0); s0 = dot_i16(a.y, v1, s0); s0 = dot_i16(a.z, v2, s0); s0 = dot_i16(a.w, v3, s0);
                s1 = dot_i16(b.x, v0, s1); s1 = dot_i16(b.y, v1, s1); s1 = dot_i16(b.z, v2, s1); s1 = dot_i16(b.w, v3, s1);
              }
            }
            const float t0 = __fadd_rn(__fmul_rn(__fmul_rn(__int2float_rn(s0), __ldg(sc + j)), istd), __ldg(bi + j));
            const float t1 = __fadd_rn(__fmul_rn(__fmul_rn(__int2float_rn(s1), __ldg(sc + nns + j)), istd), __ldg(bi + nns + j));
            // e0_m16_FMA3: min 80, max -80 (NaN -> 80), fma by (1 / ln 2) * 2^23 and the bias, round to int, reinterpret
            float xe = t0 < 80.0f ? t0 : 80.0f;
            xe = xe > -80.0f ? xe : -80.0f;
            const float ex = __int_as_float(__float2int_rn(__fmaf_rn(xe, 12102203.161561486f, 1064866805.0f)));
            // elliott with rcpps
            const float el = __fmul_rn(t1, rcp_hw(net.rcp_tab, __fadd_rn(fabsf(t1), 1.0f)));
            const int ln = j & 7;
            lw[ln] = __fadd_rn(lw[ln], ex);
            lv[ln] = __fmaf_rn(ex, el, lv[ln]);
          }
          // horizontal sums: (l0 + l4) + (l2 + l6) + ((l1 + l5) + (l3 + l7))
          const float aw0 = __fadd_rn(lw[0], lw[4]), aw1 = __fadd_rn(lw[1], lw[5]), aw2 = __fadd_rn(lw[2], lw[6]), aw3 = __fadd_rn(lw[3], lw[7]);
          const float av0 = __fadd_rn(lv[0], lv[4]), av1 = __fadd_rn(lv[1], lv[5]), av2 = __fadd_rn(lv[2], lv[6]), av3 = __fadd_rn(lv[3], lv[7]);
          const float wsum = __fadd_rn(__fadd_rn(aw0, aw2), __fadd_rn(aw1, aw3));
          float vsum = __fadd_rn(__fadd_rn(av0, av2), __fadd_rn(av1, av3));
          if (wsum > 1.0e-10f) // vcomiss / jbe min_weight_sum
            vsum = __fmul_rn(__fmul_rn(vsum, 5.0f), rcp_hw(net.rcp_tab, wsum));
          else
            vsum = 0.0f;
          acc = __fadd_rn(__fadd_rn(__fmul_rn(vsum, sd), mean), acc);
        }
        // castScale_AVX(_16): truncate(val * (1 / qual) + 0.5), clamp; cvttss2si gives INT_MIN for NaN / out of range
        const float scale = net.qual == 2 ? 0.5f : 1.0f;
        const float tv = __fadd_rn(__fmul_rn(acc, scale), 0.5f);
        int iv = (tv > -2147483648.0f && tv < 2147483648.0f) ? __float2int_rz(tv) : (int)0x80000000;
        if (iv >= pl.vmax[p])
          iv = pl.vmax[p];
        if (iv <= pl.vmin[p])
          iv = pl.vmin[p];
        pl.dst[p][(size_t)r * pl.pitch_d[p] + c] = (T)iv;
      }
      __syncthreads(); // the next round reuses the shared window
    }
  }

  // mma m16n8k32 with 8-bit A (u8 / s8) and B (s8 / u8), int32 accumulators (sm_80 and newer)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define NN_MMA(NAME, AT, BT)                                                                                          \
  __device__ __forceinline__ void NAME(int (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1)                 \
  {                                                                                                                   \
    asm("mma.sync.aligned.m16n8k32.row.col.s32." AT "." BT ".s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])                                                              \
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));                                              \
  }
  NN_MMA(mma_u8s8, "u8", "s8")
  NN_MMA(mma_u8u8, "u8", "u8")
  NN_MMA(mma_s8s8, "s8", "s8")
  NN_MMA(mma_s8u8, "s8", "u8")
#undef NN_MMA
#endif

  constexpr int MMA_WARPS = 4; // warps per CTA, 16 queued pixels per warp and round

  // k_nn_pred with the integer dot products on the tensor cores. Each int16 weight is split into a signed high byte
  // and an unsigned low byte (w = 256 * hi + lo), 16-bit samples the same way (v = 256 * vh + vl), and each dot product
  // is put together from the partial sums in int32: the same value as the plugin's pmaddwd / paddd sums (integer sums,
  // wrapping, in any order). Rows of the mma = 16 pixels, columns = 8 neurons, k = 32 taps per step (the taps beyond
  // asize have weight 0). In the accumulator layout lane (g, t) holds neurons 2t and 2t + 1 of every group of 8 for
  // pixels g and g + 8: exactly the plugin's ymm lanes j & 7, visited in increasing j, so every float step and its order
  // is that of k_nn_pred.
  template <typename T, int KS>
  __global__ void __launch_bounds__(32 * MMA_WARPS) k_nn_pred_mma(NnNet net, NnPlanes<T> pl, int off, const unsigned *queue, const unsigned *count)
  {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int NP = (int)sizeof(T); // sample parts: low bytes, high bytes (16-bit samples)
    constexpr int S = KS * 32 + 16;    // window row stride in bytes: the 8 rows of a fragment load on different banks
    __shared__ __align__(16) uint8_t shw[MMA_WARPS][NP][16 * S];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const unsigned n = *count;
    const int xdia = net.xdia, ydia = net.ydia, nns = net.nns;
    for (unsigned base = (blockIdx.x * MMA_WARPS + warp) * 16u; base < n; base += gridDim.x * MMA_WARPS * 16u)
    {
      // window of pixel i = lane & 15, columns of parity lane >> 4; sum and sum of squares as extract_m8_i16 (int32)
      const int i = lane & 15, h = lane >> 4;
      const bool act = base + i < n;
      const unsigned e = act ? queue[base + i] : 0u;
      const int p = (int)(e >> 30), r = (int)((e >> 16) & 0x3fff), c = (int)(e & 0xffff);
      int sum = 0, sumsq = 0;
      if (act)
      {
        const int L0 = (r - off - (ydia - 1)) >> 1;
        const int x0 = c - ((xdia >> 1) - 1);
        for (int y = 0; y < ydia; y++)
          for (int x = h; x < xdia; x += 2)
          {
            const int v = net_in<T>(fld(pl, p, off, L0 + y, x0 + x));
            const int k = i * S + y * xdia + x;
            shw[warp][0][k] = (uint8_t)(v & 255);
            if (NP == 2)
              shw[warp][NP - 1][k] = (uint8_t)((v >> 8) & 255);
            sum += v;
            sumsq += v * v;
          }
      }
      sum += __shfl_xor_sync(0xffffffffu, sum, 16);
      sumsq += __shfl_xor_sync(0xffffffffu, sumsq, 16);
      const float mean = __fmul_rn(__int2float_rn(sum), net.r_asize);
      const float msq = __fmul_rn(__int2float_rn(sumsq), net.r_asize);
      const float var = __fsub_rn(msq, __fmul_rn(mean, mean));
      float istd = 0.0f, sd = 0.0f;
      if (var > 1.192092896e-07f)
      {
        istd = rsq_hw(net.rsq_tab, var);
        sd = rcp_hw(net.rcp_tab, istd);
      }
      __syncwarp();

      // A fragments of all k steps
      unsigned a[NP][KS][4];
      #pragma unroll
      for (int pt = 0; pt < NP; pt++)
        #pragma unroll
        for (int ks = 0; ks < KS; ks++)
        {
          const uint8_t *s0 = shw[warp][pt] + g * S + ks * 32 + 4 * t;
          a[pt][ks][0] = *reinterpret_cast<const unsigned *>(s0);
          a[pt][ks][1] = *reinterpret_cast<const unsigned *>(s0 + 8 * S);
          a[pt][ks][2] = *reinterpret_cast<const unsigned *>(s0 + 16);
          a[pt][ks][3] = *reinterpret_cast<const unsigned *>(s0 + 8 * S + 16);
        }
      const float istd0 = __shfl_sync(0xffffffffu, istd, g), istd1 = __shfl_sync(0xffffffffu, istd, g + 8);
      // lanes t = 0 / 1 finish pixel g / g + 8
      const int fp = g + 8 * (t & 1);
      const float mean_f = __shfl_sync(0xffffffffu, mean, fp), sd_f = __shfl_sync(0xffffffffu, sd, fp);
      const unsigned e_f = __shfl_sync(0xffffffffu, e, fp);
      const bool act_f = t < 2 && base + fp < n;

      float acc = 0.0f;
      for (int q = 0; q < net.qual; q++)
      {
        const uint4 *wb = net.pred_mma[q] + lane;
        const float *sc = net.pred_scale[q], *bi = net.pred_bias[q];
        float lw[2][2] = { { 0.0f, 0.0f }, { 0.0f, 0.0f } }, lv[2][2] = { { 0.0f, 0.0f }, { 0.0f, 0.0f } }; // [pixel][column]
        for (int jt = 0; jt < nns / 8; jt++)
        {
          int ds[2 * NP][4], de[2 * NP][4];
          #pragma unroll
          for (int u = 0; u < 2 * NP; u++)
            #pragma unroll
            for (int x = 0; x < 4; x++)
              ds[u][x] = de[u][x] = 0;
          #pragma unroll
          for (int ks = 0; ks < KS; ks++)
          {
            const uint4 bs = __ldg(wb + ((jt * 2 + 0) * KS + ks) * 32);
            const uint4 be = __ldg(wb + ((jt * 2 + 1) * KS + ks) * 32);
            mma_u8s8(ds[0], a[0][ks], bs.x, bs.y);
            mma_u8u8(ds[1], a[0][ks], bs.z, bs.w);
            mma_u8s8(de[0], a[0][ks], be.x, be.y);
            mma_u8u8(de[1], a[0][ks], be.z, be.w);
            if (NP == 2)
            {
              mma_s8s8(ds[2], a[NP - 1][ks], bs.x, bs.y);
              mma_s8u8(ds[3], a[NP - 1][ks], bs.z, bs.w);
              mma_s8s8(de[2], a[NP - 1][ks], be.x, be.y);
              mma_s8u8(de[3], a[NP - 1][ks], be.z, be.w);
            }
          }
          const int j = jt * 8 + 2 * t;
          const float2 scS = __ldg(reinterpret_cast<const float2 *>(sc + j)), scE = __ldg(reinterpret_cast<const float2 *>(sc + nns + j));
          const float2 biS = __ldg(reinterpret_cast<const float2 *>(bi + j)), biE = __ldg(reinterpret_cast<const float2 *>(bi + nns + j));
          #pragma unroll
          for (int x = 0; x < 4; x++)
          {
            const int col = x & 1, pi = x >> 1;
            unsigned s0 = (unsigned)ds[0][x] * 256u + (unsigned)ds[1][x];
            unsigned s1 = (unsigned)de[0][x] * 256u + (unsigned)de[1][x];
            if (NP == 2)
            {
              s0 += ((unsigned)ds[2][x] * 256u + (unsigned)ds[3][x]) * 256u;
              s1 += ((unsigned)de[2][x] * 256u + (unsigned)de[3][x]) * 256u;
            }
            const float is = pi ? istd1 : istd0;
            const float t0 = __fadd_rn(__fmul_rn(__fmul_rn(__int2float_rn((int)s0), col ? scS.y : scS.x), is), col ? biS.y : biS.x);
            const float t1 = __fadd_rn(__fmul_rn(__fmul_rn(__int2float_rn((int)s1), col ? scE.y : scE.x), is), col ? biE.y : biE.x);
            float xe = t0 < 80.0f ? t0 : 80.0f;
            xe = xe > -80.0f ? xe : -80.0f;
            const float ex = __int_as_float(__float2int_rn(__fmaf_rn(xe, 12102203.161561486f, 1064866805.0f)));
            const float el = __fmul_rn(t1, rcp_hw(net.rcp_tab, __fadd_rn(fabsf(t1), 1.0f)));
            lw[pi][col] = __fadd_rn(lw[pi][col], ex);
            lv[pi][col] = __fmaf_rn(ex, el, lv[pi][col]);
          }
        }
        // the 8 ymm lanes of the finishing pixel: lane ln sits in column ln & 1 of lane (g, ln >> 1)
        float W[8], V[8];
        #pragma unroll
        for (int s = 0; s < 4; s++)
          #pragma unroll
          for (int col = 0; col < 2; col++)
          {
            const int src = g * 4 + s;
            const float w0 = __shfl_sync(0xffffffffu, lw[0][col], src), w1 = __shfl_sync(0xffffffffu, lw[1][col], src);
            const float v0 = __shfl_sync(0xffffffffu, lv[0][col], src), v1 = __shfl_sync(0xffffffffu, lv[1][col], src);
            W[2 * s + col] = (t & 1) ? w1 : w0;
            V[2 * s + col] = (t & 1) ? v1 : v0;
          }
        const float aw0 = __fadd_rn(W[0], W[4]), aw1 = __fadd_rn(W[1], W[5]), aw2 = __fadd_rn(W[2], W[6]), aw3 = __fadd_rn(W[3], W[7]);
        const float av0 = __fadd_rn(V[0], V[4]), av1 = __fadd_rn(V[1], V[5]), av2 = __fadd_rn(V[2], V[6]), av3 = __fadd_rn(V[3], V[7]);
        const float wsum = __fadd_rn(__fadd_rn(aw0, aw2), __fadd_rn(aw1, aw3));
        float vsum = __fadd_rn(__fadd_rn(av0, av2), __fadd_rn(av1, av3));
        if (wsum > 1.0e-10f)
          vsum = __fmul_rn(__fmul_rn(vsum, 5.0f), rcp_hw(net.rcp_tab, wsum));
        else
          vsum = 0.0f;
        acc = __fadd_rn(__fadd_rn(__fmul_rn(vsum, sd_f), mean_f), acc);
      }
      if (act_f)
      {
        const int pf = (int)(e_f >> 30), rf = (int)((e_f >> 16) & 0x3fff), cf = (int)(e_f & 0xffff);
        const float scale = net.qual == 2 ? 0.5f : 1.0f;
        const float tv = __fadd_rn(__fmul_rn(acc, scale), 0.5f);
        int iv = (tv > -2147483648.0f && tv < 2147483648.0f) ? __float2int_rz(tv) : (int)0x80000000;
        if (iv >= pl.vmax[pf])
          iv = pl.vmax[pf];
        if (iv <= pl.vmin[pf])
          iv = pl.vmin[pf];
        pl.dst[pf][(size_t)rf * pl.pitch_d[pf] + cf] = (T)iv;
      }
      __syncwarp(); // the next round reuses the shared window
    }
#endif
  }

  template <typename T, int KS>
  void launch_pred_mma_ks(const NnNet &nv, const NnPlanes<T> &pl, int off, const unsigned *qbuf, const unsigned *count, cudaStream_t st)
  {
    k_nn_pred_mma<T, KS><<<1024, 32 * MMA_WARPS, 0, st>>>(nv, pl, off, qbuf, count);
  }

  template <typename T>
  void launch_pred_mma(const NnNet &nv, const NnPlanes<T> &pl, int off, const unsigned *qbuf, const unsigned *count, cudaStream_t st)
  {
    switch (nv.ks)
    {
    case 1: launch_pred_mma_ks<T, 1>(nv, pl, off, qbuf, count, st); break;
    case 2: launch_pred_mma_ks<T, 2>(nv, pl, off, qbuf, count, st); break;
    case 3: launch_pred_mma_ks<T, 3>(nv, pl, off, qbuf, count, st); break;
    case 4: launch_pred_mma_ks<T, 4>(nv, pl, off, qbuf, count, st); break;
    case 6: launch_pred_mma_ks<T, 6>(nv, pl, off, qbuf, count, st); break;
    default: launch_pred_mma_ks<T, 9>(nv, pl, off, qbuf, count, st); break;
    }
  }

  template <typename T>
  T *upload(const T *src, size_t count)
  {
    T *d = nullptr;
    if (!src || cudaMalloc(&d, count * sizeof(T)) != cudaSuccess)
      return nullptr;
    if (cudaMemcpy(d, src, count * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess)
    {
      cudaFree(d);
      return nullptr;
    }
    return d;
  }

  unsigned nn_seq = 0; // pipeline order of the calls (all sample types: they share the queue counters)

  template <typename T>
  int run(const NnNet &n, const mvc::DevFrame &fs, const mvc::DevFrame &dst, int field_n, int plane_mask,
    const int val_min[3], const int val_max[3])
  {
    const MvCudaFrameFormat &f = fs.f;
    NnPlanes<T> pl = {};
    int max_w = 0, max_h = 0;
    size_t queue_max = 0;
    for (int p = 0; p < f.planes; p++)
    {
      pl.src[p] = (const T *)fs.plane[p];
      pl.dst[p] = (T *)dst.plane[p];
      pl.pitch_s[p] = fs.pitch[p] / sizeof(T);
      pl.pitch_d[p] = dst.pitch[p] / sizeof(T);
      pl.width[p] = fs.w[p];
      pl.height[p] = fs.h[p];
      pl.interp[p] = (plane_mask >> p) & 1;
      pl.vmin[p] = val_min[p];
      pl.vmax[p] = val_max[p];
      if (pl.interp[p] && (fs.h[p] < 8 || (fs.h[p] & 1) || fs.w[p] < 32 || fs.w[p] > 65535 || fs.h[p] > 16383))
        return -4;
      max_w = fs.w[p] > max_w ? fs.w[p] : max_w;
      max_h = fs.h[p] > max_h ? fs.h[p] : max_h;
      queue_max += (size_t)fs.w[p] * (fs.h[p] / 2);
    }
    const size_t qbytes = (queue_max ? queue_max : 1) * sizeof(unsigned);
    unsigned *counts = (unsigned *)mvc::pipe_scratch(mvc::SCRATCH_NNEDI3, 2 * sizeof(unsigned));
    unsigned *qbuf = counts ? (unsigned *)mvc::pipe_alloc(qbytes) : nullptr;
    if (!qbuf)
      return -2;
    const int off = 1 - field_n;
    const NnNet nv = n; // device pointers only; the net is freed after pipe_flush
    const dim3 grid((max_w + PRE_NT - 1) / PRE_NT, max_h, f.planes);
    mvc::pipe_submit([=](cudaStream_t st) { // issued by the pipeline submitter (mvcuda_internal.h)
      // the calls alternate between two queue counters (pipeline order): each clears the other one for the next call
      const unsigned seq = nn_seq++;
      unsigned *count = counts + (seq & 1), *count_next = counts + ((seq + 1) & 1);
      k_nn_pre<T><<<grid, PRE_NT, 0, st>>>(nv, pl, off, qbuf, count, count_next);
      if (nv.mma)
        launch_pred_mma<T>(nv, pl, off, qbuf, count, st);
      else
      {
        const int nt = 64;
        k_nn_pred<T><<<1024, nt, (size_t)nv.asize * nt * sizeof(T), st>>>(nv, pl, off, qbuf, count);
      }
    });
    mvc::pipe_free(qbuf, qbytes); // stream-ordered reuse: later pipeline work only
    return 0;
  }
}

extern "C" void *mvcuda_nnedi3_create(const MvCudaNnediParams *p)
{
  if (!p || !mvc::use_device() || (p->qual != 1 && p->qual != 2) || p->nns < 16 || p->nns % 16 != 0 || p->xdia < 8
    || p->xdia % 8 != 0 || p->ydia < 4 || (p->ydia & 1) || p->xdia * p->ydia > 48 * 6)
    return nullptr;
  NnNet *n = new NnNet();
  n->xdia = p->xdia;
  n->ydia = p->ydia;
  n->nns = p->nns;
  n->qual = p->qual;
  n->asize = p->xdia * p->ydia;
  bool ok = (n->pre_w = upload((const int16_t *)p->pre_w, 4 * 64)) && (n->pre_f = upload(p->pre_f, 28))
    && (n->rcp_tab = upload(p->rcp_tab, 2048)) && (n->rsq_tab = upload(p->rsq_tab, 2048));
  for (int q = 0; q < p->qual && ok; q++)
    ok = (n->pred_w[q] = upload((const int16_t *)p->pred_w[q], (size_t)2 * n->nns * n->asize))
      && (n->pred_scale[q] = upload(p->pred_scale[q], (size_t)2 * n->nns)) && (n->pred_bias[q] = upload(p->pred_bias[q], (size_t)2 * n->nns));
  // rcpss of the window size, from the same table (the plugin computes it per pixel with vrcpss)
  {
    const float fa = (float)n->asize;
    unsigned u;
    std::memcpy(&u, &fa, 4);
    const int e = (int)(u >> 23) - 127;
    const unsigned r = p->rcp_tab ? p->rcp_tab[(u >> 12) & 2047u] - ((unsigned)e << 23) : 0;
    std::memcpy(&n->r_asize, &r, 4);
  }
  // tensor core predictor on sm_80 and newer (MVCUDA_NNEDI3_MMA=0: the plain kernel)
  {
    int dev = 0, major = 0;
    const char *ev = std::getenv("MVCUDA_NNEDI3_MMA");
    n->ks = (n->asize + 31) / 32;
    n->mma = ok && cudaGetDevice(&dev) == cudaSuccess && cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess
      && major >= 8 && !(ev && ev[0] == '0') && (n->ks <= 4 || n->ks == 6 || n->ks == 9);
    const int nns = n->nns, asize = n->asize, ks = n->ks;
    for (int q = 0; q < p->qual && n->mma; q++)
    {
      const int16_t *w = (const int16_t *)p->pred_w[q]; // [2 * nns][asize]
      std::vector<uint4> fr((size_t)(nns / 8) * 2 * ks * 32);
      // 4 taps from k on as bytes (tap k in the lowest), high or low byte of the weights, 0 beyond asize
      auto pack = [&](int neuron, int k, bool hi) {
        unsigned u = 0;
        for (int b = 0; b < 4; b++)
        {
          const int wv = k + b < asize ? w[(size_t)neuron * asize + k + b] : 0;
          u |= (unsigned)((hi ? wv >> 8 : wv) & 255) << (8 * b);
        }
        return u;
      };
      for (int jt = 0; jt < nns / 8; jt++)
        for (int part = 0; part < 2; part++)
          for (int s = 0; s < ks; s++)
            for (int lane = 0; lane < 32; lane++)
            {
              // B fragment of mma m16n8k32: b0 = taps 4t .. 4t + 3, b1 = taps 16 + 4t .., column (neuron) g
              const int g = lane >> 2, t = lane & 3, neuron = part * nns + jt * 8 + g, k0 = s * 32 + 4 * t;
              fr[(((size_t)jt * 2 + part) * ks + s) * 32 + lane] =
                make_uint4(pack(neuron, k0, true), pack(neuron, k0 + 16, true), pack(neuron, k0, false), pack(neuron, k0 + 16, false));
            }
      ok = (n->pred_mma[q] = upload(fr.data(), fr.size())) != nullptr;
    }
  }
  if (!ok)
  {
    mvcuda_nnedi3_free(n);
    return nullptr;
  }
  return n;
}

extern "C" void mvcuda_nnedi3_free(void *net)
{
  NnNet *n = static_cast<NnNet *>(net);
  if (!n)
    return;
  mvc::pipe_flush(); // queued frames may still read the weights
  cudaFree(n->pre_w);
  cudaFree(n->pre_f);
  cudaFree(n->rcp_tab);
  cudaFree(n->rsq_tab);
  for (int q = 0; q < 2; q++)
  {
    cudaFree(n->pred_w[q]);
    cudaFree(n->pred_scale[q]);
    cudaFree(n->pred_bias[q]);
    cudaFree(n->pred_mma[q]);
  }
  delete n;
}

extern "C" int mvcuda_nnedi3(void *net, unsigned long long src, int field_n, int plane_mask, const int val_min[3],
  const int val_max[3], unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  NnNet *n = static_cast<NnNet *>(net);
  if (!n || !val_min || !val_max || !out || (field_n != 0 && field_n != 1))
    return -4;
  if (!mvc::use_device())
    return -1;
  auto fs = mvc::frame_get(src);
  if (!fs)
    return MVCUDA_TOKEN_GONE;
  const MvCudaFrameFormat &f = fs->f;
  if (f.bytes_per_sample != 1 && f.bytes_per_sample != 2)
    return -4;
  auto dst = mvc::frame_alloc(f);
  if (!dst)
    return -2;
  const int rc = f.bytes_per_sample == 1 ? run<uint8_t>(*n, *fs, *dst, field_n, plane_mask, val_min, val_max)
                                         : run<uint16_t>(*n, *fs, *dst, field_n, plane_mask, val_min, val_max);
  if (rc != 0)
    return rc;
  mvc::frame_put(dst, producer, frame, *out);
  return 0;
}
