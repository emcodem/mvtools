// NNEDI3 field interpolation on device frames, bit for bit with plugins_JPSDR 4.1.0 nnedi3 (nnedi3.cpp evalFunc_1 /
// evalFunc_2) for 8-bit data as it runs on AVX2 + FMA3 CPUs without AVX-512 (opt 6): new prescreener with int16 weights
// (computeNetwork0new_AVX2, processLine0_AVX2_ASM) and the int16 predictor (extract_m8_i16_AVX2,
// dotProd_m32/48_m16_i16_AVX2, e0_m16_FMA3, weightedAvgElliottMul5_m16_FMA3, castScale_AVX).
//
// Every floating point step follows the order of the asm (explicit __f*_rn intrinsics: no contraction into FMA unless the
// asm uses one). The asm uses the CPU's approximate rcpps / rsqrtps, whose results are implementation specific: the host
// passes tables measured on its own CPU (on the CPUs checked the result depends only on the top 11 / 10 mantissa bits
// and the exponent, see cudafilters\CudaNnedi3.cpp, which verifies that before using them).
//
// The plugin pads the kept field by mirroring (32 columns, 3 field lines on each side, the edge not repeated); the
// kernels read the field through the same reflection. Work per frame: k_nn_pre makes every missing pixel by the cubic
// kernel and queues the ones the prescreener sends to the predictor; k_nn_pred runs the predictor on the queue.

#include "mvcuda_internal.h"

#include <cstdint>
#include <cstring>

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
  };

  struct NnPlanes
  {
    const uint8_t *src[3];
    uint8_t *dst[3];
    size_t pitch_s[3], pitch_d[3];
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
  __device__ __forceinline__ int fld(const NnPlanes &pl, int p, int off, int L, int c)
  {
    const int nl = pl.height[p] >> 1;
    L = reflect(L, nl);
    c = reflect(c, pl.width[p]);
    return pl.src[p][(size_t)(2 * L + off) * pl.pitch_s[p] + c];
  }

  // one thread per output sample: kept lines copied, missing lines made by the cubic kernel; prescreened pixels queued.
  // count_next (the queue counter of the next call, used by the previous one) is cleared here.
  __global__ void k_nn_pre(NnNet net, NnPlanes pl, int off, unsigned *queue, unsigned *count, unsigned *count_next)
  {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y, p = blockIdx.z;
    if ((c | r | p) == 0)
      *count_next = 0;
    if (c >= pl.width[p] || r >= pl.height[p])
      return;
    uint8_t *drow = pl.dst[p] + (size_t)r * pl.pitch_d[p];
    if (!pl.interp[p] || (r & 1) == off)
    {
      drow[c] = pl.src[p][(size_t)r * pl.pitch_s[p] + c];
      return;
    }
    const int L0 = (r - off - 3) >> 1; // field line of row r - 3 (r - off is odd)

    // processLine0_AVX2_ASM: (19 * (b + c) -us 3 * (a + d) +us 16) >> 5, clamped, written for every missing pixel
    {
      const int a = fld(pl, p, off, L0, c), b = fld(pl, p, off, L0 + 1, c), cc = fld(pl, p, off, L0 + 2, c), d = fld(pl, p, off, L0 + 3, c);
      int t = 19 * (b + cc) - 3 * (a + d);
      t = t < 0 ? 0 : t;
      t = (t + 16) >> 5;
      t = t > pl.vmax[p] ? pl.vmax[p] : t;
      t = t < pl.vmin[p] ? pl.vmin[p] : t;
      drow[c] = (uint8_t)(t < 0 ? 0 : t > 255 ? 255 : t);
    }

    // computeNetwork0new_AVX2 for the group of 4 pixels this one belongs to: 4 field lines x 16 columns from gx - 6
    const int gx = c & ~3, lane = c & 3;
    int s[4] = { 0, 0, 0, 0 };
    for (int y = 0; y < 4; y++)
      for (int x = 0; x < 16; x++)
      {
        const int v = fld(pl, p, off, L0 + y, gx - 6 + x);
        const int k = y * 16 + x;
        for (int j = 0; j < 4; j++)
          s[j] += v * (int)__ldg(net.pre_w + j * 64 + k);
      }
    float h[4];
    for (int j = 0; j < 4; j++)
    {
      const float t = __fadd_rn(__fmul_rn(__int2float_rn(s[j]), __ldg(net.pre_f + j)), __ldg(net.pre_f + 4 + j));
      h[j] = __fmul_rn(rcp_hw(net.rcp_tab, __fadd_rn(fabsf(t), 1.0f)), t);
    }
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
  __global__ void k_nn_pred(NnNet net, NnPlanes pl, int off, const unsigned *queue, const unsigned *count)
  {
    extern __shared__ uint8_t sh[]; // window, [k][thread]
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
          for (int x = 0; x < net.xdia; x++)
          {
            const int v = fld(pl, p, off, L0 + y, x0 + x);
            sh[(y * net.xdia + x) * nt + tid] = (uint8_t)v;
            sum += v;
            sumsq += v * v;
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
            int s0 = 0, s1 = 0;
            const int16_t *w0 = w + (size_t)j * asize, *w1 = w + (size_t)(nns + j) * asize;
            for (int k = 0; k < asize; k++)
            {
              const int v = sh[k * nt + tid];
              s0 += v * (int)__ldg(w0 + k);
              s1 += v * (int)__ldg(w1 + k);
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
        // castScale_AVX: truncate(val * (1 / qual) + 0.5), clamp; cvttss2si gives INT_MIN for NaN / out of range
        const float scale = net.qual == 2 ? 0.5f : 1.0f;
        const float tv = __fadd_rn(__fmul_rn(acc, scale), 0.5f);
        int iv = (tv > -2147483648.0f && tv < 2147483648.0f) ? __float2int_rz(tv) : (int)0x80000000;
        if (iv >= pl.vmax[p])
          iv = pl.vmax[p];
        if (iv <= pl.vmin[p])
          iv = pl.vmin[p];
        pl.dst[p][(size_t)r * pl.pitch_d[p] + c] = (uint8_t)iv;
      }
      __syncthreads(); // the next round reuses the shared window
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
}

extern "C" void *mvcuda_nnedi3_create(const MvCudaNnediParams *p)
{
  if (!p || !mvc::use_device() || (p->qual != 1 && p->qual != 2) || p->nns < 16 || p->nns % 16 != 0 || p->xdia < 8
    || p->ydia < 4 || (p->ydia & 1) || p->xdia * p->ydia > 48 * 6)
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
  if (f.bytes_per_sample != 1)
    return -4;
  NnPlanes pl = {};
  int max_w = 0, max_h = 0;
  size_t queue_max = 0;
  auto dst = mvc::frame_alloc(f);
  if (!dst)
    return -2;
  for (int p = 0; p < f.planes; p++)
  {
    pl.src[p] = (const uint8_t *)fs->plane[p];
    pl.dst[p] = (uint8_t *)dst->plane[p];
    pl.pitch_s[p] = fs->pitch[p];
    pl.pitch_d[p] = dst->pitch[p];
    pl.width[p] = fs->w[p];
    pl.height[p] = fs->h[p];
    pl.interp[p] = (plane_mask >> p) & 1;
    pl.vmin[p] = val_min[p];
    pl.vmax[p] = val_max[p];
    if (pl.interp[p] && (fs->h[p] < 8 || (fs->h[p] & 1) || fs->w[p] < 32 || fs->w[p] > 65535 || fs->h[p] > 16383))
      return -4;
    max_w = fs->w[p] > max_w ? fs->w[p] : max_w;
    max_h = fs->h[p] > max_h ? fs->h[p] : max_h;
    queue_max += (size_t)fs->w[p] * (fs->h[p] / 2);
  }
  const size_t qbytes = (queue_max ? queue_max : 1) * sizeof(unsigned);
  unsigned *counts = (unsigned *)mvc::pipe_scratch(mvc::SCRATCH_NNEDI3, 2 * sizeof(unsigned));
  unsigned *qbuf = counts ? (unsigned *)mvc::pipe_alloc(qbytes) : nullptr;
  if (!qbuf)
    return -2;
  const int off = 1 - field_n;
  const NnNet nv = *n; // device pointers only; the net is freed after pipe_flush
  const dim3 grid((max_w + 127) / 128, max_h, f.planes);
  mvc::pipe_submit([=](cudaStream_t st) { // issued by the pipeline submitter (mvcuda_internal.h)
    // the calls alternate between two queue counters (pipeline order): each clears the other one for the next call
    static unsigned seq = 0;
    unsigned *count = counts + (seq & 1), *count_next = counts + ((seq + 1) & 1);
    seq++;
    k_nn_pre<<<grid, 128, 0, st>>>(nv, pl, off, qbuf, count, count_next);
    const int nt = 64;
    k_nn_pred<<<1024, nt, (size_t)nv.asize * nt, st>>>(nv, pl, off, qbuf, count);
  });
  mvc::pipe_free(qbuf, qbytes); // stream-ordered reuse: later pipeline work only
  mvc::frame_put(dst, producer, frame, *out);
  return 0;
}
