// 4 neighbouring samples per thread for the per-sample kernels: loads and stores of 4 consecutive samples (8-bit samples
// or 16-bit containers) as aligned words, so a warp reads and writes whole rows with few wide accesses. ld4 reads only
// the words that hold the 4 samples (any sample alignment); st4 stores one word / two words where the address allows,
// else per sample.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace mvc
{
  template <class T>
  __device__ __forceinline__ void ld4(const T *q, int v[4])
  {
    const size_t a = (size_t)q;
    const unsigned *w = (const unsigned *)(a & ~(size_t)3);
    const int s = (int)(a & 3) * 8;
    if (sizeof(T) == 1)
    {
      unsigned u = w[0];
      if (s)
        u = __funnelshift_r(u, w[1], s);
      #pragma unroll
      for (int k = 0; k < 4; k++)
        v[k] = (u >> (8 * k)) & 0xffu;
    }
    else
    {
      unsigned u0 = w[0], u1 = w[1];
      if (s)
      {
        const unsigned u2 = w[2];
        u0 = __funnelshift_r(u0, u1, s);
        u1 = __funnelshift_r(u1, u2, s);
      }
      v[0] = u0 & 0xffffu;
      v[1] = u0 >> 16;
      v[2] = u1 & 0xffffu;
      v[3] = u1 >> 16;
    }
  }

  // the first n (1..4) samples: ld4 for n == 4, else per sample (the words past a row end may lie past the buffer)
  template <class T>
  __device__ __forceinline__ void ld4n(const T *q, int n, int v[4])
  {
    if (n == 4)
      ld4<T>(q, v);
    else
      for (int k = 0; k < 4; k++)
        v[k] = k < n ? (int)q[k] : 0;
  }

  // the first n (1..4) of v (values in range) to d
  template <class T>
  __device__ __forceinline__ void st4n(T *d, int n, const int v[4])
  {
    if (n == 4 && ((size_t)d & (4 * sizeof(T) - 1)) == 0)
    {
      if (sizeof(T) == 1)
        *(unsigned *)d = (unsigned)v[0] | ((unsigned)v[1] << 8) | ((unsigned)v[2] << 16) | ((unsigned)v[3] << 24);
      else
        *(uint2 *)d = make_uint2((unsigned)v[0] | ((unsigned)v[1] << 16), (unsigned)v[2] | ((unsigned)v[3] << 16));
      return;
    }
    for (int k = 0; k < n; k++)
      d[k] = (T)v[k];
  }
}
