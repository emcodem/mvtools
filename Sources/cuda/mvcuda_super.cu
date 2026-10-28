// GPU version of MSuper (GPU_PLAN.md M2): pyramid reduction (rfilter=4), edge padding and the pel=2 Wiener
// sub-pel planes, all in integer arithmetic so the result can be bit-exact against the CPU code
// (MVPlane.cpp, Interpolation.cpp RB2Cubic* / *Wiener, Padding.cpp). 16-bit container samples only.
//
// Plane layout everywhere (host and device): packed rows, ew = w + 2*hpad samples per row, eh = h + 2*vpad rows,
// i.e. the padded plane without the pitch gap. This is also the layout of the golden-data plane dumps.

#include "mvcuda_internal.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
      fprintf(stderr, "mvcuda: %s failed: %s\n", #call, cudaGetErrorString(err_)); \
      rc = -2; \
      goto cleanup; \
    } \
  } while (0)

namespace MVC_NS
{
  using namespace ::mvc;
  typedef mvc_pix pix;

  __device__ __forceinline__ int at(const pix *plane, const MvCudaPlaneDims &d, int x, int y)
  {
    return plane[(size_t)(y + d.vpad) * d.ew + (x + d.hpad)];
  }

  // pitch in samples
  __global__ void k_copy_in(pix *dst, MvCudaPlaneDims d, const pix *src, int src_pitch)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= d.w || y >= d.h)
      return;
    dst[(size_t)(y + d.vpad) * d.ew + (x + d.hpad)] = src[(size_t)y * src_pitch + x];
  }

  // Padding::PadReferenceFrame: edge replication, corners take the corner sample
  __global__ void k_pad(pix *plane, MvCudaPlaneDims d)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= d.ew || y >= d.eh)
      return;
    const int ix = x - d.hpad, iy = y - d.vpad;
    if (ix >= 0 && ix < d.w && iy >= 0 && iy < d.h)
      return;
    const int cx = min(max(ix, 0), d.w - 1), cy = min(max(iy, 0), d.h - 1);
    plane[(size_t)y * d.ew + x] = plane[(size_t)(cy + d.vpad) * d.ew + (cx + d.hpad)];
  }

  // RB2CubicVertical (cubic = 1) / RB2BilinearFilteredVertical (cubic = 0): dst rows nh, width 2*nw (the horizontal
  // pass halves it). The cubic filter has 6 taps 1 5 10 10 5 1 / 32 on rows 2y-2..2y+3, the bilinear one 4 taps
  // 1 3 3 1 / 8 on rows 2y-1..2y+2; the first and last row are a plain average of two rows in both.
  __global__ void k_reduce_v(pix *tmp, const pix *src, MvCudaPlaneDims sd, int nw, int nh, int cubic)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= 2 * nw || y >= nh)
      return;
    int v;
    if (y == 0 || y == nh - 1)
      v = (at(src, sd, x, 2 * y) + at(src, sd, x, 2 * y + 1) + 1) >> 1;
    else if (!cubic)
      v = (at(src, sd, x, 2 * y - 1) + at(src, sd, x, 2 * y) * 3 + at(src, sd, x, 2 * y + 1) * 3
        + at(src, sd, x, 2 * y + 2) + 4) >> 3;
    else
      v = (at(src, sd, x, 2 * y - 2) + at(src, sd, x, 2 * y - 1) * 5 + at(src, sd, x, 2 * y) * 10
        + at(src, sd, x, 2 * y + 1) * 10 + at(src, sd, x, 2 * y + 2) * 5 + at(src, sd, x, 2 * y + 3) + 16) >> 5;
    tmp[(size_t)y * 2 * nw + x] = (pix)v;
  }

  // RB2CubicHorizontalInplace / RB2BilinearFilteredHorizontalInplace, out of place
  __global__ void k_reduce_h(pix *dst, MvCudaPlaneDims dd, const pix *tmp, int nw, int nh, int cubic)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nw || y >= nh)
      return;
    const pix *r = tmp + (size_t)y * 2 * nw;
    int v;
    if (x == 0 || x == nw - 1)
      v = (r[2 * x] + r[2 * x + 1] + 1) >> 1;
    else if (!cubic)
      v = (r[2 * x - 1] + r[2 * x] * 3 + r[2 * x + 1] * 3 + r[2 * x + 2] + 4) >> 3;
    else
      v = (r[2 * x - 2] + r[2 * x - 1] * 5 + r[2 * x] * 10 + r[2 * x + 1] * 10 + r[2 * x + 2] * 5 + r[2 * x + 3] + 16) >> 5;
    dst[(size_t)(y + dd.vpad) * dd.ew + (x + dd.hpad)] = (pix)v;
  }

  __device__ __forceinline__ int wiener6(int a, int b, int c, int d, int e, int f, int maxv)
  {
    const int v = (a + (-b + (c << 2) + (d << 2) - e) * 5 + f + 16) >> 5;
    return min(maxv, max(0, v));
  }

  // HorizontalWiener over the whole extended plane (W = ew, H = eh)
  __global__ void k_wiener_h(pix *dst, const pix *src, int W, int H, int maxv)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
      return;
    const pix *s = src + (size_t)y * W;
    int v;
    if (x < 2)
      v = (s[x] + s[x + 1] + 1) >> 1;
    else if (x < W - 4)
      v = wiener6(s[x - 2], s[x - 1], s[x], s[x + 1], s[x + 2], s[x + 3], maxv);
    else if (x < W - 1)
      v = (s[x] + s[x + 1] + 1) >> 1;
    else
      v = s[x];
    dst[(size_t)y * W + x] = (pix)v;
  }

  // VerticalWiener over the whole extended plane
  __global__ void k_wiener_v(pix *dst, const pix *src, int W, int H, int maxv)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
      return;
    const pix *s = src + (size_t)y * W + x;
    int v;
    if (y < 2)
      v = (s[0] + s[W] + 1) >> 1;
    else if (y < H - 4)
      v = wiener6(s[-2 * W], s[-W], s[0], s[W], s[2 * W], s[3 * W], maxv);
    else if (y < H - 1)
      v = (s[0] + s[W] + 1) >> 1;
    else
      v = s[0];
    dst[(size_t)y * W + x] = (pix)v;
  }

  dim3 grid_for(int w, int h)
  {
    return dim3((w + 31) / 32, (h + 7) / 8);
  }
  const dim3 BLOCK(32, 8);

  int super_build(const MvCudaSuperParams &p, const void *const src[3], const int src_pitch_bytes[3], DevSuper &out,
    cudaStream_t st)
  {
    int rc = 0;
    out.p = p;
    out.esize = (int)sizeof(pix);
    // supported: pel 1 or 2, Wiener sharp, cubic reduction (rfilter 4); bit depth matches the sample type
    if ((sizeof(pix) == 1 ? (p.bits_per_pixel != 8) : (p.bits_per_pixel < 9 || p.bits_per_pixel > 16))
      || (p.pel != 1 && p.pel != 2) || p.levels < 1 || p.levels > MVCUDA_MAX_LEVELS || p.rfilter != 4 || p.sharp != 2)
      return -4;
    if (!use_device())
      return -1;

    const int maxv = (1 << p.bits_per_pixel) - 1;
    const int nplanes = p.chroma ? 3 : 1;
    pix *d_src = nullptr, *d_tmp = nullptr;
    size_t src_bytes = 0, tmp_bytes = 0;

    for (int plane = 0; plane < nplanes; plane++)
    {
      MvCudaPlaneDims d0;
      size_t tmp_elems = 0;
      plane_dims(p, 0, plane, d0);
      for (int l = 0; l < p.levels; l++)
      {
        MvCudaPlaneDims d;
        plane_dims(p, l, plane, d);
        if (d.w < 1 || d.h < 1)
        {
          rc = -4;
          goto cleanup;
        }
        if (l > 0)
        {
          MvCudaPlaneDims s;
          plane_dims(p, l - 1, plane, s);
          // the CPU reads one row/column beyond an unpadded level-0 plane for odd sizes: not reproducible, refuse
          if (l == 1 && (2 * d.w > s.w || 2 * d.h > s.h))
          {
            rc = -4;
            goto cleanup;
          }
          tmp_elems = std::max(tmp_elems, (size_t)2 * d.w * d.h);
        }
      }

      src_bytes = (size_t)d0.w * d0.h * sizeof(pix);
      tmp_bytes = tmp_elems * sizeof(pix);
      d_src = (pix *)dev_alloc(src_bytes);
      if (!d_src)
      {
        rc = -2;
        goto cleanup;
      }
      if (tmp_elems)
      {
        d_tmp = (pix *)dev_alloc(tmp_bytes);
        if (!d_tmp)
        {
          rc = -2;
          goto cleanup;
        }
      }
      // every element of every plane is written by the kernels below (inner area, padding, interpolation over the
      // whole extended plane), so the buffers need no clearing
      for (int l = 0; l < p.levels; l++)
      {
        DevPlane &dp = out.pl[l * 3 + plane];
        plane_dims(p, l, plane, dp.d);
        dp.nsub = (l == 0) ? p.pel * p.pel : 1;
        for (int s = 0; s < dp.nsub; s++)
        {
          dp.sub[s] = dev_alloc((size_t)dp.d.ew * dp.d.eh * sizeof(pix));
          if (!dp.sub[s])
          {
            rc = -2;
            goto cleanup;
          }
        }
      }

      // level 0: raw plane in (pitch in samples), then the pyramid; level l+1 is padded as soon as it is reduced,
      // level 0 only after the reduction (it is read unpadded, like on the CPU)
      CUDA_CHECK(cudaMemcpy2DAsync(d_src, d0.w * sizeof(pix), src[plane], src_pitch_bytes[plane], d0.w * sizeof(pix), d0.h,
        cudaMemcpyHostToDevice, st));
      k_copy_in<<<grid_for(d0.w, d0.h), BLOCK, 0, st>>>((pix *)out.pl[plane].sub[0], d0, d_src, d0.w);
      for (int l = 0; l + 1 < p.levels; l++)
      {
        const MvCudaPlaneDims &s = out.pl[l * 3 + plane].d, &d = out.pl[(l + 1) * 3 + plane].d;
        pix *cur = (pix *)out.pl[l * 3 + plane].sub[0], *next = (pix *)out.pl[(l + 1) * 3 + plane].sub[0];
        // CPU quirk: MVGroupOfFrames::set_interp only configures level 0, so only the first reduction uses rfilter
        // (cubic); every deeper level keeps the MVPlane constructor default RB2BilinearFiltered
        const int cubic = (l == 0 && p.rfilter == 4) ? 1 : 0;
        k_reduce_v<<<grid_for(2 * d.w, d.h), BLOCK, 0, st>>>(d_tmp, cur, s, d.w, d.h, cubic);
        k_reduce_h<<<grid_for(d.w, d.h), BLOCK, 0, st>>>(next, d, d_tmp, d.w, d.h, cubic);
        k_pad<<<grid_for(d.ew, d.eh), BLOCK, 0, st>>>(next, d);
      }
      k_pad<<<grid_for(d0.ew, d0.eh), BLOCK, 0, st>>>((pix *)out.pl[plane].sub[0], d0);

      if (p.pel == 2)
      {
        pix *sub[4] = { (pix *)out.pl[plane].sub[0], (pix *)out.pl[plane].sub[1], (pix *)out.pl[plane].sub[2],
          (pix *)out.pl[plane].sub[3] };
        const dim3 g = grid_for(d0.ew, d0.eh);
        k_wiener_h<<<g, BLOCK, 0, st>>>(sub[1], sub[0], d0.ew, d0.eh, maxv);
        k_wiener_v<<<g, BLOCK, 0, st>>>(sub[2], sub[0], d0.ew, d0.eh, maxv);
        k_wiener_h<<<g, BLOCK, 0, st>>>(sub[3], sub[2], d0.ew, d0.eh, maxv);
      }
      CUDA_CHECK(cudaGetLastError());
      // d_src / d_tmp are reused by the next plane: wait until this plane's kernels are done with them
      CUDA_CHECK(cudaStreamSynchronize(st));
      dev_free(d_src, src_bytes);
      dev_free(d_tmp, tmp_bytes);
      d_src = d_tmp = nullptr;
    }
    return 0;

  cleanup:
    cudaStreamSynchronize(st);
    if (d_src)
      dev_free(d_src, src_bytes);
    if (d_tmp)
      dev_free(d_tmp, tmp_bytes);
    // release what was allocated so far
    for (DevPlane &dp : out.pl)
      for (void *&b : dp.sub)
        if (b)
        {
          dev_free(b, (size_t)dp.d.ew * dp.d.eh * sizeof(pix));
          b = nullptr;
        }
    return rc;
  }
}

// host-memory test interface of the super builder: copies every plane back
extern "C" int MVC_API(mvcuda_super_host)(const MvCudaSuperParams *p, const void *const src[3], const int src_pitch_bytes[3],
  void *const out[])
{
  if (!p || !src || !src_pitch_bytes || !out)
    return -4;
  mvc::StreamLease lease;
  mvc::DevSuper s;
  int rc = MVC_NS::super_build(*p, src, src_pitch_bytes, s, lease.s);
  if (rc)
    return rc;
  for (int l = 0; l < p->levels && !rc; l++)
    for (int plane = 0; plane < (p->chroma ? 3 : 1) && !rc; plane++)
    {
      const mvc::DevPlane &dp = s.pl[l * 3 + plane];
      for (int sub = 0; sub < dp.nsub; sub++)
      {
        void *dst = out[(l * 3 + plane) * MVCUDA_MAX_SUB + sub];
        if (dst && cudaMemcpy(dst, dp.sub[sub], (size_t)dp.d.ew * dp.d.eh * sizeof(mvc_pix), cudaMemcpyDeviceToHost) != cudaSuccess)
          rc = -2;
      }
    }
  return rc;
}
