// GPU version of MSuper (GPU_PLAN.md M2): pyramid reduction (rfilter=4), edge padding and the pel=2 Wiener
// sub-pel planes, all in integer arithmetic so the result can be bit-exact against the CPU code
// (MVPlane.cpp, Interpolation.cpp RB2Cubic* / *Wiener, Padding.cpp). 16-bit container samples only.
//
// Plane layout everywhere (host and device): packed rows, ew = w + 2*hpad samples per row, eh = h + 2*vpad rows,
// i.e. the padded plane without the pitch gap. This is also the layout of the golden-data plane dumps.

#include "mvcuda_internal.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
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

  // Level 0 with its padding in one pass: every sample of the extended plane takes the source sample at the clamped
  // inner position (PadReferenceFrame replicates the edge samples; the reduction below reads only the inner area).
  __global__ void k_level0(pix *dst, MvCudaPlaneDims d, const pix *src, int src_pitch)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= d.ew || y >= d.eh)
      return;
    const int cx = min(max(x - d.hpad, 0), d.w - 1), cy = min(max(y - d.vpad, 0), d.h - 1);
    dst[(size_t)y * d.ew + x] = src[(size_t)cy * src_pitch + cx];
  }

  // the vertical pass of k_reduce_v for one sample (rounded to the sample type, as the CPU's temporary rows)
  __device__ __forceinline__ int reduce_v_at(const pix *src, const MvCudaPlaneDims &sd, int x, int y, int nh, int cubic)
  {
    if (y == 0 || y == nh - 1)
      return (at(src, sd, x, 2 * y) + at(src, sd, x, 2 * y + 1) + 1) >> 1;
    if (!cubic)
      return (at(src, sd, x, 2 * y - 1) + at(src, sd, x, 2 * y) * 3 + at(src, sd, x, 2 * y + 1) * 3 + at(src, sd, x, 2 * y + 2) + 4) >> 3;
    return (at(src, sd, x, 2 * y - 2) + at(src, sd, x, 2 * y - 1) * 5 + at(src, sd, x, 2 * y) * 10 + at(src, sd, x, 2 * y + 1) * 10
      + at(src, sd, x, 2 * y + 2) * 5 + at(src, sd, x, 2 * y + 3) + 16) >> 5;
  }

  // One reduction level with its padding in one pass (k_reduce_v + k_reduce_h + k_pad): each sample of the extended
  // plane of level l + 1 computes the reduced sample at its clamped inner position from level l's inner area.
  __global__ void k_level(pix *dst, MvCudaPlaneDims dd, const pix *src, MvCudaPlaneDims sd, int cubic)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dd.ew || y >= dd.eh)
      return;
    const int nw = dd.w, nh = dd.h;
    const int cx = min(max(x - dd.hpad, 0), nw - 1), cy = min(max(y - dd.vpad, 0), nh - 1);
#define RV(c) reduce_v_at(src, sd, (c), cy, nh, cubic)
    int v;
    if (cx == 0 || cx == nw - 1)
      v = (RV(2 * cx) + RV(2 * cx + 1) + 1) >> 1;
    else if (!cubic)
      v = (RV(2 * cx - 1) + RV(2 * cx) * 3 + RV(2 * cx + 1) * 3 + RV(2 * cx + 2) + 4) >> 3;
    else
      v = (RV(2 * cx - 2) + RV(2 * cx - 1) * 5 + RV(2 * cx) * 10 + RV(2 * cx + 1) * 10 + RV(2 * cx + 2) * 5 + RV(2 * cx + 3) + 16) >> 5;
#undef RV
    dst[(size_t)y * dd.ew + x] = (pix)v;
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

  // src_device: src[] are device pointers (planes of a device frame, pitch in bytes), already ordered before st by the
  // caller; otherwise host planes that are uploaded through pinned staging
  static int super_build_impl(const MvCudaSuperParams &p, const void *const src[3], const int src_pitch_bytes[3], bool src_device,
    DevSuper &out, cudaStream_t st)
  {
    int rc = 0;
    out.p = p;
    out.esize = (int)sizeof(pix);
    // supported: pel 1 or 2, Wiener sharp, cubic reduction (rfilter 4); bit depth matches the sample type
    if ((sizeof(pix) == 1 ? (p.bits_per_pixel != 8) : (p.bits_per_pixel < 9 || p.bits_per_pixel > 16))
      || (p.pel != 1 && p.pel != 2) || p.levels < 1 || p.levels > MVCUDA_MAX_LEVELS || (p.rfilter != 4 && p.rfilter != 2) || p.sharp != 2)
      return -4;
    if (!use_device())
      return -1;

    const int maxv = (1 << p.bits_per_pixel) - 1;
    const int nplanes = p.chroma ? 3 : 1;
    pix *d_src = nullptr, *d_tmp = nullptr;
    void *pin = nullptr;
    const bool timing = std::getenv("MVCUDA_TIMING") != nullptr;
    double t_alloc = 0, t_copy = 0, t_sync = 0;
    auto now = [] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    const double t_start = now();
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
      double t0 = now();
      if (!src_device)
      {
        d_src = (pix *)dev_alloc(src_bytes);
        if (!d_src)
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

      // level 0 with its padding, then each pyramid level with its padding (one kernel per level, k_level0 / k_level:
      // the reduction reads only the inner area of the level above, so padding it first changes nothing)
      t_alloc += now() - t0;
      t0 = now();
      if (src_device)
      {
        // a device frame (token): straight into the padded level-0 plane
        k_level0<<<grid_for(d0.ew, d0.eh), BLOCK, 0, st>>>((pix *)out.pl[plane].sub[0], d0, (const pix *)src[plane],
          src_pitch_bytes[plane] / (int)sizeof(pix));
      }
      else
      {
        pin = pin_alloc(src_bytes);
        if (pin)
        {
          // rows into the pinned buffer (plain memcpy), then one DMA; the buffer goes back once this plane has synchronised
          for (int y = 0; y < d0.h; y++)
            std::memcpy((char *)pin + (size_t)y * d0.w * sizeof(pix), (const char *)src[plane] + (size_t)y * src_pitch_bytes[plane], d0.w * sizeof(pix));
          CUDA_CHECK(cudaMemcpyAsync(d_src, pin, src_bytes, cudaMemcpyHostToDevice, st));
        }
        else
          CUDA_CHECK(cudaMemcpy2DAsync(d_src, d0.w * sizeof(pix), src[plane], src_pitch_bytes[plane], d0.w * sizeof(pix), d0.h,
            cudaMemcpyHostToDevice, st));
        k_level0<<<grid_for(d0.ew, d0.eh), BLOCK, 0, st>>>((pix *)out.pl[plane].sub[0], d0, d_src, d0.w);
      }
      t_copy += now() - t0;
      for (int l = 0; l + 1 < p.levels; l++)
      {
        const MvCudaPlaneDims &s = out.pl[l * 3 + plane].d, &d = out.pl[(l + 1) * 3 + plane].d;
        pix *cur = (pix *)out.pl[l * 3 + plane].sub[0], *next = (pix *)out.pl[(l + 1) * 3 + plane].sub[0];
        // CPU quirk: MVGroupOfFrames::set_interp only configures level 0, so only the first reduction uses rfilter
        // (cubic); every deeper level keeps the MVPlane constructor default RB2BilinearFiltered
        const int cubic = (l == 0 && p.rfilter == 4) ? 1 : 0;
        k_level<<<grid_for(d.ew, d.eh), BLOCK, 0, st>>>(next, d, cur, s, cubic);
      }

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
      if (!src_device)
      {
        // the pinned staging buffer and d_src go back to their pools (other threads' streams): wait for this plane
        t0 = now();
        CUDA_CHECK(cudaStreamSynchronize(st));
        t_sync += now() - t0;
        pin_free(pin, src_bytes);
        pin = nullptr;
        dev_free(d_src, src_bytes);
        d_src = nullptr;
      }
    }
    // from a device frame nothing was waited for: readers on other streams wait for this event (super_get)
    if (src_device)
    {
      out.ready = event_get();
      if (!out.ready || cudaEventRecord(out.ready, st) != cudaSuccess)
      {
        rc = -2;
        goto cleanup;
      }
    }
    if (timing)
      std::fprintf(stderr, "mvcuda super_build: total %.2f ms (alloc %.2f, host copy+pin %.2f, sync %.2f)\n", now() - t_start, t_alloc, t_copy, t_sync);
    return 0;

  cleanup:
    cudaStreamSynchronize(st);
    if (pin)
      pin_free(pin, src_bytes);
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

  int super_build(const MvCudaSuperParams &p, const void *const src[3], const int src_pitch_bytes[3], DevSuper &out,
    cudaStream_t st)
  {
    return super_build_impl(p, src, src_pitch_bytes, false, out, st);
  }

  int super_build_dev(const MvCudaSuperParams &p, const void *const src[3], const int src_pitch_bytes[3], DevSuper &out,
    cudaStream_t st)
  {
    return super_build_impl(p, src, src_pitch_bytes, true, out, st);
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
