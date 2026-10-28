// Internal types shared by the .cu files of mvtools_cuda.dll (not part of the C API).
//
// The kernel files (mvcuda_super.cu, mvcuda_recalc.cu, mvcuda_analyse.cu) are compiled twice, through the wrappers
// *_8.cu / *_16.cu: MVC_PIX8 selects 8-bit samples (uint8_t), otherwise 16-bit samples (uint16_t, 9..16 bit data).
// pix, MVC_NS and MVC_API() are the per-type names; everything in namespace mvc is type independent.
#ifndef MVCUDA_INTERNAL_H
#define MVCUDA_INTERNAL_H

#include "mvcuda.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>

#ifdef MVC_PIX8
  typedef uint8_t mvc_pix;
  #define MVC_NS mvc8
  #define MVC_API(name) name##_8
#else
  typedef uint16_t mvc_pix;
  #define MVC_NS mvc16
  #define MVC_API(name) name##_16
#endif

namespace mvc
{
  // ---- device selection (mvcuda_core.cu) -------------------------------------------------------------------------
  // The device used: MVCUDA_DEVICE if set, else the first with compute capability >= 8.0. Every public entry point
  // calls use_device() first (cudaSetDevice is per thread); it returns false when there is no usable device.
  int usable_device_count();
  bool use_device();

  // ---- pooled device memory and streams (mvcuda_core.cu) ---------------------------------------------------------
  // cudaMalloc / cudaFree synchronise the whole device and cost milliseconds; frame-sized buffers repeat every call,
  // so they are kept in a pool by exact size and reused. Thread safe.
  void *dev_alloc(size_t bytes);              // nullptr on failure
  void dev_free(void *p, size_t bytes);
  // page-locked host staging buffers (pageable cudaMemcpy runs at about 3 GB/s and blocks the thread; a plain memcpy into
  // pinned memory plus a DMA is several times faster); nullptr when none can be had, the caller then copies directly
  void *pin_alloc(size_t bytes);
  void pin_free(void *p, size_t bytes);
  void pool_trim();                           // returns all pooled (free) buffers to the driver

  // one stream per running call, from a pool (non-blocking streams: concurrent calls overlap on the GPU)
  struct StreamLease
  {
    cudaStream_t s;
    StreamLease();
    ~StreamLease();
    StreamLease(const StreamLease &) = delete;
    StreamLease &operator=(const StreamLease &) = delete;
  };

  // ---- super frame on the device ---------------------------------------------------------------------------------
  // one padded plane (packed rows, ew x eh samples of esize bytes); level 0 has pel*pel sub-pel planes, other levels one
  struct DevPlane
  {
    MvCudaPlaneDims d = {};
    int nsub = 0;
    void *sub[MVCUDA_MAX_SUB] = {};
  };

  // super frame on the device, index [level * 3 + plane]; the buffers go back to the pool on destruction
  struct DevSuper
  {
    MvCudaSuperParams p = {};
    int esize = 2;   // bytes per sample
    DevPlane pl[MVCUDA_MAX_LEVELS * 3];
    DevSuper() = default;
    DevSuper(const DevSuper &) = delete;
    DevSuper &operator=(const DevSuper &) = delete;
    ~DevSuper();
  };

  // geometry of one plane of one level (same formulas as MVSuper.h PlaneWidthLuma / PlaneHeightLuma)
  inline int level_size(int size, int level, int ratio, int pad)
  {
    for (int i = 1; i <= level; i++)
      size = pad >= ratio ? ((size / ratio + 1) / 2) * ratio : ((size / ratio) / 2) * ratio;
    return size;
  }

  inline void plane_dims(const MvCudaSuperParams &p, int level, int plane, MvCudaPlaneDims &d)
  {
    const int xr = plane == 0 ? 1 : p.x_ratio_uv;
    const int yr = plane == 0 ? 1 : p.y_ratio_uv;
    d.w = level_size(p.width, level, p.x_ratio_uv, p.hpad) / xr;
    d.h = level_size(p.height, level, p.y_ratio_uv, p.vpad) / yr;
    d.hpad = p.hpad / xr;
    d.vpad = p.vpad / yr;
    d.ew = d.w + 2 * d.hpad;
    d.eh = d.h + 2 * d.vpad;
  }

  // builds the super frame from the raw planes (host memory) on stream s and waits for it; 0 on success
  typedef int (*SuperBuildFn)(const MvCudaSuperParams &p, const void *const src[3], const int src_pitch_bytes[3],
    DevSuper &out, cudaStream_t s);

}

namespace MVC_NS
{
  // the per-type builder (mvcuda_super.cu), what mvc::super_get calls on a cache miss
  int super_build(const MvCudaSuperParams &p, const void *const src[3], const int src_pitch_bytes[3], mvc::DevSuper &out,
    cudaStream_t s);
}

namespace mvc
{
  // Builds through the frame cache: key != 0 identifies the frame content (caller guarantees: same key and same
  // parameters = same pixels); a hit costs nothing, concurrent requests for one key build it once.
  int super_get(SuperBuildFn build, const MvCudaSuperParams &p, unsigned long long key, const void *const src[3],
    const int src_pitch_bytes[3], cudaStream_t s, std::shared_ptr<DevSuper> &out);
  void cache_clear();
}

#endif
