// Internal types shared by the .cu files of mvtools_cuda.dll (not part of the C API).
#ifndef MVCUDA_INTERNAL_H
#define MVCUDA_INTERNAL_H

#include "mvcuda.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>

namespace mvc
{
  typedef uint16_t pix;

  // ---- pooled device memory and streams (mvcuda_core.cu) ---------------------------------------------------------
  // cudaMalloc / cudaFree synchronise the whole device and cost milliseconds; frame-sized buffers repeat every call,
  // so they are kept in a pool by exact size and reused. Thread safe.
  void *dev_alloc(size_t bytes);              // nullptr on failure
  void dev_free(void *p, size_t bytes);
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
  // one padded plane (packed rows, ew x eh samples); level 0 has pel*pel sub-pel planes, other levels one
  struct DevPlane
  {
    MvCudaPlaneDims d = {};
    int nsub = 0;
    pix *sub[MVCUDA_MAX_SUB] = {};
  };

  // super frame on the device, index [level * 3 + plane]; the buffers go back to the pool on destruction
  struct DevSuper
  {
    MvCudaSuperParams p = {};
    DevPlane pl[MVCUDA_MAX_LEVELS * 3];
    DevSuper() = default;
    DevSuper(const DevSuper &) = delete;
    DevSuper &operator=(const DevSuper &) = delete;
    ~DevSuper();
  };

  void plane_dims(const MvCudaSuperParams &p, int level, int plane, MvCudaPlaneDims &d);

  // builds the super frame from the raw planes (host memory) on stream s and waits for it; 0 on success
  int super_build(const MvCudaSuperParams &p, const void *const src[3], const int src_pitch_bytes[3], DevSuper &out,
    cudaStream_t s);

  // Same, through the frame cache: key != 0 identifies the frame content (caller guarantees: same key and same
  // parameters = same pixels); a hit costs nothing, concurrent requests for one key build it once.
  int super_get(const MvCudaSuperParams &p, unsigned long long key, const void *const src[3], const int src_pitch_bytes[3],
    cudaStream_t s, std::shared_ptr<DevSuper> &out);
  void cache_clear();
}

#endif
