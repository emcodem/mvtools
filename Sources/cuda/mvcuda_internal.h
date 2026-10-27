// Internal types shared by the .cu files of mvtools_cuda.dll (not part of the C API).
#ifndef MVCUDA_INTERNAL_H
#define MVCUDA_INTERNAL_H

#include "mvcuda.h"

#include <cstdint>

namespace mvc
{
  typedef uint16_t pix;

  // one padded plane on the device (packed rows, ew x eh samples); level 0 has pel*pel sub-pel planes, other levels one
  struct DevPlane
  {
    MvCudaPlaneDims d = {};
    int nsub = 0;
    pix *sub[MVCUDA_MAX_SUB] = {};
  };

  // super frame on the device, index [level * 3 + plane]
  struct DevSuper
  {
    MvCudaSuperParams p = {};
    DevPlane pl[MVCUDA_MAX_LEVELS * 3];
  };

  void plane_dims(const MvCudaSuperParams &p, int level, int plane, MvCudaPlaneDims &d);

  // 0 on success; allocates device memory in s (freed by super_free, also called on failure)
  int super_build(const MvCudaSuperParams &p, const void *const src[3], const int src_pitch_bytes[3], DevSuper &s);
  void super_free(DevSuper &s);
}

#endif
