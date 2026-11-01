// mvtools_vulkan: super (port of Sources/cuda/mvcuda_super.cu). STUB: every entry point reports "not supported", so the
// callers use their CPU code. Replace with the Vulkan implementation.

#include "vk_internal.h"

extern "C" int mvcuda_super_host(const MvCudaSuperParams *p, const void *const src[3], const int src_pitch_bytes[3], void *const out[])
{
  return -4;
}

extern "C" int mvcuda_super_plane_dims(const MvCudaSuperParams *p, int level, int plane, MvCudaPlaneDims *d)
{
  if (!p || !d || level < 0 || level >= p->levels || level >= MVCUDA_MAX_LEVELS || plane < 0 || plane > 2 || (plane > 0 && !p->chroma))
    return -4;
  mvk::plane_dims(*p, level, plane, *d);
  return 0;
}

namespace mvk
{
  int super_build(const MvCudaSuperParams &, const DevFrame &, DevSuper &, vkrt::Cmd &)
  {
    return -4;
  }
}
