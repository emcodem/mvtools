// mvtools_vulkan: comp (port of Sources/cuda/mvcuda_comp.cu). STUB: every entry point reports "not supported", so the
// callers use their CPU code. Replace with the Vulkan implementation.

#include "vk_internal.h"

extern "C" int mvcuda_compensate(const MvCudaBlockGeom *g, unsigned long long src, unsigned long long ref, unsigned long long fill, const int *blocks, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  return -4;
}

extern "C" int mvcuda_degrain(const MvCudaBlockGeom *g, int level, unsigned long long src, const unsigned long long *ref_b, const unsigned long long *ref_f, const int *blocks, int plane_mask, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  return -4;
}
