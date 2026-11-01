// mvtools_vulkan: nnedi3 (port of Sources/cuda/mvcuda_nnedi3.cu). STUB: every entry point reports "not supported", so the
// callers use their CPU code. Replace with the Vulkan implementation.

#include "vk_internal.h"

extern "C" void * mvcuda_nnedi3_create(const MvCudaNnediParams *p)
{
  return nullptr;
}

extern "C" void mvcuda_nnedi3_free(void *net)
{

}

extern "C" int mvcuda_nnedi3(void *net, unsigned long long src, int field_n, int plane_mask, const int val_min[3], const int val_max[3], unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  return -4;
}
