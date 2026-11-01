// mvtools_vulkan: nvof (port of Sources/cuda/mvcuda_nvof.cu). STUB: every entry point reports "not supported", so the
// callers use their CPU code. Replace with the Vulkan implementation.

#include "vk_internal.h"

extern "C" int mvcuda_nvof_token(const MvCudaSuperParams *sp, const MvCudaNvofParams *np, unsigned long long src_token, unsigned long long ref_token, int *payload, int payload_words)
{
  return MVCUDA_NEEDS_CPU;
}

extern "C" int mvcuda_nvof_available(void)
{
  return 0;
}
