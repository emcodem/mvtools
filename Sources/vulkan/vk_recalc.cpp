// mvtools_vulkan: recalc (port of Sources/cuda/mvcuda_recalc.cu). STUB: every entry point reports "not supported", so the
// callers use their CPU code. Replace with the Vulkan implementation.

#include "vk_internal.h"

extern "C" int mvcuda_recalc_host(const MvCudaSuperParams *sp, const MvCudaRecalcParams *rp, const void *const src[3], const int src_pitch_bytes[3], const void *const ref[3], const int ref_pitch_bytes[3], const int *old_vectors, int *out_vectors)
{
  return -4;
}

extern "C" int mvcuda_recalc_token(const MvCudaSuperParams *sp, const MvCudaRecalcParams *rp, unsigned long long src_token, unsigned long long ref_token, const int *old_vectors, int *out_vectors)
{
  return -4;
}
