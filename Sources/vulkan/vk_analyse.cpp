// mvtools_vulkan: analyse (port of Sources/cuda/mvcuda_analyse.cu). STUB: every entry point reports "not supported", so the
// callers use their CPU code. Replace with the Vulkan implementation.

#include "vk_internal.h"

extern "C" int mvcuda_analyse_host(const MvCudaSuperParams *sp, const MvCudaAnalyseParams *ap, const void *const src[3], const int src_pitch_bytes[3], const void *const ref[3], const int ref_pitch_bytes[3], int *payload, int payload_words)
{
  return -4;
}

extern "C" int mvcuda_analyse_token(const MvCudaSuperParams *sp, const MvCudaAnalyseParams *ap, unsigned long long src_token, unsigned long long ref_token, int *payload, int payload_words)
{
  return -4;
}
