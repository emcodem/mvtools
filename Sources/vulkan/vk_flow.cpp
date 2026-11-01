// mvtools_vulkan: flow (port of Sources/cuda/mvcuda_flow.cu). STUB: every entry point reports "not supported", so the
// callers use their CPU code. Replace with the Vulkan implementation.

#include "vk_internal.h"

extern "C" int mvcuda_flow_host(const MvCudaSuperParams *sp, const MvCudaFlowParams *fp, const void *const src_raw[3], const int src_pitch_bytes[3], const void *const ref_raw[3], const int ref_pitch_bytes[3], const int *vec_b, const int *vec_f, const int *vec_bb, const int *vec_ff, void *const dst[3], const int dst_pitch_bytes[3], void *const dbg[MVCUDA_FLOW_DEBUG_SLOTS])
{
  return -4;
}

extern "C" int mvcuda_flow_token(const MvCudaSuperParams *sp, const MvCudaFlowParams *fp, unsigned long long left, unsigned long long right, const int *vec_b, const int *vec_f, const int *vec_bb, const int *vec_ff, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  return -4;
}

extern "C" int mvcuda_flow_token_host(const MvCudaSuperParams *sp, const MvCudaFlowParams *fp, unsigned long long left, unsigned long long right, const int *vec_b, const int *vec_f, const int *vec_bb, const int *vec_ff, void *const dst[3], const int dst_pitch_bytes[3])
{
  return -4;
}
