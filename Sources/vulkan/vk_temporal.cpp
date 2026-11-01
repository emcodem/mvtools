// mvtools_vulkan: temporal (port of Sources/cuda/mvcuda_temporal.cu). STUB: every entry point reports "not supported", so the
// callers use their CPU code. Replace with the Vulkan implementation.

#include "vk_internal.h"

extern "C" int mvcuda_temporal_soften(const unsigned long long *tokens, int radius, int luma_thresh, int chroma_thresh, int scenechange, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  return -4;
}

extern "C" int mvcuda_temporal_binomial(const unsigned long long tokens[5], int luma_thresh, int chroma_thresh, int scenechange, int w1, int inv1, int w2, int inv2, int tail_y, int merge_chroma, int tail_c, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  return -4;
}
