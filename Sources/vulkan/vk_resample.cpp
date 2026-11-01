// mvtools_vulkan: resample (port of Sources/cuda/mvcuda_resample.cu). STUB: every entry point reports "not supported", so the
// callers use their CPU code. Replace with the Vulkan implementation.

#include "vk_internal.h"

extern "C" unsigned long long mvcuda_resample_program_create(int target_size, int taps, const int *offset, const short *coeff)
{
  return 0;
}

extern "C" void mvcuda_resample_program_free(unsigned long long program)
{

}

extern "C" int mvcuda_resample_v(const unsigned long long prog[3], unsigned long long src_token, const int row_start[3], const int row_step[3], const MvCudaFrameFormat *out_fmt, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  return -4;
}

extern "C" int mvcuda_resample_2d(const unsigned long long prog_h[3], const unsigned long long prog_v[3], int h_first, unsigned long long src_token, const MvCudaFrameFormat *out_fmt, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  return -4;
}
