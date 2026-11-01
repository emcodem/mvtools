// mvtools_vulkan: kobs (port of Sources/cuda/mvcuda_kobs.cu). STUB: every entry point reports "not supported", so the
// callers use their CPU code. Replace with the Vulkan implementation.

#include "vk_internal.h"

extern "C" unsigned long long mvcuda_lut2_create(const unsigned char *table)
{
  return 0;
}

extern "C" unsigned long long mvcuda_lut2_create_bits(const void *luma, const void *chroma, int bits)
{
  return 0;
}

extern "C" void mvcuda_lut2_free(unsigned long long lut)
{

}

extern "C" int mvcuda_kobs(unsigned long long input, unsigned long long ref, int rep, int planes_mask, unsigned long long lut1, unsigned long long lut2, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  return -4;
}

extern "C" int mvcuda_removegrain(unsigned long long src, const int mode[3], unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  return -4;
}

extern "C" int mvcuda_mt(const int op[3], unsigned long long a, unsigned long long b, unsigned long long c, unsigned long long lut, int p1, int p2, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  return -4;
}

extern "C" int mvcuda_qtgmc_sharp(const MvCudaQtgmcSharp *p, unsigned long long edi, unsigned long long deg1, unsigned long long deg2, const unsigned long long *refs, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  return -4;
}

extern "C" int mvcuda_qtgmc_backblend(unsigned long long thin, unsigned long long lossed, unsigned long long prog_h, unsigned long long prog_v, int h_first, const unsigned long long *refs, int nref, int sovs, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  return -4;
}
