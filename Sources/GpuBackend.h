// Optional GPU backend: loads mvtools_cuda.dll (next to mvtools2.dll) at run time. No CUDA types or link
// dependencies in this plugin; if the library, a device or a matching API version is missing, usable() is false and the
// filters run their CPU code. See GPU_PLAN.md.

#ifndef __MV_GPU_BACKEND__
#define __MV_GPU_BACKEND__

#include "cuda/mvcuda.h"

class GpuBackend
{
public:
  static GpuBackend &instance();

  bool usable() const { return _usable; }

  int (*analyse_host)(const MvCudaSuperParams *, const MvCudaAnalyseParams *, const void *const[3], const int[3],
    const void *const[3], const int[3], int *, int);
  int (*recalc_host)(const MvCudaSuperParams *, const MvCudaRecalcParams *, const void *const[3], const int[3],
    const void *const[3], const int[3], const int *, int *);

  // number of block-row slices to use: the CPU thread count when mt is on (so the result equals the CPU result with
  // that many threads), 1 without mt; 24 when mt is on but no thread pool is available (the GPU needs slices)
  static int slice_threads(bool mt_flag);

  // MVTOOLS_GPU_LOG=1 prints every GPU call / fallback to stderr
  static bool log_enabled();

private:
  GpuBackend();
  bool _usable;
};

#endif
