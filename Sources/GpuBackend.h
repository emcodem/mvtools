// Optional GPU backend: loads mvtools_cuda.dll (next to mvtools2.dll) at run time. No CUDA types or link
// dependencies in this plugin; if the library, a device or a matching API version is missing, usable() is false and the
// filters run their CPU code. See GPU_PLAN.md.

#ifndef __MV_GPU_BACKEND__
#define __MV_GPU_BACKEND__

#include "cuda/mvcuda.h"

#include <chrono>

// MVTOOLS_GPU_PROFILE=1: wall time per named phase (sum, count), printed to stderr when the process ends
void gpu_prof_add(const char *name, double ms);
bool gpu_prof_enabled();

class GpuScope
{
public:
  explicit GpuScope(const char *name)
    : _name(name), _on(gpu_prof_enabled()), _t(_on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point())
  {
  }
  ~GpuScope()
  {
    if (_on)
      gpu_prof_add(_name, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _t).count());
  }
  GpuScope(const GpuScope &) = delete;
  GpuScope &operator=(const GpuScope &) = delete;

private:
  const char *_name;
  bool _on;
  std::chrono::steady_clock::time_point _t;
};

class GpuBackend
{
public:
  static GpuBackend &instance();

  bool usable() const { return _usable; }

  int (*analyse_host)(const MvCudaSuperParams *, const MvCudaAnalyseParams *, const void *const[3], const int[3],
    const void *const[3], const int[3], int *, int);
  int (*recalc_host)(const MvCudaSuperParams *, const MvCudaRecalcParams *, const void *const[3], const int[3],
    const void *const[3], const int[3], const int *, int *);
  int (*flow_host)(const MvCudaSuperParams *, const MvCudaFlowParams *, const void *const[3], const int[3],
    const void *const[3], const int[3], const int *, const int *, const int *, const int *, void *const[3], const int[3],
    void *const[]);
  void (*cache_clear)();

  // number of block-row slices to use: the CPU thread count when mt is on (so the result equals the CPU result with
  // that many threads), 1 without mt; 24 when mt is on but no thread pool is available (the GPU needs slices)
  static int slice_threads(bool mt_flag);

  // MVTOOLS_GPU_LOG=1 prints every GPU call / fallback to stderr
  static bool log_enabled();
  static bool force_fallback(); // MVTOOLS_GPU_FORCE_FALLBACK: every GPU call reports "use the CPU" (test of the fallback paths)

  // identifier of the content of frame n of the clip c (same clip object and frame number = same pixels), for the
  // library's super frame cache
  static unsigned long long frame_key(const void *clip, int n);

  // filters that use the GPU register here; when the last one is gone the library's frame cache is dropped
  void add_user();
  void remove_user();

private:
  GpuBackend();
  bool _usable;
  long _users;
};

#endif
