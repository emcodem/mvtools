// Optional GPU backend: loads mvtools_cuda.dll (next to mvtools2.dll) at run time. No CUDA types or link
// dependencies in this plugin; if the library, a device or a matching API version is missing, usable() is false and the
// filters run their CPU code. See GPU_PLAN.md.

#ifndef __MV_GPU_BACKEND__
#define __MV_GPU_BACKEND__

#include "avisynth.h"
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
  int (*analyse_token)(const MvCudaSuperParams *, const MvCudaAnalyseParams *, unsigned long long, unsigned long long, int *, int);
  int (*frame_download)(unsigned long long, void *const[3], const int[3]);
  int (*token_regenerate)(unsigned long long, int, void *, MvCudaTokenHeader *);
  int (*token_pin)(unsigned long long);
  void (*token_unpin)(unsigned long long);
  unsigned long long (*producer_register)(MvCudaRegenFn, void *);
  void (*producer_unregister)(unsigned long long);
  int (*compensate)(const MvCudaBlockGeom *, unsigned long long, unsigned long long, unsigned long long, const int *,
    unsigned long long, int, MvCudaTokenHeader *);
  int (*degrain)(const MvCudaBlockGeom *, int, unsigned long long, const unsigned long long *, const unsigned long long *,
    const int *, int, unsigned long long, int, MvCudaTokenHeader *);

  // A picture that is a token frame (cudafilters output: a MvCudaTokenHeader in its first bytes instead of pixels).
  // p: first byte of the picture's luma plane, row_bytes: its row size.
  static bool read_token(const void *p, int row_bytes, MvCudaTokenHeader &h);

  // Copies the device frame of a token frame into host planes (dst[plane], pitch in bytes); when the token was evicted,
  // its producer makes it again (env: the caller's IScriptEnvironment). False when that fails.
  bool download_token(const MvCudaTokenHeader &h, void *const dst[3], const int dst_pitch[3], void *env);

  // the token of a frame of a lite super clip (MSuper(gpu=true) of a token clip: the header at the picture position)
  static bool lite_token(const PVideoFrame &f, const VideoInfo &super_vi, MvCudaTokenHeader &h);
  // the token of an ordinary token frame (the header at the start of the luma plane)
  static bool frame_token(const PVideoFrame &f, const VideoInfo &vi, MvCudaTokenHeader &h);
  // f itself, or when it is a token frame its picture downloaded into a new frame (for CPU code); throws when the device
  // frame cannot be made again
  PVideoFrame resolve_frame(const PVideoFrame &f, const VideoInfo &vi, IScriptEnvironment *env);
  // writes a token header into the start of the luma plane of a new frame
  static void write_token(PVideoFrame &dst, const MvCudaTokenHeader &h);

  // Runs op(tokens) on the device frames of the given token headers; when op reports MVCUDA_TOKEN_GONE, the tokens still
  // there are pinned, the others made again by their producers (pinned right away), and op runs once more.
  template <class Op>
  int with_tokens(const MvCudaTokenHeader *hs, int count, IScriptEnvironment *env, Op op)
  {
    unsigned long long tk[16];
    bool pinned[16] = {};
    if (count > 16)
      return -4;
    for (int i = 0; i < count; i++)
      tk[i] = hs[i].token;
    int rc = op(tk);
    if (rc != MVCUDA_TOKEN_GONE)
      return rc;
    rc = 0;
    for (int i = 0; i < count && rc == 0; i++)
    {
      bool dup = false; // the same token twice: pinned once is enough
      for (int k = 0; k < i; k++)
        dup = dup || (pinned[k] && tk[k] == hs[i].token);
      if (dup)
        continue;
      if (token_pin(tk[i]) == 0)
      {
        pinned[i] = true;
        continue;
      }
      MvCudaTokenHeader again;
      if (!hs[i].producer || token_regenerate(hs[i].producer, hs[i].frame, env, &again) != 0 || token_pin(again.token) != 0)
        rc = MVCUDA_TOKEN_GONE;
      else
      {
        for (int k = i; k < count; k++) // later entries with the same frame use the new token too
          if (hs[k].token == hs[i].token)
            tk[k] = again.token;
        pinned[i] = true;
      }
    }
    if (rc == 0)
      rc = op(tk);
    for (int i = 0; i < count; i++)
      if (pinned[i])
        token_unpin(tk[i]);
    return rc;
  }

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

  // Search configurations the GPU kernels implement (mvcuda_refine.cuh). type is a SearchType.h value. MAnalyse: p_fine is
  // pelsearch (level 0), p_coarse is searchparam (the coarse levels run exhaustive for every type but horizontal /
  // vertical); MRecalculate has one level, p_fine = p_coarse = searchparam, n_levels = 1.
  static bool search_supported(int type, int p_fine, int p_coarse, int n_levels)
  {
    const bool known = type == 1 || type == 2 || type == 4 || type == 8 || type == 16 || type == 32 || type == 64 || type == 128;
    if (!known || p_fine < 1 || p_coarse < 1)
      return false;
    const bool horv = type == 64 || type == 128;
    if (p_fine > (type == 8 ? 3 : 24))
      return false;
    if (n_levels > 1 && p_coarse > (horv ? 24 : 3))
      return false;
    return true;
  }

private:
  GpuBackend();
  bool _usable;
  long _users;
};

#endif
