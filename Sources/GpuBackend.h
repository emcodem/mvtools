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

  void (*cache_clear)();
  int (*analyse_token)(const MvCudaSuperParams *, const MvCudaAnalyseParams *, unsigned long long, unsigned long long, int *, int);
  int (*recalc_token)(const MvCudaSuperParams *, const MvCudaRecalcParams *, unsigned long long, unsigned long long, const int *, int *);
  int (*flow_token)(const MvCudaSuperParams *, const MvCudaFlowParams *, unsigned long long, unsigned long long, const int *,
    const int *, const int *, const int *, unsigned long long, int, MvCudaTokenHeader *);
  int (*flow_token_host)(const MvCudaSuperParams *, const MvCudaFlowParams *, unsigned long long, unsigned long long, const int *,
    const int *, const int *, const int *, void *const[3], const int[3]);
  int (*frame_upload_keyed)(const MvCudaFrameFormat *, const void *const[3], const int[3], unsigned long long, MvCudaTokenHeader *);
  int (*frame_download)(unsigned long long, void *const[3], const int[3]);
  void (*frame_stats)(MvCudaFrameStats *);
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
  // writes a token header into the start of the luma plane of a new frame, and black rows every 8 rows below it (as
  // cudafilters does: a filter that does not understand tokens shows obvious stripes, not a plausible picture)
  static void write_token(PVideoFrame &dst, const MvCudaTokenHeader &h);

  // One input picture of a GPU call (GPU_PIPELINE_PLAN.md section 25, usage rule: token frames in -> token frames out,
  // normal frames in -> normal frames out): a device frame already (token), or a host picture that with_inputs uploads
  // once under a content key (the clip and the frame number), shared by every GPU filter reading that picture.
  struct Input
  {
    MvCudaTokenHeader h = {};    // the token (set by with_inputs for host pictures)
    bool host = false;           // a host picture: planes / pitch / fmt / key below
    const void *planes[3] = {};
    int pitch[3] = {};
    MvCudaFrameFormat fmt = {};
    unsigned long long key = 0;
  };
  // The picture of frame f of the super clip `clip` (MSuper with or without gpu=true: level 0 inside the padding; a token
  // frame's header at the picture position) as the input of frame n. False for formats the device cannot take.
  static bool super_input(const PVideoFrame &f, const VideoInfo &super_vi, const void *clip, int n, Input &in);
  // the same for frame n of an ordinary clip (a token frame's header at the start of the luma plane)
  static bool frame_input(const PVideoFrame &f, const VideoInfo &vi, const void *clip, int n, Input &in);

  // Runs op(tokens) on the device frames of the inputs (host pictures uploaded first, or found on the device under their
  // key). When op reports MVCUDA_TOKEN_GONE, the inputs still there are pinned, the evicted ones made again (token inputs
  // by their producers, host pictures uploaded again; pinned right away, so that making one cannot evict another under a
  // small budget), and op runs once more.
  template <class Op>
  int with_inputs(Input *in, int count, IScriptEnvironment *env, Op op)
  {
    unsigned long long tk[16];
    bool pinned[16] = {};
    if (count > 16)
      return -4;
    for (int i = 0; i < count; i++)
    {
      if (in[i].host)
      {
        const int rc = frame_upload_keyed(&in[i].fmt, in[i].planes, in[i].pitch, in[i].key, &in[i].h);
        if (rc)
          return rc;
      }
      tk[i] = in[i].h.token;
    }
    int rc = op(tk);
    if (rc != MVCUDA_TOKEN_GONE)
      return rc;
    rc = 0;
    // token inputs first: their regeneration may evict uploaded pictures, which are cheap to send again
    for (int pass = 0; pass < 2 && rc == 0; pass++)
      for (int i = 0; i < count && rc == 0; i++)
      {
        if (in[i].host != (pass == 1))
          continue;
        if (token_pin(tk[i]) == 0) // counted: an input given twice is pinned twice
        {
          pinned[i] = true;
          continue;
        }
        MvCudaTokenHeader again;
        const bool ok = in[i].host ? frame_upload_keyed(&in[i].fmt, in[i].planes, in[i].pitch, in[i].key, &again) == 0
                                   : in[i].h.producer && token_regenerate(in[i].h.producer, in[i].h.frame, env, &again) == 0;
        if (!ok || token_pin(again.token) != 0)
        {
          rc = MVCUDA_TOKEN_GONE;
          break;
        }
        const unsigned long long old = tk[i];
        for (int k = i; k < count; k++) // later entries with the same frame use the new token too
          if (tk[k] == old)
            tk[k] = again.token;
        pinned[i] = true;
      }
    if (rc == 0)
      rc = op(tk);
    for (int i = 0; i < count; i++)
      if (pinned[i])
        token_unpin(tk[i]);
    return rc;
  }

  // copies the device frame h (made by this thread a moment ago, so still reachable) into the planes of dst; false on failure
  bool download_frame(const MvCudaTokenHeader &h, PVideoFrame &dst, const VideoInfo &vi);

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
    // the coarse levels search exhaustively (GroupOfPlanes::SearchMVs), except for the horizontal / vertical types;
    // exhaustive radii up to 64 (mvc_refine::max_param), every other type up to 24
    if (p_fine > (type == 8 ? 64 : 24))
      return false;
    if (n_levels > 1 && p_coarse > (horv ? 24 : 64))
      return false;
    return true;
  }

private:
  GpuBackend();
  bool _usable;
  long _users;
};

#endif
