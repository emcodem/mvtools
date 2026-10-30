// Internal types shared by the .cu files of mvtools_cuda.dll (not part of the C API).
//
// The kernel files (mvcuda_super.cu, mvcuda_recalc.cu, mvcuda_analyse.cu) are compiled twice, through the wrappers
// *_8.cu / *_16.cu: MVC_PIX8 selects 8-bit samples (uint8_t), otherwise 16-bit samples (uint16_t, 9..16 bit data).
// pix, MVC_NS and MVC_API() are the per-type names; everything in namespace mvc is type independent.
#ifndef MVCUDA_INTERNAL_H
#define MVCUDA_INTERNAL_H

#include "mvcuda.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#ifdef MVC_PIX8
  typedef uint8_t mvc_pix;
  #define MVC_NS mvc8
  #define MVC_API(name) name##_8
#else
  typedef uint16_t mvc_pix;
  #define MVC_NS mvc16
  #define MVC_API(name) name##_16
#endif

namespace mvc
{
  // ---- device selection (mvcuda_core.cu) -------------------------------------------------------------------------
  // The device used: MVCUDA_DEVICE if set, else the first with compute capability >= 8.0. Every public entry point
  // calls use_device() first (cudaSetDevice is per thread); it returns false when there is no usable device.
  int usable_device_count();
  bool use_device();

  // ---- pooled device memory and streams (mvcuda_core.cu) ---------------------------------------------------------
  // cudaMalloc / cudaFree synchronise the whole device and cost milliseconds; frame-sized buffers repeat every call,
  // so they are kept in a pool by exact size and reused. Thread safe.
  void *dev_alloc(size_t bytes);              // nullptr on failure
  void dev_free(void *p, size_t bytes);
  // page-locked host staging buffers (pageable cudaMemcpy runs at about 3 GB/s and blocks the thread; a plain memcpy into
  // pinned memory plus a DMA is several times faster); nullptr when none can be had, the caller then copies directly
  void *pin_alloc(size_t bytes);
  void pin_free(void *p, size_t bytes);
  void pool_trim();                           // returns all pooled (free) buffers to the driver

  // one stream per running call, from a pool (non-blocking streams: concurrent calls overlap on the GPU)
  struct StreamLease
  {
    cudaStream_t s;
    StreamLease();
    ~StreamLease();
    StreamLease(const StreamLease &) = delete;
    StreamLease &operator=(const StreamLease &) = delete;
  };

  // ---- super frame on the device ---------------------------------------------------------------------------------
  // one padded plane (packed rows, ew x eh samples of esize bytes); level 0 has pel*pel sub-pel planes, other levels one
  struct DevPlane
  {
    MvCudaPlaneDims d = {};
    int nsub = 0;
    void *sub[MVCUDA_MAX_SUB] = {};
  };

  // super frame on the device, index [level * 3 + plane]; the buffers go back to the pool on destruction
  struct DevSuper
  {
    MvCudaSuperParams p = {};
    int esize = 2;   // bytes per sample
    DevPlane pl[MVCUDA_MAX_LEVELS * 3];
    cudaEvent_t ready = nullptr; // recorded after the build when the builder did not wait for it (pooled event)
    DevSuper() = default;
    DevSuper(const DevSuper &) = delete;
    DevSuper &operator=(const DevSuper &) = delete;
    ~DevSuper();
  };

  // geometry of one plane of one level (same formulas as MVSuper.h PlaneWidthLuma / PlaneHeightLuma)
  inline int level_size(int size, int level, int ratio, int pad)
  {
    for (int i = 1; i <= level; i++)
      size = pad >= ratio ? ((size / ratio + 1) / 2) * ratio : ((size / ratio) / 2) * ratio;
    return size;
  }

  inline void plane_dims(const MvCudaSuperParams &p, int level, int plane, MvCudaPlaneDims &d)
  {
    const int xr = plane == 0 ? 1 : p.x_ratio_uv;
    const int yr = plane == 0 ? 1 : p.y_ratio_uv;
    d.w = level_size(p.width, level, p.x_ratio_uv, p.hpad) / xr;
    d.h = level_size(p.height, level, p.y_ratio_uv, p.vpad) / yr;
    d.hpad = p.hpad / xr;
    d.vpad = p.vpad / yr;
    d.ew = d.w + 2 * d.hpad;
    d.eh = d.h + 2 * d.vpad;
  }

  // builds the super frame from a device frame on stream s without a host wait (records DevSuper::ready); 0 on success
  typedef int (*SuperBuildFn)(const MvCudaSuperParams &p, const void *const src[3], const int src_pitch_bytes[3],
    DevSuper &out, cudaStream_t s);

  // device bytes of a super frame with parameters p (all levels, sub-pel planes; esize bytes per sample)
  size_t super_bytes(const MvCudaSuperParams &p, int esize);

}

namespace MVC_NS
{
  // the per-type builder (mvcuda_super.cu): src[] are device plane pointers (pitch in bytes), ordered before s by the caller
  int super_build_dev(const MvCudaSuperParams &p, const void *const src[3], const int src_pitch_bytes[3], mvc::DevSuper &out,
    cudaStream_t s);
}

namespace mvc
{
  // ---- the pipeline stream (mvcuda_frames.cu) -----------------------------------------------------------------------
  // Every operation on device frames (the cudafilters stages, MCompensate / MDegrain, the waits for uploads) runs on this
  // one stream, issued by one submitter thread in submission order, without a host wait: the submission order of the
  // AviSynth filters already orders producers before consumers. Uploads are copied ahead on their own stream, downloads
  // and the analysis wait on their own streams for the ops that made their frames (marks). Buffers of these operations
  // come from their own pool and go back to it at once: a buffer is only reused by work queued later on the same stream
  // (stream-ordered reuse), never by the analysis code, which has its own pool and streams.
  cudaStream_t pipeline_stream();
  // second stream for device -> host copies of pipeline results (ordered after the pipeline by events)
  cudaStream_t copy_stream();
  // makes st wait for the pipeline ops up to sequence number seq (DevFrame::seq: the op that made a frame), not for what
  // was queued after it (no host wait); false on a CUDA error
  bool pipeline_order(cudaStream_t st, unsigned long long seq);
  // an event recorded on the pipeline stream after op seq; nullptr when there is nothing to wait for
  cudaEvent_t mark_after(unsigned long long seq);
  unsigned long long pipe_last_seq();          // sequence number of the calling thread's last queued op
  // For a pipeline op that orders another stream after op seq: mark_request (before queueing the op) asks the submitter
  // to record a mark right after op seq; mark_in_op (inside the op) returns the earliest mark at or after seq, recording
  // one at the current position when there is none.
  void mark_request(unsigned long long seq);
  cudaEvent_t mark_in_op(unsigned long long seq, cudaStream_t s);
  // Every pipeline-stream operation goes through the submitter thread: pipe_submit queues op (it runs later, in
  // submission order, with the pipeline stream as argument); pipe_call queues op and returns once it has run; pipe_flush
  // returns once everything queued has finished on the GPU. Ops may queue further ops (pipe_submit) but must not wait
  // (pipe_call, pipe_flush): the submitter would wait for itself.
  void pipe_submit(std::function<void(cudaStream_t)> op);
  void pipe_call(std::function<void(cudaStream_t)> op);
  void pipe_flush();
  // pooled events (timing disabled); nullptr on failure. Re-recording a pooled event is safe: a stream wait binds to the
  // recording current at the time of the wait.
  cudaEvent_t event_get();
  void event_put(cudaEvent_t e);
  void *pipe_alloc(size_t bytes);             // nullptr on failure
  // host array -> new device buffer, copied ahead of the pipeline stream (free with up_free after the work using it is
  // queued); nullptr on failure
  void *pipe_upload(const void *host, size_t bytes);
  void *up_alloc(size_t bytes);                // upload buffer (no queued pipeline work uses it); nullptr on failure
  void up_free(void *p, size_t bytes);         // back to the upload pool once the pipeline stream has passed this point
  void pipe_free(void *p, size_t bytes);
  // Inside a pipeline op: holds obj until the pipeline stream has passed the work queued so far (for objects whose memory
  // is not stream-ordered, e.g. a DevSuper the op reads: its buffers come from dev_alloc and may go to any stream at once)
  void pipe_keep(std::shared_ptr<void> obj);
  struct DevFrame;
  // Runs issue(st) on a leased stream, issued by the submitter thread (no launches from the worker threads), with st
  // ordered after the work that made the input frames (an uploaded frame: its copy on the upload stream; else the
  // pipeline op that wrote it); then waits for st. issue must not wait for the submitter. Returns issue's code, or -2 when
  // the wait fails; st has finished in every case on return.
  int leased_call(const DevFrame *const *frames, int count, const std::function<int(cudaStream_t)> &issue);
  // Device scratch shared by all pipeline ops of one kind (id), zeroed once at creation and never freed; nullptr on
  // failure. Its users run one after the other on the pipeline stream and leave it zeroed for the next one, which saves
  // a memset per op (a memset between kernels costs an engine switch on the stream).
  void *pipe_scratch(int id, size_t bytes);
  enum { SCRATCH_TEMPORAL = 0, SCRATCH_NNEDI3 = 1, SCRATCH_BINOMIAL = 2, SCRATCH_COUNT = 3 };

  // ---- device frames behind tokens (mvcuda_frames.cu) -----------------------------------------------------------
  // The pel-2 sub-pel planes of a frame as MSuper(pel=2, sharp=2) makes them (mvcuda_comp.cu): for each plane the three
  // Wiener planes (horizontal, vertical, both) of the padded picture, ew x eh samples each, packed. Made once on the
  // pipeline stream by the first consumer that needs them and kept with the frame.
  struct DevSubpel
  {
    int hpad = 0, vpad = 0;     // luma padding they were made for
    void *buf = nullptr;
    size_t bytes = 0;
    const unsigned char *sub[3][3] = {}; // [plane][0: horizontal, 1: vertical, 2: both]
    int ew[3] = {}, eh[3] = {}, hp[3] = {}, vp[3] = {};
    DevSubpel() = default;
    DevSubpel(const DevSubpel &) = delete;
    DevSubpel &operator=(const DevSubpel &) = delete;
    ~DevSubpel();
  };

  // one buffer, planes packed (pitch = width * bytes per sample) at 256-byte aligned offsets; the buffer goes back to its
  // pool when the last holder lets go, so a frame evicted from the registry stays valid for a call that still holds it
  struct DevFrame
  {
    MvCudaFrameFormat f = {};
    void *plane[3] = {};
    int w[3] = {}, h[3] = {};   // plane sizes in samples
    size_t pitch[3] = {};       // bytes per row
    size_t bytes = 0;           // total device bytes
    bool upload = false;        // buffer from up_alloc (frame_alloc(f, true)), else from pipe_alloc
    cudaEvent_t up_done = nullptr; // filled from the host: recorded on the upload stream after the copy (null: see seq)
    unsigned long long seq = 0; // the pipeline op that writes it (set by frame_put)
    unsigned long long token = 0; // registry key (set by frame_put)
    std::mutex extra_m;        // guards subpel and supers
    std::shared_ptr<DevSubpel> subpel;
    std::vector<std::shared_ptr<DevSuper>> supers; // super frames (pyramids) made from this picture (frame_super)
    size_t extra_acct = 0;      // bytes of subpel / supers counted in the registry budget (while the frame is registered)
    DevFrame() = default;
    DevFrame(const DevFrame &) = delete;
    DevFrame &operator=(const DevFrame &) = delete;
    ~DevFrame();
  };
  // nullptr on failure or bad format; upload: the buffer comes from up_alloc (for frames filled from the host)
  std::shared_ptr<DevFrame> frame_alloc(const MvCudaFrameFormat &f, bool upload = false);
  void frame_put(const std::shared_ptr<DevFrame> &fr, unsigned long long producer, int frame, MvCudaTokenHeader &out);
  std::shared_ptr<DevFrame> frame_get(unsigned long long token);            // nullptr when gone
  // counts bytes of data kept with a frame (its sub-pel planes) in the registry budget while the frame is registered
  void frame_account_extra(DevFrame &fr, size_t bytes);
  // the pel-2 sub-pel planes of fr for a super clip with luma padding hpad / vpad (8 bit), made on first use (queued on
  // the pipeline stream before anything the caller queues later); nullptr on failure
  std::shared_ptr<DevSubpel> frame_subpel(const std::shared_ptr<DevFrame> &fr, int hpad, int vpad);
  // The super frame (MSuper pyramid with sub-pel planes) of fr for the parameters p, kept with the frame and counted in the
  // registry budget (GPU_PIPELINE_PLAN.md section 25, M2). A stored one serves when it covers p: same padding, pel and
  // sharp, at least p's levels and chroma, and the same rfilter when p has more than one level (only the first reduction
  // uses it; level 0 does not depend on it). Otherwise build makes it on s, which the caller has ordered after the work
  // that made fr; a stored one built on another stream is waited for on s. The caller keeps out until s has passed its
  // reads (host wait, or pipe_keep on the pipeline stream). 0, -2 or build's code (-4: parameters not supported).
  int frame_super(SuperBuildFn build, const std::shared_ptr<DevFrame> &fr, const MvCudaSuperParams &p, cudaStream_t s,
    std::shared_ptr<DevSuper> &out);
  // the format of the pictures a super with parameters p is made from (chroma: 3 planes)
  MvCudaFrameFormat super_format(const MvCudaSuperParams &p, int esize);
  // whether device frame f can be the picture of a super with parameters p (sizes, depth, chroma layout)
  bool super_fits(const DevFrame &f, const MvCudaSuperParams &p, int esize);
  // a host picture (planes of p's format) as a transient device frame (for the host-memory test entry points); nullptr on failure
  // forgets the upload keys (mvcuda_frame_upload_keyed): the clips they name may be gone, and a new clip may get the address
  void keyed_clear();
  std::shared_ptr<DevFrame> upload_picture(const MvCudaSuperParams &p, int esize, const void *const src[3], const int pitch[3]);
}

#endif
