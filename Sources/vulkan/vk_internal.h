// Internal types of mvtools_vulkan (the Vulkan build of the mvcuda.h C API). Mirrors Sources/cuda/mvcuda_internal.h with
// device addresses (vkrt::dptr) in place of device pointers and without streams: every operation is one vkrt::Cmd that
// has finished when the entry point returns (see vkrt.h), so there are no events, marks or stream-ordered pools.
//
// Porting rule: the results must equal the CUDA library's bit for bit (and therefore the CPU's). The host-side logic of
// each Sources/cuda/mvcuda_*.cu file (parameter checks, tables, launch geometry) is kept as close to the original as
// possible; the kernels become shaders/*.comp.
#ifndef MVTOOLS_VK_INTERNAL_H
#define MVTOOLS_VK_INTERNAL_H

#include "../cuda/mvcuda.h"
#include "vkrt.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace mvk
{
  using vkrt::dptr;

  // ---- super frame on the device (built by vk_super.cpp) ----------------------------------------------------------
  // one padded plane (packed rows, ew x eh samples of esize bytes); level 0 has pel*pel sub-pel planes, other levels one
  struct DevPlane
  {
    MvCudaPlaneDims d = {};
    int nsub = 0;
    dptr sub[MVCUDA_MAX_SUB] = {};
  };

  // super frame on the device, index [level * 3 + plane]. bufs own the memory (freed with the object).
  struct DevSuper
  {
    MvCudaSuperParams p = {};
    int esize = 2;   // bytes per sample
    DevPlane pl[MVCUDA_MAX_LEVELS * 3];
    std::vector<vkrt::DevBuf> bufs;
    DevSuper() = default;
    DevSuper(const DevSuper &) = delete;
    DevSuper &operator=(const DevSuper &) = delete;
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

  size_t super_bytes(const MvCudaSuperParams &p, int esize);

  // ---- device frames behind tokens (vk_frames.cpp) ------------------------------------------------------------------
  // The pel-2 sub-pel planes of a frame (MCompensate / MDegrain, vk_comp.cpp): for each plane the three Wiener planes
  // (horizontal, vertical, both) of the padded picture, ew x eh samples each, packed.
  struct DevSubpel
  {
    int hpad = 0, vpad = 0;
    vkrt::DevBuf buf;
    dptr sub[3][3] = {};
    int ew[3] = {}, eh[3] = {}, hp[3] = {}, vp[3] = {};
  };

  // one buffer, planes packed (pitch = width * bytes per sample) at 256-byte aligned offsets; the buffer goes back to the
  // pool when the last holder lets go, so a frame evicted from the registry stays valid for a call that still holds it
  struct DevFrame
  {
    MvCudaFrameFormat f = {};
    dptr plane[3] = {};
    int w[3] = {}, h[3] = {};   // plane sizes in samples
    size_t pitch[3] = {};       // bytes per row
    size_t bytes = 0;           // total device bytes
    unsigned long long token = 0;
    std::mutex extra_m;         // guards subpel and supers
    std::shared_ptr<DevSubpel> subpel;
    std::vector<std::shared_ptr<DevSuper>> supers;
    size_t extra_acct = 0;
    DevFrame() = default;
    DevFrame(const DevFrame &) = delete;
    DevFrame &operator=(const DevFrame &) = delete;
    ~DevFrame();
  };

  bool valid_format(const MvCudaFrameFormat &f);
  // nullptr on failure or bad format
  std::shared_ptr<DevFrame> frame_alloc(const MvCudaFrameFormat &f);
  // registers a frame whose contents are complete (the Cmd that wrote it has finished) and fills out
  void frame_put(const std::shared_ptr<DevFrame> &fr, unsigned long long producer, int frame, MvCudaTokenHeader &out);
  std::shared_ptr<DevFrame> frame_get(unsigned long long token); // nullptr when gone
  void frame_account_extra(DevFrame &fr, size_t bytes);

  // Builds the super frame of src (a device frame of super_format(p)) into out: records the work into cmd (the caller
  // submits). 0, -2 or -4 (parameters not supported). Implemented in vk_super.cpp.
  int super_build(const MvCudaSuperParams &p, const DevFrame &src, DevSuper &out, vkrt::Cmd &cmd);

  // The super frame of fr for the parameters p, kept with the frame and counted in the registry budget (as the CUDA
  // library's frame_super). A stored one serves when it covers p. A new one is built and submitted (waited for) inside.
  // 0, -2 or -4.
  int frame_super(const std::shared_ptr<DevFrame> &fr, const MvCudaSuperParams &p, std::shared_ptr<DevSuper> &out);
  MvCudaFrameFormat super_format(const MvCudaSuperParams &p, int esize);
  bool super_fits(const DevFrame &f, const MvCudaSuperParams &p, int esize);
  void keyed_clear();
  // a host picture (planes of p's format) as a transient device frame (host-memory test entry points); nullptr on failure
  std::shared_ptr<DevFrame> upload_picture(const MvCudaSuperParams &p, int esize, const void *const src[3], const int pitch[3]);

  // Device copy of a small host table that lives as long as the returned object (e.g. per-call block tables)
  inline vkrt::DevBuf upload_table(const void *host, size_t bytes, int &rc)
  {
    vkrt::DevBuf b(bytes ? bytes : 4);
    if (!b)
    {
      rc = -2;
      return b;
    }
    vkrt::Cmd c;
    c.upload(b.p, host, bytes);
    rc = c.submit_wait();
    return b;
  }
}

#endif
