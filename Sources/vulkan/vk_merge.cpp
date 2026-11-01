// mvtools_vulkan: AviSynth+ Merge / MergeLuma / MergeChroma on device frames, port of Sources/cuda/mvcuda_merge.cu
// (bit for bit: see that file for the rounding rules). Also the reference for the other ports: host logic unchanged,
// the kernel launch becomes Cmd::dispatch with an argument struct that shaders/merge.pix.comp declares identically.

#include "vk_internal.h"

#include <cstdint>

namespace
{
  struct MergeArgs
  {
    uint64_t a[3], b[3], dst[3];
    uint64_t pitch_a[3], pitch_b[3], pitch_d[3]; // in samples
    int width[3], height[3], mode[3], tail[3];
    int weight_i, invweight_i, bits, pad_;
  };
}

extern "C" int mvcuda_merge(unsigned long long a, unsigned long long b, const int mode[3], int weight_i, int invweight_i,
  const int tail_start[3], unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  if (!mode || !tail_start || !out)
    return -4;
  if (!vkrt::ready())
    return -1;
  auto fa = mvk::frame_get(a);
  auto fb = mvk::frame_get(b);
  if (!fa || !fb)
    return MVCUDA_TOKEN_GONE;
  const MvCudaFrameFormat &f = fa->f;
  if (fb->f.width != f.width || fb->f.height != f.height || fb->f.planes != f.planes || fb->f.bytes_per_sample != f.bytes_per_sample
    || fb->f.bits_per_sample != f.bits_per_sample || fb->f.x_ratio_uv != f.x_ratio_uv || fb->f.y_ratio_uv != f.y_ratio_uv)
    return -4;
  auto dst = mvk::frame_alloc(f);
  if (!dst)
    return -2;
  MergeArgs prm = {};
  int max_w = 0, max_h = 0;
  for (int p = 0; p < f.planes; p++)
  {
    prm.a[p] = fa->plane[p];
    prm.b[p] = fb->plane[p];
    prm.dst[p] = dst->plane[p];
    prm.pitch_a[p] = fa->pitch[p] / f.bytes_per_sample;
    prm.pitch_b[p] = fb->pitch[p] / f.bytes_per_sample;
    prm.pitch_d[p] = dst->pitch[p] / f.bytes_per_sample;
    prm.width[p] = fa->w[p];
    prm.height[p] = fa->h[p];
    prm.mode[p] = mode[p];
    prm.tail[p] = tail_start[p];
    max_w = fa->w[p] > max_w ? fa->w[p] : max_w;
    max_h = fa->h[p] > max_h ? fa->h[p] : max_h;
  }
  prm.weight_i = weight_i;
  prm.invweight_i = invweight_i;
  prm.bits = f.bits_per_sample;
  vkrt::Kernel *k = vkrt::kernel(f.bytes_per_sample == 1 ? "merge_8" : "merge_16");
  if (!k)
    return -2;
  vkrt::Cmd cmd;
  cmd.dispatch(k, (max_w + 127) / 128, max_h, f.planes, prm);
  const int rc = cmd.submit_wait();
  if (rc)
    return rc;
  mvk::frame_put(dst, producer, frame, *out);
  return 0;
}
