// AviSynth+ Merge / MergeLuma / MergeChroma on device frames (GPU_PIPELINE_PLAN.md section 15, stages 3, 5, 12, 13), bit
// for bit with what AviSynth+ 3.7.5 runs on SSE2 / AVX2 CPUs (avs_core\filters\merge.cpp merge_plane,
// intel\merge_avx2.cpp / merge_sse.cpp): weighted merge (a * invweight + b * weight + 16384) >> 15 with weights scaled to
// 32767; for 16-bit data the vector body works on samples shifted to signed, which gives one more than the scalar formula
// (weight + invweight = 32767), while the scalar tail of a row uses the plain formula; average (a + b + 1) >> 1.

#include "mvcuda_internal.h"

#include <cstdint>

namespace
{
  struct MergeParams
  {
    const void *a[3], *b[3];
    void *dst[3];
    size_t pitch_a[3], pitch_b[3], pitch_d[3]; // in samples
    int width[3], height[3], mode[3], tail[3];
  };

  template <typename pixel_t>
  __global__ void k_merge(MergeParams prm, int weight_i, int invweight_i, int bits)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y, p = blockIdx.z;
    if (x >= prm.width[p] || y >= prm.height[p])
      return;
    const int a = ((const pixel_t *)prm.a[p])[(size_t)y * prm.pitch_a[p] + x];
    const int mode = prm.mode[p];
    int r = a;
    if (mode != 0)
    {
      const int b = ((const pixel_t *)prm.b[p])[(size_t)y * prm.pitch_b[p] + x];
      if (mode == 3)
        r = b;
      else if (mode == 2)
        r = (a + b + 1) >> 1;
      else if (sizeof(pixel_t) == 2 && bits == 16 && x < prm.tail[p])
      {
        // weighted_merge_planar_uint16_*<false>: shift to signed, madd, round, arithmetic shift, signed saturation, shift back
        int s = ((a - 32768) * invweight_i + (b - 32768) * weight_i + 16384) >> 15;
        s = s < -32768 ? -32768 : s > 32767 ? 32767 : s;
        r = (s + 32768) & 0xffff;
      }
      else
        r = (a * invweight_i + b * weight_i + 16384) >> 15;
    }
    ((pixel_t *)prm.dst[p])[(size_t)y * prm.pitch_d[p] + x] = (pixel_t)r;
  }
}

extern "C" int mvcuda_merge(unsigned long long a, unsigned long long b, const int mode[3], int weight_i, int invweight_i,
  const int tail_start[3], unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  if (!mode || !tail_start || !out)
    return -4;
  if (!mvc::use_device())
    return -1;
  auto fa = mvc::frame_get(a);
  auto fb = mvc::frame_get(b);
  if (!fa || !fb)
    return MVCUDA_TOKEN_GONE;
  const MvCudaFrameFormat &f = fa->f;
  if (fb->f.width != f.width || fb->f.height != f.height || fb->f.planes != f.planes || fb->f.bytes_per_sample != f.bytes_per_sample
    || fb->f.bits_per_sample != f.bits_per_sample || fb->f.x_ratio_uv != f.x_ratio_uv || fb->f.y_ratio_uv != f.y_ratio_uv)
    return -4;
  auto dst = mvc::frame_alloc(f);
  if (!dst)
    return -2;
  MergeParams prm = {};
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
  const dim3 grid((max_w + 127) / 128, max_h, f.planes);
  const int bytes = f.bytes_per_sample, bits = f.bits_per_sample;
  mvc::pipe_submit([=](cudaStream_t st) { // issued by the pipeline submitter (mvcuda_internal.h)
    if (bytes == 1)
      k_merge<uint8_t><<<grid, 128, 0, st>>>(prm, weight_i, invweight_i, bits);
    else
      k_merge<uint16_t><<<grid, 128, 0, st>>>(prm, weight_i, invweight_i, bits);
  });
  mvc::frame_put(dst, producer, frame, *out);
  return 0;
}
