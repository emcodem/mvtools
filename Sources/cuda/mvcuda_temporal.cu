// AviSynth+ TemporalSoften on device frames (GPU_PIPELINE_PLAN.md section 15, stage 2), bit for bit with
// avs_core\filters\focus.cpp of AviSynth+ 3.7.5 (TemporalSoften::GetFrame, accumulate_line_c / accumulate_line_16_sse41,
// calculate_sad_c).
//
// CPU order of work, reproduced here: the planes with a threshold are processed in the order Y, V, U. For each plane the
// SAD of the centre against every neighbour decides (scene change) which neighbours are averaged: backwards from the
// nearest and forwards from the nearest, the first neighbour at or above the limit disables itself and everything
// beyond it, also for the following planes. With no neighbour left the filter returns at once: the planes before stay
// processed, this and later planes keep the centre's samples.
//
// Queued on the pipeline stream without a host wait (mvcuda_internal.h); the read-only parameters travel by value as
// kernel arguments, the SADs and decisions live in a scratch shared by all calls (mvc::pipe_scratch: k_decide clears
// the SADs for the next call).

#include "mvcuda_internal.h"

#include <cstdint>

namespace
{
  const int MAX_NEIGH = 32;         // 2 * MAX_RADIUS of the CPU filter (MAX_RADIUS = 7) fits easily
  const int MAX_FRAMES = 2 * 7 + 1;

  struct TsParams
  {
    const void *planes[MAX_FRAMES * 3]; // [frame * 3 + plane], the centre is frame radius
    void *dst[3];
    size_t pitch[3];                    // in samples (all frames of the window have the same geometry)
    int width[3], height[3], plane_ids[3], slot_of_plane[3];
  };

  struct TsScratch
  {
    unsigned long long sad[3 * MAX_NEIGH];
    int mask[3], count[3];
  };

  // SAD of every (processed plane, neighbour): one CUDA block per (row, neighbour, plane slot); row sums in int as the CPU,
  // accumulated in 64 bit
  template <typename pixel_t>
  __global__ void k_sad(TsParams prm, int n_neigh, TsScratch *scr)
  {
    const int y = blockIdx.x, j = blockIdx.y, slot = blockIdx.z;
    const int p = prm.plane_ids[slot];
    if (y >= prm.height[p])
      return;
    const int radius = n_neigh / 2;
    const int k = j < radius ? j : j + 1;
    const pixel_t *c = (const pixel_t *)prm.planes[radius * 3 + p] + (size_t)y * prm.pitch[p];
    const pixel_t *o = (const pixel_t *)prm.planes[k * 3 + p] + (size_t)y * prm.pitch[p];
    int s = 0;
    for (int x = threadIdx.x; x < prm.width[p]; x += blockDim.x)
      s += abs((int)c[x] - (int)o[x]);
    __shared__ int red[256];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int o2 = blockDim.x / 2; o2 > 0; o2 >>= 1)
    {
      if (threadIdx.x < o2)
        red[threadIdx.x] += red[threadIdx.x + o2];
      __syncthreads();
    }
    if (threadIdx.x == 0)
      atomicAdd(&scr->sad[slot * MAX_NEIGH + j], (unsigned long long)(long long)red[0]);
  }

  // the decisions of TemporalSoften::GetFrame: for each processed plane the bit mask of averaged neighbours (bit j =
  // neighbour j in the CPU's planeP order) and the count; count < 0 marks "unchanged" (the early return). sad(c, j): the
  // SAD of slot c against neighbour j.
  template <class Sad>
  __device__ void decide(int n_slots, int n_neigh, int scenechange, int sad_shift, Sad sad, int *mask, int *count)
  {
    const int radius = n_neigh / 2;
    bool disabled[MAX_NEIGH];
    for (int i = 0; i < n_neigh; i++)
      disabled[i] = false;
    bool stopped = false;
    for (int c = 0; c < n_slots; c++)
    {
      if (stopped)
      {
        mask[c] = 0;
        count[c] = -1;
        continue;
      }
      int m = 0, d = n_neigh;
      if (scenechange > 0)
      {
        d = 0;
        bool skiprest = false;
        for (int i = radius - 1; i >= 0; i--)
        {
          if (!skiprest && !disabled[i])
          {
            const int s = (int)(long long)(sad(c, i) >> sad_shift);
            if (s < scenechange)
            {
              m |= 1 << i;
              d++;
            }
            else
              skiprest = true;
            disabled[i] = skiprest;
          }
          else
            disabled[i] = true;
        }
        skiprest = false;
        for (int i = radius; i < 2 * radius; i++)
        {
          if (!skiprest && !disabled[i])
          {
            const int s = (int)(long long)(sad(c, i) >> sad_shift);
            if (s < scenechange)
            {
              m |= 1 << i;
              d++;
            }
            else
              skiprest = true;
            disabled[i] = skiprest;
          }
          else
            disabled[i] = true;
        }
      }
      else
        m = (1 << n_neigh) - 1;
      if (d < 1)
      {
        stopped = true;
        mask[c] = 0;
        count[c] = -1;
        continue;
      }
      mask[c] = m;
      count[c] = d;
    }
  }

  __global__ void k_decide(int n_slots, int n_neigh, int scenechange, int sad_shift, TsScratch *scr)
  {
    decide(n_slots, n_neigh, scenechange, sad_shift, [&](int c, int j) { return scr->sad[c * MAX_NEIGH + j]; }, scr->mask, scr->count);
    for (int i = 0; i < 3 * MAX_NEIGH; i++)
      scr->sad[i] = 0; // the scratch is shared by all calls: the next k_sad starts from zero (mvc::pipe_scratch)
  }

  // one sample of accumulate_line: cur and the averaged neighbours q(j) (bit j of m, d of them)
  template <typename pixel_t, class Q>
  __device__ __forceinline__ pixel_t ts_value(int cur, int n_neigh, int m, int d, int threshold, int bits, Q q)
  {
    const bool max_thr = threshold == 255;
    int thr = threshold;
    if constexpr (sizeof(pixel_t) == 2)
      thr = threshold * (1 << (bits - 8));
    int sum = cur;
    // the CPU visits the averaged neighbours from the last to the first; integer sums do not depend on the order
    for (int j = 0; j < n_neigh; j++)
    {
      if (!(m & (1 << j)))
        continue;
      const int v = q(j);
      if (max_thr)
        sum += v;
      else
        sum += abs(cur - v) <= thr ? v : cur;
    }
    if constexpr (sizeof(pixel_t) == 1)
    {
      // accumulate_line_c / the SSE2 and SSSE3 paths agree: fixed point divide
      const int div = 32768 / (d + 1);
      return (pixel_t)((sum * div + 16384) >> 15);
    }
    else
    {
      // 10..16 bit: what the CPU runs on every SSE4.1 machine (accumulate_line_16_sse41, focus_sse.cpp): float multiply
      // by 1 / (d + 1), round to nearest even (cvtps_epi32), unsigned saturation to 16 bit, then the bit-depth limit.
      // (The C fallback divides in fixed point and can differ by one; CPUs without SSE4.1 are not a target.)
      const float r = __fmul_rn((float)sum, __fdiv_rn(1.0f, (float)(d + 1)));
      int v = __float2int_rn(r);
      v = v < 0 ? 0 : v > 65535 ? 65535 : v;
      if (bits < 16)
        v = v > (1 << bits) - 1 ? (1 << bits) - 1 : v;
      return (pixel_t)v;
    }
  }
  // accumulate_line for one plane (blockIdx.z; luma and chroma have their own threshold), or a copy of the centre when
  // the plane is not processed / the filter stopped
  template <typename pixel_t>
  __global__ void k_accumulate(TsParams prm, int n_neigh, const TsScratch *scr, int thr_y, int thr_c, int bits)
  {
    const int p = blockIdx.z;
    const int threshold = p == 0 ? thr_y : thr_c;
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
    if (x >= prm.width[p] || y >= prm.height[p])
      return;
    const int radius = n_neigh / 2;
    const size_t at = (size_t)y * prm.pitch[p] + x;
    const pixel_t cur = ((const pixel_t *)prm.planes[radius * 3 + p])[at];
    const int slot = prm.slot_of_plane[p];
    pixel_t out = cur;
    if (slot >= 0 && scr->count[slot] >= 0)
      out = ts_value<pixel_t>(cur, n_neigh, scr->mask[slot], scr->count[slot], threshold, bits,
        [&](int j) { return (int)((const pixel_t *)prm.planes[(j < radius ? j : j + 1) * 3 + p])[at]; });
    ((pixel_t *)prm.dst[p])[at] = out;
  }
}

extern "C" int mvcuda_temporal_soften(const unsigned long long *tokens, int radius, int luma_thresh, int chroma_thresh,
  int scenechange, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  if (!tokens || !out || radius < 1 || 2 * radius + 1 > MAX_FRAMES)
    return -4;
  if (!mvc::use_device())
    return -1;
  const int n_frames = 2 * radius + 1, n_neigh = 2 * radius;
  std::shared_ptr<mvc::DevFrame> fr[MAX_FRAMES];
  for (int k = 0; k < n_frames; k++)
  {
    fr[k] = mvc::frame_get(tokens[k]);
    if (!fr[k])
      return MVCUDA_TOKEN_GONE;
    if (k > 0 && (fr[k]->f.width != fr[0]->f.width || fr[k]->f.height != fr[0]->f.height || fr[k]->f.planes != fr[0]->f.planes
      || fr[k]->f.bytes_per_sample != fr[0]->f.bytes_per_sample || fr[k]->f.x_ratio_uv != fr[0]->f.x_ratio_uv
      || fr[k]->f.y_ratio_uv != fr[0]->f.y_ratio_uv))
      return -4;
  }
  const mvc::DevFrame &cf = *fr[radius];
  const MvCudaFrameFormat &f = cf.f;
  luma_thresh = luma_thresh > 255 ? 255 : luma_thresh;
  chroma_thresh = chroma_thresh > 255 ? 255 : chroma_thresh;
  // processing order and per-plane slots (TemporalSoften constructor: Y if luma_thresh, then V, U if chroma_thresh)
  TsParams prm = {};
  int n_slots = 0, thr_of_slot[3] = {};
  for (int p = 0; p < 3; p++)
    prm.slot_of_plane[p] = -1;
  if (luma_thresh > 0)
  {
    prm.slot_of_plane[0] = n_slots;
    thr_of_slot[n_slots] = luma_thresh;
    prm.plane_ids[n_slots++] = 0;
  }
  if (chroma_thresh > 0 && f.planes == 3)
  {
    prm.slot_of_plane[2] = n_slots;
    thr_of_slot[n_slots] = chroma_thresh;
    prm.plane_ids[n_slots++] = 2;
    prm.slot_of_plane[1] = n_slots;
    thr_of_slot[n_slots] = chroma_thresh;
    prm.plane_ids[n_slots++] = 1;
  }
  auto dst = mvc::frame_alloc(f);
  if (!dst)
    return -2;
  for (int k = 0; k < n_frames; k++)
    for (int p = 0; p < f.planes; p++)
      prm.planes[k * 3 + p] = fr[k]->plane[p];
  for (int p = 0; p < f.planes; p++)
  {
    prm.dst[p] = dst->plane[p];
    prm.pitch[p] = cf.pitch[p] / f.bytes_per_sample;
    prm.width[p] = cf.w[p];
    prm.height[p] = cf.h[p];
  }
  TsScratch *scr = (TsScratch *)mvc::pipe_scratch(mvc::SCRATCH_TEMPORAL, sizeof(TsScratch));
  if (!scr)
    return -2;
  const int sad_shift = f.bytes_per_sample == 2 ? f.bits_per_sample - 8 : 0;
  int max_h = 0;
  for (int s = 0; s < n_slots; s++)
    max_h = cf.h[prm.plane_ids[s]] > max_h ? cf.h[prm.plane_ids[s]] : max_h;
  const bool do_sad = n_slots > 0 && scenechange > 0;

  const int thr_y = prm.slot_of_plane[0] >= 0 ? thr_of_slot[prm.slot_of_plane[0]] : 0;
  const int thr_c = (f.planes == 3 && prm.slot_of_plane[1] >= 0) ? thr_of_slot[prm.slot_of_plane[1]] : 0;
  const dim3 grid_sad(max_h > 0 ? max_h : 1, n_neigh, n_slots > 0 ? n_slots : 1);
  const dim3 grid_a((cf.w[0] + 127) / 128, cf.h[0], f.planes); // luma is the largest plane
  const bool b8 = f.bytes_per_sample == 1;
  const int bits = f.bits_per_sample;
  mvc::pipe_submit([=](cudaStream_t st) { // issued by the pipeline submitter (mvcuda_internal.h)

    if (do_sad)
    {
      if (b8)
        k_sad<uint8_t><<<grid_sad, 256, 0, st>>>(prm, n_neigh, scr);
      else
        k_sad<uint16_t><<<grid_sad, 256, 0, st>>>(prm, n_neigh, scr);
    }
    k_decide<<<1, 1, 0, st>>>(n_slots, n_neigh, scenechange, sad_shift, scr);
    if (b8)
      k_accumulate<uint8_t><<<grid_a, 128, 0, st>>>(prm, n_neigh, scr, thr_y, thr_c, bits);
    else
      k_accumulate<uint16_t><<<grid_a, 128, 0, st>>>(prm, n_neigh, scr, thr_y, thr_c, bits);
  });

  mvc::frame_put(dst, producer, frame, *out);
  return 0;
}

// ---- QTGMC's search-clip binomial: TemporalSoften radius 1 and 2, merged (CudaQtgmcBinomial0) ------------------------
// MergeLuma(MergeLuma(TemporalSoften(b, 1, ...), TemporalSoften(b, 2, ...), w1), b, w2) of QTGMC.avsi (TR0 = 2) in three
// kernels instead of eight: the SADs of radius 1 are those of radius 2 against the nearer neighbours (same frame pairs),
// both filters decide from them, one kernel makes both results per sample and merges them.
namespace
{
  struct TbScratch
  {
    unsigned long long sad[3 * MAX_NEIGH]; // radius 2: neighbours n - 2, n - 1, n + 1, n + 2
    int mask[3], count[3];                 // radius 2
    int mask1[3], count1[3];               // radius 1
  };

  __global__ void k_decide_pair(int n_slots, int scenechange, int sad_shift, TbScratch *scr)
  {
    decide(n_slots, 4, scenechange, sad_shift, [&](int c, int j) { return scr->sad[c * MAX_NEIGH + j]; }, scr->mask, scr->count);
    decide(n_slots, 2, scenechange, sad_shift, [&](int c, int j) { return scr->sad[c * MAX_NEIGH + (j == 0 ? 1 : 2)]; }, scr->mask1,
      scr->count1);
    for (int i = 0; i < 3 * MAX_NEIGH; i++)
      scr->sad[i] = 0; // shared by all calls (mvc::pipe_scratch)
  }

  // Merge / MergeLuma weighted mode (mvcuda_merge.cu): 16-bit samples in the SIMD part of a row (x < tail) go through
  // the signed variant
  template <typename pixel_t>
  __device__ __forceinline__ int merge_w(int a, int b, int w, int inv, int bits, bool simd)
  {
    if (sizeof(pixel_t) == 2 && bits == 16 && simd)
    {
      int s = ((a - 32768) * inv + (b - 32768) * w + 16384) >> 15;
      s = s < -32768 ? -32768 : s > 32767 ? 32767 : s;
      return (s + 32768) & 0xffff;
    }
    return (a * inv + b * w + 16384) >> 15;
  }

  // frames 0..4 = n - 2 .. n + 2; luma: MergeLuma(MergeLuma(ts1, ts2, w1), centre, w2), chroma: ts1's, or with merge_chroma
  // merged the same way (Merge instead of MergeLuma; tail_c: the chroma rows' SIMD tail)
  template <typename pixel_t>
  __global__ void k_binomial(TsParams prm, const TbScratch *scr, int thr_y, int thr_c, int bits, int w1, int inv1, int w2, int inv2,
    int tail_y, int merge_chroma, int tail_c)
  {
    const int p = blockIdx.z;
    const int threshold = p == 0 ? thr_y : thr_c;
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
    if (x >= prm.width[p] || y >= prm.height[p])
      return;
    const size_t at = (size_t)y * prm.pitch[p] + x;
    auto px = [&](int k) { return (int)((const pixel_t *)prm.planes[k * 3 + p])[at]; };
    const int cur = px(2);
    const int slot = prm.slot_of_plane[p];
    int ts1 = cur, ts2 = cur;
    if (slot >= 0 && scr->count1[slot] >= 0)
      ts1 = ts_value<pixel_t>(cur, 2, scr->mask1[slot], scr->count1[slot], threshold, bits, [&](int j) { return px(j == 0 ? 1 : 3); });
    pixel_t out = (pixel_t)ts1;
    if (p == 0 || merge_chroma)
    {
      if (slot >= 0 && scr->count[slot] >= 0)
        ts2 = ts_value<pixel_t>(cur, 4, scr->mask[slot], scr->count[slot], threshold, bits, [&](int j) { return px(j < 2 ? j : j + 1); });
      const bool simd = x < (p == 0 ? tail_y : tail_c);
      out = (pixel_t)merge_w<pixel_t>(merge_w<pixel_t>(ts1, ts2, w1, inv1, bits, simd), cur, w2, inv2, bits, simd);
    }
    ((pixel_t *)prm.dst[p])[at] = out;
  }
}

extern "C" int mvcuda_temporal_binomial(const unsigned long long tokens[5], int luma_thresh, int chroma_thresh, int scenechange,
  int w1, int inv1, int w2, int inv2, int tail_y, int merge_chroma, int tail_c, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  if (!tokens || !out)
    return -4;
  if (!mvc::use_device())
    return -1;
  std::shared_ptr<mvc::DevFrame> fr[5];
  for (int k = 0; k < 5; k++)
  {
    fr[k] = mvc::frame_get(tokens[k]);
    if (!fr[k])
      return MVCUDA_TOKEN_GONE;
    if (k > 0 && (fr[k]->f.width != fr[0]->f.width || fr[k]->f.height != fr[0]->f.height || fr[k]->f.planes != fr[0]->f.planes
      || fr[k]->f.bytes_per_sample != fr[0]->f.bytes_per_sample || fr[k]->f.x_ratio_uv != fr[0]->f.x_ratio_uv
      || fr[k]->f.y_ratio_uv != fr[0]->f.y_ratio_uv))
      return -4;
  }
  const mvc::DevFrame &cf = *fr[2];
  const MvCudaFrameFormat &f = cf.f;
  luma_thresh = luma_thresh > 255 ? 255 : luma_thresh;
  chroma_thresh = chroma_thresh > 255 ? 255 : chroma_thresh;
  // slots as mvcuda_temporal_soften (the same for both radii)
  TsParams prm = {};
  int n_slots = 0;
  for (int p = 0; p < 3; p++)
    prm.slot_of_plane[p] = -1;
  if (luma_thresh > 0)
  {
    prm.slot_of_plane[0] = n_slots;
    prm.plane_ids[n_slots++] = 0;
  }
  if (chroma_thresh > 0 && f.planes == 3)
  {
    prm.slot_of_plane[2] = n_slots;
    prm.plane_ids[n_slots++] = 2;
    prm.slot_of_plane[1] = n_slots;
    prm.plane_ids[n_slots++] = 1;
  }
  auto dst = mvc::frame_alloc(f);
  if (!dst)
    return -2;
  for (int k = 0; k < 5; k++)
    for (int p = 0; p < f.planes; p++)
      prm.planes[k * 3 + p] = fr[k]->plane[p];
  for (int p = 0; p < f.planes; p++)
  {
    prm.dst[p] = dst->plane[p];
    prm.pitch[p] = cf.pitch[p] / f.bytes_per_sample;
    prm.width[p] = cf.w[p];
    prm.height[p] = cf.h[p];
  }
  TbScratch *scr = (TbScratch *)mvc::pipe_scratch(mvc::SCRATCH_BINOMIAL, sizeof(TbScratch));
  if (!scr)
    return -2;
  const int sad_shift = f.bytes_per_sample == 2 ? f.bits_per_sample - 8 : 0;
  int max_h = 0;
  for (int s = 0; s < n_slots; s++)
    max_h = cf.h[prm.plane_ids[s]] > max_h ? cf.h[prm.plane_ids[s]] : max_h;
  const bool do_sad = n_slots > 0 && scenechange > 0;
  const dim3 grid_sad(max_h > 0 ? max_h : 1, 4, n_slots > 0 ? n_slots : 1);
  const dim3 grid_a((cf.w[0] + 127) / 128, cf.h[0], f.planes);
  const bool b8 = f.bytes_per_sample == 1;
  const int bits = f.bits_per_sample;
  // k_sad works on a TsScratch: TbScratch begins with the same SAD array
  TsScratch *scr_sad = (TsScratch *)scr;
  mvc::pipe_submit([=](cudaStream_t st) { // issued by the pipeline submitter (mvcuda_internal.h)
    if (do_sad)
    {
      if (b8)
        k_sad<uint8_t><<<grid_sad, 256, 0, st>>>(prm, 4, scr_sad);
      else
        k_sad<uint16_t><<<grid_sad, 256, 0, st>>>(prm, 4, scr_sad);
    }
    k_decide_pair<<<1, 1, 0, st>>>(n_slots, scenechange, sad_shift, scr);
    if (b8)
      k_binomial<uint8_t><<<grid_a, 128, 0, st>>>(prm, scr, luma_thresh, chroma_thresh, bits, w1, inv1, w2, inv2, tail_y, merge_chroma, tail_c);
    else
      k_binomial<uint16_t><<<grid_a, 128, 0, st>>>(prm, scr, luma_thresh, chroma_thresh, bits, w1, inv1, w2, inv2, tail_y, merge_chroma, tail_c);
  });
  mvc::frame_put(dst, producer, frame, *out);
  return 0;
}