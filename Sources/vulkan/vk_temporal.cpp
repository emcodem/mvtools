// mvtools_vulkan: AviSynth+ TemporalSoften and QTGMC's search-clip binomial on device frames, port of
// Sources/cuda/mvcuda_temporal.cu (bit for bit with avs_core\filters\focus.cpp of AviSynth+ 3.7.5; see that file for the
// order of work). Differences to the CUDA version: the row SADs go to a per-call buffer and are summed in tmp_decide (no
// 64-bit atomics, no shared scratch), and the decisions go to a per-call buffer.
// Kernels: shaders/tmp_sad.pix.comp, shaders/tmp_decide.comp, shaders/tmp_acc.pix.comp.

#include "vk_internal.h"
#include "vk_cf_log.h"

#include <atomic>
#include <cstdint>
#include <cstdio>

namespace
{
  const int MAX_NEIGH = 32;
  const int MAX_FRAMES = 2 * 7 + 1;

  struct SadArgs
  {
    uint64_t planes[45];
    uint64_t rowsad;
    int pitch[3], width[3], height[3], plane_ids[3];
    int n_neigh, max_h, pad0, pad1;
  };

  struct DecideArgs
  {
    uint64_t rowsad, outp;
    int height_of_slot[3];
    int n_slots, n_neigh, scenechange, sad_shift, max_h, do_sad, pair;
  };

  struct AccArgs
  {
    uint64_t planes[45];
    uint64_t dst[3];
    uint64_t dec;
    float rcp[16];
    int pitch[3], width[3], height[3], slot_of_plane[3];
    int n_neigh, thr_y, thr_c, bits;
    int w1, inv1, w2, inv2, tail_y, merge_chroma, tail_c, pad0;
  };


  // the common part of both entry points: frames of the window, slots, and the three dispatches
  //   n_frames: 2 * radius + 1 (binomial: 5, radius 2); pair: binomial
  int run(const unsigned long long *tokens, int n_frames, int luma_thresh, int chroma_thresh, int scenechange, bool pair,
    AccArgs &acc, unsigned long long producer, int frame, MvCudaTokenHeader *out)
  {
    if (!vkrt::ready())
      return -1;
    const int radius = n_frames / 2, n_neigh = 2 * radius;
    std::shared_ptr<mvk::DevFrame> fr[MAX_FRAMES];
    for (int k = 0; k < n_frames; k++)
    {
      fr[k] = mvk::frame_get(tokens[k]);
      if (!fr[k])
        return MVCUDA_TOKEN_GONE;
      if (k > 0 && (fr[k]->f.width != fr[0]->f.width || fr[k]->f.height != fr[0]->f.height || fr[k]->f.planes != fr[0]->f.planes
        || fr[k]->f.bytes_per_sample != fr[0]->f.bytes_per_sample || fr[k]->f.x_ratio_uv != fr[0]->f.x_ratio_uv
        || fr[k]->f.y_ratio_uv != fr[0]->f.y_ratio_uv))
        return -4;
    }
    const mvk::DevFrame &cf = *fr[radius];
    const MvCudaFrameFormat &f = cf.f;
    luma_thresh = luma_thresh > 255 ? 255 : luma_thresh;
    chroma_thresh = chroma_thresh > 255 ? 255 : chroma_thresh;
    // processing order and per-plane slots (TemporalSoften constructor: Y if luma_thresh, then V, U if chroma_thresh)
    int n_slots = 0, plane_ids[3] = {};
    for (int p = 0; p < 3; p++)
      acc.slot_of_plane[p] = -1;
    if (luma_thresh > 0)
    {
      acc.slot_of_plane[0] = n_slots;
      plane_ids[n_slots++] = 0;
    }
    if (chroma_thresh > 0 && f.planes == 3)
    {
      acc.slot_of_plane[2] = n_slots;
      plane_ids[n_slots++] = 2;
      acc.slot_of_plane[1] = n_slots;
      plane_ids[n_slots++] = 1;
    }
    auto dst = mvk::frame_alloc(f);
    if (!dst)
      return -2;
    SadArgs sa = {};
    for (int k = 0; k < n_frames; k++)
      for (int p = 0; p < f.planes; p++)
        acc.planes[k * 3 + p] = sa.planes[k * 3 + p] = fr[k]->plane[p];
    for (int p = 0; p < f.planes; p++)
    {
      acc.dst[p] = dst->plane[p];
      acc.pitch[p] = sa.pitch[p] = (int)(cf.pitch[p] / f.bytes_per_sample);
      acc.width[p] = sa.width[p] = cf.w[p];
      acc.height[p] = sa.height[p] = cf.h[p];
    }
    const int sad_shift = f.bytes_per_sample == 2 ? f.bits_per_sample - 8 : 0;
    int max_h = 0;
    for (int s = 0; s < n_slots; s++)
      max_h = cf.h[plane_ids[s]] > max_h ? cf.h[plane_ids[s]] : max_h;
    const bool do_sad = n_slots > 0 && scenechange > 0;
    // TemporalSoften's thresholds per plane (only read for processed planes)
    acc.thr_y = luma_thresh;
    acc.thr_c = chroma_thresh;
    acc.n_neigh = n_neigh;
    acc.bits = f.bits_per_sample;
    for (int d = 0; d < 16; d++)
      acc.rcp[d] = 1.0f / (float)(d + 1);

    vkrt::DevBuf rowsad(do_sad ? sizeof(int) * (size_t)n_slots * n_neigh * max_h : 4), dec(sizeof(int) * 16);
    if (!rowsad || !dec)
      return -2;
    vkrt::Kernel *ks = do_sad ? vkrt::kernel(f.bytes_per_sample == 1 ? "tmp_sad_8" : "tmp_sad_16") : nullptr;
    vkrt::Kernel *kd = vkrt::kernel("tmp_decide");
    vkrt::Kernel *ka = vkrt::kernel(f.bytes_per_sample == 1 ? "tmp_acc_8" : "tmp_acc_16", { pair ? 1 : 0 });
    if ((do_sad && !ks) || !kd || !ka)
      return -2;
    vkrt::Cmd cmd;
    if (do_sad)
    {
      for (int s = 0; s < n_slots; s++)
        sa.plane_ids[s] = plane_ids[s];
      sa.rowsad = rowsad.p;
      sa.n_neigh = n_neigh;
      sa.max_h = max_h;
      cmd.dispatch(ks, max_h, n_neigh, n_slots, sa);
    }
    DecideArgs da = {};
    da.rowsad = rowsad.p;
    da.outp = dec.p;
    for (int s = 0; s < n_slots; s++)
      da.height_of_slot[s] = cf.h[plane_ids[s]];
    da.n_slots = n_slots;
    da.n_neigh = n_neigh;
    da.scenechange = scenechange;
    da.sad_shift = sad_shift;
    da.max_h = max_h;
    da.do_sad = do_sad ? 1 : 0;
    da.pair = pair ? 1 : 0;
    cmd.dispatch(kd, 1, 1, 1, da);
    acc.dec = dec.p;
    cmd.dispatch(ka, (cf.w[0] + 127) / 128, cf.h[0], f.planes, acc); // luma is the largest plane
    const int rc = cmd.submit_wait();
    if (rc)
      return rc;
    mvk::frame_put(dst, producer, frame, *out);
    return 0;
  }
}

static int temporal_soften_impl(const unsigned long long *tokens, int radius, int luma_thresh, int chroma_thresh,
  int scenechange, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  if (!tokens || !out || radius < 1 || 2 * radius + 1 > MAX_FRAMES)
    return -4;
  AccArgs acc = {};
  const int rc = run(tokens, 2 * radius + 1, luma_thresh, chroma_thresh, scenechange, false, acc, producer, frame, out);
  return rc;
}

static int temporal_binomial_impl(const unsigned long long tokens[5], int luma_thresh, int chroma_thresh, int scenechange,
  int w1, int inv1, int w2, int inv2, int tail_y, int merge_chroma, int tail_c, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  if (!tokens || !out)
    return -4;
  AccArgs acc = {};
  acc.w1 = w1;
  acc.inv1 = inv1;
  acc.w2 = w2;
  acc.inv2 = inv2;
  acc.tail_y = tail_y;
  acc.merge_chroma = merge_chroma;
  acc.tail_c = tail_c;
  const int rc = run(tokens, 5, luma_thresh, chroma_thresh, scenechange, true, acc, producer, frame, out);
  return rc;
}

// ---- exported entry points (counted, see vk_cf_log.h) ----
extern "C" int mvcuda_temporal_soften(const unsigned long long *tokens, int radius, int luma_thresh, int chroma_thresh,
  int scenechange, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  static mvk::CallLog log("temporal_soften");
  return log(temporal_soften_impl(tokens, radius, luma_thresh, chroma_thresh, scenechange, producer, frame, out));
}

extern "C" int mvcuda_temporal_binomial(const unsigned long long tokens[5], int luma_thresh, int chroma_thresh, int scenechange,
  int w1, int inv1, int w2, int inv2, int tail_y, int merge_chroma, int tail_c, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  static mvk::CallLog log("temporal_binomial");
  return log(temporal_binomial_impl(tokens, luma_thresh, chroma_thresh, scenechange, w1, inv1, w2, inv2, tail_y, merge_chroma, tail_c, producer, frame, out));
}
