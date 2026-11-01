// mvtools_vulkan: NNEDI3 field interpolation on device frames, port of Sources/cuda/mvcuda_nnedi3.cu (bit for bit with
// plugins_JPSDR nnedi3's AVX2 + FMA3 path and so with the CUDA library; see that file for the arithmetic). Kernels:
// shaders/nn_pre.pix.comp (copy, cubic kernel, prescreener, queue) and shaders/nn_pred.pix.comp (predictor on the queue),
// shared declarations in shaders/nn_common.glsl. One call = one Cmd: clear the counter, nn_pre, nn_pred.

#include "vk_internal.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace
{
  // same layout as NnArgs in shaders/nn_common.glsl (64-bit members first)
  struct NnArgs
  {
    uint64_t src[3], dst[3];
    uint64_t pitch_s[3], pitch_d[3];
    uint64_t pre_w, pre_f;
    uint64_t pred_w[2];
    uint64_t pred_scale[2], pred_bias[2];
    uint64_t rcp_tab, rsq_tab;
    uint64_t queue, count;
    int width[3], height[3], interp[3], vmin[3], vmax[3];
    int off, xdia, ydia, nns, qual, asize;
    float r_asize;
    int pad_;
  };

  struct NnNet
  {
    int xdia = 0, ydia = 0, nns = 0, qual = 0, asize = 0;
    vkrt::DevBuf pre_w, pre_f, pred_w[2], pred_scale[2], pred_bias[2], rcp_tab, rsq_tab;
    float r_asize = 0.0f; // rcpss((float)asize) as the CPU computes it
  };

  const int NN_NT = 64; // invocations per workgroup of nn_pred (its specialization constant 4)

  bool upload(vkrt::DevBuf &b, const void *src, size_t bytes)
  {
    if (!src)
      return false;
    int rc = 0;
    b = mvk::upload_table(src, bytes, rc);
    return b && rc == 0;
  }
}

extern "C" void *mvcuda_nnedi3_create(const MvCudaNnediParams *p)
{
  if (!p || !vkrt::ready() || (p->qual != 1 && p->qual != 2) || p->nns < 16 || p->nns % 16 != 0 || p->xdia < 8
    || p->ydia < 4 || (p->ydia & 1) || p->xdia * p->ydia > 48 * 6)
    return nullptr;
  // the predictor reads the weights as uvec4 (8 int16) and the window in words of 4 / 2 samples: xdia a multiple of 8
  // (all nsize values)
  if (p->xdia % 8 != 0)
    return nullptr;
  NnNet *n = new NnNet();
  n->xdia = p->xdia;
  n->ydia = p->ydia;
  n->nns = p->nns;
  n->qual = p->qual;
  n->asize = p->xdia * p->ydia;
  bool ok = upload(n->pre_w, p->pre_w, 4 * 64 * sizeof(int16_t)) && upload(n->pre_f, p->pre_f, 28 * sizeof(float))
    && upload(n->rcp_tab, p->rcp_tab, 2048 * sizeof(unsigned)) && upload(n->rsq_tab, p->rsq_tab, 2048 * sizeof(unsigned));
  for (int q = 0; q < p->qual && ok; q++)
    ok = upload(n->pred_w[q], p->pred_w[q], (size_t)2 * n->nns * n->asize * sizeof(int16_t))
      && upload(n->pred_scale[q], p->pred_scale[q], (size_t)2 * n->nns * sizeof(float))
      && upload(n->pred_bias[q], p->pred_bias[q], (size_t)2 * n->nns * sizeof(float));
  // rcpss of the window size, from the same table (the plugin computes it per pixel with vrcpss)
  if (ok)
  {
    const float fa = (float)n->asize;
    unsigned u;
    std::memcpy(&u, &fa, 4);
    const int e = (int)(u >> 23) - 127;
    const unsigned r = p->rcp_tab[(u >> 12) & 2047u] - ((unsigned)e << 23);
    std::memcpy(&n->r_asize, &r, 4);
  }
  // the pipelines now, so that a failure (e.g. too little shared memory) means the CPU from the start
  for (int bytes = 1; bytes <= 2 && ok; bytes++)
    ok = vkrt::kernel(bytes == 1 ? "nn_pre_8" : "nn_pre_16")
      && vkrt::kernel(bytes == 1 ? "nn_pred_8" : "nn_pred_16", { n->xdia, n->ydia, n->nns, n->qual, NN_NT });
  if (!ok)
  {
    if (vkrt::log_enabled())
      std::fprintf(stderr, "mvtools_vulkan: nnedi3 network %dx%d nns %d qual %d not created\n", p->xdia, p->ydia, p->nns, p->qual);
    delete n;
    return nullptr;
  }
  return n;
}

extern "C" void mvcuda_nnedi3_free(void *net)
{
  delete static_cast<NnNet *>(net); // every call has finished (submit_wait) before it returns
}

static int nn_call(void *net, unsigned long long src, int field_n, int plane_mask, const int val_min[3],
  const int val_max[3], unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  NnNet *n = static_cast<NnNet *>(net);
  if (!n || !val_min || !val_max || !out || (field_n != 0 && field_n != 1))
    return -4;
  if (!vkrt::ready())
    return -1;
  auto fs = mvk::frame_get(src);
  if (!fs)
    return MVCUDA_TOKEN_GONE;
  const MvCudaFrameFormat &f = fs->f;
  if (f.bytes_per_sample != 1 && f.bytes_per_sample != 2)
    return -4;
  NnArgs a = {};
  int max_w = 0, max_h = 0;
  size_t queue_max = 0;
  for (int p = 0; p < f.planes; p++)
  {
    a.src[p] = fs->plane[p];
    a.pitch_s[p] = fs->pitch[p] / f.bytes_per_sample;
    a.width[p] = fs->w[p];
    a.height[p] = fs->h[p];
    a.interp[p] = (plane_mask >> p) & 1;
    a.vmin[p] = val_min[p];
    a.vmax[p] = val_max[p];
    if (a.interp[p] && (fs->h[p] < 8 || (fs->h[p] & 1) || fs->w[p] < 32 || fs->w[p] > 65535 || fs->h[p] > 16383))
      return -4;
    max_w = fs->w[p] > max_w ? fs->w[p] : max_w;
    max_h = fs->h[p] > max_h ? fs->h[p] : max_h;
    queue_max += (size_t)fs->w[p] * (fs->h[p] / 2);
  }
  const bool b8 = f.bytes_per_sample == 1;
  vkrt::Kernel *kpre = vkrt::kernel(b8 ? "nn_pre_8" : "nn_pre_16");
  vkrt::Kernel *kpred = vkrt::kernel(b8 ? "nn_pred_8" : "nn_pred_16", { n->xdia, n->ydia, n->nns, n->qual, NN_NT });
  if (!kpre || !kpred)
    return -2;
  auto dst = mvk::frame_alloc(f);
  if (!dst)
    return -2;
  vkrt::DevBuf qbuf((queue_max ? queue_max : 1) * sizeof(unsigned)), cnt(sizeof(unsigned));
  if (!qbuf || !cnt)
    return -2;
  for (int p = 0; p < f.planes; p++)
  {
    a.dst[p] = dst->plane[p];
    a.pitch_d[p] = dst->pitch[p] / f.bytes_per_sample;
  }
  a.pre_w = n->pre_w.p;
  a.pre_f = n->pre_f.p;
  for (int q = 0; q < n->qual; q++)
  {
    a.pred_w[q] = n->pred_w[q].p;
    a.pred_scale[q] = n->pred_scale[q].p;
    a.pred_bias[q] = n->pred_bias[q].p;
  }
  a.rcp_tab = n->rcp_tab.p;
  a.rsq_tab = n->rsq_tab.p;
  a.queue = qbuf.p;
  a.count = cnt.p;
  a.off = 1 - field_n;
  a.xdia = n->xdia;
  a.ydia = n->ydia;
  a.nns = n->nns;
  a.qual = n->qual;
  a.asize = n->asize;
  a.r_asize = n->r_asize;

  vkrt::Cmd cmd;
  cmd.fill(cnt.p, sizeof(unsigned), 0);
  cmd.barrier();
  cmd.dispatch(kpre, (max_w + 127) / 128, max_h, f.planes, a);
  cmd.barrier();
  // enough workgroups of 64 for every pixel that can be queued (the ones beyond the count leave at once); nn_pred loops
  // when there are more than 65535 workgroups' worth
  size_t groups = (queue_max + NN_NT - 1) / NN_NT;
  groups = groups < 1 ? 1 : groups > 65535 ? 65535 : groups;
  cmd.dispatch(kpred, (uint32_t)groups, 1, 1, a);
  const int rc = cmd.submit_wait();
  if (vkrt::log_enabled())
  {
    static std::atomic<unsigned> calls{ 0 };
    if (!rc && calls.fetch_add(1) % 1000 == 0)
      std::fprintf(stderr, "mvtools_vulkan: nnedi3 %dx%d nns %d qual %d, %d-bit, field %d: ok (logged every 1000 calls)\n",
        n->xdia, n->ydia, n->nns, n->qual, f.bits_per_sample, field_n);
  }
  if (rc)
    return rc;
  mvk::frame_put(dst, producer, frame, *out);
  return 0;
}

extern "C" int mvcuda_nnedi3(void *net, unsigned long long src, int field_n, int plane_mask, const int val_min[3],
  const int val_max[3], unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  const int rc = nn_call(net, src, field_n, plane_mask, val_min, val_max, producer, frame, out);
  if (rc && vkrt::log_enabled())
    std::fprintf(stderr, "mvtools_vulkan: nnedi3 frame %d returns %d (caller uses the CPU)\n", frame, rc);
  return rc;
}
