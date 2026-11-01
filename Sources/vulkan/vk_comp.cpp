// mvtools_vulkan: MCompensate and MDegrain1..6 on device frames, port of Sources/cuda/mvcuda_comp.cu (bit for bit: see
// that file for the rounding rules), and the pel-2 Wiener sub-pel planes kept with a device frame (frame_subpel).
// Kernels: shaders/comp_blocks.pix.comp (spec constants MODE, PEL) and shaders/comp_subpel.pix.comp.
// One API call = one Cmd: the upload of the block table and windows, the sub-pel planes of references that have none yet,
// and the block kernel. New sub-pel planes are stored with their frame only after the Cmd finished, so another thread never
// reads unfinished planes (two threads that need the same new planes at once both make them; one copy is kept).

#include "vk_internal.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace
{
  const int MAXR = 2 * MVCUDA_MAX_DEGRAIN; // reference frames of one call

  struct CompArgs
  {
    uint64_t src[3], fill[3], dst[3];
    uint64_t ref[MAXR * 3];        // [r * 3 + p]
    uint64_t sub[MAXR * 9];        // [(r * 3 + p) * 3 + s]
    uint64_t win[3];
    uint64_t blocks;
    uint64_t pitch[3];             // in samples
    int maxv, lstep_x, lstep_y, nbx;
    int width[3], height[3];
    int bw[3], bh[3], step_x[3], step_y[3], log_x[3], log_y[3];
    int wb[3], hb[3];
    int ew[3], eh[3], hp[3], vp[3];
    int act[3];
    int nby, ov, L, pad_;
  };

  struct SubArgs
  {
    uint64_t src[3];
    uint64_t ps[3];                // in samples
    uint64_t dst[9];               // [plane * 3 + sub]
    int w[3], h[3], hp[3], vp[3], ew[3], eh[3];
    int maxv, pad_;
  };

  bool frames_match(const mvk::DevFrame &a, const mvk::DevFrame &b)
  {
    return a.f.width == b.f.width && a.f.height == b.f.height && a.f.planes == b.f.planes && a.f.bytes_per_sample == b.f.bytes_per_sample
      && a.f.x_ratio_uv == b.f.x_ratio_uv && a.f.y_ratio_uv == b.f.y_ratio_uv;
  }

  int log2i(int v)
  {
    int l = 0;
    while ((1 << l) < v)
      l++;
    return l;
  }

  // The sub-pel planes of fr for (hpad, vpad): the stored ones (fresh = false), or new ones whose build is recorded into
  // cmd (fresh = true; store them with subpel_publish after the Cmd finished). nullptr on failure.
  std::shared_ptr<mvk::DevSubpel> subpel_get(const std::shared_ptr<mvk::DevFrame> &fr, int hpad, int vpad, vkrt::Cmd &cmd, bool &fresh)
  {
    fresh = false;
    if (!fr || (fr->f.bytes_per_sample != 1 && fr->f.bytes_per_sample != 2))
      return nullptr;
    {
      std::lock_guard<std::mutex> lk(fr->extra_m);
      if (fr->subpel && fr->subpel->hpad == hpad && fr->subpel->vpad == vpad)
        return fr->subpel;
    }
    const int bps = fr->f.bytes_per_sample;
    vkrt::Kernel *k = vkrt::kernel(bps == 1 ? "comp_subpel_8" : "comp_subpel_16");
    if (!k)
      return nullptr;
    auto s = std::make_shared<mvk::DevSubpel>();
    s->hpad = hpad;
    s->vpad = vpad;
    size_t off[3][3] = {}, total = 0;
    int max_ew = 0, max_eh = 0;
    for (int p = 0; p < fr->f.planes; p++)
    {
      const int xr = p ? fr->f.x_ratio_uv : 1, yr = p ? fr->f.y_ratio_uv : 1;
      s->hp[p] = hpad / xr;
      s->vp[p] = vpad / yr;
      s->ew[p] = fr->w[p] + 2 * s->hp[p];
      s->eh[p] = fr->h[p] + 2 * s->vp[p];
      for (int q = 0; q < 3; q++)
      {
        off[p][q] = total;
        total += ((size_t)s->ew[p] * s->eh[p] * bps + 255) & ~(size_t)255;
      }
      max_ew = s->ew[p] > max_ew ? s->ew[p] : max_ew;
      max_eh = s->eh[p] > max_eh ? s->eh[p] : max_eh;
    }
    s->buf = vkrt::DevBuf(total);
    if (!s->buf)
      return nullptr;
    SubArgs b = {};
    b.maxv = bps == 1 ? 255 : (1 << fr->f.bits_per_sample) - 1;
    for (int p = 0; p < fr->f.planes; p++)
    {
      b.src[p] = fr->plane[p];
      b.ps[p] = fr->pitch[p] / bps;
      b.w[p] = fr->w[p];
      b.h[p] = fr->h[p];
      b.hp[p] = s->hp[p];
      b.vp[p] = s->vp[p];
      b.ew[p] = s->ew[p];
      b.eh[p] = s->eh[p];
      for (int q = 0; q < 3; q++)
      {
        s->sub[p][q] = s->buf.p + off[p][q];
        b.dst[p * 3 + q] = s->sub[p][q];
      }
    }
    if (!cmd.dispatch(k, (max_ew + 127) / 128, max_eh, fr->f.planes * 3, b))
      return nullptr;
    fresh = true;
    return s;
  }

  void subpel_publish(const std::shared_ptr<mvk::DevFrame> &fr, const std::shared_ptr<mvk::DevSubpel> &s)
  {
    {
      std::lock_guard<std::mutex> lk(fr->extra_m);
      if (fr->subpel && fr->subpel->hpad == s->hpad && fr->subpel->vpad == s->vpad)
        return; // another thread stored the same planes meanwhile
      fr->subpel = s;
    }
    mvk::frame_account_extra(*fr, s->buf.bytes);
  }

  // refs: nref reference tokens (compensate 1, degrain 2 * L: 2k backward, 2k + 1 forward of radius k + 1)
  template <int MODE>
  int run(const MvCudaBlockGeom *g, int L, unsigned long long t_src, const unsigned long long *t_refs, int nref,
    unsigned long long t_fill, const int *blocks, int plane_mask, unsigned long long producer, int frame, MvCudaTokenHeader *out)
  {
    if (!g || !blocks || !out || g->n_blk_x < 1 || g->n_blk_y < 1 || g->blk_x < 1 || g->blk_y < 1 || g->ov_x >= g->blk_x || g->ov_y >= g->blk_y
      || ((g->ov_x || g->ov_y) && (!g->win_luma)) || (g->pel != 1 && g->pel != 2) || g->hpad < 0 || g->vpad < 0 || nref < 1 || nref > MAXR)
      return -4;
    if (!vkrt::ready())
      return -1;
    auto fs = mvk::frame_get(t_src), ff = mvk::frame_get(t_fill);
    std::shared_ptr<mvk::DevFrame> fr[MAXR];
    if (!fs || !ff)
      return MVCUDA_TOKEN_GONE;
    for (int r = 0; r < nref; r++)
      if (!(fr[r] = mvk::frame_get(t_refs[r])))
        return MVCUDA_TOKEN_GONE;
    if ((fs->f.bytes_per_sample != 1 && fs->f.bytes_per_sample != 2) || !frames_match(*fs, *ff))
      return -4;
    for (int r = 0; r < nref; r++)
      if (!frames_match(*fs, *fr[r]))
        return -4;
    const MvCudaFrameFormat &f = fs->f;
    if (f.planes == 3 && (g->ov_x || g->ov_y) && !g->win_chroma)
      return -4;
    const int lx = log2i(f.x_ratio_uv), ly = log2i(f.y_ratio_uv);
    if (f.planes == 3 && g->pel == 2 && (g->hpad % f.x_ratio_uv || g->vpad % f.y_ratio_uv))
      return -4;
    const int bps = f.bytes_per_sample;
    vkrt::Kernel *k = vkrt::kernel(bps == 1 ? "comp_blocks_8" : "comp_blocks_16", { MODE, g->pel });
    if (!k)
      return -2;
    vkrt::Cmd cmd;
    // pel 2: the sub-pel planes of every reference (made in this Cmd if this is their first use)
    std::shared_ptr<mvk::DevSubpel> sp[MAXR];
    bool fresh[MAXR] = {};
    bool any_fresh = false;
    if (g->pel == 2)
      for (int r = 0; r < nref; r++)
      {
        for (int q = 0; q < r && !sp[r]; q++)
          if (fr[q] == fr[r])
            sp[r] = sp[q]; // the same frame twice in this call
        if (sp[r])
          continue;
        if (!(sp[r] = subpel_get(fr[r], g->hpad, g->vpad, cmd, fresh[r])))
          return -2;
        any_fresh |= fresh[r];
      }
    const int ov = (g->ov_x || g->ov_y) ? 1 : 0;
    const int nblocks = g->n_blk_x * g->n_blk_y;
    const int stride = MODE == 0 ? 3 : 4 * L + 2 * (1 + 2 * L);
    const size_t bbytes = (size_t)nblocks * stride * sizeof(int);
    // the overlap windows: 9 luma windows, then 9 chroma windows
    const size_t wl = (size_t)9 * g->blk_x * g->blk_y, wc = (size_t)9 * (g->blk_x >> lx) * (g->blk_y >> ly);
    auto dst = mvk::frame_alloc(f);
    if (!dst)
      return -2;
    // one device buffer per call: the block table, then (with overlap) the luma and the chroma windows; the table (MBs
    // for small blocks) goes straight from the caller's memory into the staging buffer, the windows (small) after it
    const size_t wbytes = ov ? (wl + (f.planes == 3 ? wc : 0)) * sizeof(short) : 0;
    const size_t woff = (bbytes + 15) & ~(size_t)15;
    const size_t up_bytes = woff + wbytes;
    vkrt::DevBuf up((up_bytes + 3) & ~(size_t)3);
    if (!up)
      return -2;
    cmd.upload(up.p, blocks, bbytes);
    if (ov)
    {
      std::vector<short> win(wbytes / sizeof(short));
      std::memcpy(win.data(), g->win_luma, wl * sizeof(short));
      if (f.planes == 3)
        std::memcpy(win.data() + wl, g->win_chroma, wc * sizeof(short));
      cmd.upload(up.p + woff, win.data(), wbytes);
    }
    CompArgs pl = {};
    pl.maxv = bps == 1 ? 255 : (1 << f.bits_per_sample) - 1;
    pl.lstep_x = g->blk_x - g->ov_x;
    pl.lstep_y = g->blk_y - g->ov_y;
    pl.nbx = g->n_blk_x;
    pl.nby = g->n_blk_y;
    pl.ov = ov;
    pl.L = L;
    pl.blocks = up.p;
    int max_w = 0, max_h = 0;
    const int wbl = g->n_blk_x * (g->blk_x - g->ov_x) + g->ov_x, hbl = g->n_blk_y * (g->blk_y - g->ov_y) + g->ov_y;
    for (int p = 0; p < f.planes; p++)
    {
      const int sx = p ? lx : 0, sy = p ? ly : 0;
      pl.src[p] = fs->plane[p];
      pl.fill[p] = ff->plane[p];
      for (int r = 0; r < nref; r++)
      {
        pl.ref[r * 3 + p] = fr[r]->plane[p];
        if (sp[r])
          for (int s = 0; s < 3; s++)
            pl.sub[(r * 3 + p) * 3 + s] = sp[r]->sub[p][s];
      }
      if (sp[0])
      {
        pl.ew[p] = sp[0]->ew[p];
        pl.eh[p] = sp[0]->eh[p];
        pl.hp[p] = sp[0]->hp[p];
        pl.vp[p] = sp[0]->vp[p];
      }
      pl.dst[p] = dst->plane[p];
      pl.pitch[p] = fs->pitch[p] / bps;
      pl.width[p] = fs->w[p];
      pl.height[p] = fs->h[p];
      pl.bw[p] = g->blk_x >> sx;
      pl.bh[p] = g->blk_y >> sy;
      pl.step_x[p] = (g->blk_x - g->ov_x) >> sx;
      pl.step_y[p] = (g->blk_y - g->ov_y) >> sy;
      pl.log_x[p] = sx;
      pl.log_y[p] = sy;
      pl.wb[p] = wbl >> sx;
      pl.hb[p] = hbl >> sy;
      pl.win[p] = ov ? up.p + woff + (p ? wl * sizeof(short) : 0) : 0;
      pl.act[p] = (plane_mask >> p) & 1;
      max_w = fs->w[p] > max_w ? fs->w[p] : max_w;
      max_h = fs->h[p] > max_h ? fs->h[p] : max_h;
    }
    cmd.barrier();
    cmd.dispatch(k, (max_w + 127) / 128, max_h, f.planes, pl);
    const int rc = cmd.submit_wait();
    if (rc)
      return rc;
    if (any_fresh)
      for (int r = 0; r < nref; r++)
        if (fresh[r])
          subpel_publish(fr[r], sp[r]);
    mvk::frame_put(dst, producer, frame, *out);
    return 0;
  }
}

extern "C" int mvcuda_compensate(const MvCudaBlockGeom *g, unsigned long long src, unsigned long long ref, unsigned long long fill,
  const int *blocks, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  return run<0>(g, 1, src, &ref, 1, fill, blocks, 7, producer, frame, out);
}

extern "C" int mvcuda_degrain(const MvCudaBlockGeom *g, int level, unsigned long long src, const unsigned long long *ref_b,
  const unsigned long long *ref_f, const int *blocks, int plane_mask, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  if (level < 1 || level > MVCUDA_MAX_DEGRAIN || !ref_b || !ref_f)
    return -4;
  unsigned long long refs[MAXR];
  for (int k = 0; k < level; k++)
  {
    refs[2 * k] = ref_b[k];
    refs[2 * k + 1] = ref_f[k];
  }
  return run<1>(g, level, src, refs, 2 * level, src, blocks, plane_mask, producer, frame, out);
}
