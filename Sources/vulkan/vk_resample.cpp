// mvtools_vulkan: resampling with AviSynth+ FilteredResizeV / FilteredResizeH semantics and QTGMC's back blend, port of
// Sources/cuda/mvcuda_resample.cu (bit for bit: see that file for the arithmetic). The programs are made on the host by the
// plugin with AviSynth's own code; this file only evaluates them. Kernels: shaders/rs_pass.pix.comp, shaders/rs_bb.pix.comp.

#include "vk_internal.h"
#include "vk_cf_log.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <unordered_map>

namespace
{
  // a resample program on the device; the handle given to the caller is a registry id (a freed or unknown handle is
  // detected, and a program stays alive while a call uses it)
  struct DevProgram
  {
    int target_size = 0, taps = 0;
    vkrt::DevBuf offset;   // int[target_size]
    vkrt::DevBuf coeff;    // short[target_size * taps]
  };

  std::mutex g_prog_mutex;
  std::unordered_map<unsigned long long, std::shared_ptr<DevProgram>> g_programs;
  std::atomic<unsigned long long> g_next_program{ 1 };

  std::shared_ptr<DevProgram> program_get(unsigned long long h)
  {
    std::lock_guard<std::mutex> lk(g_prog_mutex);
    auto it = g_programs.find(h);
    return it == g_programs.end() ? nullptr : it->second;
  }


  // shaders/rs_pass.pix.comp Args
  struct RsArgs
  {
    uint64_t src[3], dst[3], off[3], coeff[3];
    int src_pitch[3], dst_pitch[3], width[3], height[3], taps[3], row_start[3], row_step[3];
    int bits, horizontal, pad0, pad1;
  };

  // records one pass over all planes
  bool rs_record(vkrt::Cmd &cmd, const RsArgs &r, int planes, int bytes_per_sample)
  {
    int max_w = 0, max_h = 0;
    for (int p = 0; p < planes; p++)
    {
      max_w = r.width[p] > max_w ? r.width[p] : max_w;
      max_h = r.height[p] > max_h ? r.height[p] : max_h;
    }
    vkrt::Kernel *k = vkrt::kernel(bytes_per_sample == 1 ? "rs_pass_8" : "rs_pass_16");
    if (!k)
      return false;
    return cmd.dispatch(k, (max_w + 127) / 128, max_h, planes, r);
  }

  // checks one pass over all planes of src into dst (dst sized for the pass) and describes it in r; 0 when it fits
  int resample_check(const mvk::DevFrame &src, const mvk::DevFrame &dst, const std::shared_ptr<DevProgram> *progs, bool horizontal, RsArgs &r)
  {
    const int bps = src.f.bytes_per_sample;
    r = RsArgs{};
    r.bits = src.f.bits_per_sample;
    r.horizontal = horizontal ? 1 : 0;
    for (int p = 0; p < src.f.planes; p++)
    {
      const DevProgram &pr = *progs[p];
      if (pr.target_size != (horizontal ? dst.w[p] : dst.h[p]) || (horizontal ? src.h[p] != dst.h[p] : src.w[p] != dst.w[p]))
        return -4;
      r.src[p] = src.plane[p];
      r.dst[p] = dst.plane[p];
      r.off[p] = pr.offset.p;
      r.coeff[p] = pr.coeff.p;
      r.src_pitch[p] = (int)(src.pitch[p] / bps);
      r.dst_pitch[p] = (int)(dst.pitch[p] / bps);
      r.width[p] = dst.w[p];
      r.height[p] = dst.h[p];
      r.taps[p] = pr.taps;
      r.row_start[p] = 0;
      r.row_step[p] = 1;
    }
    return 0;
  }

}

extern "C" unsigned long long mvcuda_resample_program_create(int target_size, int taps, const int *offset, const short *coeff)
{
  if (target_size <= 0 || taps <= 0 || !offset || !coeff || !vkrt::ready())
    return 0;
  auto p = std::make_shared<DevProgram>();
  p->target_size = target_size;
  p->taps = taps;
  const size_t ob = sizeof(int) * (size_t)target_size, cb = sizeof(short) * (size_t)target_size * taps;
  p->offset = vkrt::DevBuf(ob);
  p->coeff = vkrt::DevBuf((cb + 3) & ~(size_t)3);
  if (!p->offset || !p->coeff)
    return 0;
  vkrt::Cmd cmd;
  cmd.upload(p->offset.p, offset, ob);
  cmd.upload(p->coeff.p, coeff, cb);
  if (cmd.submit_wait())
    return 0;
  const unsigned long long h = g_next_program.fetch_add(1);
  std::lock_guard<std::mutex> lk(g_prog_mutex);
  g_programs[h] = p;
  return h;
}

extern "C" void mvcuda_resample_program_free(unsigned long long program)
{
  std::shared_ptr<DevProgram> keep; // released outside the lock
  std::lock_guard<std::mutex> lk(g_prog_mutex);
  auto it = g_programs.find(program);
  if (it != g_programs.end())
  {
    keep = it->second;
    g_programs.erase(it);
  }
}

static int resample_v_impl(const unsigned long long prog[3], unsigned long long src_token, const int row_start[3],
  const int row_step[3], const MvCudaFrameFormat *out_fmt, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  if (!prog || !row_start || !row_step || !out_fmt || !out)
    return -4;
  if (!vkrt::ready())
    return -1;
  auto src = mvk::frame_get(src_token);
  if (!src)
    return MVCUDA_TOKEN_GONE;
  const MvCudaFrameFormat &sf = src->f;
  if (sf.planes != out_fmt->planes || sf.bytes_per_sample != out_fmt->bytes_per_sample || sf.bits_per_sample != out_fmt->bits_per_sample
    || sf.width != out_fmt->width)
    return -4;
  auto dst = mvk::frame_alloc(*out_fmt);
  if (!dst)
    return -2;
  std::shared_ptr<DevProgram> progs[3];
  for (int p = 0; p < sf.planes; p++)
  {
    progs[p] = program_get(prog[p]);
    if (!progs[p] || progs[p]->target_size != dst->h[p] || row_step[p] < 1 || row_start[p] < 0 || dst->w[p] != src->w[p])
      return -4;
  }
  RsArgs r = {};
  r.bits = sf.bits_per_sample;
  r.horizontal = 0;
  const int bps = sf.bytes_per_sample;
  for (int p = 0; p < sf.planes; p++)
  {
    r.src[p] = src->plane[p];
    r.dst[p] = dst->plane[p];
    r.off[p] = progs[p]->offset.p;
    r.coeff[p] = progs[p]->coeff.p;
    r.src_pitch[p] = (int)(src->pitch[p] / bps);
    r.dst_pitch[p] = (int)(dst->pitch[p] / bps);
    r.width[p] = dst->w[p];
    r.height[p] = dst->h[p];
    r.taps[p] = progs[p]->taps;
    r.row_start[p] = row_start[p];
    r.row_step[p] = row_step[p];
  }
  vkrt::Cmd cmd;
  if (!rs_record(cmd, r, sf.planes, bps))
    return -2;
  const int rc = cmd.submit_wait();
  if (rc)
    return rc;
  mvk::frame_put(dst, producer, frame, *out);
  return 0;
}

static int resample_2d_impl(const unsigned long long prog_h[3], const unsigned long long prog_v[3], int h_first,
  unsigned long long src_token, const MvCudaFrameFormat *out_fmt, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  if (!prog_h || !prog_v || !out_fmt || !out)
    return -4;
  if (!vkrt::ready())
    return -1;
  auto src = mvk::frame_get(src_token);
  if (!src)
    return MVCUDA_TOKEN_GONE;
  const MvCudaFrameFormat &sf = src->f;
  if (sf.planes != out_fmt->planes || sf.bytes_per_sample != out_fmt->bytes_per_sample || sf.bits_per_sample != out_fmt->bits_per_sample)
    return -4;
  const bool do_h = prog_h[0] != 0, do_v = prog_v[0] != 0;
  std::shared_ptr<DevProgram> ph[3], pv[3];
  for (int p = 0; p < sf.planes; p++)
  {
    if (do_h && !(ph[p] = program_get(prog_h[p])))
      return -4;
    if (do_v && !(pv[p] = program_get(prog_v[p])))
      return -4;
  }
  auto dst = mvk::frame_alloc(*out_fmt);
  if (!dst)
    return -2;
  const int np = sf.planes, bps = sf.bytes_per_sample;
  vkrt::Cmd cmd;
  std::shared_ptr<mvk::DevFrame> tmp; // intermediate, alive until submit_wait returned
  if (do_h && do_v)
  {
    // intermediate: the size after the first pass
    MvCudaFrameFormat mid = sf;
    if (h_first)
      mid.width = out_fmt->width;
    else
      mid.height = out_fmt->height;
    tmp = mvk::frame_alloc(mid);
    if (!tmp)
      return -2;
    RsArgs p1, p2;
    if (resample_check(*src, *tmp, h_first ? ph : pv, h_first != 0, p1) || resample_check(*tmp, *dst, h_first ? pv : ph, h_first == 0, p2))
      return -4;
    if (!rs_record(cmd, p1, np, bps) || !rs_record(cmd, p2, np, bps))
      return -2;
  }
  else if (do_h || do_v)
  {
    RsArgs p1;
    if (resample_check(*src, *dst, do_h ? ph : pv, do_h, p1))
      return -4;
    if (!rs_record(cmd, p1, np, bps))
      return -2;
  }
  else
    cmd.copy(dst->plane[0], src->plane[0], dst->bytes); // same format: same layout (frame_alloc)
  const int rc = cmd.submit_wait();
  if (rc)
    return rc;
  mvk::frame_put(dst, producer, frame, *out);
  return 0;
}

// ---- QTGMC's back blend (Sbb), see mvcuda.h ---------------------------------------------------------------------------
namespace
{
  const int BB_REFS = 5;

  // shaders/rs_bb.pix.comp Args
  struct BbArgs
  {
    uint64_t thin[3], ref[BB_REFS][3], dst[3];
    uint64_t lossed, diff, mid, off1, off2, c1, c2;
    int pitch[3], w[3], h[3];
    int taps1, taps2, h_first, nref, sovs, bits, maxv, half_;
  };
}

static int qtgmc_backblend_impl(unsigned long long thin, unsigned long long lossed, unsigned long long prog_h,
  unsigned long long prog_v, int h_first, const unsigned long long *refs, int nref, int sovs, unsigned long long producer, int frame,
  MvCudaTokenHeader *out)
{
  if (!out || nref < 0 || nref > BB_REFS || (nref && !refs) || sovs < 0 || sovs > 65535)
    return -4;
  if (!vkrt::ready())
    return -1;
  auto ft = mvk::frame_get(thin), fl = mvk::frame_get(lossed);
  std::shared_ptr<mvk::DevFrame> fr[BB_REFS];
  if (!ft || !fl)
    return MVCUDA_TOKEN_GONE;
  for (int j = 0; j < nref; j++)
    if (!(fr[j] = mvk::frame_get(refs[j])))
      return MVCUDA_TOKEN_GONE;
  auto same = [&](const mvk::DevFrame &a) {
    return a.f.width == ft->f.width && a.f.height == ft->f.height && a.f.planes == ft->f.planes
      && a.f.bytes_per_sample == ft->f.bytes_per_sample && a.f.bits_per_sample == ft->f.bits_per_sample
      && a.f.x_ratio_uv == ft->f.x_ratio_uv && a.f.y_ratio_uv == ft->f.y_ratio_uv;
  };
  const int bps = ft->f.bytes_per_sample, bits = ft->f.bits_per_sample;
  if (!((bps == 1 && bits == 8) || (bps == 2 && bits > 8 && bits <= 16)) || !same(*fl))
    return -4;
  for (int j = 0; j < nref; j++)
    if (!same(*fr[j]))
      return -4;
  auto ph = program_get(prog_h), pv = program_get(prog_v);
  if (!ph || !pv || ph->target_size != ft->w[0] || pv->target_size != ft->h[0])
    return -4;
  const int w = ft->w[0], h = ft->h[0];
  const size_t tmp = ((size_t)w * h * bps + 3) & ~(size_t)3;
  auto dst = mvk::frame_alloc(ft->f);
  vkrt::DevBuf diff(tmp), mid(tmp);
  if (!dst || !diff || !mid)
    return -2;
  BbArgs b = {};
  int max_w = 0, max_h = 0;
  for (int p = 0; p < ft->f.planes; p++)
  {
    b.thin[p] = ft->plane[p];
    for (int j = 0; j < nref; j++)
      b.ref[j][p] = fr[j]->plane[p];
    b.dst[p] = dst->plane[p];
    b.pitch[p] = (int)(ft->pitch[p] / bps);
    b.w[p] = ft->w[p];
    b.h[p] = ft->h[p];
    max_w = b.w[p] > max_w ? b.w[p] : max_w;
    max_h = b.h[p] > max_h ? b.h[p] : max_h;
  }
  b.lossed = fl->plane[0];
  b.diff = diff.p;
  b.mid = mid.p;
  const DevProgram &first = h_first ? *ph : *pv, &second = h_first ? *pv : *ph;
  b.off1 = first.offset.p;
  b.c1 = first.coeff.p;
  b.taps1 = first.taps;
  b.off2 = second.offset.p;
  b.c2 = second.coeff.p;
  b.taps2 = second.taps;
  b.h_first = h_first ? 1 : 0;
  b.nref = nref;
  b.sovs = sovs;
  b.bits = bits;
  b.maxv = (1 << bits) - 1;
  b.half_ = 1 << (bits - 1);
  const char *name = bps == 2 ? "rs_bb_16" : "rs_bb_8";
  vkrt::Kernel *k0 = vkrt::kernel(name, { 0 }), *k1 = vkrt::kernel(name, { 1 }), *k2 = vkrt::kernel(name, { 2 });
  if (!k0 || !k1 || !k2)
    return -2;
  vkrt::Cmd cmd;
  cmd.dispatch(k0, (w + 127) / 128, h, 1, b);
  cmd.dispatch(k1, (w + 127) / 128, h, 1, b);
  cmd.dispatch(k2, (max_w + 127) / 128, max_h, ft->f.planes, b);
  const int rc = cmd.submit_wait();
  if (rc)
    return rc;
  mvk::frame_put(dst, producer, frame, *out);
  return 0;
}

// ---- exported entry points (counted, see vk_cf_log.h) ----
extern "C" int mvcuda_resample_v(const unsigned long long prog[3], unsigned long long src_token, const int row_start[3],
  const int row_step[3], const MvCudaFrameFormat *out_fmt, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  static mvk::CallLog log("resample_v");
  return log(resample_v_impl(prog, src_token, row_start, row_step, out_fmt, producer, frame, out));
}

extern "C" int mvcuda_resample_2d(const unsigned long long prog_h[3], const unsigned long long prog_v[3], int h_first,
  unsigned long long src_token, const MvCudaFrameFormat *out_fmt, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  static mvk::CallLog log("resample_2d");
  return log(resample_2d_impl(prog_h, prog_v, h_first, src_token, out_fmt, producer, frame, out));
}

extern "C" int mvcuda_qtgmc_backblend(unsigned long long thin, unsigned long long lossed, unsigned long long prog_h,
  unsigned long long prog_v, int h_first, const unsigned long long *refs, int nref, int sovs, unsigned long long producer, int frame,
  MvCudaTokenHeader *out)
{
  static mvk::CallLog log("qtgmc_backblend");
  return log(qtgmc_backblend_impl(thin, lossed, prog_h, prog_v, h_first, refs, nref, sovs, producer, frame, out));
}
