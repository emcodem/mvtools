// Resampling with AviSynth+ FilteredResizeV semantics (GPU_PIPELINE_PLAN.md section 15, stage 1: Bob). The programs are
// made on the host by the plugin with AviSynth's own code (cudafilters\Resample.cpp); this file only evaluates them,
// with the integer arithmetic of AviSynth's C resampler (avs_core\filters\resample.cpp, resize_v_c_planar).

#include "mvcuda_internal.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace
{
  struct DevProgram
  {
    int target_size = 0, taps = 0;
    int *offset = nullptr;     // device
    short *coeff = nullptr;    // device, target_size * taps
    ~DevProgram()
    {
      mvc::pipe_flush(); // queued work may still read it (filter teardown only; never released inside a pipeline op)
      mvc::dev_free(offset, sizeof(int) * (size_t)target_size);
      mvc::dev_free(coeff, sizeof(short) * (size_t)target_size * taps);
    }
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

  // one plane of one pass: pitches in samples; the vertical pass reads source row row_start + i * row_step
  struct RsPlane
  {
    const void *src;
    void *dst;
    size_t src_pitch, dst_pitch;
    int width, height;          // destination size
    const int *offset;
    const short *coeff;
    int taps, row_start, row_step;
  };
  struct RsPlanes
  {
    RsPlane p[3];
    int bits;
  };

  // One output sample per thread, blockIdx.z = plane. 8-bit: 14-bit coefficients, rounder 1 << 13; 10..15-bit: 13-bit
  // coefficients, rounder 1 << 12; 16-bit: samples shifted to signed first and back after the sum (AviSynth does this to
  // stay in int16 SIMD).
  template <typename pixel_t>
  __device__ __forceinline__ pixel_t rs_sum(const pixel_t *s, size_t step, const short *c, int taps, int bits)
  {
    int result;
    if constexpr (sizeof(pixel_t) == 1)
    {
      result = 1 << 13;
      for (int i = 0; i < taps; i++)
        result += (int)s[i * step] * c[i];
      result >>= 14;
      result = result > 255 ? 255 : result < 0 ? 0 : result;
    }
    else
    {
      const int limit = (1 << bits) - 1;
      result = 1 << 12;
      for (int i = 0; i < taps; i++)
      {
        int v = s[i * step];
        if (bits == 16)
          v -= 32768;
        result += v * c[i];
      }
      if (bits == 16)
        result += 32768 << 13;
      result >>= 13;
      result = result > limit ? limit : result < 0 ? 0 : result;
    }
    return (pixel_t)result;
  }

  // vertical (resize_v_c_planar): the program runs over the rows
  template <typename pixel_t>
  __global__ void k_resample_v(RsPlanes r)
  {
    const RsPlane &q = r.p[blockIdx.z];
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
    if (x >= q.width || y >= q.height)
      return;
    const size_t step = (size_t)q.row_step * q.src_pitch;
    const pixel_t *s = (const pixel_t *)q.src + (size_t)(q.row_start + q.offset[y] * q.row_step) * q.src_pitch + x;
    ((pixel_t *)q.dst)[(size_t)y * q.dst_pitch + x] = rs_sum<pixel_t>(s, step, q.coeff + (size_t)y * q.taps, q.taps, r.bits);
  }

  // the horizontal counterpart (resize_h_c_planar): the program runs over the columns
  template <typename pixel_t>
  __global__ void k_resample_h(RsPlanes r)
  {
    const RsPlane &q = r.p[blockIdx.z];
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
    if (x >= q.width || y >= q.height)
      return;
    const pixel_t *s = (const pixel_t *)q.src + (size_t)y * q.src_pitch + q.offset[x];
    ((pixel_t *)q.dst)[(size_t)y * q.dst_pitch + x] = rs_sum<pixel_t>(s, 1, q.coeff + (size_t)x * q.taps, q.taps, r.bits);
  }

  // all planes of one pass in one launch
  void rs_launch(const RsPlanes &r, int planes, int bytes_per_sample, bool horizontal, cudaStream_t st)
  {
    int max_w = 0, max_h = 0;
    for (int p = 0; p < planes; p++)
    {
      max_w = r.p[p].width > max_w ? r.p[p].width : max_w;
      max_h = r.p[p].height > max_h ? r.p[p].height : max_h;
    }
    const dim3 grid((max_w + 127) / 128, max_h, planes);
    if (horizontal)
    {
      if (bytes_per_sample == 1)
        k_resample_h<uint8_t><<<grid, 128, 0, st>>>(r);
      else
        k_resample_h<uint16_t><<<grid, 128, 0, st>>>(r);
    }
    else if (bytes_per_sample == 1)
      k_resample_v<uint8_t><<<grid, 128, 0, st>>>(r);
    else
      k_resample_v<uint16_t><<<grid, 128, 0, st>>>(r);
  }

  // checks one pass over all planes of src into dst (dst sized for the pass) and describes it in r; 0 when it fits.
  // Programs go in as raw pointers: they are freed only after pipe_flush.
  int resample_check(const mvc::DevFrame &src, const mvc::DevFrame &dst, const std::shared_ptr<DevProgram> *progs, bool horizontal, RsPlanes &r)
  {
    const int bps = src.f.bytes_per_sample;
    r.bits = src.f.bits_per_sample;
    for (int p = 0; p < src.f.planes; p++)
    {
      const DevProgram &pr = *progs[p];
      if (pr.target_size != (horizontal ? dst.w[p] : dst.h[p]) || (horizontal ? src.h[p] != dst.h[p] : src.w[p] != dst.w[p]))
        return -4;
      r.p[p] = RsPlane{ src.plane[p], dst.plane[p], src.pitch[p] / bps, dst.pitch[p] / bps, dst.w[p], dst.h[p], pr.offset, pr.coeff, pr.taps, 0, 1 };
    }
    return 0;
  }
}
extern "C" int mvcuda_resample_2d(const unsigned long long prog_h[3], const unsigned long long prog_v[3], int h_first,
  unsigned long long src_token, const MvCudaFrameFormat *out_fmt, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  if (!prog_h || !prog_v || !out_fmt || !out)
    return -4;
  if (!mvc::use_device())
    return -1;
  auto src = mvc::frame_get(src_token);
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
  auto dst = mvc::frame_alloc(*out_fmt);
  if (!dst)
    return -2;
  // checked here, issued by the pipeline submitter (mvcuda_internal.h); buffers released here are only reused by work
  // submitted later (stream-ordered reuse), so the closures carry plain pointers
  RsPlanes p1 = {}, p2 = {};
  const int np = sf.planes, bps = sf.bytes_per_sample;
  if (do_h && do_v)
  {
    // intermediate: the size after the first pass
    MvCudaFrameFormat mid = sf;
    if (h_first)
      mid.width = out_fmt->width;
    else
      mid.height = out_fmt->height;
    std::shared_ptr<mvc::DevFrame> tmp = mvc::frame_alloc(mid);
    if (!tmp)
      return -2;
    if (resample_check(*src, *tmp, h_first ? ph : pv, h_first != 0, p1) || resample_check(*tmp, *dst, h_first ? pv : ph, h_first == 0, p2))
      return -4;
    const bool hf = h_first != 0;
    mvc::pipe_submit([=](cudaStream_t st) {
      rs_launch(p1, np, bps, hf, st);
      rs_launch(p2, np, bps, !hf, st);
    });
  }
  else if (do_h || do_v)
  {
    if (resample_check(*src, *dst, do_h ? ph : pv, do_h, p1))
      return -4;
    mvc::pipe_submit([=](cudaStream_t st) { rs_launch(p1, np, bps, do_h, st); });
  }
  else
  {
    void *d = dst->plane[0];
    const void *s = src->plane[0];
    const size_t bytes = dst->bytes; // same format: same layout (frame_alloc)
    mvc::pipe_submit([=](cudaStream_t st) { cudaMemcpyAsync(d, s, bytes, cudaMemcpyDeviceToDevice, st); });
  }
  mvc::frame_put(dst, producer, frame, *out);
  return 0;
}

// ---- QTGMC's back blend (Sbb), see mvcuda.h ---------------------------------------------------------------------------
namespace
{
  const int BB_REFS = 5;

  // sample pointers are bytes, read as pixel_t; pitches in samples
  struct BbParams
  {
    const uint8_t *thin[3], *lossed;  // lossed: luma only
    const uint8_t *ref[BB_REFS][3];
    uint8_t *dst[3], *diff, *mid;     // diff / mid: luma-sized scratch planes, pitch w[0]
    size_t pitch[3];
    int w[3], h[3];
    const int *off1, *off2;           // first / second pass programs (luma)
    const short *c1, *c2;
    int taps1, taps2, h_first, nref, sovs;
    int bits, maxv, half;
  };

  template <typename pixel_t>
  __device__ __forceinline__ int bb_d(const BbParams &b, int x, int y)
  {
    const size_t o = (size_t)y * b.pitch[0] + x;
    const int v = (int)((const pixel_t *)b.thin[0])[o] - (int)((const pixel_t *)b.lossed)[o] + b.half;
    return v < 0 ? 0 : v > b.maxv ? b.maxv : v;
  }

  __device__ __forceinline__ int bb_avg(int a, int c) { return (a + c + 1) >> 1; }

  // mt_makediff(thin, lossed) (luma), then RemoveGrain 12 (borders copied; rg_mode12_sse_16 is the same cascade)
  template <typename pixel_t>
  __global__ void k_bb_diff(BbParams b)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y, w = b.w[0], h = b.h[0];
    if (x >= w || y >= h)
      return;
    int r;
    if (x == 0 || y == 0 || x == w - 1 || y == h - 1)
      r = bb_d<pixel_t>(b, x, y);
    else
    {
      const int a123 = bb_avg(bb_d<pixel_t>(b, x, y - 1), bb_avg(bb_d<pixel_t>(b, x - 1, y - 1), bb_d<pixel_t>(b, x + 1, y - 1)));
      const int a678 = bb_avg(bb_d<pixel_t>(b, x, y + 1), bb_avg(bb_d<pixel_t>(b, x - 1, y + 1), bb_d<pixel_t>(b, x + 1, y + 1)));
      const int a4c5 = bb_avg(bb_d<pixel_t>(b, x, y), bb_avg(bb_d<pixel_t>(b, x - 1, y), bb_d<pixel_t>(b, x + 1, y)));
      int a123678 = bb_avg(a123, a678);
      a123678 = a123678 > 0 ? a123678 - 1 : 0;
      r = bb_avg(a4c5, a123678);
    }
    ((pixel_t *)b.diff)[(size_t)y * w + x] = (pixel_t)r;
  }

  // first resize pass, diff -> mid
  template <typename pixel_t>
  __global__ void k_bb_pass1(BbParams b)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y, w = b.w[0], h = b.h[0];
    if (x >= w || y >= h)
      return;
    const pixel_t *diff = (const pixel_t *)b.diff;
    const pixel_t *s = b.h_first ? diff + (size_t)y * w + b.off1[x] : diff + (size_t)b.off1[y] * w + x;
    ((pixel_t *)b.mid)[(size_t)y * w + x] = rs_sum<pixel_t>(s, b.h_first ? 1 : (size_t)w, b.c1 + (size_t)(b.h_first ? x : y) * b.taps1, b.taps1, b.bits);
  }

  // second pass (luma) and mt_makediff(thin, G); chroma is thin's; then the optional clamp; blockIdx.z = plane
  template <typename pixel_t>
  __global__ void k_bb_final(BbParams b)
  {
    const int p = blockIdx.z, x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
    if (x >= b.w[p] || y >= b.h[p])
      return;
    const size_t o = (size_t)y * b.pitch[p] + x;
    int v = ((const pixel_t *)b.thin[p])[o];
    if (p == 0)
    {
      const int w = b.w[0];
      const pixel_t *mid = (const pixel_t *)b.mid;
      const pixel_t *s = b.h_first ? mid + (size_t)b.off2[y] * w + x : mid + (size_t)y * w + b.off2[x];
      const int g = rs_sum<pixel_t>(s, b.h_first ? (size_t)w : 1, b.c2 + (size_t)(b.h_first ? y : x) * b.taps2, b.taps2, b.bits);
      v = v - g + b.half;
      v = v < 0 ? 0 : v > b.maxv ? b.maxv : v;
    }
    if (b.nref)
    {
      int up = 0, lo = b.maxv;
      for (int j = 0; j < b.nref; j++)
      {
        const int r = ((const pixel_t *)b.ref[j][p])[o];
        up = r > up ? r : up;
        lo = r < lo ? r : lo;
      }
      up = up + b.sovs > b.maxv ? b.maxv : up + b.sovs;
      lo = lo - b.sovs < 0 ? 0 : lo - b.sovs;
      v = v < up ? v : up;
      v = v > lo ? v : lo;
    }
    ((pixel_t *)b.dst[p])[o] = (pixel_t)v;
  }
}

extern "C" int mvcuda_qtgmc_backblend(unsigned long long thin, unsigned long long lossed, unsigned long long prog_h,
  unsigned long long prog_v, int h_first, const unsigned long long *refs, int nref, int sovs, unsigned long long producer, int frame,
  MvCudaTokenHeader *out)
{
  if (!out || nref < 0 || nref > BB_REFS || (nref && !refs) || sovs < 0 || sovs > 65535)
    return -4;
  if (!mvc::use_device())
    return -1;
  auto ft = mvc::frame_get(thin), fl = mvc::frame_get(lossed);
  std::shared_ptr<mvc::DevFrame> fr[BB_REFS];
  if (!ft || !fl)
    return MVCUDA_TOKEN_GONE;
  for (int j = 0; j < nref; j++)
    if (!(fr[j] = mvc::frame_get(refs[j])))
      return MVCUDA_TOKEN_GONE;
  auto same = [&](const mvc::DevFrame &a) {
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
  const size_t tmp = (size_t)w * h * bps;
  auto dst = mvc::frame_alloc(ft->f);
  uint8_t *diff = (uint8_t *)mvc::pipe_alloc(tmp), *mid = (uint8_t *)mvc::pipe_alloc(tmp);
  if (!dst || !diff || !mid)
  {
    mvc::pipe_free(diff, tmp);
    mvc::pipe_free(mid, tmp);
    return -2;
  }
  BbParams b = {};
  int max_w = 0, max_h = 0;
  for (int p = 0; p < ft->f.planes; p++)
  {
    b.thin[p] = (const uint8_t *)ft->plane[p];
    for (int j = 0; j < nref; j++)
      b.ref[j][p] = (const uint8_t *)fr[j]->plane[p];
    b.dst[p] = (uint8_t *)dst->plane[p];
    b.pitch[p] = ft->pitch[p] / bps;
    b.w[p] = ft->w[p];
    b.h[p] = ft->h[p];
    max_w = b.w[p] > max_w ? b.w[p] : max_w;
    max_h = b.h[p] > max_h ? b.h[p] : max_h;
  }
  b.lossed = (const uint8_t *)fl->plane[0];
  b.diff = diff;
  b.mid = mid;
  const DevProgram &first = h_first ? *ph : *pv, &second = h_first ? *pv : *ph;
  b.off1 = first.offset;
  b.c1 = first.coeff;
  b.taps1 = first.taps;
  b.off2 = second.offset;
  b.c2 = second.coeff;
  b.taps2 = second.taps;
  b.h_first = h_first ? 1 : 0;
  b.nref = nref;
  b.sovs = sovs;
  b.bits = bits;
  b.maxv = (1 << bits) - 1;
  b.half = 1 << (bits - 1);
  const int np = ft->f.planes;
  mvc::pipe_submit([=](cudaStream_t st) {
    const dim3 gl((w + 127) / 128, h), gf((max_w + 127) / 128, max_h, np);
    if (bps == 2)
    {
      k_bb_diff<uint16_t><<<gl, 128, 0, st>>>(b);
      k_bb_pass1<uint16_t><<<gl, 128, 0, st>>>(b);
      k_bb_final<uint16_t><<<gf, 128, 0, st>>>(b);
    }
    else
    {
      k_bb_diff<uint8_t><<<gl, 128, 0, st>>>(b);
      k_bb_pass1<uint8_t><<<gl, 128, 0, st>>>(b);
      k_bb_final<uint8_t><<<gf, 128, 0, st>>>(b);
    }
  });
  mvc::pipe_free(diff, tmp); // stream-ordered reuse
  mvc::pipe_free(mid, tmp);
  mvc::frame_put(dst, producer, frame, *out);
  return 0;
}

extern "C" unsigned long long mvcuda_resample_program_create(int target_size, int taps, const int *offset, const short *coeff)
{
  if (target_size <= 0 || taps <= 0 || !offset || !coeff || !mvc::use_device())
    return 0;
  auto p = std::make_shared<DevProgram>();
  p->target_size = target_size;
  p->taps = taps;
  p->offset = (int *)mvc::dev_alloc(sizeof(int) * (size_t)target_size);
  p->coeff = (short *)mvc::dev_alloc(sizeof(short) * (size_t)target_size * taps);
  if (!p->offset || !p->coeff
    || cudaMemcpy(p->offset, offset, sizeof(int) * (size_t)target_size, cudaMemcpyHostToDevice) != cudaSuccess
    || cudaMemcpy(p->coeff, coeff, sizeof(short) * (size_t)target_size * taps, cudaMemcpyHostToDevice) != cudaSuccess)
  {
    cudaGetLastError();
    return 0;
  }
  const unsigned long long h = g_next_program.fetch_add(1);
  std::lock_guard<std::mutex> lk(g_prog_mutex);
  g_programs[h] = p;
  return h;
}

extern "C" void mvcuda_resample_program_free(unsigned long long program)
{
  std::lock_guard<std::mutex> lk(g_prog_mutex);
  g_programs.erase(program);
}

extern "C" int mvcuda_resample_v(const unsigned long long prog[3], unsigned long long src_token, const int row_start[3],
  const int row_step[3], const MvCudaFrameFormat *out_fmt, unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  if (!prog || !row_start || !row_step || !out_fmt || !out)
    return -4;
  if (!mvc::use_device())
    return -1;
  auto src = mvc::frame_get(src_token);
  if (!src)
    return MVCUDA_TOKEN_GONE;
  const MvCudaFrameFormat &sf = src->f;
  if (sf.planes != out_fmt->planes || sf.bytes_per_sample != out_fmt->bytes_per_sample || sf.bits_per_sample != out_fmt->bits_per_sample
    || sf.width != out_fmt->width)
    return -4;
  auto dst = mvc::frame_alloc(*out_fmt);
  if (!dst)
    return -2;
  std::shared_ptr<DevProgram> progs[3];
  for (int p = 0; p < sf.planes; p++)
  {
    progs[p] = program_get(prog[p]);
    if (!progs[p] || progs[p]->target_size != dst->h[p] || row_step[p] < 1 || row_start[p] < 0
      || dst->w[p] != src->w[p])
      return -4;
  }
  RsPlanes r = {};
  r.bits = sf.bits_per_sample;
  const int bps = sf.bytes_per_sample;
  for (int p = 0; p < sf.planes; p++)
    r.p[p] = RsPlane{ src->plane[p], dst->plane[p], src->pitch[p] / bps, dst->pitch[p] / bps, dst->w[p], dst->h[p], progs[p]->offset,
      progs[p]->coeff, progs[p]->taps, row_start[p], row_step[p] };
  const int np = sf.planes;
  mvc::pipe_submit([=](cudaStream_t st) { rs_launch(r, np, bps, false, st); }); // issued by the pipeline submitter (mvcuda_internal.h)
  mvc::frame_put(dst, producer, frame, *out);
  return 0;
}
