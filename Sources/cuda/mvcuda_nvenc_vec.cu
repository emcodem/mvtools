// Motion vectors from the video encoder's motion estimation (NVENC ME-only mode) for the nvencmv plugin's
// CudaNVENCAnalyse: one vector field in MAnalyse's payload format with a single level. NOT MAnalyse's search (an
// encoder's block matching, not bit-exact with anything on the CPU); MAnalyse does not use this unit.
//
// Per call:
// 1. (leased GPU call) the grey pictures NVENC reads, written into the session's NV12 buffers: the luma, or three colour
//    channels (Y, U, V with the chroma at luma size; or R, G, B), 8 bit, optionally downscaled 2x2;
// 2. (this thread; NVENC calls block, so never on the submitter thread) one ME run per channel, plus optionally one run
//    of the first channel with the pictures swapped; each gives vectors on an 8x8 grid of the ME picture, quarter pel;
// 3. (leased GPU call) per grid cell the candidate with the lowest cost: the runs' vectors (and their 8 neighbours'),
//    the negated swapped field; cost = luma + scaled chroma SAD at integer pel + smoothness + consistency terms. Then per
//    block the median (x and y separately) of the chosen cells under it, rounded to 1/pel pixel, clamped to the area
//    MAnalyse's search can reach, and its SAD against the sub-pel planes of the reference's super frame.

#include "mvcuda_internal.h"
#include "mvcuda_nvenc.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace
{
  typedef mvc_pix pix;

  struct NvPlanes
  {
    const pix *p[3];   // integer-pel plane of the super frame (padded)
    int ew[3], hpad[3], vpad[3];
  };

  struct PicArgs
  {
    NvPlanes s;
    int w, h, W, H, half, channels, shift, log_x_uv, log_y_uv, has_chroma;
    uint8_t *dst[3];
    size_t pitch;
  };

  __device__ __forceinline__ int to8(int v, int shift)
  {
    if (shift > 0)
      v = (v + (1 << (shift - 1))) >> shift;
    return min(v, 255);
  }

  // the three channel values (8 bit) of source pixel (x, y), clamped into the picture
  __device__ void chan3(const PicArgs &a, int x, int y, int c[3])
  {
    x = min(max(x, 0), a.w - 1);
    y = min(max(y, 0), a.h - 1);
    const int yy = to8(a.s.p[0][(size_t)(a.s.vpad[0] + y) * a.s.ew[0] + a.s.hpad[0] + x], a.shift);
    if (a.channels == 0 || !a.has_chroma)
    {
      c[0] = yy;
      c[1] = c[2] = 128;
      return;
    }
    const int cx = x >> a.log_x_uv, cy = y >> a.log_y_uv;
    const int u = to8(a.s.p[1][(size_t)(a.s.vpad[1] + cy) * a.s.ew[1] + a.s.hpad[1] + cx], a.shift);
    const int v = to8(a.s.p[2][(size_t)(a.s.vpad[2] + cy) * a.s.ew[2] + a.s.hpad[2] + cx], a.shift);
    if (a.channels == 1)
    {
      c[0] = yy;
      c[1] = u;
      c[2] = v;
      return;
    }
    // BT.709 limited range -> R, G, B
    const float fy = 1.16438f * (yy - 16), fu = u - 128.0f, fv = v - 128.0f;
    c[0] = min(255, max(0, __float2int_rn(fy + 1.79274f * fv)));
    c[1] = min(255, max(0, __float2int_rn(fy - 0.21325f * fu - 0.53291f * fv)));
    c[2] = min(255, max(0, __float2int_rn(fy + 2.11240f * fu)));
  }

  // the grey ME pictures (W x H, even) of one frame: channel k into dst[k]
  __global__ void k_nv_pic(PicArgs a)
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= a.W || y >= a.H)
      return;
    const int nch = a.channels ? 3 : 1;
    int c[3];
    if (a.half)
    {
      int s[3] = { 0, 0, 0 };
      for (int d = 0; d < 4; d++)
      {
        chan3(a, 2 * x + (d & 1), 2 * y + (d >> 1), c);
        for (int k = 0; k < nch; k++)
          s[k] += c[k];
      }
      for (int k = 0; k < nch; k++)
        a.dst[k][(size_t)y * a.pitch + x] = (uint8_t)((s[k] + 2) >> 2);
    }
    else
    {
      chan3(a, x, y, c);
      for (int k = 0; k < nch; k++)
        a.dst[k][(size_t)y * a.pitch + x] = (uint8_t)c[k];
    }
  }

  __device__ __forceinline__ int warp_sum(int v)
  {
    for (int o = 16; o > 0; o >>= 1)
      v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
  }

  __device__ __forceinline__ int scale_chroma(int sad, int scale)
  {
    if (scale == 0)
      return sad;
    return scale > 0 ? sad >> scale : sad << (-scale);
  }

  // quarter pel of the ME picture -> units of 1/den source pixel (s: 1, or 2 when half), rounded half away from zero
  __device__ __forceinline__ int q_to(int q, int s, int den)
  {
    const int num = q * s * den;
    return num >= 0 ? (num + 2) / 4 : -((-num + 2) / 4);
  }

  struct SelArgs
  {
    NvPlanes src, ref;
    int w, h, cs, s, gw, gh, nfields, nb, lambda, fb_penalty, has_chroma, log_x_uv, log_y_uv, chroma_scale;
    const short2 *field[4]; // candidate fields (the swapped one already negated)
    const short2 *rev;      // the swapped field as it came (reference -> source), or null
    short2 *sel;
    int *stats;             // 64 counters or null
  };

  // integer-pel SAD of the w x h area at (x0, y0) of plane p (inside the padding) against the reference moved by (dx, dy),
  // the reference position clamped into the padded plane (eh rows)
  __device__ int sad_int(const NvPlanes &s, const NvPlanes &r, int p, int x0, int y0, int w, int h, int dx, int dy, int eh, int lane)
  {
    const int ew = r.ew[p];
    const int rx = min(max(x0 + dx, 0), ew - w), ry = min(max(y0 + dy, 0), eh - h);
    int acc = 0;
    for (int i = lane; i < w * h; i += 32)
    {
      const int rr = i / w, cc = i - rr * w;
      acc += abs((int)s.p[p][(size_t)(y0 + rr) * s.ew[p] + x0 + cc] - (int)r.p[p][(size_t)(ry + rr) * ew + rx + cc]);
    }
    return warp_sum(acc);
  }

  // One warp per grid cell: the candidate with the lowest cost.
  __global__ void k_nv_select(SelArgs a, int eh0, int eh1)
  {
    const int wib = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int idx = blockIdx.x * 4 + wib;
    if (idx >= a.gw * a.gh)
      return;
    const int gx = idx % a.gw, gy = idx / a.gw;
    const int px = gx * a.cs, py = gy * a.cs;
    const short2 own0 = a.field[0][idx];
    if (px >= a.w || py >= a.h)
    {
      if (lane == 0)
        a.sel[idx] = own0;
      return;
    }
    const int cw = min(a.cs, a.w - px), chh = min(a.cs, a.h - py);
    int medx = 0, medy = 0;
    if (a.lambda)
    {
      int vx[9], vy[9];
      for (int t = 0; t < 9; t++)
      {
        const int ax = min(max(gx + t % 3 - 1, 0), a.gw - 1), ay = min(max(gy + t / 3 - 1, 0), a.gh - 1);
        const short2 v = a.field[0][(size_t)ay * a.gw + ax];
        vx[t] = v.x;
        vy[t] = v.y;
      }
      // lower median of 9 (insertion sort, every lane the same)
      for (int i = 1; i < 9; i++)
        for (int j = i; j > 0 && vx[j - 1] > vx[j]; j--)
        {
          const int t = vx[j]; vx[j] = vx[j - 1]; vx[j - 1] = t;
        }
      for (int i = 1; i < 9; i++)
        for (int j = i; j > 0 && vy[j - 1] > vy[j]; j--)
        {
          const int t = vy[j]; vy[j] = vy[j - 1]; vy[j - 1] = t;
        }
      medx = vx[4];
      medy = vy[4];
    }
    // candidate order: per field the own cell, then the neighbours (raster, own skipped); ties keep the earlier one
    const int OX[9] = { 0, -1, 0, 1, -1, 1, -1, 0, 1 }, OY[9] = { 0, -1, -1, -1, 0, 0, 1, 1, 1 };
    long long best = -1;
    int bi = 0, idx_c = 0;
    short2 bv = own0;
    const int x0 = a.src.hpad[0] + px, y0 = a.src.vpad[0] + py;
    for (int f = 0; f < a.nfields; f++)
      for (int t = 0; t < (a.nb ? 9 : 1); t++, idx_c++)
      {
        const int ax = min(max(gx + OX[t], 0), a.gw - 1), ay = min(max(gy + OY[t], 0), a.gh - 1);
        const short2 m = a.field[f][(size_t)ay * a.gw + ax];
        const int dx = q_to(m.x, a.s, 1), dy = q_to(m.y, a.s, 1);
        long long cost = sad_int(a.src, a.ref, 0, x0, y0, cw, chh, dx, dy, eh0, lane);
        if (a.has_chroma)
        {
          const int ccw = max(cw >> a.log_x_uv, 1), cch = max(chh >> a.log_y_uv, 1);
          const int cx0 = a.src.hpad[1] + (px >> a.log_x_uv), cy0 = a.src.vpad[1] + (py >> a.log_y_uv);
          const int cdx = mvc::mv_to_chroma(dx, a.log_x_uv), cdy = mvc::mv_to_chroma(dy, a.log_y_uv);
          const int sc = sad_int(a.src, a.ref, 1, cx0, cy0, ccw, cch, cdx, cdy, eh1, lane) + sad_int(a.src, a.ref, 2, cx0, cy0, ccw, cch, cdx, cdy, eh1, lane);
          cost += scale_chroma(sc, a.chroma_scale);
        }
        if (a.lambda)
        {
          const long long ddx = (m.x - medx) / 2, ddy = (m.y - medy) / 2;
          cost += ((long long)a.lambda * (ddx * ddx + ddy * ddy)) >> 8;
        }
        if (a.fb_penalty && a.rev)
        {
          const int tx = min(max((gx * 8 + 4 + (m.x >= 0 ? (m.x + 2) / 4 : -((-m.x + 2) / 4))) / 8, 0), a.gw - 1);
          const int ty = min(max((gy * 8 + 4 + (m.y >= 0 ? (m.y + 2) / 4 : -((-m.y + 2) / 4))) / 8, 0), a.gh - 1);
          const short2 r = a.rev[(size_t)ty * a.gw + tx];
          if (abs(m.x + r.x) > 4 || abs(m.y + r.y) > 4)
            cost += a.fb_penalty;
        }
        if (best < 0 || cost < best)
        {
          best = cost;
          bi = idx_c;
          bv = m;
        }
      }
    if (lane == 0)
    {
      a.sel[idx] = bv;
      if (a.stats)
        atomicAdd(&a.stats[min(bi, 63)], 1);
    }
  }

  struct BlkArgs
  {
    int n_blk_x, n_blk_y, bs_x, bs_y, ov_x, ov_y, cw, ch, log_x_uv, log_y_uv;
    int pel, log_pel, ext_w, ext_h, ew[3], hpad[3], vpad[3];
    const pix *src[3];
    const pix *ref[3][4];
    int has_chroma, chroma_scale, cs, s, gw, gh;
    const short2 *sel;
    int *vec; // (vx, vy, sad) per block
  };

  // SAD of the w x h block at (x0, y0) of plane p of the source against the reference at (xp, yp) in 1/pel units
  __device__ int sad_block(const BlkArgs &a, int p, int x0, int y0, int w, int h, int xp, int yp, int lane)
  {
    const int sub = (xp & (a.pel - 1)) | ((yp & (a.pel - 1)) << a.log_pel);
    const pix *rp = a.ref[p][sub];
    const int rx = xp >> a.log_pel, ry = yp >> a.log_pel;
    const int ew = a.ew[p];
    int s = 0;
    for (int i = lane; i < w * h; i += 32)
    {
      const int r = i / w, c = i - r * w;
      s += abs((int)a.src[p][(size_t)(y0 + r) * ew + x0 + c] - (int)rp[(size_t)(ry + r) * ew + rx + c]);
    }
    return warp_sum(s);
  }

  // One warp per block: the lower median (x and y separately) of the chosen cells under the block, at 1/pel, clamped to
  // MAnalyse's search area, then its SAD. Blocks up to 64x64 cover at most 9 x 9 cells of 8 pixels.
  __global__ void k_nv_blocks(BlkArgs a)
  {
    __shared__ short2 cells[4][81];
    const int wib = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int idx = blockIdx.x * 4 + wib;
    if (idx >= a.n_blk_x * a.n_blk_y)
      return;
    const int bx = idx % a.n_blk_x, by = idx / a.n_blk_x;
    const int xi = bx * (a.bs_x - a.ov_x), yi = by * (a.bs_y - a.ov_y);
    const int gx0 = min(xi / a.cs, a.gw - 1), gx1 = min((xi + a.bs_x - 1) / a.cs, a.gw - 1);
    const int gy0 = min(yi / a.cs, a.gh - 1), gy1 = min((yi + a.bs_y - 1) / a.cs, a.gh - 1);
    const int cw_ = gx1 - gx0 + 1, n = min(cw_ * (gy1 - gy0 + 1), 81);
    for (int i = lane; i < n; i += 32)
    {
      const int r = i / cw_, c = i - r * cw_;
      cells[wib][i] = a.sel[(size_t)(gy0 + r) * a.gw + gx0 + c];
    }
    __syncwarp();
    const int k = (n - 1) / 2;
    int med[2] = { 0, 0 };
    for (int comp = 0; comp < 2; comp++)
    {
      int found = 0, val = 0;
      for (int j = lane; j < n; j += 32)
      {
        const int c = comp ? cells[wib][j].y : cells[wib][j].x;
        int lt = 0, le = 0;
        for (int i = 0; i < n; i++)
        {
          const int v = comp ? cells[wib][i].y : cells[wib][i].x;
          lt += v < c;
          le += v <= c;
        }
        if (!found && lt <= k && k < le)
        {
          found = 1;
          val = c;
        }
      }
      const unsigned b = __ballot_sync(0xffffffffu, found);
      med[comp] = __shfl_sync(0xffffffffu, val, __ffs(b) - 1);
    }
    const int x0 = a.hpad[0] + xi, y0 = a.vpad[0] + yi;
    const int dx_min = -a.pel * x0, dy_min = -a.pel * y0;
    const int dx_max = a.pel * (a.ext_w - x0 - a.bs_x), dy_max = a.pel * (a.ext_h - y0 - a.bs_y);
    const int vx = max(dx_min, min(dx_max - 1, q_to(med[0], a.s, a.pel)));
    const int vy = max(dy_min, min(dy_max - 1, q_to(med[1], a.s, a.pel)));
    int sad = sad_block(a, 0, x0, y0, a.bs_x, a.bs_y, x0 * a.pel + vx, y0 * a.pel + vy, lane);
    if (a.has_chroma)
    {
      int sc = 0;
      for (int p = 1; p < 3; p++)
      {
        const int cx0 = a.hpad[p] + (xi >> a.log_x_uv), cy0 = a.vpad[p] + (yi >> a.log_y_uv);
        sc += sad_block(a, p, cx0, cy0, a.cw, a.ch, cx0 * a.pel + mvc::mv_to_chroma(vx, a.log_x_uv), cy0 * a.pel + mvc::mv_to_chroma(vy, a.log_y_uv), lane);
      }
      sad += scale_chroma(sc, a.chroma_scale);
    }
    if (lane == 0)
    {
      a.vec[3 * idx + 0] = vx;
      a.vec[3 * idx + 1] = vy;
      a.vec[3 * idx + 2] = sad;
    }
  }

  int ilog2(int v)
  {
    int l = 0;
    while ((1 << l) < v)
      l++;
    return l;
  }

  void planes_of(const mvc::DevSuper &s, int np, NvPlanes &o)
  {
    std::memset(&o, 0, sizeof o);
    for (int p = 0; p < np; p++)
    {
      o.p[p] = (const pix *)s.pl[p].sub[0];
      o.ew[p] = s.pl[p].d.ew;
      o.hpad[p] = s.pl[p].d.hpad;
      o.vpad[p] = s.pl[p].d.vpad;
    }
  }

  // device buffers of one call, freed on scope exit
  struct Bufs
  {
    struct B
    {
      void *p;
      size_t n;
      bool pinned;
    };
    std::vector<B> v;
    void *dev(size_t n)
    {
      void *p = mvc::dev_alloc(n);
      if (p)
        v.push_back({ p, n, false });
      return p;
    }
    void *pin(size_t n)
    {
      void *p = mvc::pin_alloc(n);
      if (p)
        v.push_back({ p, n, true });
      return p;
    }
    ~Bufs()
    {
      for (auto &b : v)
        b.pinned ? mvc::pin_free(b.p, b.n) : mvc::dev_free(b.p, b.n);
    }
  };
}

static int nvenc_dev(const MvCudaSuperParams *sp, const MvCudaNvencParams *ep, const std::shared_ptr<mvc::DevFrame> &fs,
  const std::shared_ptr<mvc::DevFrame> &fr, const short *hints, int *payload, int payload_words, short *grid_out, int grid_words,
  long long *stats)
{
  if (sp->levels != 1 || (sp->pel != 1 && sp->pel != 2) || ep->preset < 1 || ep->preset > 7 || ep->n_blk_x < 1 || ep->n_blk_y < 1
    || ep->blk_x < 1 || ep->blk_y < 1 || ep->blk_x > 64 || ep->blk_y > 64 || ep->channels < 0 || ep->channels > 2 || ep->hints < 0
    || ep->hints > 4 || ep->qp > 51 || ep->fresh < 0 || ep->fresh > 2 || (ep->fb_penalty && !ep->swap))
    return -4;
  const int nblk = ep->n_blk_x * ep->n_blk_y, words = 3 + 3 * nblk;
  if (payload_words < words)
    return -4;
  const int has_chroma = sp->chroma ? 1 : 0;                   // the chroma planes are there (channels need them)
  const int sad_chroma = has_chroma && ep->sad_chroma ? 1 : 0; // and count in the SADs
  if (ep->channels && !has_chroma)
    return -4;
  const int s = ep->half ? 2 : 1, cs = 8 * s;
  const int W0 = (sp->width + s - 1) / s, H0 = (sp->height + s - 1) / s;
  mvc_nvenc::Config cfg;
  cfg.width = (W0 + 1) & ~1;
  cfg.height = (H0 + 1) & ~1;
  cfg.hevc = ep->codec ? 1 : 0;
  cfg.preset = ep->preset;
  cfg.qp = ep->qp;
  cfg.no_intra = ep->no_intra;
  cfg.p16 = ep->p16;
  cfg.hints = cfg.hevc ? 0 : ep->hints;
  const int nch = ep->channels ? 3 : 1;
  cfg.slots = 2 * nch;
  const int gw = cfg.hevc ? (cfg.width + 7) / 8 : (cfg.width + 15) / 16 * 2;
  const int gh = cfg.hevc ? (cfg.height + 7) / 8 : (cfg.height + 15) / 16 * 2;
  if (grid_out && grid_words < 2 * gw * gh)
    return -4;
  const int nfields = nch + (ep->swap ? 1 : 0);
  const size_t gcells = (size_t)gw * gh, gbytes = gcells * sizeof(short2);

  cudaFree(nullptr); // the primary context current on this thread (NVENC takes it)
  std::string err;
  mvc_nvenc::Session *ses = mvc_nvenc::acquire(cfg, err);
  if (!ses)
  {
    static bool told = false;
    if (!told)
      fprintf(stderr, "mvcuda nvenc: %s\n", err.c_str());
    told = true;
    return MVCUDA_NEEDS_CPU;
  }
  struct Rel
  {
    mvc_nvenc::Session *s;
    ~Rel() { mvc_nvenc::release(s); }
  } rel{ ses };

  Bufs b;
  short2 *h_grid = (short2 *)b.pin(gbytes * (nfields + 1)); // the fields, then the swapped one as it came
  short2 *d_grid = (short2 *)b.dev(gbytes * (nfields + 1));
  short2 *d_sel = (short2 *)b.dev(gbytes);
  const size_t vec_bytes = (size_t)nblk * 3 * sizeof(int);
  int *d_vec = (int *)b.dev(vec_bytes);
  int *h_vec = (int *)b.pin(vec_bytes);
  int *d_stats = stats ? (int *)b.dev(64 * sizeof(int)) : nullptr;
  int *h_stats = stats ? (int *)b.pin(64 * sizeof(int)) : nullptr;
  short2 *h_sel = grid_out ? (short2 *)b.pin(gbytes) : nullptr;
  if (!h_grid || !d_grid || !d_sel || !d_vec || !h_vec || (stats && (!d_stats || !h_stats)) || (grid_out && !h_sel))
    return -2;

  const mvc::DevFrame *in[2] = { fs.get(), fr.get() };
  std::shared_ptr<mvc::DevSuper> s_src, s_ref;
#define NV_CHECK(call) \
  do { \
    cudaError_t e_ = (call); \
    if (e_ != cudaSuccess) { \
      fprintf(stderr, "mvcuda: %s failed: %s\n", #call, cudaGetErrorString(e_)); \
      return -2; \
    } \
  } while (0)
  // 1. super frames and the grey pictures
  int rc = mvc::leased_call(in, 2, [&](cudaStream_t st) -> int {
    int r = mvc::frame_super(&MVC_NS::super_build_dev, fs, *sp, st, s_src);
    if (!r)
      r = mvc::frame_super(&MVC_NS::super_build_dev, fr, *sp, st, s_ref);
    if (r)
      return r;
    PicArgs a = {};
    a.w = sp->width;
    a.h = sp->height;
    a.W = cfg.width;
    a.H = cfg.height;
    a.half = ep->half ? 1 : 0;
    a.channels = ep->channels;
    a.shift = sp->bits_per_pixel > 8 ? sp->bits_per_pixel - 8 : 0;
    a.has_chroma = has_chroma;
    a.log_x_uv = has_chroma ? ilog2(sp->x_ratio_uv) : 0;
    a.log_y_uv = has_chroma ? ilog2(sp->y_ratio_uv) : 0;
    a.pitch = ses->pitch;
    const dim3 bl(32, 8), gr((cfg.width + 31) / 32, (cfg.height + 7) / 8);
    for (int f = 0; f < 2; f++)
    {
      planes_of(f ? *s_ref : *s_src, has_chroma ? 3 : 1, a.s);
      for (int k = 0; k < nch; k++)
        a.dst[k] = ses->pic[f * nch + k];
      k_nv_pic<<<gr, bl, 0, st>>>(a);
      NV_CHECK(cudaGetLastError());
    }
    return 0;
  });
  if (rc)
    return rc;

  // 2. the ME runs (this thread)
  std::vector<mvc_nvenc::Mv> tmp(gcells);
  auto put = [&](int f, bool negate) {
    for (size_t i = 0; i < gcells; i++)
      h_grid[(size_t)f * gcells + i] = negate ? make_short2((short)-tmp[i].x, (short)-tmp[i].y) : make_short2(tmp[i].x, tmp[i].y);
  };
  const mvc_nvenc::Mv *hp = (hints && cfg.hints > 0) ? reinterpret_cast<const mvc_nvenc::Mv *>(hints) : nullptr;
  for (int k = 0; k < nch; k++)
  {
    if (mvc_nvenc::run(ses, k, nch + k, ep->fresh, hp, tmp.data(), err))
    {
      fprintf(stderr, "mvcuda nvenc: %s\n", err.c_str());
      return MVCUDA_NEEDS_CPU;
    }
    put(k, false);
  }
  if (ep->swap)
  {
    if (mvc_nvenc::run(ses, nch, 0, ep->fresh, nullptr, tmp.data(), err))
    {
      fprintf(stderr, "mvcuda nvenc: %s\n", err.c_str());
      return MVCUDA_NEEDS_CPU;
    }
    put(nch, true);
    put(nfields, false);
  }

  // 3. selection, blocks, read back
  rc = mvc::leased_call(in, 2, [&](cudaStream_t st) -> int {
    NV_CHECK(cudaMemcpyAsync(d_grid, h_grid, gbytes * (nfields + (ep->swap ? 1 : 0)), cudaMemcpyHostToDevice, st));
    if (d_stats)
      NV_CHECK(cudaMemsetAsync(d_stats, 0, 64 * sizeof(int), st));
    SelArgs a = {};
    planes_of(*s_src, has_chroma ? 3 : 1, a.src);
    planes_of(*s_ref, has_chroma ? 3 : 1, a.ref);
    a.w = sp->width;
    a.h = sp->height;
    a.cs = cs;
    a.s = s;
    a.gw = gw;
    a.gh = gh;
    a.nfields = nfields;
    a.nb = ep->neighbours ? 1 : 0;
    a.lambda = ep->lambda;
    a.fb_penalty = ep->fb_penalty;
    a.has_chroma = (sad_chroma || ep->channels) ? has_chroma : 0; // colour runs are chosen by the colour SAD
    a.log_x_uv = has_chroma ? ilog2(sp->x_ratio_uv) : 0;
    a.log_y_uv = has_chroma ? ilog2(sp->y_ratio_uv) : 0;
    a.chroma_scale = ep->chroma_sad_scale;
    for (int f = 0; f < nfields; f++)
      a.field[f] = d_grid + (size_t)f * gcells;
    a.rev = ep->swap ? d_grid + (size_t)nfields * gcells : nullptr;
    a.sel = d_sel;
    a.stats = d_stats;
    const int eh0 = s_ref->pl[0].d.eh, eh1 = has_chroma ? s_ref->pl[1].d.eh : 0;
    k_nv_select<<<(unsigned)((gcells + 3) / 4), 128, 0, st>>>(a, eh0, eh1);
    NV_CHECK(cudaGetLastError());
    BlkArgs k = {};
    k.n_blk_x = ep->n_blk_x;
    k.n_blk_y = ep->n_blk_y;
    k.bs_x = ep->blk_x;
    k.bs_y = ep->blk_y;
    k.ov_x = ep->overlap_x;
    k.ov_y = ep->overlap_y;
    k.pel = sp->pel;
    k.log_pel = ilog2(sp->pel);
    k.has_chroma = sad_chroma;
    k.log_x_uv = a.log_x_uv;
    k.log_y_uv = a.log_y_uv;
    k.cw = sad_chroma ? k.bs_x >> k.log_x_uv : 0;
    k.ch = sad_chroma ? k.bs_y >> k.log_y_uv : 0;
    k.chroma_scale = ep->chroma_sad_scale;
    for (int p = 0; p < (has_chroma ? 3 : 1); p++)
    {
      const mvc::DevPlane &dr = s_ref->pl[p];
      k.src[p] = (const pix *)s_src->pl[p].sub[0];
      for (int i = 0; i < 4; i++)
        k.ref[p][i] = (const pix *)dr.sub[i < dr.nsub ? i : 0];
      k.ew[p] = dr.d.ew;
      k.hpad[p] = dr.d.hpad;
      k.vpad[p] = dr.d.vpad;
    }
    k.ext_w = s_ref->pl[0].d.ew;
    k.ext_h = s_ref->pl[0].d.eh;
    k.cs = cs;
    k.s = s;
    k.gw = gw;
    k.gh = gh;
    k.sel = d_sel;
    k.vec = d_vec;
    k_nv_blocks<<<(nblk + 3) / 4, 128, 0, st>>>(k);
    NV_CHECK(cudaGetLastError());
    NV_CHECK(cudaMemcpyAsync(h_vec, d_vec, vec_bytes, cudaMemcpyDeviceToHost, st));
    if (h_stats)
      NV_CHECK(cudaMemcpyAsync(h_stats, d_stats, 64 * sizeof(int), cudaMemcpyDeviceToHost, st));
    if (h_sel)
      NV_CHECK(cudaMemcpyAsync(h_sel, d_sel, gbytes, cudaMemcpyDeviceToHost, st));
    return 0;
  });
#undef NV_CHECK
  if (rc)
    return rc;
  std::memcpy(payload + 3, h_vec, vec_bytes);
  payload[0] = words;
  payload[1] = 1;
  payload[2] = 1 + 3 * nblk;
  if (grid_out)
    std::memcpy(grid_out, h_sel, gbytes);
  if (stats)
    for (int i = 0; i < 64; i++)
      stats[i] += h_stats[i];
  return 0;
}

extern "C" int MVC_API(mvcuda_nvenc_token)(const MvCudaSuperParams *sp, const MvCudaNvencParams *ep, unsigned long long src_token,
  unsigned long long ref_token, const short *hints, int *payload, int payload_words, short *grid_out, int grid_words, long long *stats)
{
  if (!sp || !ep || !payload)
    return -4;
  if (!mvc::use_device())
    return -1;
  auto fs = mvc::frame_get(src_token), fr = mvc::frame_get(ref_token);
  if (!fs || !fr)
    return MVCUDA_TOKEN_GONE;
  if (!mvc::super_fits(*fs, *sp, (int)sizeof(mvc_pix)) || !mvc::super_fits(*fr, *sp, (int)sizeof(mvc_pix)))
    return -4;
  return nvenc_dev(sp, ep, fs, fr, hints, payload, payload_words, grid_out, grid_words, stats);
}
