// MCompensate and MDegrain1..6 on device frames (8 to 16 bit, pel 1 or 2, one-level super), bit for bit with MVTools'
// MVCompensate::compensate_slice_overlap / _normal and MVDegrainX::GetFrame (Degrain1to6_C, Overlaps_C, Short2Bytes,
// Short2Bytes_Int32toWord16):
// - a block's samples: compensation takes the reference block at the vector (or the source block when the block is
//   bad), degrain forms (src * WSrc + sum(refF * WF + refB * WB) + 128) >> 8 with the host's normalised weights (they sum
//   to 256, so the 8-bit SIMD code's 16-bit sums cannot overflow and equal the C code);
// - with overlap every block adds (sample * window + 32) >> 6 to a 16-bit sum, which becomes min(255, (sum + 16) >> 5)
//   (Short2Bytes_sse2: arithmetic shift and unsigned saturation); above 8 bit the sum of sample * window is a 32-bit int
//   and becomes min((sum + 1024) >> 11, max pixel value); without overlap the block sample is the output;
// - the area right of / below the block grid is copied from the fill frame.
// The super frame of a pel-1 one-level MSuper is the picture with replicated edges (PadReferenceFrame), so a reference
// sample at a vector reaching into the padding is the clamped picture sample. With pel 2 a block starts at
// MVPlane::GetPointer(blx, bly): sub-pel plane (blx & 1) | (bly & 1) << 1 at (blx >> 1, bly >> 1); plane 0 is the padded
// picture, planes 1..3 the Wiener planes of the padded picture (Padding + Refine with sharp = 2: horizontal, vertical,
// horizontal of the vertical one, MVPlane.cpp / Interpolation.cpp, clamped to the max pixel value), made once per device
// frame (frame_subpel).
// One thread per output sample sums the (at most 2 x 2) blocks covering it; the sum is integer, so the order does not matter.

#include "mvcuda_internal.h"
#include "mvcuda_vec4.cuh"

#include <cstdint>
#include <cstring>
#include <vector>

namespace
{
  const int MAXR = 2 * MVCUDA_MAX_DEGRAIN; // reference frames of one call

  // sample pointers are bytes; the kernels read them as T (uint8_t or uint16_t)
  struct CompPlanes
  {
    const uint8_t *src[3], *fill[3];
    const uint8_t *ref[MAXR][3];      // compensate: ref[0]; degrain: ref[2k] backward, ref[2k + 1] forward of radius k + 1
    const uint8_t *sub[MAXR][3][3];   // pel 2: the Wiener planes of each reference (DevSubpel::sub)
    uint8_t *dst[3];
    size_t pitch[3];                  // in samples; all frames of a call have one format (packed rows)
    int maxv;                         // max pixel value (255 for 8 bit)
    int width[3], height[3];
    int bw[3], bh[3], step_x[3], step_y[3], log_x[3], log_y[3];
    int lstep_x, lstep_y;             // luma block step: block origins are pos_to_chroma(luma origin * pel, vector, chroma shift)
    int wb[3], hb[3];                 // width / height of the block grid in this plane
    int ew[3], eh[3], hp[3], vp[3];   // pel 2: padded sub-pel plane geometry
    const short *win[3];              // overlap windows of this plane (luma or chroma set), nullptr without overlap
    int active[3];
    int grp4[3];                      // block step and size multiples of 4: k_blocks' 4-sample groups
  };

  template <class T>
  __device__ __forceinline__ int at(const uint8_t *p, size_t pitch, int x, int y)
  {
    return ((const T *)p)[(size_t)y * pitch + x];
  }

  template <class T>
  __device__ __forceinline__ int px(const uint8_t *p, size_t pitch, int w, int h, int x, int y)
  {
    x = x < 0 ? 0 : x >= w ? w - 1 : x;
    y = y < 0 ? 0 : y >= h ? h - 1 : y;
    return at<T>(p, pitch, x, y);
  }

  // sample (i, j) of the reference block that starts at (X, Y) (plane coordinates in 1 / PEL units) of reference r
  template <class T, int PEL>
  __device__ __forceinline__ int ref_at(const CompPlanes &pl, int r, int p, int X, int Y, int i, int j)
  {
    if (PEL == 1)
      return px<T>(pl.ref[r][p], pl.pitch[p], pl.width[p], pl.height[p], X + i, Y + j);
    const int idx = (X & 1) | ((Y & 1) << 1);
    const int x = (X >> 1) + i, y = (Y >> 1) + j;
    if (idx == 0)
      return px<T>(pl.ref[r][p], pl.pitch[p], pl.width[p], pl.height[p], x, y);
    const int ew = pl.ew[p], eh = pl.eh[p];
    int ex = x + pl.hp[p], ey = y + pl.vp[p];
    ex = ex < 0 ? 0 : ex >= ew ? ew - 1 : ex; // vectors stay inside the padding; clamped only for safety
    ey = ey < 0 ? 0 : ey >= eh ? eh - 1 : ey;
    return at<T>(pl.sub[r][p][idx - 1], ew, ex, ey);
  }

  // the blocks along one axis covering coordinate v: [*b0, *b1]
  __device__ __forceinline__ void cover(int v, int step, int bsize, int nb, int &b0, int &b1)
  {
    int hi = v / step;
    if (hi > nb - 1)
      hi = nb - 1;
    int lo = hi - 1;
    if (lo < 0 || v >= lo * step + bsize)
      lo = hi;
    b0 = lo;
    b1 = hi;
  }

  __device__ __forceinline__ int window_index(int bx, int by, int nbx, int nby)
  {
    const int wby = (by == 0) ? 0 : (by == nby - 1) ? 6 : 3;
    const int wbx = (bx == 0) ? 0 : (bx == nbx - 1) ? 2 : 1;
    return wby + wbx;
  }

  // the value of block b at sample (i, j) of the block, plane sample (x, y): compensate (MODE 0) or degrain of radius L
  template <class T, int MODE, int PEL>
  __device__ __forceinline__ int block_value(const CompPlanes &pl, const int *b, int bx, int by, int p, int x, int y, int i, int j, int L)
  {
    // block origin in the plane: pos_to_chroma(luma origin * pel, vector, chroma shift), as the CPU since 2.7.48 (#59)
    const int lx = bx * pl.lstep_x * PEL, ly = by * pl.lstep_y * PEL;
    if (MODE == 0)
    {
      if (b[2])
        return ref_at<T, PEL>(pl, 0, p, mvc::pos_to_chroma(lx, b[0], pl.log_x[p]), mvc::pos_to_chroma(ly, b[1], pl.log_y[p]), i, j);
      return at<T>(pl.src[p], pl.pitch[p], x, y);
    }
    const int *w = b + 4 * L + (p == 0 ? 0 : 1 + 2 * L); // WSrc, WRefB[L], WRefF[L]
    int s = at<T>(pl.src[p], pl.pitch[p], x, y) * w[0];
    for (int k = 0; k < L; k++)
    {
      const int *mv = b + 4 * k;
      const int rb = ref_at<T, PEL>(pl, 2 * k, p, mvc::pos_to_chroma(lx, mv[0], pl.log_x[p]), mvc::pos_to_chroma(ly, mv[1], pl.log_y[p]), i, j);
      const int rf = ref_at<T, PEL>(pl, 2 * k + 1, p, mvc::pos_to_chroma(lx, mv[2], pl.log_x[p]), mvc::pos_to_chroma(ly, mv[3], pl.log_y[p]), i, j);
      s += rf * w[1 + L + k] + rb * w[1 + k];
    }
    return (s + 128) >> 8;
  }

  // ref_at for samples i0 .. i0 + 3: one ld4 where none of them needs the clamp, else ref_at per sample
  template <class T, int PEL>
  __device__ __forceinline__ void ref4(const CompPlanes &pl, int r, int p, int X, int Y, int i0, int j, int v[4])
  {
    const T *q = nullptr;
    if (PEL == 1 || ((X | Y) & 1) == 0)
    {
      const int x = (PEL == 1 ? X : X >> 1) + i0, y = (PEL == 1 ? Y : Y >> 1) + j;
      if (x >= 0 && x + 3 < pl.width[p] && y >= 0 && y < pl.height[p])
        q = (const T *)pl.ref[r][p] + (size_t)y * pl.pitch[p] + x;
    }
    else
    {
      const int ex = (X >> 1) + i0 + pl.hp[p], ey = (Y >> 1) + j + pl.vp[p];
      if (ex >= 0 && ex + 3 < pl.ew[p] && ey >= 0 && ey < pl.eh[p])
        q = (const T *)pl.sub[r][p][((X & 1) | ((Y & 1) << 1)) - 1] + (size_t)ey * pl.ew[p] + ex;
    }
    if (q)
      mvc::ld4<T>(q, v);
    else
    {
      #pragma unroll
      for (int k = 0; k < 4; k++)
        v[k] = ref_at<T, PEL>(pl, r, p, X, Y, i0 + k, j);
    }
  }

  // block_value for the 4 samples x0 .. x0 + 3 (block samples i0 .. i0 + 3), sv = their source samples
  template <class T, int MODE, int PEL>
  __device__ __forceinline__ void block_value4(const CompPlanes &pl, const int *b, int bx, int by, int p, const int sv[4], int i0, int j, int L, int v[4])
  {
    const int lx = bx * pl.lstep_x * PEL, ly = by * pl.lstep_y * PEL;
    if (MODE == 0)
    {
      if (b[2])
        ref4<T, PEL>(pl, 0, p, mvc::pos_to_chroma(lx, b[0], pl.log_x[p]), mvc::pos_to_chroma(ly, b[1], pl.log_y[p]), i0, j, v);
      else
        for (int k = 0; k < 4; k++)
          v[k] = sv[k];
      return;
    }
    const int *w = b + 4 * L + (p == 0 ? 0 : 1 + 2 * L); // WSrc, WRefB[L], WRefF[L]
    int s[4];
    #pragma unroll
    for (int k = 0; k < 4; k++)
      s[k] = sv[k] * w[0];
    for (int m = 0; m < L; m++)
    {
      const int *mv = b + 4 * m;
      int rb[4], rf[4];
      ref4<T, PEL>(pl, 2 * m, p, mvc::pos_to_chroma(lx, mv[0], pl.log_x[p]), mvc::pos_to_chroma(ly, mv[1], pl.log_y[p]), i0, j, rb);
      ref4<T, PEL>(pl, 2 * m + 1, p, mvc::pos_to_chroma(lx, mv[2], pl.log_x[p]), mvc::pos_to_chroma(ly, mv[3], pl.log_y[p]), i0, j, rf);
      const int wb = w[1 + m], wf = w[1 + L + m];
      #pragma unroll
      for (int k = 0; k < 4; k++)
        s[k] += rf[k] * wf + rb[k] * wb;
    }
    #pragma unroll
    for (int k = 0; k < 4; k++)
      v[k] = (s[k] + 128) >> 8;
  }

  // the overlap sum's final scaling (8 bit: the CPU's signed 16-bit sum, then packus; 16-bit containers: 32-bit sum)
  template <class T>
  __device__ __forceinline__ T ov_out(int acc, int maxv)
  {
    if (sizeof(T) == 1)
    {
      int r = (short)(acc + 16) >> 5; // _mm_srai_epi16 on the signed 16-bit sum, then packus
      return (T)(r < 0 ? 0 : r > 255 ? 255 : r);
    }
    int r = (acc + 1024) >> 11; // Short2Bytes_Int32toWord16(_sse4): srai, packus, min with the max pixel value
    return (T)(r < 0 ? 0 : r > maxv ? maxv : r);
  }
  template <class T>
  __device__ __forceinline__ int ov_add(int acc, int v, int wv)
  {
    if (sizeof(T) == 1)
      return (int)(short)(acc + ((v * wv + 32) >> 6)); // 16-bit sum, as the CPU's short buffer
    return acc + v * wv;                               // 32-bit sum (Overlaps_C<uint16_t>)
  }

  // MODE 0: compensate (blocks: mvx, mvy, use_ref), MODE 1: degrain of radius L (blocks: see mvcuda_degrain).
  // 4 neighbouring samples per thread: with block steps and sizes that are multiples of 4 (CompPlanes::grp4) the 4 are
  // covered by the same blocks, so the cover, the block entry and the block origins are computed once for the 4; the
  // sums per sample are those of the single-sample path (blocks_px), which takes the planes without grp4 and the edges.
  constexpr int BLK_NT = 128;
  template <class T, int MODE, int PEL>
  __device__ __forceinline__ void blocks_px(const CompPlanes &pl, const int *blocks, int nbx, int nby, int ov, int L, int x, int y, int p);

  template <class T, int MODE, int PEL>
  __global__ void __launch_bounds__(BLK_NT) k_blocks(CompPlanes pl, const int *blocks, int nbx, int nby, int ov, int L)
  {
    const int x0 = (blockIdx.x * BLK_NT + threadIdx.x) * 4, y = blockIdx.y, p = blockIdx.z;
    if (x0 >= pl.width[p] || y >= pl.height[p])
      return;
    if (!pl.active[p] || !pl.grp4[p] || x0 + 4 > pl.wb[p] || x0 + 4 > pl.width[p] || y >= pl.hb[p])
    {
      for (int k = 0; k < 4 && x0 + k < pl.width[p]; k++)
        blocks_px<T, MODE, PEL>(pl, blocks, nbx, nby, ov, L, x0 + k, y, p);
      return;
    }
    const int bw = pl.bw[p], bh = pl.bh[p], sx = pl.step_x[p], sy = pl.step_y[p];
    const int stride = MODE == 0 ? 3 : 4 * L + 2 * (1 + 2 * L);
    int bx0, bx1, by0, by1;
    cover(x0, sx, bw, nbx, bx0, bx1);
    cover(y, sy, bh, nby, by0, by1);
    int acc[4] = { 0, 0, 0, 0 }, single[4] = { 0, 0, 0, 0 }, sv[4];
    mvc::ld4<T>((const T *)pl.src[p] + (size_t)y * pl.pitch[p] + x0, sv);
    for (int by = by0; by <= by1; by++)
      for (int bx = bx0; bx <= bx1; bx++)
      {
        const int i0 = x0 - bx * sx, j = y - by * sy;
        if (i0 < 0 || i0 >= bw || j < 0 || j >= bh) // (i0 and bw are multiples of 4: the whole group or none)
          continue;
        const int *b = blocks + stride * (by * nbx + bx);
        int v[4];
        block_value4<T, MODE, PEL>(pl, b, bx, by, p, sv, i0, j, L, v);
        if (ov)
        {
          // the 4 window weights: 8 bytes, aligned (window offsets and j * bw + i0 are multiples of 4 samples)
          const short *wr = pl.win[p] + window_index(bx, by, nbx, nby) * bw * bh + j * bw + i0;
          int wv[4];
          if (((size_t)wr & 7) == 0)
          {
            const uint2 ww = *(const uint2 *)wr;
            wv[0] = (short)(ww.x & 0xffffu);
            wv[1] = (short)(ww.x >> 16);
            wv[2] = (short)(ww.y & 0xffffu);
            wv[3] = (short)(ww.y >> 16);
          }
          else
          {
            #pragma unroll
            for (int k = 0; k < 4; k++)
              wv[k] = wr[k];
          }
          #pragma unroll
          for (int k = 0; k < 4; k++)
            acc[k] = ov_add<T>(acc[k], v[k], wv[k]);
        }
        else
        {
          #pragma unroll
          for (int k = 0; k < 4; k++)
            single[k] = v[k];
        }
      }
    T *d = (T *)pl.dst[p] + (size_t)y * pl.pitch[p] + x0;
    #pragma unroll
    for (int k = 0; k < 4; k++)
      d[k] = ov ? ov_out<T>(acc[k], pl.maxv) : (T)single[k];
  }

  template <class T, int MODE, int PEL>
  __device__ __forceinline__ void blocks_px(const CompPlanes &pl, const int *blocks, int nbx, int nby, int ov, int L, int x, int y, int p)
  {
    T *d = (T *)pl.dst[p] + (size_t)y * pl.pitch[p] + x;
    if (!pl.active[p] || x >= pl.wb[p] || y >= pl.hb[p])
    {
      *d = (T)at<T>(pl.fill[p], pl.pitch[p], x, y);
      return;
    }
    const int bw = pl.bw[p], bh = pl.bh[p], sx = pl.step_x[p], sy = pl.step_y[p];
    const int stride = MODE == 0 ? 3 : 4 * L + 2 * (1 + 2 * L);
    int bx0, bx1, by0, by1;
    cover(x, sx, bw, nbx, bx0, bx1);
    cover(y, sy, bh, nby, by0, by1);
    int acc = 0;
    int single = 0;
    for (int by = by0; by <= by1; by++)
      for (int bx = bx0; bx <= bx1; bx++)
      {
        const int i = x - bx * sx, j = y - by * sy;
        if (i < 0 || i >= bw || j < 0 || j >= bh)
          continue;
        const int *b = blocks + stride * (by * nbx + bx);
        const int v = block_value<T, MODE, PEL>(pl, b, bx, by, p, x, y, i, j, L);
        if (ov)
          acc = ov_add<T>(acc, v, pl.win[p][window_index(bx, by, nbx, nby) * bw * bh + j * bw + i]);
        else
          single = v;
      }
    *d = ov ? ov_out<T>(acc, pl.maxv) : (T)single;
  }

  // ---- pel-2 sub-pel planes (MSuper sharp = 2: HorizontalWiener / VerticalWiener over the padded plane) ----------------
  struct SubBuild
  {
    const uint8_t *src[3];
    size_t ps[3];                     // in samples
    int w[3], h[3], hp[3], vp[3], ew[3], eh[3];
    uint8_t *dst[3][3];
    int maxv;
  };

  __device__ __forceinline__ int wiener6(int a, int b, int c, int d, int e, int f, int maxv)
  {
    const int v = (a + (-b + (c << 2) + (d << 2) - e) * 5 + f + 16) >> 5;
    return min(maxv, max(0, v));
  }

  // one sample of a Wiener pass at position t of a line of n samples, s(k) = sample k of the line (Interpolation.cpp: two
  // samples averaged at the first two and the last four positions, the last one copied)
  template <class S>
  __device__ __forceinline__ int wiener_at(int t, int n, int maxv, S s)
  {
    if (t < 2 || (t >= n - 4 && t < n - 1))
      return (s(t) + s(t + 1) + 1) >> 1;
    if (t >= n - 1)
      return s(t);
    return wiener6(s(t - 2), s(t - 1), s(t), s(t + 1), s(t + 2), s(t + 3), maxv);
  }

  // the three sub-pel planes of one padded-plane row segment: 0 horizontal, 1 vertical, 2 horizontal of vertical.
  // blockIdx.z = plane, SUBPEL_NT threads per CTA. The segment's padded source row and its vertical pass (with the
  // columns a horizontal pass reads on each side) go to shared memory once, so plane 2 takes the finished vertical
  // values instead of recomputing the vertical pass for each of its six taps (same arithmetic).
  constexpr int SUBPEL_NT = 128;
  template <class T>
  __global__ void __launch_bounds__(SUBPEL_NT) k_subpel(SubBuild b)
  {
    const int p = blockIdx.z;
    const int x0 = blockIdx.x * SUBPEL_NT, y = blockIdx.y;
    const int ew = b.ew[p], eh = b.eh[p];
    if (x0 >= ew || y >= eh)
      return; // the whole CTA
    const T *src = (const T *)b.src[p];
    const size_t ps = b.ps[p];
    const int w = b.w[p], h = b.h[p], hp = b.hp[p], vp = b.vp[p], maxv = b.maxv;
    auto P = [&](int px_, int py_) {
      const int cx = min(max(px_ - hp, 0), w - 1), cy = min(max(py_ - vp, 0), h - 1);
      return (int)src[(size_t)cy * ps + cx];
    };
    // tile entry i = column x0 - 2 + i; a horizontal pass at column x reads columns x - 2 .. x + 3, all inside the
    // padded row, so the clamped entries outside it are never read
    constexpr int TW = SUBPEL_NT + 5;
    __shared__ int row[TW], col[TW];
    for (int i = threadIdx.x; i < TW; i += SUBPEL_NT)
    {
      const int k = min(max(x0 - 2 + i, 0), ew - 1);
      row[i] = P(k, y);
      col[i] = wiener_at(y, eh, maxv, [&](int m) { return P(k, m); });
    }
    __syncthreads();
    const int x = x0 + threadIdx.x;
    if (x >= ew)
      return;
    const size_t o = (size_t)y * ew + x;
    ((T *)b.dst[p][0])[o] = (T)wiener_at(x, ew, maxv, [&](int k) { return row[k - x0 + 2]; });
    ((T *)b.dst[p][1])[o] = (T)col[threadIdx.x + 2];
    ((T *)b.dst[p][2])[o] = (T)wiener_at(x, ew, maxv, [&](int k) { return col[k - x0 + 2]; });
  }

  bool frames_match(const mvc::DevFrame &a, const mvc::DevFrame &b)
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

  // refs: nref reference tokens (compensate 1, degrain 2 * L, in the CompPlanes::ref order)
  template <int MODE>
  int run(const MvCudaBlockGeom *g, int L, unsigned long long t_src, const unsigned long long *t_refs, int nref,
    unsigned long long t_fill, const int *blocks, int plane_mask, unsigned long long producer, int frame, MvCudaTokenHeader *out)
  {
    if (!g || !blocks || !out || g->n_blk_x < 1 || g->n_blk_y < 1 || g->blk_x < 1 || g->blk_y < 1 || g->ov_x >= g->blk_x || g->ov_y >= g->blk_y
      || ((g->ov_x || g->ov_y) && (!g->win_luma)) || (g->pel != 1 && g->pel != 2) || g->hpad < 0 || g->vpad < 0 || nref < 1 || nref > MAXR)
      return -4;
    if (!mvc::use_device())
      return -1;
    auto fs = mvc::frame_get(t_src), ff = mvc::frame_get(t_fill);
    std::shared_ptr<mvc::DevFrame> fr[MAXR];
    if (!fs || !ff)
      return MVCUDA_TOKEN_GONE;
    for (int r = 0; r < nref; r++)
      if (!(fr[r] = mvc::frame_get(t_refs[r])))
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
    // pel 2: the sub-pel planes of every reference (made now if this is their first use; queued before the kernel)
    std::shared_ptr<mvc::DevSubpel> sp[MAXR];
    if (g->pel == 2)
      for (int r = 0; r < nref; r++)
        if (!(sp[r] = mvc::frame_subpel(fr[r], g->hpad, g->vpad)))
          return -2;
    const int ov = (g->ov_x || g->ov_y) ? 1 : 0;
    const int nblocks = g->n_blk_x * g->n_blk_y;
    const int stride = MODE == 0 ? 3 : 4 * L + 2 * (1 + 2 * L);
    const size_t bbytes = ((size_t)nblocks * stride * sizeof(int) + 15) & ~(size_t)15; // the windows start 16-byte aligned
    // the overlap windows: 9 luma windows, then 9 chroma windows
    const size_t wl = (size_t)9 * g->blk_x * g->blk_y, wc = (size_t)9 * (g->blk_x >> lx) * (g->blk_y >> ly);
    auto dst = mvc::frame_alloc(f);
    if (!dst)
      return -2;
    // one upload per call: the block table, then (with overlap) the luma and the chroma windows; the windows are constant
    // per filter but small, and sending them along keeps the API stateless
    const size_t win_bytes = ov ? (wl + (f.planes == 3 ? wc : 0)) * sizeof(short) : 0;
    const size_t up_bytes = bbytes + win_bytes;
    std::vector<unsigned char> host(up_bytes);
    std::memcpy(host.data(), blocks, (size_t)nblocks * stride * sizeof(int)); // (the padding stays zero)
    if (ov)
    {
      std::memcpy(host.data() + bbytes, g->win_luma, wl * sizeof(short));
      if (f.planes == 3)
        std::memcpy(host.data() + bbytes + wl * sizeof(short), g->win_chroma, wc * sizeof(short));
    }
    unsigned char *d_up = (unsigned char *)mvc::pipe_upload(host.data(), up_bytes);
    if (!d_up)
      return -2;
    int *d_blocks = (int *)d_up;
    short *d_win = ov ? (short *)(d_up + bbytes) : nullptr;
    const int bps = f.bytes_per_sample;
    CompPlanes pl = {};
    pl.maxv = bps == 1 ? 255 : (1 << f.bits_per_sample) - 1;
    pl.lstep_x = g->blk_x - g->ov_x;
    pl.lstep_y = g->blk_y - g->ov_y;
    int max_w = 0, max_h = 0;
    const int wbl = g->n_blk_x * (g->blk_x - g->ov_x) + g->ov_x, hbl = g->n_blk_y * (g->blk_y - g->ov_y) + g->ov_y;
    for (int p = 0; p < f.planes; p++)
    {
      const int sx = p ? lx : 0, sy = p ? ly : 0;
      pl.src[p] = (const uint8_t *)fs->plane[p];
      pl.fill[p] = (const uint8_t *)ff->plane[p];
      for (int r = 0; r < nref; r++)
      {
        pl.ref[r][p] = (const uint8_t *)fr[r]->plane[p];
        if (sp[r])
          for (int s = 0; s < 3; s++)
            pl.sub[r][p][s] = sp[r]->sub[p][s];
      }
      if (sp[0])
      {
        pl.ew[p] = sp[0]->ew[p];
        pl.eh[p] = sp[0]->eh[p];
        pl.hp[p] = sp[0]->hp[p];
        pl.vp[p] = sp[0]->vp[p];
      }
      pl.dst[p] = (uint8_t *)dst->plane[p];
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
      pl.win[p] = ov ? d_win + (p ? wl : 0) : nullptr;
      pl.active[p] = (plane_mask >> p) & 1;
      pl.grp4[p] = pl.step_x[p] % 4 == 0 && pl.bw[p] % 4 == 0;
      max_w = fs->w[p] > max_w ? fs->w[p] : max_w;
      max_h = fs->h[p] > max_h ? fs->h[p] : max_h;
    }
    const dim3 grid((max_w + 4 * BLK_NT - 1) / (4 * BLK_NT), max_h, f.planes);
    const int nbx = g->n_blk_x, nby = g->n_blk_y;
    if (bps == 2)
    {
      if (g->pel == 2)
        mvc::pipe_submit([=](cudaStream_t s) { k_blocks<uint16_t, MODE, 2><<<grid, BLK_NT, 0, s>>>(pl, d_blocks, nbx, nby, ov, L); });
      else
        mvc::pipe_submit([=](cudaStream_t s) { k_blocks<uint16_t, MODE, 1><<<grid, BLK_NT, 0, s>>>(pl, d_blocks, nbx, nby, ov, L); });
    }
    else if (g->pel == 2)
      mvc::pipe_submit([=](cudaStream_t s) { k_blocks<uint8_t, MODE, 2><<<grid, BLK_NT, 0, s>>>(pl, d_blocks, nbx, nby, ov, L); });
    else
      mvc::pipe_submit([=](cudaStream_t s) { k_blocks<uint8_t, MODE, 1><<<grid, BLK_NT, 0, s>>>(pl, d_blocks, nbx, nby, ov, L); });
    mvc::up_free(d_up, up_bytes); // reused once the pipeline stream has passed this point
    mvc::frame_put(dst, producer, frame, *out);
    return 0;
  }
}

namespace mvc
{
  DevSubpel::~DevSubpel()
  {
    pipe_free(buf, bytes); // stream-ordered reuse, as the frame buffers
  }

  std::shared_ptr<DevSubpel> frame_subpel(const std::shared_ptr<DevFrame> &fr, int hpad, int vpad)
  {
    if (!fr || (fr->f.bytes_per_sample != 1 && fr->f.bytes_per_sample != 2))
      return nullptr;
    const int bps = fr->f.bytes_per_sample;
    std::lock_guard<std::mutex> lk(fr->extra_m);
    if (fr->subpel && fr->subpel->hpad == hpad && fr->subpel->vpad == vpad)
      return fr->subpel;
    auto s = std::make_shared<DevSubpel>();
    s->hpad = hpad;
    s->vpad = vpad;
    SubBuild b = {};
    size_t off[3][3] = {}, total = 0;
    int max_ew = 0, max_eh = 0;
    for (int p = 0; p < fr->f.planes; p++)
    {
      const int xr = p ? fr->f.x_ratio_uv : 1, yr = p ? fr->f.y_ratio_uv : 1;
      s->hp[p] = hpad / xr;
      s->vp[p] = vpad / yr;
      s->ew[p] = fr->w[p] + 2 * s->hp[p];
      s->eh[p] = fr->h[p] + 2 * s->vp[p];
      for (int k = 0; k < 3; k++)
      {
        off[p][k] = total;
        total += ((size_t)s->ew[p] * s->eh[p] * bps + 255) & ~(size_t)255;
      }
      max_ew = s->ew[p] > max_ew ? s->ew[p] : max_ew;
      max_eh = s->eh[p] > max_eh ? s->eh[p] : max_eh;
    }
    s->buf = pipe_alloc(total);
    if (!s->buf)
      return nullptr;
    s->bytes = total;
    b.maxv = bps == 1 ? 255 : (1 << fr->f.bits_per_sample) - 1;
    for (int p = 0; p < fr->f.planes; p++)
    {
      b.src[p] = (const uint8_t *)fr->plane[p];
      b.ps[p] = fr->pitch[p] / bps;
      b.w[p] = fr->w[p];
      b.h[p] = fr->h[p];
      b.hp[p] = s->hp[p];
      b.vp[p] = s->vp[p];
      b.ew[p] = s->ew[p];
      b.eh[p] = s->eh[p];
      for (int k = 0; k < 3; k++)
      {
        b.dst[p][k] = (uint8_t *)s->buf + off[p][k];
        s->sub[p][k] = b.dst[p][k];
      }
    }
    const dim3 grid((max_ew + SUBPEL_NT - 1) / SUBPEL_NT, max_eh, fr->f.planes);
    if (bps == 2)
      pipe_submit([=](cudaStream_t st) { k_subpel<uint16_t><<<grid, SUBPEL_NT, 0, st>>>(b); });
    else
      pipe_submit([=](cudaStream_t st) { k_subpel<uint8_t><<<grid, SUBPEL_NT, 0, st>>>(b); });
    fr->subpel = s;
    frame_account_extra(*fr, total);
    return s;
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
