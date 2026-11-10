// splat_core: the exact CPU path of SplatFps' sample I/O (input, output) and the picture build (Laplacian-pyramid
// forward splatting); see splat_core.h for the description and the exactness rules.
#include "splat_core.h"
#include "splat_const.h"

#include <algorithm>
#include <cmath>

namespace splat
{
  float exp32(float x) { return (float)std::exp((double)x); }

  // ---------------------------------------------------------------------------------------------------- input / output
  float to_val(int code10) { return ((float)code10 - 512.0f) * 0.25f; }

  int to_code(float v)
  {
    float t = v * 4.0f + 0.5f;
    int c = 512 + (int)std::trunc(t);
    return c < 0 ? 0 : (c > 1023 ? 1023 : c);
  }

  // half-band taps: CHROMA_HB (splat_const.h)
  static const float *const HB = CHROMA_HB;

  // centre-sited chroma: input j sits half way between outputs 2 j and 2 j + 1, a quarter of an input sample from
  // each; both by the quadratic interpolation CHROMA_420 (the same geometry as 4:2:0 chroma between two rows):
  // input j, its neighbour on the side of the output, the neighbour on the other side (edge clamp)
  void chroma_upsample_row_centre(const float *in, int w2, float *out)
  {
    for (int j = 0; j < w2; j++)
    {
      const float l = in[j > 0 ? j - 1 : 0], r = in[j + 1 < w2 ? j + 1 : w2 - 1];
      float a = CHROMA_420[0] * in[j];
      a = a + CHROMA_420[1] * l;
      a = a + CHROMA_420[2] * r;
      out[2 * j] = a;
      float b = CHROMA_420[0] * in[j];
      b = b + CHROMA_420[1] * r;
      b = b + CHROMA_420[2] * l;
      out[2 * j + 1] = b;
    }
  }

  void chroma_upsample_row(const float *in, int w2, float *out)
  {
    for (int j = 0; j < w2; j++)
    {
      out[2 * j] = in[j];
      float acc = 0.0f;
      for (int k = 0; k < CHROMA_HB_TAPS; k++)
      {
        int ia = j - k, ib = j + 1 + k;
        ia = ia < 0 ? 0 : (ia >= w2 ? w2 - 1 : ia);
        ib = ib < 0 ? 0 : (ib >= w2 ? w2 - 1 : ib);
        float s = in[ia] + in[ib];
        acc = acc + HB[k] * s;
      }
      out[2 * j + 1] = acc;
    }
  }

  float to_val(int code, int bits)
  {
    const float mid = (float)(1 << (bits - 1)), inv = 1.0f / (float)(1 << (bits - 8)); // exact powers of two
    return ((float)code - mid) * inv;
  }

  int to_code(float v, int bits)
  {
    const float t = v * (float)(1 << (bits - 8)) + 0.5f;
    const int c = (1 << (bits - 1)) + (int)std::floor(t), top = (1 << bits) - 1;
    return c < 0 ? 0 : (c > top ? top : c);
  }

  namespace
  {
    template <class T> const T *row_of(const void *plane, int pitch, int y)
    {
      return reinterpret_cast<const T *>(static_cast<const unsigned char *>(plane) + (size_t)y * pitch);
    }
    template <class T> T *row_of(void *plane, int pitch, int y)
    {
      return reinterpret_cast<T *>(static_cast<unsigned char *>(plane) + (size_t)y * pitch);
    }

    template <class T>
    void input_rows_t(const void *const planes[3], const int pitch[3], const IoFormat &f, int y0, int y1, Pic3 &out)
    {
      const int W = out.c[0].w, H = out.c[0].h, cw = W >> f.sub_w, ch = H >> f.sub_h, b = f.bits;
      std::vector<float> row(cw);
      for (int y = y0; y < y1; y++)
      {
        const T *py = row_of<T>(planes[0], pitch[0], y);
        float *oy = &out.c[0].at(0, y);
        for (int x = 0; x < W; x++)
          oy[x] = to_val(py[x], b);
        for (int c = 1; c < 3; c++)
        {
          float *r = f.sub_w ? row.data() : &out.c[c].at(0, y);
          if (!f.sub_h)
          {
            const T *s = row_of<T>(planes[c], pitch[c], y);
            for (int x = 0; x < cw; x++)
              r[x] = to_val(s[x], b);
          }
          else
          {
            // chroma row j, the neighbour toward output row y and the one away from it (edge clamp)
            const int j = y >> 1, t = (y & 1) ? std::min(j + 1, ch - 1) : std::max(j - 1, 0),
                      a = (y & 1) ? std::max(j - 1, 0) : std::min(j + 1, ch - 1);
            const T *s0 = row_of<T>(planes[c], pitch[c], j), *st = row_of<T>(planes[c], pitch[c], t),
                    *sa = row_of<T>(planes[c], pitch[c], a);
            for (int x = 0; x < cw; x++)
            {
              float acc = CHROMA_420[0] * to_val(s0[x], b);
              acc = acc + CHROMA_420[1] * to_val(st[x], b);
              acc = acc + CHROMA_420[2] * to_val(sa[x], b);
              r[x] = acc;
            }
          }
          if (f.sub_w && f.centre)
            chroma_upsample_row_centre(r, cw, &out.c[c].at(0, y));
          else if (f.sub_w)
            chroma_upsample_row(r, cw, &out.c[c].at(0, y));
        }
      }
    }

    // the half-band interpolation of row s (W samples) half way between X and X + 1 (edge clamp)
    inline float half_way(const float *s, int W, int X)
    {
      float acc = 0.0f;
      for (int k = 0; k < CHROMA_HB_TAPS; k++)
        acc = acc + CHROMA_HB[k] * (s[X - k < 0 ? 0 : X - k] + s[X + 1 + k > W - 1 ? W - 1 : X + 1 + k]);
      return acc;
    }

    template <class T>
    void output_rows_t(const Pic3 &in, const IoFormat &f, int y0, int y1, void *const planes[3], const int pitch[3])
    {
      const int W = in.c[0].w, H = in.c[0].h, cw = W >> f.sub_w, b = f.bits;
      for (int y = y0; y < y1; y++)
      {
        T *py = row_of<T>(planes[0], pitch[0], y);
        const float *iy = in.c[0].d.data() + (size_t)y * W;
        for (int x = 0; x < W; x++)
          py[x] = (T)to_code(iy[x], b);
        if (!f.sub_h)
          for (int c = 1; c < 3; c++)
          {
            T *d = row_of<T>(planes[c], pitch[c], y);
            const float *s = in.c[c].d.data() + (size_t)y * W;
            for (int x = 0; x < cw; x++)
              d[x] = (T)to_code(f.sub_w && f.centre ? half_way(s, W, 2 * x) : s[x << f.sub_w], b);
          }
      }
      if (!f.sub_h)
        return;
      // 4:2:0: chroma row j half way between rows 2 j and 2 j + 1, the half-band pairs down the column (edge clamp)
      const float *const HBv = CHROMA_HB;
      for (int j = (y0 + 1) / 2; 2 * j < y1; j++)
        for (int c = 1; c < 3; c++)
        {
          const float *ra[CHROMA_HB_TAPS], *rb[CHROMA_HB_TAPS];
          for (int k = 0; k < CHROMA_HB_TAPS; k++)
          {
            ra[k] = in.c[c].d.data() + (size_t)std::max(2 * j - k, 0) * W;
            rb[k] = in.c[c].d.data() + (size_t)std::min(2 * j + 1 + k, H - 1) * W;
          }
          T *d = row_of<T>(planes[c], pitch[c], j);
          auto vert = [&](int xs) {
            float acc = 0.0f;
            for (int k = 0; k < CHROMA_HB_TAPS; k++)
              acc = acc + HBv[k] * (ra[k][xs] + rb[k][xs]);
            return acc;
          };
          if (f.centre)
          {
            // centre siting: the vertical half-band row, then the horizontal half-band pairs around 2 x + 1/2
            std::vector<float> v(W);
            for (int X = 0; X < W; X++)
              v[X] = vert(X);
            for (int x = 0; x < cw; x++)
              d[x] = (T)to_code(half_way(v.data(), W, 2 * x), b);
          }
          else
            for (int x = 0; x < cw; x++)
              d[x] = (T)to_code(vert(x << f.sub_w), b);
        }
    }
  }

  void input_rows(const void *const planes[3], const int pitch[3], const IoFormat &f, int y0, int y1, Pic3 &out)
  {
    if (f.bits == 8)
      input_rows_t<uint8_t>(planes, pitch, f, y0, y1, out);
    else
      input_rows_t<uint16_t>(planes, pitch, f, y0, y1, out);
  }

  void output_rows(const Pic3 &in, const IoFormat &f, int y0, int y1, void *const planes[3], const int pitch[3])
  {
    if (f.bits == 8)
      output_rows_t<uint8_t>(in, f, y0, y1, planes, pitch);
    else
      output_rows_t<uint16_t>(in, f, y0, y1, planes, pitch);
  }

  // ----------------------------------------------------------------------------------------------------------- pyramid
  Plane pad_edge(const Plane &src, int gw, int gh, int px, int py)
  {
    Plane o(gw, gh);
    for (int y = 0; y < gh; y++)
      for (int x = 0; x < gw; x++)
        o.at(x, y) = src.clamped(x - px, y - py);
    return o;
  }

  // one output of the reduce filter (symmetric 7 taps D: centre, distance 1, 2, 3) from the centre sample and the
  // pairs at distance 1, 2, 3
  static inline float dn_tap(const float *D, float g0, float gm1, float gp1, float gm2, float gp2, float gm3,
    float gp3)
  {
    float r = D[0] * g0;
    r = r + D[1] * (gm1 + gp1);
    r = r + D[2] * (gm2 + gp2);
    r = r + D[3] * (gm3 + gp3);
    return r;
  }

  Plane pyr_reduce(const Plane &s, const float *D)
  {
    const int oh = (s.h + 1) / 2, ow = (s.w + 1) / 2;
    Plane v(s.w, oh); // vertical pass first
    for (int j = 0; j < oh; j++)
      for (int x = 0; x < s.w; x++)
      {
        int y = 2 * j;
        v.at(x, j) = dn_tap(D, s.clamped(x, y), s.clamped(x, y - 1), s.clamped(x, y + 1), s.clamped(x, y - 2),
          s.clamped(x, y + 2), s.clamped(x, y - 3), s.clamped(x, y + 3));
      }
    Plane o(ow, oh);
    for (int y = 0; y < oh; y++)
      for (int i = 0; i < ow; i++)
      {
        int x = 2 * i;
        o.at(i, y) = dn_tap(D, v.clamped(x, y), v.clamped(x - 1, y), v.clamped(x + 1, y), v.clamped(x - 2, y),
          v.clamped(x + 2, y), v.clamped(x - 3, y), v.clamped(x + 3, y));
      }
    return o;
  }

  // pyramid expand kernels, one per output phase (PYR_EXPAND_A / B / C, described in splat_const.h).
  // Exact reconstruction does not depend on these: band = low - pyr_expand(low1) stores whatever pyr_expand() misses.
  constexpr float A0 = PYR_EXPAND_A[0], A1 = PYR_EXPAND_A[1], A2 = PYR_EXPAND_A[2];
  constexpr float B0 = PYR_EXPAND_B[0], B1 = PYR_EXPAND_B[1], B2 = PYR_EXPAND_B[2], B3 = PYR_EXPAND_B[3];
  constexpr float C0 = PYR_EXPAND_C[0], C1 = PYR_EXPAND_C[1], C2 = PYR_EXPAND_C[2];

  // low sample (x, y) writes the 4 outputs (2x .. 2x+1, 2y .. 2y+1); the loop runs over the low samples
  Plane pyr_expand(const Plane &l, int ow, int oh)
  {
    Plane o(ow, oh);
    for (int y = 0; y < l.h; y++)
      for (int x = 0; x < l.w; x++)
      {
        auto C = [&](int dy, int dx) { return l.clamped(x + dx, y + dy); };
        const float c = C(0, 0), n = C(-1, 0), wst = C(0, -1), e = C(0, 1), s = C(1, 0);
        const float nw = C(-1, -1), ne = C(-1, 1), sw = C(1, -1), se = C(1, 1);
        const float c02 = C(0, 2), cm12 = C(-1, 2), c12 = C(1, 2), c20 = C(2, 0), c2m1 = C(2, -1), c21 = C(2, 1),
          c22 = C(2, 2);
        const int ox = 2 * x, oy = 2 * y;
        if (oy < oh && ox < ow)
        {
          float r = A0 * c;
          r = r + A1 * (((n + wst) + e) + s);
          r = r + A2 * (((nw + ne) + sw) + se);
          o.at(ox, oy) = r;
        }
        if (oy < oh && ox + 1 < ow)
        {
          float r = B0 * (c + e);
          r = r + B1 * (wst + c02);
          r = r + B2 * (((n + ne) + s) + se);
          r = r + B3 * (((nw + cm12) + sw) + c12);
          o.at(ox + 1, oy) = r;
        }
        if (oy + 1 < oh && ox < ow)
        {
          float r = B0 * (c + s);
          r = r + B1 * (n + c20);
          r = r + B2 * (((wst + e) + sw) + se);
          r = r + B3 * (((nw + ne) + c2m1) + c21);
          o.at(ox, oy + 1) = r;
        }
        if (oy + 1 < oh && ox + 1 < ow)
        {
          float r = C0 * (((c + e) + s) + se);
          r = r + C1 * ((((((((n + ne) + c20) + c21) + wst) + c02) + sw) + c12));
          r = r + C2 * (((nw + cm12) + c2m1) + c22);
          o.at(ox + 1, oy + 1) = r;
        }
      }
    return o;
  }

  // ------------------------------------------------------------------------------------------------------------ fields
  // Projection field of one side. g: the fraction of its vector the side travels to reach the output time (P: d along
  // its forward vector, N: 1 - d along its backward vector); pr = 1 - g: the side's temporal proximity weight.
  // Channel 2 becomes the splat weight splat_gain * pr, or with splat_err_scale > 0
  // splat_gain * max(splat_min_weight, exp(-err / splat_err_scale)) * pr (a confidence that falls with the match error
  // but never to zero); channel 3 the fallback weight * pr. The border (the padding) takes the nearest cell inside.
  Field4 border_weight(const Field4 &F, float d, bool n_side, const Config &cfg)
  {
    const float g = n_side ? 1.0f - d : d;
    const float pr = 1.0f - g;
    const int w = F.c[0].w, h = F.c[0].h, XB = cfg.pad_x / cfg.scale, YB = cfg.pad_y / cfg.scale;
    Field4 G;
    for (int c = 0; c < 4; c++)
      G.c[c] = Plane(w, h);
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++)
      {
        // the border (the padding) takes the nearest cell inside the picture
        const int sx = std::min(std::max(x, XB), w - 1 - XB), sy = std::min(std::max(y, YB), h - 1 - YB);
        G.c[0].at(x, y) = F.c[0].at(sx, sy) * g;
        G.c[1].at(x, y) = F.c[1].at(sx, sy) * g;
        const float ex = cfg.splat_err_scale > 0.0f ? exp32(-F.c[2].at(sx, sy) / cfg.splat_err_scale) : 1.0f;
        const float m = std::fmax(cfg.splat_min_weight, ex);
        G.c[2].at(x, y) = (cfg.splat_gain * m) * pr;
        G.c[3].at(x, y) = F.c[3].at(sx, sy) * pr;
      }
    return G;
  }

  // ME resolution -> full resolution: ME pixel (x, y) and its right / lower neighbours (edge clamp) are interpolated
  // bilinearly at the s x s positions fx / s, fy / s; vectors are scaled by s into full-resolution pixels
  Field4 field_upsample(const Field4 &G, int s, int ow, int oh)
  {
    const int w = G.c[0].w, h = G.c[0].h, sh = ilog2(s), msk = s - 1;
    const float fs = (float)s;
    if (ow <= 0 || oh <= 0)
    {
      ow = s * w;
      oh = s * h;
    }
    Field4 o;
    for (int c = 0; c < 4; c++)
    {
      Plane A = G.c[c];
      if (c < 2)
        for (float &v : A.d)
          v = v * fs;
      o.c[c] = Plane(ow, oh);
      for (int Y = 0; Y < oh; Y++)
        for (int X = 0; X < ow; X++)
        {
          const int x = std::min(X >> sh, w - 1), y = std::min(Y >> sh, h - 1);
          const float a = A.clamped(x, y), b = A.clamped(x + 1, y), cc = A.clamped(x, y + 1), e = A.clamped(x + 1, y + 1);
          const float tx = (float)(X & msk) / fs, ty = (float)(Y & msk) / fs;
          const float l6 = a * (1.0f - tx) + b * tx;
          const float l18 = cc * (1.0f - tx) + e * tx;
          o.c[c].at(X, Y) = l6 * (1.0f - ty) + l18 * ty;
        }
    }
    return o;
  }

  // field of the next coarser level: 3x3 binomial ([1 2 1] x [1 2 1] / 16, the reduce step of P. J. Burt and E. H.
  // Adelson, 1983) at the even positions, edge clamp, vectors halved to stay in the pixels of the smaller grid;
  // weights are averaged the same way
  Field4 field_reduce2(const Field4 &F)
  {
    const int w = F.c[0].w, h = F.c[0].h, ow = w / 2, oh = h / 2;
    Field4 o;
    for (int c = 0; c < 4; c++)
    {
      const Plane &s = F.c[c];
      // edge clamp: positions outside the field read the nearest edge sample
      auto g = [&](int y, int x) { return s.at(std::min(std::max(x, 0), w - 1), std::min(std::max(y, 0), h - 1)); };
      o.c[c] = Plane(ow, oh);
      for (int j = 0; j < oh; j++)
        for (int i = 0; i < ow; i++)
        {
          const int y = 2 * j, x = 2 * i;
          float r = 0.0625f * (((g(y - 1, x - 1) + g(y - 1, x + 1)) + g(y + 1, x - 1)) + g(y + 1, x + 1));
          r = r + 0.125f * (((g(y - 1, x) + g(y, x - 1)) + g(y, x + 1)) + g(y + 1, x));
          r = r + 0.25f * g(y, x);
          if (c < 2)
            r = r / 2.0f;
          o.c[c].at(i, j) = r;
        }
    }
    return o;
  }

  // --------------------------------------------------------------------------------------------------------- one level
  Pic3 build_level(const Pic3 &P, const Pic3 &N, const Field4 &FP, const Field4 &FN, int stride, float d,
    const Config &cfg, LevelDebug *dbg)
  {
    const int w = P.c[0].w, h = P.c[0].h;
    const float s = (float)stride;
    // per side and band sample: vector in this level's pixels (full-res vector / stride), splat weight mc, in-place
    // weight st (fallback weight, plus mc for zero vectors) and the zero-vector flag
    struct Side { Plane vx, vy, mc, st; std::vector<uint8_t> zero; };
    Side sd[2];
    const Field4 *F[2] = { &FP, &FN };
    for (int k = 0; k < 2; k++)
    {
      Side &t = sd[k];
      t.vx = Plane(w, h); t.vy = Plane(w, h); t.mc = Plane(w, h); t.st = Plane(w, h);
      t.zero.assign((size_t)w * h, 0);
      for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
          // edge clamp: a coarse level can be one sample larger than its field (the picture pyramid rounds up)
          const int fx = std::min(x * stride, F[k]->c[0].w - 1), fy = std::min(y * stride, F[k]->c[0].h - 1);
          const float vx = F[k]->c[0].at(fx, fy) / s, vy = F[k]->c[1].at(fx, fy) / s;
          const float mc = F[k]->c[2].at(fx, fy);
          float st = F[k]->c[3].at(fx, fy);
          const bool z = std::sqrt(vx * vx + vy * vy) <= cfg.still_vector;
          if (z)
            st = st + mc;
          t.vx.at(x, y) = vx; t.vy.at(x, y) = vy; t.mc.at(x, y) = mc; t.st.at(x, y) = st;
          t.zero[(size_t)y * w + x] = z;
        }
    }
    // accumulator (w + 1) x (h + 1): 3 channels + weight (one extra row / column so the corner splats of a position
    // on the last row / column stay inside). It starts with the in-place part: P st_P + N st_N.
    const int aw = w + 1, ah = h + 1;
    std::vector<float> acc((size_t)aw * ah * 4, 0.0f);
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++)
      {
        float *a = &acc[((size_t)y * aw + x) * 4];
        const float sP = sd[0].st.at(x, y), sN = sd[1].st.at(x, y);
        for (int c = 0; c < 3; c++)
          a[c] = P.c[c].at(x, y) * sP + N.c[c].at(x, y) * sN;
        a[3] = sP + sN;
      }
    auto snap = [&](Plane *out) {
      for (int c = 0; c < 4; c++)
      {
        out[c] = Plane(w, h);
        for (int y = 0; y < h; y++)
          for (int x = 0; x < w; x++)
            out[c].at(x, y) = acc[((size_t)y * aw + x) * 4 + c];
      }
    };
    if (dbg)
      snap(dbg->acc_init);
    // bilinear forward splats: each sample with a non-zero vector lands at (x + vx, y + vy) (targets outside the band
    // are dropped) and adds sample * mc and mc to the 4 surrounding cells with the bilinear corner weights.
    // Order: side by side, corner by corner, pixels in raster order (the order of the sums)
    const Pic3 *pic[2] = { &P, &N };
    const float wl = (float)(w - 1), hl = (float)(h - 1);
    for (int k = 0; k < 2; k++)
    {
      const Side &t = sd[k];
      for (int corner = 0; corner < 4; corner++)
        for (int y = 0; y < h; y++)
          for (int x = 0; x < w; x++)
          {
            if (t.zero[(size_t)y * w + x])
              continue;
            const float px = (float)x + t.vx.at(x, y), py = (float)y + t.vy.at(x, y);
            if (!(px >= 0.0f && px <= wl && py >= 0.0f && py <= hl))
              continue;
            const float ix = std::floor(px), iy = std::floor(py);
            const float fx = px - ix, fy = py - iy;
            float a;
            int ox = 0, oy = 0;
            switch (corner)
            {
            case 0: a = (1.0f - fx) * (1.0f - fy); break;
            case 1: a = fx * (1.0f - fy); ox = 1; break;
            case 2: a = (1.0f - fx) * fy; oy = 1; break;
            default: a = fx * fy; ox = 1; oy = 1; break;
            }
            const float wt = t.mc.at(x, y);
            float *dst = &acc[(((size_t)iy + oy) * aw + (size_t)ix + ox) * 4];
            for (int c = 0; c < 3; c++)
            {
              const float lt = pic[k]->c[c].at(x, y) * wt;
              dst[c] = dst[c] + lt * a;
            }
            dst[3] = dst[3] + a * wt;
          }
    }
    if (dbg)
      snap(dbg->acc_splat);
    // normalisation; holes (total weight below hole_weight) get the cross-fade of the two bands at phase d
    Pic3 out;
    for (int c = 0; c < 3; c++)
      out.c[c] = Plane(w, h);
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++)
      {
        const float *a = &acc[((size_t)y * aw + x) * 4];
        const float sw = a[3];
        for (int c = 0; c < 3; c++)
          out.c[c].at(x, y) = sw < cfg.hole_weight ? P.c[c].at(x, y) * (1.0f - d) + N.c[c].at(x, y) * d : a[c] / sw;
      }
    (void)ah;
    return out;
  }

  // ------------------------------------------------------------------------------------------------------- whole build
  static Pic3 sub(const Pic3 &a, const Pic3 &b)
  {
    Pic3 o = a;
    for (int c = 0; c < 3; c++)
      for (size_t i = 0; i < o.c[c].d.size(); i++)
        o.c[c].d[i] = a.c[c].d[i] - b.c[c].d[i];
    return o;
  }

  static Pic3 up3(const Pic3 &l, int ow, int oh)
  {
    Pic3 o;
    for (int c = 0; c < 3; c++)
      o.c[c] = pyr_expand(l.c[c], ow, oh);
    return o;
  }

  Pic3 build(const Pic3 &Pin, const Pic3 &Nin, const Field4 &MEP, const Field4 &MEN, float d, const Config &cfg,
    BuildDebug *dbg)
  {
    const Geometry g(Pin.c[0].w, Pin.c[0].h, cfg);
    const int L = cfg.levels, stride_max = cfg.stride_max, PADX = cfg.pad_x, PADY = cfg.pad_y;
    std::vector<Pic3> lowP(L + 1), lowN(L + 1);
    for (int c = 0; c < 3; c++)
    {
      lowP[0].c[c] = pad_edge(Pin.c[c], g.GW, g.GH, PADX, PADY);
      lowN[0].c[c] = pad_edge(Nin.c[c], g.GW, g.GH, PADX, PADY);
    }
    for (int i = 0; i < L; i++)
      for (int c = 0; c < 3; c++)
      {
        lowP[i + 1].c[c] = pyr_reduce(lowP[i].c[c], cfg.pyr_reduce);
        lowN[i + 1].c[c] = pyr_reduce(lowN[i].c[c], cfg.pyr_reduce);
      }
    std::vector<Pic3> bandP(L + 1), bandN(L + 1);
    for (int i = 0; i < L; i++)
    {
      bandP[i] = sub(lowP[i], up3(lowP[i + 1], lowP[i].c[0].w, lowP[i].c[0].h));
      bandN[i] = sub(lowN[i], up3(lowN[i + 1], lowN[i].c[0].w, lowN[i].c[0].h));
    }
    bandP[L] = lowP[L];
    bandN[L] = lowN[L];
    // fields per level: levels with stride 2^i <= stride_max sample the full-res projection field; the first coarser
    // level reduces the ME-res field (resolution 1 / scale) i - log2(scale) times, each further level once more
    const Field4 gP = border_weight(MEP, d, false, cfg), gN = border_weight(MEN, d, true, cfg);
    const Field4 fullP = field_upsample(gP, cfg.scale, g.GW, g.GH), fullN = field_upsample(gN, cfg.scale, g.GW, g.GH);
    const int log2max = ilog2(cfg.scale);
    std::vector<Field4> fP(L + 1), fN(L + 1);
    std::vector<int> stride(L + 1);
    bool have_coarse = false;
    Field4 curP, curN;
    for (int i = 0; i <= L; i++)
    {
      const int st = 1 << i;
      if (st <= stride_max)
      {
        stride[i] = st;
        continue;
      }
      if (!have_coarse)
      {
        curP = gP; curN = gN;
        for (int r = 0; r < i - 1 - log2max + 1; r++)
        {
          curP = field_reduce2(curP);
          curN = field_reduce2(curN);
        }
        have_coarse = true;
      }
      else
      {
        curP = field_reduce2(curP);
        curN = field_reduce2(curN);
      }
      fP[i] = curP; fN[i] = curN;
      stride[i] = 0;
    }
    std::vector<Pic3> built(L + 1);
    std::vector<LevelDebug> lv(dbg ? L + 1 : 0);
    for (int i = 0; i <= L; i++)
    {
      LevelDebug *ld = dbg ? &lv[i] : nullptr;
      if (stride[i])
        built[i] = build_level(bandP[i], bandN[i], fullP, fullN, stride[i], d, cfg, ld);
      else
        built[i] = build_level(bandP[i], bandN[i], fP[i], fN[i], 1, d, cfg, ld);
    }
    // Laplacian reconstruction, coarse to fine: out[i] = built[i] + pyr_expand(out[i+1]); clamp to the sample range at the end
    std::vector<Pic3> recon(L + 1);
    Pic3 out = built[L];
    recon[L] = out;
    for (int i = L - 1; i >= 0; i--)
    {
      const Pic3 u = up3(out, built[i].c[0].w, built[i].c[0].h);
      Pic3 o = built[i];
      for (int c = 0; c < 3; c++)
        for (size_t k = 0; k < o.c[c].d.size(); k++)
        {
          float v = built[i].c[c].d[k] + u.c[c].d[k];
          if (i == 0)
            v = std::fmin(std::fmax(v, SAMPLE_MIN), SAMPLE_MAX);
          o.c[c].d[k] = v;
        }
      out = o;
      recon[i] = out;
    }
    Pic3 res;
    for (int c = 0; c < 3; c++)
    {
      res.c[c] = Plane(g.W, g.H);
      for (int y = 0; y < g.H; y++)
        for (int x = 0; x < g.W; x++)
          res.c[c].at(x, y) = out.c[c].at(x + PADX, y + PADY);
    }
    if (dbg)
    {
      dbg->lowP = lowP; dbg->lowN = lowN; dbg->bandP = bandP; dbg->bandN = bandN; dbg->built = built;
      dbg->recon = recon; dbg->projP = gP; dbg->projN = gN; dbg->fullP = fullP; dbg->fullN = fullN;
      dbg->fieldP = fP; dbg->fieldN = fN; dbg->stride = stride; dbg->levels = lv;
    }
    return res;
  }
}
