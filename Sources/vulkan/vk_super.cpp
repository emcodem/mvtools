// mvtools_vulkan: MSuper on the device, port of Sources/cuda/mvcuda_super.cu (bit for bit: pyramid reduction rfilter 4 / 2,
// edge padding, the pel 2 Wiener and pel 4 quarter-pel planes). Kernels: shaders/super_level0, super_level, super_wiener,
// super_quarter (.pix.comp). Every kernel handles all planes (or several sub-plane jobs) in one dispatch (z = plane / job).
//
// Plane layout: packed rows, ew = w + 2*hpad samples per row, eh rows (the padded plane without a pitch gap). All sub-planes
// of one super frame live in one device buffer (DevSuper::bufs[0]) at 256-byte aligned offsets.

#include "vk_internal.h"

#include <cstring>

namespace
{
  struct Level0Args
  {
    uint64_t dst[3], src[3];
    int d[18];
    int src_pitch[3];
    int pad_;
  };

  struct LevelArgs
  {
    uint64_t dst[3], src[3];
    int dd[18];
    int sd[18];
    int cubic, pad_;
  };

  struct WienerArgs
  {
    uint64_t dst[6], src[6];
    int W[6], H[6], vert[6];
    int maxv, njobs;
  };

  struct QuarterArgs
  {
    uint64_t q[48];
    int W[3], H[3];
  };

  void put_dims(int *a, const MvCudaPlaneDims &d)
  {
    a[0] = d.w;
    a[1] = d.h;
    a[2] = d.hpad;
    a[3] = d.vpad;
    a[4] = d.ew;
    a[5] = d.eh;
  }

  inline size_t al256(size_t b) { return (b + 255) & ~(size_t)255; }
}

namespace mvk
{
  int super_build(const MvCudaSuperParams &p, const DevFrame &src, DevSuper &out, vkrt::Cmd &cmd)
  {
    const int esize = src.f.bytes_per_sample;
    out.p = p;
    out.esize = esize;
    if ((esize == 1 ? (p.bits_per_pixel != 8) : (esize != 2 || p.bits_per_pixel < 9 || p.bits_per_pixel > 16))
      || (p.pel != 1 && p.pel != 2 && p.pel != 4) || p.levels < 1 || p.levels > MVCUDA_MAX_LEVELS || (p.rfilter != 4 && p.rfilter != 2)
      || p.sharp != 2)
      return -4;
    if (!super_fits(src, p, esize))
      return -4;
    if (!vkrt::ready())
      return -1;
    const int nplanes = p.chroma ? 3 : 1;
    const char *sfx = esize == 1 ? "_8" : "_16";
    auto kn = [&](const char *base) {
      char name[64];
      std::strcpy(name, base);
      std::strcat(name, sfx);
      return vkrt::kernel(name);
    };
    vkrt::Kernel *k_level0 = kn("super_level0"), *k_level = kn("super_level"), *k_wiener = kn("super_wiener"),
      *k_quarter = kn("super_quarter");
    if (!k_level0 || !k_level || !k_wiener || !k_quarter)
      return -2;

    // geometry checks (the CPU reads beyond an unpadded level-0 plane for odd sizes: not reproducible, refuse) and layout
    size_t total = 0;
    for (int plane = 0; plane < nplanes; plane++)
    {
      for (int l = 1; l < p.levels; l++)
      {
        MvCudaPlaneDims d, s;
        plane_dims(p, l, plane, d);
        plane_dims(p, l - 1, plane, s);
        if (d.w < 1 || d.h < 1 || (l == 1 && (2 * d.w > s.w || 2 * d.h > s.h)))
          return -4;
      }
      for (int l = 0; l < p.levels; l++)
      {
        DevPlane &dp = out.pl[l * 3 + plane];
        plane_dims(p, l, plane, dp.d);
        dp.nsub = (l == 0) ? p.pel * p.pel : 1;
        total += al256((size_t)dp.d.ew * dp.d.eh * esize) * dp.nsub;
      }
    }
    // every sample of every plane is written by the kernels below: no clearing
    vkrt::DevBuf buf(total);
    if (!buf)
      return -2;
    {
      size_t o = 0;
      for (int plane = 0; plane < nplanes; plane++)
        for (int l = 0; l < p.levels; l++)
        {
          DevPlane &dp = out.pl[l * 3 + plane];
          for (int s = 0; s < dp.nsub; s++)
          {
            dp.sub[s] = buf.p + o;
            o += al256((size_t)dp.d.ew * dp.d.eh * esize);
          }
        }
    }
    out.bufs.clear();
    out.bufs.push_back(std::move(buf));

    auto gx = [](int w) { return (uint32_t)((w + 31) / 32); };
    auto gy = [](int h) { return (uint32_t)((h + 7) / 8); };

    // level 0 with its padding straight from the frame
    {
      Level0Args a = {};
      int mw = 0, mh = 0;
      for (int plane = 0; plane < nplanes; plane++)
      {
        const DevPlane &dp = out.pl[plane];
        a.dst[plane] = dp.sub[0];
        a.src[plane] = src.plane[plane];
        put_dims(a.d + plane * 6, dp.d);
        a.src_pitch[plane] = (int)(src.pitch[plane] / esize);
        mw = dp.d.ew > mw ? dp.d.ew : mw;
        mh = dp.d.eh > mh ? dp.d.eh : mh;
      }
      cmd.dispatch(k_level0, gx(mw), gy(mh), nplanes, a);
    }
    // each pyramid level with its padding. CPU quirk: only the first reduction uses rfilter (cubic), every deeper level
    // keeps the MVPlane default RB2BilinearFiltered
    for (int l = 0; l + 1 < p.levels; l++)
    {
      LevelArgs a = {};
      int mw = 0, mh = 0;
      for (int plane = 0; plane < nplanes; plane++)
      {
        const DevPlane &s = out.pl[l * 3 + plane], &d = out.pl[(l + 1) * 3 + plane];
        a.dst[plane] = d.sub[0];
        a.src[plane] = s.sub[0];
        put_dims(a.dd + plane * 6, d.d);
        put_dims(a.sd + plane * 6, s.d);
        mw = d.d.ew > mw ? d.d.ew : mw;
        mh = d.d.eh > mh ? d.d.eh : mh;
      }
      a.cubic = (l == 0 && p.rfilter == 4) ? 1 : 0;
      cmd.dispatch(k_level, gx(mw), gy(mh), nplanes, a);
    }

    if (p.pel == 2 || p.pel == 4)
    {
      const int maxv = (1 << p.bits_per_pixel) - 1;
      // pel 2: sub 1 = H(sub 0), sub 2 = V(sub 0), then sub 3 = H(sub 2); pel 4: the same into 2, 8, 10, then the quarters
      const int ih = p.pel == 2 ? 1 : 2, iv = p.pel == 2 ? 2 : 8, ihv = p.pel == 2 ? 3 : 10;
      int mw = 0, mh = 0;
      WienerArgs a = {}, b = {};
      for (int plane = 0; plane < nplanes; plane++)
      {
        const DevPlane &dp = out.pl[plane];
        mw = dp.d.ew > mw ? dp.d.ew : mw;
        mh = dp.d.eh > mh ? dp.d.eh : mh;
        for (int j = 0; j < 2; j++)
        {
          const int k = plane * 2 + j;
          a.dst[k] = dp.sub[j == 0 ? ih : iv];
          a.src[k] = dp.sub[0];
          a.W[k] = dp.d.ew;
          a.H[k] = dp.d.eh;
          a.vert[k] = j;
        }
        b.dst[plane] = dp.sub[ihv];
        b.src[plane] = dp.sub[iv];
        b.W[plane] = dp.d.ew;
        b.H[plane] = dp.d.eh;
        b.vert[plane] = 0;
      }
      a.maxv = b.maxv = maxv;
      a.njobs = 2 * nplanes;
      b.njobs = nplanes;
      cmd.dispatch(k_wiener, gx(mw), gy(mh), 2 * nplanes, a);
      cmd.dispatch(k_wiener, gx(mw), gy(mh), nplanes, b);
      if (p.pel == 4)
      {
        QuarterArgs q = {};
        for (int plane = 0; plane < nplanes; plane++)
        {
          const DevPlane &dp = out.pl[plane];
          for (int i = 0; i < 16; i++)
            q.q[plane * 16 + i] = dp.sub[i];
          q.W[plane] = dp.d.ew;
          q.H[plane] = dp.d.eh;
        }
        cmd.dispatch(k_quarter, gx(mw), gy(mh), nplanes, q);
      }
    }
    return cmd.ok() ? 0 : -2;
  }
}

// host-memory test interface of the super builder: copies every plane back
extern "C" int mvcuda_super_host(const MvCudaSuperParams *p, const void *const src[3], const int src_pitch_bytes[3], void *const out[])
{
  if (!p || !src || !src_pitch_bytes || !out)
    return -4;
  if (!vkrt::ready())
    return -1;
  const int esize = p->bits_per_pixel <= 8 ? 1 : 2;
  auto fr = mvk::upload_picture(*p, esize, src, src_pitch_bytes);
  if (!fr)
    return -4;
  mvk::DevSuper s;
  vkrt::Cmd cmd;
  int rc = mvk::super_build(*p, *fr, s, cmd);
  if (rc)
    return rc;
  for (int l = 0; l < p->levels; l++)
    for (int plane = 0; plane < (p->chroma ? 3 : 1); plane++)
    {
      const mvk::DevPlane &dp = s.pl[l * 3 + plane];
      for (int sub = 0; sub < dp.nsub; sub++)
      {
        void *dst = out[(l * 3 + plane) * MVCUDA_MAX_SUB + sub];
        if (dst)
          cmd.download(dst, dp.sub[sub], (size_t)dp.d.ew * dp.d.eh * esize);
      }
    }
  return cmd.submit_wait();
}

extern "C" int mvcuda_super_plane_dims(const MvCudaSuperParams *p, int level, int plane, MvCudaPlaneDims *d)
{
  if (!p || !d || level < 0 || level >= p->levels || level >= MVCUDA_MAX_LEVELS || plane < 0 || plane > 2 || (plane > 0 && !p->chroma))
    return -4;
  mvk::plane_dims(*p, level, plane, *d);
  return 0;
}
