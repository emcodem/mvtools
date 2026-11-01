// mvtools_vulkan: library entry points (version, devices), the super frames kept with device frames, cache clearing.

#include "vk_internal.h"

#include <cstdio>
#include <cstring>

namespace mvk
{
  size_t super_bytes(const MvCudaSuperParams &p, int esize)
  {
    size_t total = 0;
    for (int plane = 0; plane < (p.chroma ? 3 : 1); plane++)
      for (int l = 0; l < p.levels; l++)
      {
        MvCudaPlaneDims d;
        plane_dims(p, l, plane, d);
        total += (size_t)d.ew * d.eh * esize * (l == 0 ? p.pel * p.pel : 1);
      }
    return total;
  }

  MvCudaFrameFormat super_format(const MvCudaSuperParams &p, int esize)
  {
    MvCudaFrameFormat f = {};
    f.width = p.width;
    f.height = p.height;
    f.planes = p.chroma ? 3 : 1;
    f.bytes_per_sample = esize;
    f.bits_per_sample = p.bits_per_pixel;
    f.x_ratio_uv = p.x_ratio_uv;
    f.y_ratio_uv = p.y_ratio_uv;
    return f;
  }

  bool super_fits(const DevFrame &f, const MvCudaSuperParams &p, int esize)
  {
    return f.f.width == p.width && f.f.height == p.height && f.f.bits_per_sample == p.bits_per_pixel && f.f.bytes_per_sample == esize
      && (!p.chroma || (f.f.planes == 3 && f.f.x_ratio_uv == p.x_ratio_uv && f.f.y_ratio_uv == p.y_ratio_uv));
  }

  namespace
  {
    bool covers(const MvCudaSuperParams &e, const MvCudaSuperParams &p)
    {
      return e.width == p.width && e.height == p.height && e.x_ratio_uv == p.x_ratio_uv && e.y_ratio_uv == p.y_ratio_uv
        && e.hpad == p.hpad && e.vpad == p.vpad && e.pel == p.pel && e.sharp == p.sharp && e.bits_per_pixel == p.bits_per_pixel
        && e.chroma >= p.chroma && e.levels >= p.levels && (p.levels == 1 || e.rfilter == p.rfilter);
    }
  }

  int frame_super(const std::shared_ptr<DevFrame> &fr, const MvCudaSuperParams &p, std::shared_ptr<DevSuper> &out)
  {
    if (!fr || !super_fits(*fr, p, fr->f.bytes_per_sample))
      return -4;
    size_t made = 0;
    {
      // concurrent requests for one frame: the first builds (and waits), the others then find it
      std::lock_guard<std::mutex> lk(fr->extra_m);
      for (const auto &e : fr->supers)
        if (covers(e->p, p))
        {
          out = e;
          return 0;
        }
      auto sp = std::make_shared<DevSuper>();
      vkrt::Cmd cmd;
      int rc = super_build(p, *fr, *sp, cmd);
      if (rc)
        return rc;
      rc = cmd.submit_wait();
      if (rc)
        return rc;
      fr->supers.push_back(sp);
      out = sp;
      made = super_bytes(p, fr->f.bytes_per_sample);
    }
    frame_account_extra(*fr, made);
    return 0;
  }
}

extern "C" int mvcuda_api_version()
{
  return MVCUDA_API_VERSION;
}

extern "C" int mvcuda_device_count()
{
  return vkrt::device_count();
}

extern "C" int mvcuda_device_info(int device, char *name, int name_len, int *cc_major, int *cc_minor, unsigned long long *total_mem)
{
  if (device != 0 || !vkrt::ready())
    return -1;
  if (name && name_len > 0)
  {
    std::strncpy(name, vkrt::device_name(), name_len - 1);
    name[name_len - 1] = 0;
  }
  // no compute capability under Vulkan: 0.0 marks the Vulkan backend
  if (cc_major) *cc_major = 0;
  if (cc_minor) *cc_minor = 0;
  if (total_mem) *total_mem = vkrt::device_memory();
  return 0;
}

extern "C" void mvcuda_cache_clear()
{
  mvk::keyed_clear();
  vkrt::pool_trim();
}

// quick check that the device runs a kernel: merge of two small frames against the CPU formula
extern "C" int mvcuda_selftest()
{
  if (!vkrt::ready())
    return -1;
  MvCudaFrameFormat f = {};
  f.width = 64;
  f.height = 8;
  f.planes = 1;
  f.bytes_per_sample = 1;
  f.bits_per_sample = 8;
  f.x_ratio_uv = f.y_ratio_uv = 1;
  unsigned char a[64 * 8], b[64 * 8], r[64 * 8];
  for (int i = 0; i < 64 * 8; i++)
  {
    a[i] = (unsigned char)(i * 7);
    b[i] = (unsigned char)(255 - i * 3);
  }
  const void *pa[3] = { a, nullptr, nullptr }, *pb[3] = { b, nullptr, nullptr };
  void *pr[3] = { r, nullptr, nullptr };
  const int pitch[3] = { 64, 0, 0 };
  MvCudaTokenHeader ha, hb, hr;
  if (mvcuda_frame_upload(&f, pa, pitch, 0, 0, &ha) || mvcuda_frame_upload(&f, pb, pitch, 0, 0, &hb))
    return -2;
  const int mode[3] = { 1, 0, 0 }, tail[3] = { 64, 0, 0 };
  if (mvcuda_merge(ha.token, hb.token, mode, 12000, 32767 - 12000, tail, 0, 0, &hr) || mvcuda_frame_download(hr.token, pr, pitch))
    return -2;
  for (int i = 0; i < 64 * 8; i++)
    if (r[i] != ((a[i] * (32767 - 12000) + b[i] * 12000 + 16384) >> 15))
      return -3;
  return 0;
}
