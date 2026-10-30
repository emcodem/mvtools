// Pooled device memory, stream pool and the super frames kept with device frames, of mvtools_cuda.dll.

#include "mvcuda_internal.h"

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace mvc
{
  namespace
  {
    struct DeviceChoice
    {
      int index = -1;
      int count = 0;
    };

    const DeviceChoice &device_choice()
    {
      static const DeviceChoice c = [] {
        DeviceChoice r;
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess)
          return r;
        const char *e = std::getenv("MVCUDA_DEVICE");
        const int wanted = e ? std::atoi(e) : -1;
        for (int i = 0; i < n; i++)
        {
          cudaDeviceProp p;
          if (cudaGetDeviceProperties(&p, i) != cudaSuccess || p.major < 8)
            continue;
          r.count++;
          if (r.index < 0 && (wanted < 0 || wanted == i))
            r.index = i;
        }
        return r;
      }();
      return c;
    }
  }

  int usable_device_count()
  {
    return device_choice().index >= 0 ? device_choice().count : 0;
  }

  bool use_device()
  {
    const int d = device_choice().index;
    if (d < 0)
      return false;
    // waiting for a stream spins a CPU core by default; blocking sync lets the thread sleep (set once, before the
    // context is created). MVCUDA_SPIN=1 keeps the spinning mode.
    static const bool flags_set = [] {
      if (!std::getenv("MVCUDA_SPIN"))
        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
      return true;
    }();
    (void)flags_set;
    return cudaSetDevice(d) == cudaSuccess;
  }

  namespace
  {
    std::mutex g_pool_mutex;
    std::multimap<size_t, void *> g_pool;           // free buffers by size
    std::mutex g_pin_mutex;
    std::multimap<size_t, void *> g_pin;            // free pinned staging buffers by size
    std::mutex g_stream_mutex;
    std::vector<cudaStream_t> g_streams;            // free streams
  }

  void *dev_alloc(size_t bytes)
  {
    {
      std::lock_guard<std::mutex> lk(g_pool_mutex);
      auto it = g_pool.find(bytes);
      if (it != g_pool.end())
      {
        void *p = it->second;
        g_pool.erase(it);
        return p;
      }
    }
    void *p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess)
    {
      // out of memory: give the pooled buffers back and try once more
      pool_trim();
      if (cudaMalloc(&p, bytes) != cudaSuccess)
        return nullptr;
    }
    return p;
  }

  void dev_free(void *p, size_t bytes)
  {
    if (!p)
      return;
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    g_pool.emplace(bytes, p);
  }

  void *pin_alloc(size_t bytes)
  {
    static const bool off = std::getenv("MVCUDA_NOPIN") != nullptr; // A/B test switch: pageable copies as before
    if (off)
      return nullptr;
    {
      std::lock_guard<std::mutex> lk(g_pin_mutex);
      auto it = g_pin.find(bytes);
      if (it != g_pin.end())
      {
        void *p = it->second;
        g_pin.erase(it);
        return p;
      }
    }
    void *p = nullptr;
    if (std::getenv("MVCUDA_TIMING"))
      std::fprintf(stderr, "mvcuda pin_alloc miss %zu bytes\n", bytes);
    if (cudaHostAlloc(&p, bytes, cudaHostAllocPortable) != cudaSuccess)
    {
      cudaGetLastError();
      return nullptr;
    }
    return p;
  }

  void pin_free(void *p, size_t bytes)
  {
    if (!p)
      return;
    std::lock_guard<std::mutex> lk(g_pin_mutex);
    g_pin.emplace(bytes, p);
  }

  void pool_trim()
  {
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    for (auto &e : g_pool)
      cudaFree(e.second);
    g_pool.clear();
    std::lock_guard<std::mutex> lk2(g_pin_mutex);
    for (auto &e : g_pin)
      cudaFreeHost(e.second);
    g_pin.clear();
  }

  StreamLease::StreamLease()
    : s(nullptr)
  {
    {
      std::lock_guard<std::mutex> lk(g_stream_mutex);
      if (!g_streams.empty())
      {
        s = g_streams.back();
        g_streams.pop_back();
        return;
      }
    }
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess)
      s = nullptr; // falls back to the legacy default stream
  }

  StreamLease::~StreamLease()
  {
    if (!s)
      return;
    std::lock_guard<std::mutex> lk(g_stream_mutex);
    g_streams.push_back(s);
  }

  DevSuper::~DevSuper()
  {
    event_put(ready);
    for (DevPlane &dp : pl)
      for (void *&b : dp.sub)
        if (b)
        {
          dev_free(b, (size_t)dp.d.ew * dp.d.eh * esize);
          b = nullptr;
        }
  }

  // ---- super frames kept with their device frames (section 25, M2) ---------------------------------------------------
  // They replace the earlier frame cache of their own (24 super frames by content key, outside the registry budget, also
  // built from host memory): the pictures are device frames now (uploaded once, mvcuda_frame_upload_keyed), and a super
  // frame lives as long as its picture, counted in the registry budget. MVCUDA_CACHE_FRAMES is no longer read.

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
    // a stored super frame with parameters e serves a request for p (see frame_super); the luma level sizes depend on the
    // chroma ratios too (level_size), so they must match also without chroma
    bool covers(const MvCudaSuperParams &e, const MvCudaSuperParams &p)
    {
      return e.width == p.width && e.height == p.height && e.x_ratio_uv == p.x_ratio_uv && e.y_ratio_uv == p.y_ratio_uv
        && e.hpad == p.hpad && e.vpad == p.vpad && e.pel == p.pel && e.sharp == p.sharp && e.bits_per_pixel == p.bits_per_pixel
        && e.chroma >= p.chroma && e.levels >= p.levels && (p.levels == 1 || e.rfilter == p.rfilter);
    }
  }

  int frame_super(SuperBuildFn build, const std::shared_ptr<DevFrame> &fr, const MvCudaSuperParams &p, cudaStream_t s,
    std::shared_ptr<DevSuper> &out)
  {
    if (!fr || !super_fits(*fr, p, fr->f.bytes_per_sample))
      return -4;
    std::shared_ptr<DevSuper> found;
    size_t made = 0;
    {
      // concurrent requests for one frame: the first builds (only queues kernels, no host wait), the others then find it
      std::lock_guard<std::mutex> lk(fr->extra_m);
      for (const auto &e : fr->supers)
        if (covers(e->p, p))
        {
          found = e;
          break;
        }
      if (!found)
      {
        const void *src[3] = {};
        int pitch[3] = {};
        for (int k = 0; k < fr->f.planes; k++)
        {
          src[k] = fr->plane[k];
          pitch[k] = (int)fr->pitch[k];
        }
        auto sp = std::make_shared<DevSuper>();
        const int rc = build(p, src, pitch, *sp, s);
        if (rc)
          return rc;
        fr->supers.push_back(sp);
        out = sp;
        made = super_bytes(p, fr->f.bytes_per_sample);
      }
    }
    if (made)
    {
      frame_account_extra(*fr, made);
      return 0;
    }
    out = found;
    // built on another stream without a host wait: s starts after the build
    if (found->ready && cudaStreamWaitEvent(s, found->ready, 0) != cudaSuccess)
      return -2;
    return 0;
  }
}

extern "C" void mvcuda_cache_clear()
{
  mvc::keyed_clear();
  mvc::pool_trim();
}
