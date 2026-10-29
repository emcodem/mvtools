// Pooled device memory, stream pool and the super-frame cache of mvtools_cuda.dll.

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

  // ---- frame cache ------------------------------------------------------------------------------------------------
  namespace
  {
    struct CacheKey
    {
      unsigned long long key;
      int width, height, xr, yr, hpad, vpad, pel, levels, bits, chroma, esize, rfilter;
      bool operator==(const CacheKey &o) const
      {
        return key == o.key && width == o.width && height == o.height && xr == o.xr && yr == o.yr && hpad == o.hpad
          && vpad == o.vpad && pel == o.pel && levels == o.levels && bits == o.bits && chroma == o.chroma
          && esize == o.esize && rfilter == o.rfilter;
      }
    };
    struct CacheKeyHash
    {
      size_t operator()(const CacheKey &k) const
      {
        size_t h = (size_t)k.key * 0x9E3779B97F4A7C15ull;
        h ^= (size_t)(k.width * 31 + k.height) + (h << 6) + (h >> 2);
        h ^= (size_t)(k.pel * 7 + k.levels * 13 + k.bits) + (h << 6) + (h >> 2);
        return h;
      }
    };
    struct Entry
    {
      std::shared_ptr<DevSuper> super; // null while being built
      bool failed = false;
      std::list<CacheKey>::iterator lru;
    };

    std::mutex g_cache_mutex;
    std::condition_variable g_cache_cv;
    std::unordered_map<CacheKey, Entry, CacheKeyHash> g_cache;
    std::list<CacheKey> g_lru; // front = most recently used

    size_t cache_capacity()
    {
      static const size_t cap = [] {
        const char *e = std::getenv("MVCUDA_CACHE_FRAMES");
        const int n = e ? std::atoi(e) : 24;
        return (size_t)(n < 2 ? 2 : n);
      }();
      return cap;
    }
  }

  int super_get(SuperBuildFn build, const MvCudaSuperParams &p, unsigned long long key, const void *const src[3],
    const int src_pitch_bytes[3], cudaStream_t s, std::shared_ptr<DevSuper> &out)
  {
    if (key == 0)
    {
      auto sp = std::make_shared<DevSuper>();
      const int rc = build(p, src, src_pitch_bytes, *sp, s);
      if (rc == 0)
        out = sp;
      return rc;
    }
    const CacheKey ck = { key, p.width, p.height, p.x_ratio_uv, p.y_ratio_uv, p.hpad, p.vpad, p.pel, p.levels,
      p.bits_per_pixel, p.chroma, p.bits_per_pixel <= 8 ? 1 : 2, p.rfilter };
    std::unique_lock<std::mutex> lk(g_cache_mutex);
    for (;;)
    {
      auto it = g_cache.find(ck);
      if (it == g_cache.end())
        break;
      if (it->second.super)
      {
        g_lru.splice(g_lru.begin(), g_lru, it->second.lru);
        out = it->second.super;
        lk.unlock();
        // built by another thread without a host wait: this stream starts after the build
        if (out->ready && cudaStreamWaitEvent(s, out->ready, 0) != cudaSuccess)
          return -2;
        return 0;
      }
      if (it->second.failed)
        break;
      g_cache_cv.wait(lk); // another thread is building it
    }
    // we build it: publish a pending entry first
    g_lru.push_front(ck);
    Entry &e = g_cache[ck];
    e.super.reset();
    e.failed = false;
    e.lru = g_lru.begin();
    lk.unlock();

    auto sp = std::make_shared<DevSuper>();
    const int rc = build(p, src, src_pitch_bytes, *sp, s);

    lk.lock();
    auto it = g_cache.find(ck);
    if (rc != 0)
    {
      if (it != g_cache.end())
      {
        g_lru.erase(it->second.lru);
        g_cache.erase(it);
      }
      g_cache_cv.notify_all();
      return rc;
    }
    if (it != g_cache.end())
      it->second.super = sp;
    out = sp;
    // evict the least recently used finished entries
    while (g_cache.size() > cache_capacity() && !g_lru.empty())
    {
      auto victim = g_cache.find(g_lru.back());
      if (victim != g_cache.end() && !victim->second.super)
        break; // pending: leave it
      g_cache.erase(g_lru.back());
      g_lru.pop_back();
    }
    g_cache_cv.notify_all();
    return 0;
  }

  void cache_clear()
  {
    std::lock_guard<std::mutex> lk(g_cache_mutex);
    for (auto it = g_cache.begin(); it != g_cache.end();)
    {
      if (it->second.super)
      {
        g_lru.erase(it->second.lru);
        it = g_cache.erase(it);
      }
      else
        ++it;
    }
  }
}

extern "C" void mvcuda_cache_clear()
{
  mvc::cache_clear();
  mvc::pool_trim();
}
