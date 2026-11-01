// Device frames behind tokens for mvtools_vulkan: the registry with its byte budget and LRU eviction, host upload /
// download, and the producer table used to make an evicted frame again. The registry logic is
// Sources/cuda/mvcuda_frames.cu's (budget growth, ghosts, redirects, pins, per-thread holds), unchanged; only memory and
// copies use vkrt.

#include "vk_internal.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace mvk
{
  namespace
  {
    struct Slot
    {
      std::shared_ptr<DevFrame> fr;
      std::list<unsigned long long>::iterator lru;
      unsigned long long used_at; // g_put_bytes at the last put / lookup (reuse distance, budget_on_miss)
      unsigned long long key;     // frame_key (g_latest)
      unsigned long long ukey;    // upload key (g_keyed), 0 = none
    };

    std::mutex g_mutex;
    std::unordered_map<unsigned long long, Slot> g_frames;
    std::list<unsigned long long> g_lru;     // front = most recently used
    size_t g_bytes = 0;
    struct Pin
    {
      std::shared_ptr<DevFrame> fr;
      int count;
    };
    std::unordered_map<unsigned long long, Pin> g_pins; // mvcuda_token_pin: reachable also after eviction
    std::atomic<unsigned long long> g_next_token{ 1 };
    MvCudaFrameStats g_stats = {};

    struct Producer
    {
      MvCudaRegenFn fn;
      void *user;
    };
    std::mutex g_prod_mutex;
    std::unordered_map<unsigned long long, Producer> g_producers;
    std::atomic<unsigned long long> g_next_producer{ 1 };

    // The last frames each thread created stay reachable for that thread even when other threads evict them from the
    // registry: a thread that has just made (or regenerated) a frame can always read it, also with a tiny budget and many
    // Prefetch threads. Without this, a regenerated frame could be evicted again before its download.
    const int HOLDS = 8;
    struct Held
    {
      unsigned long long token = 0;
      std::shared_ptr<DevFrame> fr;
    };
    thread_local Held t_held[HOLDS];
    thread_local unsigned t_held_pos = 0;

    // Byte budget of the registry. MVCUDA_CACHE_MB: fixed. Otherwise it starts at BUDGET_START and grows while frames
    // have to be made again after their eviction (budget_count_put / budget_count_regen), up to budget_cap(): a chain needs
    // from about 256 MB (QTGMC Fast PAL) to well over 512 MB (Slower / Placebo HD), and a budget far above the working set
    // only keeps VRAM (GPU_PIPELINE_PLAN.md sections 23, 24).
    const size_t BUDGET_START = (size_t)256 << 20;
    const long long *budget_fixed_mb()
    {
      static const long long v = [] {
        const char *e = std::getenv("MVCUDA_CACHE_MB");
        return e ? std::atoll(e) : 0LL;
      }();
      return v ? &v : nullptr;
    }
    size_t budget_cap();
    std::atomic<size_t> g_budget{ 0 };

    size_t budget()
    {
      size_t b = g_budget.load(std::memory_order_relaxed);
      if (b)
        return b;
      if (const long long *mb = budget_fixed_mb())
        b = (size_t)(*mb < 1 ? 1 : *mb) << 20;
      else
      {
        const size_t cap = budget_cap();
        b = BUDGET_START < cap ? BUDGET_START : cap;
      }
      size_t expected = 0;
      g_budget.compare_exchange_strong(expected, b);
      return g_budget.load();
    }

    // Growth by reuse distance (under g_mutex). Below the working set frames are made again many times per output frame,
    // and that feeds itself (every regenerated frame evicts another one): QTGMC Slower HD made 160,000 regenerations and
    // took 8x the time at 768 MB, 6 regenerations at 1024 MB. The registry remembers evicted tokens ("ghosts") with the
    // bytes put up to their last use; a lookup of an evicted token gives the bytes put since then, the budget LRU would
    // have needed to keep it. A few such lookups happen at any budget, with distances that grow with the run time (frames
    // evicted early, asked for again much later through AviSynth's own caches: about 5 per 512 frames made); growing on
    // every one ran the budget to the cap. A shortage gives several per frame made, and windows of 512 reacted too late
    // (HD runs fell into lasting regeneration storms). So per window of BUDGET_WINDOW frames made, at least BUDGET_MISSES
    // lookups of evicted tokens raise the budget to 1.25 times their median distance, at least by half (the cap if that
    // is beyond it). Lookups made inside a regeneration (of its inputs) do not count: one late request of an old frame
    // cascades into many (QTGMC Placebo PAL: 2 regenerations at a fixed 384 MB, 1,200 and the cap when they counted).
    // The first frame sets the start: at least BUDGET_START_FRAMES frames of its size, but not above BUDGET_START_MAX (UHD
    // frames: 128 of them are 4 GB, while the uc_FRC chain, its super pyramids counted, runs as fast with 1 GB; larger
    // needs grow the budget as above).
    const unsigned BUDGET_WINDOW = 64, BUDGET_MISSES = 8, BUDGET_START_FRAMES = 128;
    const size_t BUDGET_START_MAX = (size_t)1024 << 20;
    unsigned long long g_put_bytes = 0;                       // bytes of all frames put so far
    unsigned g_window_puts = 0;
    std::vector<unsigned long long> *g_window_dist = new std::vector<unsigned long long>; // miss distances of the window
    struct Ghost
    {
      unsigned long long used_at, key; // g_put_bytes at the last use; frame_key of its (producer, frame) or 0
    };
    std::unordered_map<unsigned long long, Ghost> *g_ghost = new std::unordered_map<unsigned long long, Ghost>;
    // Newest live token of each (producer, frame): AviSynth's frame caches keep handing out a token frame after its device
    // frame was evicted; when the frame was made again meanwhile (a regeneration, under a new token), a lookup of the old
    // token finds the new one. Without this every consumer of the cached frame made it again: once begun, regeneration
    // storms went on after the budget had grown (QTGMC Placebo HD: up to 60,000 regenerations at a 4 GB budget that,
    // fixed from the start, needed none). Same content: a producer makes frame n bit-identically every time.
    std::unordered_map<unsigned long long, unsigned long long> *g_latest = new std::unordered_map<unsigned long long, unsigned long long>;
    unsigned long long frame_key(unsigned long long producer, int frame)
    {
      return producer ? (producer << 32) ^ (unsigned)frame : 0;
    }
    // Pictures uploaded under a content key (mvcuda_frame_upload_keyed): key -> token of the live frame. The GPU filters of
    // mvtools that read the same host picture (both MAnalyse directions, MRecalculate, MFlowFps, ...) share one upload.
    std::unordered_map<unsigned long long, unsigned long long> *g_keyed = new std::unordered_map<unsigned long long, unsigned long long>;
    std::list<unsigned long long> *g_ghost_order = new std::list<unsigned long long>; // eviction order, oldest first
    const size_t GHOSTS = 65536;
    thread_local int t_regen_depth = 0; // regenerations this thread is inside (mvcuda_token_regenerate)

    void ghost_add(unsigned long long token, unsigned long long used_at, unsigned long long key)
    {
      (*g_ghost)[token] = Ghost{ used_at, key };
      g_ghost_order->push_back(token);
      if (g_ghost_order->size() > GHOSTS)
      {
        g_ghost->erase(g_ghost_order->front());
        g_ghost_order->pop_front();
      }
    }

    void budget_on_miss(unsigned long long token)
    {
      if (budget_fixed_mb())
        return;
      auto it = g_ghost->find(token);
      if (it != g_ghost->end() && g_window_dist->size() < 4096)
        g_window_dist->push_back(g_put_bytes - it->second.used_at);
    }

    // A consumer found its frame near the end of the LRU order (reuse distance above 80% of the budget): the working set
    // reaches the budget, so grow to 1.5 times that distance before evictions turn into regenerations. Misses come too
    // late for that (a regeneration storm has begun, and once begun it went on after the budget had grown) and include
    // the late requests of old frames; hits do not.
    void budget_on_hit(unsigned long long dist)
    {
      if (budget_fixed_mb())
        return;
      const size_t b = budget();
      if (dist * 5 <= (unsigned long long)b * 4)
        return;
      const size_t want = (size_t)(dist + dist / 2), cap = budget_cap();
      if (want > b)
        g_budget.store(want < cap ? want : cap);
    }

    void budget_on_put(size_t frame_bytes)
    {
      if (budget_fixed_mb())
        return;
      if (g_put_bytes == frame_bytes) // the first frame
      {
        size_t start = (size_t)BUDGET_START_FRAMES * frame_bytes;
        const size_t cap = budget_cap();
        if (start > BUDGET_START_MAX)
          start = BUDGET_START_MAX;
        if (start > budget())
          g_budget.store(start < cap ? start : cap);
      }
      if (++g_window_puts < BUDGET_WINDOW)
        return;
      std::vector<unsigned long long> &d = *g_window_dist;
      if (d.size() >= BUDGET_MISSES)
      {
        std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
        const unsigned long long med = d[d.size() / 2];
        const size_t cap = budget_cap(), b = budget();
        size_t want = (size_t)(med + med / 4);
        if (want < b + b / 2)
          want = b + b / 2;
        g_budget.store(want < cap ? want : cap);
      }
      d.clear();
      g_window_puts = 0;
    }

    size_t budget_cap()
    {
      static const size_t b = [] {
        if (const long long *mb = budget_fixed_mb())
          return (size_t)(*mb < 1 ? 1 : *mb) << 20;
        // Tokens are read shortly after they are made, so the registry needs a few hundred frames, not the whole card:
        // a large budget means freed frames almost never come back to the pipeline pool, and every new frame costs a
        // cudaMalloc, which synchronises the device and stalls every thread's launches (measured: 15900 mallocs and
        // 100 us per launch instead of 13 us in QTGMC Fast). Frames evicted too early are made again by their producers.
        // The budget grows only as far as a chain needs it (budget_count_put), so the cap is the limit for the largest
        // chains: QTGMC Placebo HD 4:2:2 needs a little more than 2 GB (35 regenerations at 2 GB, sometimes a lasting
        // regeneration storm; none at 3 GB). A third of the free memory at first use, at most 4 GB.
        const size_t free_b = (size_t)vkrt::device_free_memory();
        if (free_b)
        {
          const size_t q = free_b / 3;
          const size_t cap = (size_t)4096 << 20;
          return q < cap ? q : cap;
        }
        return (size_t)256 << 20;
      }();
      return b;
    }

    int current_device()
    {
      return 0;
    }

  }
  bool valid_format(const MvCudaFrameFormat &f)
    {
      return f.width > 0 && f.height > 0 && (f.planes == 1 || f.planes == 3) && (f.bytes_per_sample == 1 || f.bytes_per_sample == 2)
        && f.bits_per_sample >= 8 && f.bits_per_sample <= 16 && (f.bits_per_sample <= 8) == (f.bytes_per_sample == 1)
        && (f.planes == 1 || ((f.x_ratio_uv == 1 || f.x_ratio_uv == 2) && (f.y_ratio_uv == 1 || f.y_ratio_uv == 2)
          && f.width % f.x_ratio_uv == 0 && f.height % f.y_ratio_uv == 0));
    }

  DevFrame::~DevFrame()
  {
    vkrt::dev_free(plane[0], bytes); // one buffer for all planes
  }

  std::shared_ptr<DevFrame> frame_alloc(const MvCudaFrameFormat &f)
  {
    if (!valid_format(f) || !vkrt::ready())
      return nullptr;
    auto fr = std::make_shared<DevFrame>();
    fr->f = f;
    // one buffer, planes at 256-byte aligned offsets: a frame moves in one copy
    size_t off[3] = {}, total = 0;
    for (int p = 0; p < f.planes; p++)
    {
      fr->w[p] = p == 0 ? f.width : f.width / f.x_ratio_uv;
      fr->h[p] = p == 0 ? f.height : f.height / f.y_ratio_uv;
      fr->pitch[p] = (size_t)fr->w[p] * f.bytes_per_sample;
      off[p] = total;
      total += (fr->pitch[p] * (size_t)fr->h[p] + 255) & ~(size_t)255;
    }
    const dptr base = vkrt::dev_alloc(total);
    if (!base)
      return nullptr;
    fr->bytes = total;
    for (int p = 0; p < f.planes; p++)
      fr->plane[p] = base + off[p];
    return fr;
  }

  void frame_put(const std::shared_ptr<DevFrame> &fr, unsigned long long producer, int frame, MvCudaTokenHeader &out)
  {
    const unsigned long long token = g_next_token.fetch_add(1);
    fr->token = token;
    Held &hold = t_held[t_held_pos++ % HOLDS];
    hold.token = token;
    hold.fr = fr;
    std::vector<std::shared_ptr<DevFrame>> evicted;
    {
      std::lock_guard<std::mutex> lk(g_mutex);
      g_lru.push_front(token);
      g_put_bytes += fr->bytes;
      const unsigned long long key = frame_key(producer, frame);
      g_frames[token] = Slot{ fr, g_lru.begin(), g_put_bytes, key, 0 };
      if (key)
        (*g_latest)[key] = token;
      g_bytes += fr->bytes;
      budget_on_put(fr->bytes);
      // evict the least recently used frames over the budget, never the one just added
      const size_t cap = budget();
      while (g_bytes > cap && g_lru.size() > 1)
      {
        const unsigned long long victim = g_lru.back();
        auto it = g_frames.find(victim);
        if (it != g_frames.end())
        {
          g_bytes -= it->second.fr->bytes + it->second.fr->extra_acct;
          it->second.fr->extra_acct = 0;
          ghost_add(victim, it->second.used_at, it->second.key);
          auto lt = g_latest->find(it->second.key);
          if (lt != g_latest->end() && lt->second == victim)
            g_latest->erase(lt);
          if (it->second.ukey)
          {
            auto kt = g_keyed->find(it->second.ukey);
            if (kt != g_keyed->end() && kt->second == victim)
              g_keyed->erase(kt);
          }
          evicted.push_back(it->second.fr);
          g_frames.erase(it); // the memory goes back to the pool when the last holder lets go
        }
        g_lru.pop_back();
        g_stats.evictions++;
      }
    }
    // An evicted frame can stay reachable (the holds of the thread that made it, pins, readers in flight), but the data
    // kept with it (super pyramids, sub-pel planes: several times the picture) leaves the budget with it: drop it now,
    // a later reader makes it again. Readers in flight hold their own references (and pipeline ops pipe_keep them).
    for (const auto &e : evicted)
    {
      std::lock_guard<std::mutex> lk(e->extra_m);
      e->supers.clear();
      e->subpel.reset();
    }
    std::memset(&out, 0, sizeof(out));
    out.magic = MVCUDA_TOKEN_MAGIC;
    out.layout = MVCUDA_TOKEN_LAYOUT;
    out.device = current_device();
    out.token = token;
    out.producer = producer;
    out.frame = frame;
    out.format = fr->f;
  }

  std::shared_ptr<DevFrame> frame_get(unsigned long long token)
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_frames.find(token);
    if (it == g_frames.end())
    {
      auto pin = g_pins.find(token);
      if (pin != g_pins.end())
        return pin->second.fr; // evicted, but pinned by a consumer
      for (const Held &h : t_held)
        if (h.token == token)
          return h.fr; // evicted, but made by this thread a moment ago
      // evicted: a newer token of the same (producer, frame) serves as well
      auto gh = g_ghost->find(token);
      if (gh != g_ghost->end() && gh->second.key)
      {
        auto lt = g_latest->find(gh->second.key);
        if (lt != g_latest->end())
        {
          auto nw = g_frames.find(lt->second);
          if (nw != g_frames.end())
          {
            g_lru.splice(g_lru.begin(), g_lru, nw->second.lru);
            nw->second.used_at = g_put_bytes;
            g_stats.redirects++;
            return nw->second.fr;
          }
        }
      }
      g_stats.lookups_gone++;
      if (t_regen_depth == 0) // a consumer's own lookup, not one of a regeneration cascade
        budget_on_miss(token);
      return nullptr;
    }
    g_lru.splice(g_lru.begin(), g_lru, it->second.lru);
    if (t_regen_depth == 0)
      budget_on_hit(g_put_bytes - it->second.used_at);
    it->second.used_at = g_put_bytes;
    return it->second.fr;
  }

  void frame_account_extra(DevFrame &fr, size_t bytes)
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_frames.find(fr.token);
    if (it == g_frames.end() || it->second.fr.get() != &fr)
      return; // evicted already: the extra data lives as long as the holders of the frame
    fr.extra_acct += bytes;
    g_bytes += bytes; // the next frame_put evicts over the budget
    g_put_bytes += bytes;
  }
}

namespace
{
  // host <-> device copy of all planes (one Cmd, waited for)
  int copy_planes(mvk::DevFrame &fr, const void *const host_src[3], void *const host_dst[3], const int pitch_bytes[3])
  {
    vkrt::Cmd c;
    for (int p = 0; p < fr.f.planes; p++)
    {
      if (host_src)
        c.upload2d(fr.plane[p], fr.pitch[p], host_src[p], (size_t)pitch_bytes[p], fr.pitch[p], (size_t)fr.h[p]);
      else
        c.download2d(host_dst[p], (size_t)pitch_bytes[p], fr.plane[p], fr.pitch[p], fr.pitch[p], (size_t)fr.h[p]);
    }
    return c.submit_wait();
  }
}

extern "C" int mvcuda_frame_upload(const MvCudaFrameFormat *f, const void *const src[3], const int src_pitch_bytes[3],
  unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  if (!f || !src || !src_pitch_bytes || !out)
    return -4;
  if (!vkrt::ready())
    return -1;
  auto fr = mvk::frame_alloc(*f);
  if (!fr)
    return -4;
  for (int p = 0; p < f->planes; p++)
    if (!src[p] || src_pitch_bytes[p] < (int)fr->pitch[p])
      return -4;
  const int rc = copy_planes(*fr, src, nullptr, src_pitch_bytes);
  if (rc)
    return rc;
  mvk::frame_put(fr, producer, frame, *out);
  std::lock_guard<std::mutex> lk(mvk::g_mutex);
  mvk::g_stats.uploads++;
  return 0;
}

extern "C" int mvcuda_frame_upload_keyed(const MvCudaFrameFormat *f, const void *const src[3], const int src_pitch_bytes[3],
  unsigned long long key, MvCudaTokenHeader *out)
{
  if (!f || !src || !src_pitch_bytes || !out)
    return -4;
  if (!vkrt::ready())
    return -1;
  if (key)
  {
    std::lock_guard<std::mutex> lk(mvk::g_mutex);
    auto kt = mvk::g_keyed->find(key);
    auto it = kt != mvk::g_keyed->end() ? mvk::g_frames.find(kt->second) : mvk::g_frames.end();
    if (it != mvk::g_frames.end())
    {
      const MvCudaFrameFormat &g = it->second.fr->f;
      if (g.width == f->width && g.height == f->height && g.planes == f->planes && g.bytes_per_sample == f->bytes_per_sample
        && g.bits_per_sample == f->bits_per_sample && (g.planes == 1 || (g.x_ratio_uv == f->x_ratio_uv && g.y_ratio_uv == f->y_ratio_uv)))
      {
        // a hit counts as a use of the frame (LRU order, budget growth) and stays reachable for this thread
        mvk::g_lru.splice(mvk::g_lru.begin(), mvk::g_lru, it->second.lru);
        if (mvk::t_regen_depth == 0)
          mvk::budget_on_hit(mvk::g_put_bytes - it->second.used_at);
        it->second.used_at = mvk::g_put_bytes;
        mvk::Held &hold = mvk::t_held[mvk::t_held_pos++ % mvk::HOLDS];
        hold.token = kt->second;
        hold.fr = it->second.fr;
        std::memset(out, 0, sizeof(*out));
        out->magic = MVCUDA_TOKEN_MAGIC;
        out->layout = MVCUDA_TOKEN_LAYOUT;
        out->device = mvk::current_device();
        out->token = kt->second;
        out->format = g;
        mvk::g_stats.upload_hits++;
        return 0;
      }
    }
  }
  const int rc = mvcuda_frame_upload(f, src, src_pitch_bytes, 0, 0, out);
  if (rc == 0 && key)
  {
    std::lock_guard<std::mutex> lk(mvk::g_mutex);
    auto it = mvk::g_frames.find(out->token);
    if (it != mvk::g_frames.end())
    {
      it->second.ukey = key;
      (*mvk::g_keyed)[key] = out->token;
    }
  }
  return rc;
}

namespace mvk
{
  void keyed_clear()
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_keyed->clear();
    for (auto &f : g_frames)
      f.second.ukey = 0;
  }

  std::shared_ptr<DevFrame> upload_picture(const MvCudaSuperParams &p, int esize, const void *const src[3], const int pitch[3])
  {
    const MvCudaFrameFormat f = super_format(p, esize);
    MvCudaTokenHeader h;
    if (mvcuda_frame_upload(&f, src, pitch, 0, 0, &h) != 0)
      return nullptr;
    return frame_get(h.token); // reachable for this thread even when already evicted (t_held)
  }
}

extern "C" int mvcuda_frame_download(unsigned long long token, void *const dst[3], const int dst_pitch_bytes[3])
{
  if (!dst || !dst_pitch_bytes)
    return -4;
  if (!vkrt::ready())
    return -1;
  auto fr = mvk::frame_get(token);
  if (!fr)
    return MVCUDA_TOKEN_GONE;
  for (int p = 0; p < fr->f.planes; p++)
    if (!dst[p] || dst_pitch_bytes[p] < (int)fr->pitch[p])
      return -4;
  const int rc = copy_planes(*fr, nullptr, dst, dst_pitch_bytes);
  if (rc == 0)
  {
    std::lock_guard<std::mutex> lk(mvk::g_mutex);
    mvk::g_stats.downloads++;
  }
  return rc;
}

extern "C" unsigned long long mvcuda_producer_register(MvCudaRegenFn fn, void *user)
{
  if (!fn)
    return 0;
  const unsigned long long id = mvk::g_next_producer.fetch_add(1);
  std::lock_guard<std::mutex> lk(mvk::g_prod_mutex);
  mvk::g_producers[id] = mvk::Producer{ fn, user };
  return id;
}

extern "C" void mvcuda_producer_unregister(unsigned long long producer)
{
  std::lock_guard<std::mutex> lk(mvk::g_prod_mutex);
  mvk::g_producers.erase(producer);
}

extern "C" int mvcuda_token_regenerate(unsigned long long producer, int frame, void *env, MvCudaTokenHeader *out)
{
  if (!out)
    return -4;
  mvk::Producer pr;
  {
    std::lock_guard<std::mutex> lk(mvk::g_prod_mutex);
    auto it = mvk::g_producers.find(producer);
    if (it == mvk::g_producers.end())
      return -7;
    pr = it->second;
  }
  // called outside the lock: the producer makes the frame through the normal upload / kernel paths. A filter is not
  // destroyed while frames of its clip are still being requested, so the entry stays valid for the call.
  mvk::t_regen_depth++; // lookups inside a regeneration do not count for the budget (budget_on_miss)
  const int rc = pr.fn(pr.user, frame, env, out);
  mvk::t_regen_depth--;
  if (rc == 0)
  {
    std::lock_guard<std::mutex> lk(mvk::g_mutex);
    mvk::g_stats.regenerations++;
  }
  return rc;
}

extern "C" int mvcuda_token_pin(unsigned long long token)
{
  auto fr = mvk::frame_get(token);
  if (!fr)
    return MVCUDA_TOKEN_GONE;
  std::lock_guard<std::mutex> lk(mvk::g_mutex);
  auto it = mvk::g_pins.find(token);
  if (it != mvk::g_pins.end())
    it->second.count++;
  else
    mvk::g_pins[token] = mvk::Pin{ fr, 1 };
  return 0;
}

extern "C" void mvcuda_token_unpin(unsigned long long token)
{
  std::lock_guard<std::mutex> lk(mvk::g_mutex);
  auto it = mvk::g_pins.find(token);
  if (it != mvk::g_pins.end() && --it->second.count <= 0)
    mvk::g_pins.erase(it);
}

extern "C" void mvcuda_frame_stats(MvCudaFrameStats *s)
{
  if (!s)
    return;
  std::lock_guard<std::mutex> lk(mvk::g_mutex);
  *s = mvk::g_stats;
  s->bytes = mvk::g_bytes;
  s->budget = mvk::budget();
  s->frames = mvk::g_frames.size();
}

