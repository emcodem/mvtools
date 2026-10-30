// Device frames behind tokens (GPU_PIPELINE_PLAN.md section 4, milestone D): the registry with its byte budget and
// LRU eviction, host upload / download, and the producer table used to make an evicted frame again.

#include "mvcuda_internal.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <algorithm>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace mvc
{
  namespace
  {
    struct Slot
    {
      std::shared_ptr<DevFrame> fr;
      std::list<unsigned long long>::iterator lru;
      unsigned long long used_at; // g_put_bytes at the last put / lookup (reuse distance, budget_on_miss)
      unsigned long long key;     // frame_key (g_latest)
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
    // The first frame sets the start: at least BUDGET_START_FRAMES frames of its size.
    const unsigned BUDGET_WINDOW = 64, BUDGET_MISSES = 8, BUDGET_START_FRAMES = 128;
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
        const size_t start = (size_t)BUDGET_START_FRAMES * frame_bytes, cap = budget_cap();
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
        size_t free_b = 0, total_b = 0;
        if (use_device() && cudaMemGetInfo(&free_b, &total_b) == cudaSuccess)
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
      int d = -1;
      cudaGetDevice(&d);
      return d;
    }

    bool valid_format(const MvCudaFrameFormat &f)
    {
      return f.width > 0 && f.height > 0 && (f.planes == 1 || f.planes == 3) && (f.bytes_per_sample == 1 || f.bytes_per_sample == 2)
        && f.bits_per_sample >= 8 && f.bits_per_sample <= 16 && (f.bits_per_sample <= 8) == (f.bytes_per_sample == 1)
        && (f.planes == 1 || ((f.x_ratio_uv == 1 || f.x_ratio_uv == 2) && (f.y_ratio_uv == 1 || f.y_ratio_uv == 2)
          && f.width % f.x_ratio_uv == 0 && f.height % f.y_ratio_uv == 0));
    }
  }

  namespace
  {
    // free pipeline buffers by size. Deliberately never destroyed: the frame registry (a static object too) returns its
    // buffers here while it is destroyed at process exit, and the order of static destruction would otherwise free the
    // pool first.
    struct PipePool
    {
      std::mutex m;
      std::unordered_multimap<size_t, void *> free;
    };
    PipePool &pipe_pool()
    {
      static PipePool *p = new PipePool;
      return *p;
    }
  }

  namespace
  {
    // event objects for the copy ordering; reused (creating and destroying them per copy cost about 0.4 ms per frame
    // under contention). Re-recording is safe: a stream wait binds to the recording current at the time of the wait.
    std::mutex g_event_mutex;
    std::vector<cudaEvent_t> *g_events = new std::vector<cudaEvent_t>; // never destroyed (static destruction order)
  }

  cudaEvent_t event_get()
    {
      {
        std::lock_guard<std::mutex> lk(g_event_mutex);
        if (!g_events->empty())
        {
          cudaEvent_t e = g_events->back();
          g_events->pop_back();
          return e;
        }
      }
      cudaEvent_t e = nullptr;
      // blocking sync: a host wait sleeps instead of spinning (the device flag does not cover events)
      if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming | cudaEventBlockingSync) != cudaSuccess)
        e = nullptr;
      return e;
    }

  void event_put(cudaEvent_t e)
  {
    if (!e)
      return;
    std::lock_guard<std::mutex> lk(g_event_mutex);
    g_events->push_back(e);
  }

  namespace
  {
    // Buffers whose last use is queued on the pipeline stream: pinned staging buffers (their copy) and upload buffers
    // (up_alloc: their readers). They return to their pools once that work has run. Only the submitter thread touches
    // these lists: pin_release_after / up_release collect buffers, pin_seal records one event for the collected batch (when
    // the queue runs empty or the batch is large), pin_reclaim polls the events.
    struct Release
    {
      void *p;
      size_t bytes;
      bool up;    // upload buffer, else pinned staging buffer
    };
    struct PinBatch
    {
      std::vector<Release> bufs;
      cudaEvent_t ev;
    };
    std::vector<Release> *g_unsealed = new std::vector<Release>; // never destroyed (static destruction order)
    std::vector<PinBatch> *g_pending = new std::vector<PinBatch>;

    // upload buffers free on the device, by size
    std::mutex g_up_mutex;
    std::multimap<size_t, void *> *g_up = new std::multimap<size_t, void *>;

    void release_now(const Release &r)
    {
      if (r.up)
      {
        std::lock_guard<std::mutex> lk(g_up_mutex);
        g_up->emplace(r.bytes, r.p);
      }
      else
        pin_free(r.p, r.bytes);
    }

    void pin_reclaim()
    {
      for (size_t i = 0; i < g_pending->size();)
      {
        PinBatch &b = (*g_pending)[i];
        if (cudaEventQuery(b.ev) == cudaSuccess) // cudaErrorNotReady is not recorded as an error
        {
          for (const Release &r : b.bufs)
            release_now(r);
          event_put(b.ev);
          b = std::move(g_pending->back());
          g_pending->pop_back();
        }
        else
          i++;
      }
    }

    void pin_seal(cudaStream_t s)
    {
      if (g_unsealed->empty())
        return;
      PinBatch b;
      b.ev = event_get();
      if (!b.ev || cudaEventRecord(b.ev, s) != cudaSuccess)
      {
        cudaStreamSynchronize(s);
        for (const Release &r : *g_unsealed)
          release_now(r);
        g_unsealed->clear();
        event_put(b.ev);
        return;
      }
      b.bufs.swap(*g_unsealed);
      g_pending->push_back(std::move(b));
    }

    // the pinned buffer p goes back to the pool once the work queued on the pipeline stream so far has run (called by
    // pipeline ops only)
    void pin_release_after(void *p, size_t bytes, cudaStream_t)
    {
      g_unsealed->push_back(Release{ p, bytes, false });
    }

    cudaStream_t upload_stream()
    {
      static cudaStream_t s = [] {
        cudaStream_t st = nullptr;
        if (!use_device() || cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) != cudaSuccess)
          st = nullptr;
        return st;
      }();
      return s;
    }
  }

  // ---- upload buffers ----------------------------------------------------------------------------------------------
  // Host data (source frames, per-block tables) is copied on the upload stream as soon as it is ready, ahead of the
  // pipeline stream, which then only waits for an event (a copy between its kernels costs an engine switch and the copy
  // time on the critical stream). That needs destination buffers no queued pipeline work still uses: up_alloc takes them
  // from a pool that up_free refills only once the pipeline stream has passed the buffer's last use.
  void *up_alloc(size_t bytes)
  {
    {
      std::lock_guard<std::mutex> lk(g_up_mutex);
      auto it = g_up->find(bytes);
      if (it != g_up->end())
      {
        void *p = it->second;
        g_up->erase(it);
        return p;
      }
    }
    void *p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess)
    {
      cudaGetLastError();
      pipe_flush(); // everything released so far becomes free; give the pooled buffers back to the driver
      {
        std::lock_guard<std::mutex> lk(g_up_mutex);
        for (auto &e : *g_up)
          cudaFree(e.second);
        g_up->clear();
      }
      if (cudaMalloc(&p, bytes) != cudaSuccess)
      {
        cudaGetLastError();
        return nullptr;
      }
    }
    return p;
  }

  void up_free(void *p, size_t bytes)
  {
    if (p)
      pipe_submit([p, bytes](cudaStream_t) { g_unsealed->push_back(Release{ p, bytes, true }); });
  }

  // copies pinned host memory to an upload buffer on the upload stream now and orders the pipeline stream after it;
  // the pinned buffer returns to its pool afterwards. False when the copy could not be queued (nothing queued then).
  bool up_copy(void *d, void *pin, size_t bytes)
  {
    const cudaStream_t us = upload_stream();
    cudaEvent_t e = us ? event_get() : nullptr;
    if (!e)
      return false;
    if (cudaMemcpyAsync(d, pin, bytes, cudaMemcpyHostToDevice, us) != cudaSuccess || cudaEventRecord(e, us) != cudaSuccess)
    {
      cudaGetLastError();
      cudaStreamSynchronize(us);
      event_put(e);
      return false;
    }
    pipe_submit([e, pin, bytes](cudaStream_t s) {
      cudaStreamWaitEvent(s, e, 0);
      event_put(e); // a later re-recording does not affect this wait
      pin_release_after(pin, bytes, s);
    });
    return true;
  }
  // ---- the pipeline submitter ----------------------------------------------------------------------------------------
  // With 12+ Prefetch threads each issuing ~50 small pipeline operations per frame, the threads mostly waited for the
  // driver's launch lock (cudaLaunchKernel at 80-160 us instead of 5). Now one thread issues every pipeline operation, in
  // submission order (the order that already defines correctness); the worker threads only queue closures.
  namespace
  {
    struct Submitter
    {
      std::mutex m;
      std::condition_variable cv;
      std::vector<std::function<void(cudaStream_t)>> q;
      unsigned long long queued = 0;               // ops queued so far = sequence number of the last one (under m)

      // Marks: events recorded on the pipeline stream right after a given op, on demand (mark_after). A consumer outside
      // the pipeline (analysis, download) waits for the op that made its frame, not for everything queued.
      static const int MARKS = 32;
      std::mutex mark_m;
      std::condition_variable mark_cv;
      cudaEvent_t mark_ev[MARKS] = {};
      unsigned long long mark_seq[MARKS] = {};
      int mark_next = 0;
      unsigned long long marked = 0;               // the newest mark covers the ops up to here (under mark_m)
      std::atomic<unsigned long long> want{ ~0ull }; // a waiter needs a mark at or after this op
      unsigned long long issued = 0;               // ops issued so far (submitter thread only)

      Submitter()
      {
        std::thread([this] { run(); }).detach(); // lives as long as the process
      }

      // a mark after the first upto ops (all issued); submitter thread only
      void record_mark(cudaStream_t s, unsigned long long upto)
      {
        std::lock_guard<std::mutex> lk(mark_m);
        const int slot = mark_next;
        if (!mark_ev[slot] && cudaEventCreateWithFlags(&mark_ev[slot], cudaEventDisableTiming) != cudaSuccess)
          mark_ev[slot] = nullptr;
        if (!mark_ev[slot] || cudaEventRecord(mark_ev[slot], s) != cudaSuccess)
          cudaStreamSynchronize(s); // no event: everything issued so far is done, which serves as the mark
        mark_seq[slot] = upto;
        mark_next = (slot + 1) % MARKS;
        marked = upto;
        want = ~0ull; // waiters for later ops register again
        mark_cv.notify_all();
      }

      // takes everything queued at once and issues it without holding the lock
      void run()
      {
        use_device();
        cudaStream_t s = pipeline_stream();
        std::vector<std::function<void(cudaStream_t)>> batch;
        for (;;)
        {
          {
            std::unique_lock<std::mutex> lk(m);
            cv.wait(lk, [&] { return !q.empty(); });
            batch.swap(q);
          }
          for (auto &op : batch)
          {
            op(s);
            const cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
              std::fprintf(stderr, "mvcuda pipeline: %s\n", cudaGetErrorString(err));
            op = nullptr; // captured frames released at once
            ++issued;
            if (issued >= want.load())
              record_mark(s, issued);
            if (g_unsealed->size() >= 16)
              pin_seal(s);
            if ((issued & 31) == 0)
              pin_reclaim();
          }
          batch.clear();
          pin_seal(s); // the queue may run empty now: no staging buffer waits for a later batch
        }
      }
    };

    Submitter &submitter()
    {
      static Submitter *s = new Submitter; // never destroyed (static destruction order)
      return *s;
    }

    thread_local unsigned long long t_last_seq = 0; // sequence number of this thread's last queued op
  }

  void pipe_submit(std::function<void(cudaStream_t)> op)
  {
    Submitter &sb = submitter();
    bool was_empty;
    {
      std::lock_guard<std::mutex> lk(sb.m);
      was_empty = sb.q.empty();
      sb.q.push_back(std::move(op));
      t_last_seq = ++sb.queued;
    }
    if (was_empty) // otherwise the submitter is awake or has a wake-up pending
      sb.cv.notify_one();
  }

  void pipe_call(std::function<void(cudaStream_t)> op)
  {
    // the caller waits for its own op only (one signal, no wake-up of other waiters)
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
    pipe_submit([&](cudaStream_t s) {
      op(s);
      std::lock_guard<std::mutex> lk(m);
      done = true;
      cv.notify_one();
    });
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [&] { return done; });
  }

  void pipe_flush()
  {
    pipe_call([](cudaStream_t s) { cudaStreamSynchronize(s); });
  }

  unsigned long long pipe_last_seq()
  {
    return t_last_seq;
  }

  // an event recorded on the pipeline stream after op seq (and maybe later ones); nullptr: nothing to wait for
  cudaEvent_t mark_after(unsigned long long seq)
  {
    if (seq == 0)
      return nullptr;
    Submitter &sb = submitter();
    std::unique_lock<std::mutex> lk(sb.mark_m);
    while (sb.marked < seq)
    {
      unsigned long long w = sb.want.load();
      while (seq < w && !sb.want.compare_exchange_weak(w, seq))
        ;
      // wakes the submitter if it is idle; queued after seq, so the submitter passes seq before it runs out of work
      lk.unlock();
      pipe_submit([](cudaStream_t) {});
      lk.lock();
      sb.mark_cv.wait(lk, [&] { return sb.marked >= seq || sb.want.load() == ~0ull; });
    }
    // the earliest mark at or after seq (a slot recorded again later only makes the wait longer, never wrong)
    int best = -1;
    for (int i = 0; i < Submitter::MARKS; i++)
      if (sb.mark_seq[i] >= seq && (best < 0 || sb.mark_seq[i] < sb.mark_seq[best]))
        best = i;
    return best >= 0 ? sb.mark_ev[best] : nullptr;
  }

  void mark_request(unsigned long long seq)
  {
    Submitter &sb = submitter();
    unsigned long long w = sb.want.load();
    while (seq < w && !sb.want.compare_exchange_weak(w, seq))
      ;
  }

  cudaEvent_t mark_in_op(unsigned long long seq, cudaStream_t s)
  {
    if (seq == 0)
      return nullptr;
    Submitter &sb = submitter();
    {
      std::lock_guard<std::mutex> lk(sb.mark_m);
      int best = -1;
      for (int i = 0; i < Submitter::MARKS; i++)
        if (sb.mark_seq[i] >= seq && (best < 0 || sb.mark_seq[i] < sb.mark_seq[best]))
          best = i;
      if (best >= 0)
        return sb.mark_ev[best];
    }
    // no mark yet: one at the current position (after every op issued so far, which includes seq)
    sb.record_mark(s, sb.issued);
    std::lock_guard<std::mutex> lk(sb.mark_m);
    return sb.mark_ev[(sb.mark_next + Submitter::MARKS - 1) % Submitter::MARKS];
  }

  bool pipeline_order(cudaStream_t st, unsigned long long seq)
  {
    cudaEvent_t e = mark_after(seq);
    return !e || cudaStreamWaitEvent(st, e, 0) == cudaSuccess;
  }
  cudaStream_t copy_stream()
  {
    static cudaStream_t s = [] {
      cudaStream_t st = nullptr;
      if (!use_device() || cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) != cudaSuccess)
        st = pipeline_stream();
      return st;
    }();
    return s;
  }

  cudaStream_t pipeline_stream()
  {
    static cudaStream_t s = [] {
      cudaStream_t st = nullptr;
      if (!use_device() || cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) != cudaSuccess)
        st = nullptr; // legacy default stream: still ordered
      return st;
    }();
    return s;
  }

  void *pipe_alloc(size_t bytes)
  {
    {
      PipePool &pp = pipe_pool();
      std::lock_guard<std::mutex> lk(pp.m);
      auto it = pp.free.find(bytes);
      if (it != pp.free.end())
      {
        void *p = it->second;
        pp.free.erase(it);
        return p;
      }
    }
    void *p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess)
    {
      cudaGetLastError();
      // out of memory: give the free pipeline buffers back (after the queued work that may still use them) and retry
      pipe_flush();
      {
        PipePool &pp = pipe_pool();
      std::lock_guard<std::mutex> lk(pp.m);
        for (auto &e : pp.free)
          cudaFree(e.second);
        pp.free.clear();
      }
      if (cudaMalloc(&p, bytes) != cudaSuccess)
      {
        cudaGetLastError();
        return nullptr;
      }
    }
    return p;
  }

  void *pipe_scratch(int id, size_t bytes)
  {
    static std::mutex m;
    static void *buf[SCRATCH_COUNT] = {};
    static size_t size[SCRATCH_COUNT] = {};
    if (id < 0 || id >= SCRATCH_COUNT)
      return nullptr;
    std::lock_guard<std::mutex> lk(m);
    if (!buf[id])
    {
      void *p = nullptr;
      if (!use_device() || cudaMalloc(&p, bytes) != cudaSuccess || cudaMemset(p, 0, bytes) != cudaSuccess
        || cudaDeviceSynchronize() != cudaSuccess)
      {
        cudaGetLastError();
        if (p)
          cudaFree(p);
        return nullptr;
      }
      buf[id] = p;
      size[id] = bytes;
    }
    return size[id] >= bytes ? buf[id] : nullptr;
  }

  void pipe_free(void *p, size_t bytes)
  {
    if (!p)
      return;
    PipePool &pp = pipe_pool();
      std::lock_guard<std::mutex> lk(pp.m);
    pp.free.emplace(bytes, p);
  }

  DevFrame::~DevFrame()
  {
    if (upload)
      up_free(plane[0], bytes);
    else
      pipe_free(plane[0], bytes); // one buffer for all planes
  }

  std::shared_ptr<DevFrame> frame_alloc(const MvCudaFrameFormat &f, bool upload)
  {
    if (!valid_format(f) || !use_device())
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
    unsigned char *base = (unsigned char *)(upload ? up_alloc(total) : pipe_alloc(total));
    if (!base)
      return nullptr;
    fr->bytes = total;
    fr->upload = upload;
    for (int p = 0; p < f.planes; p++)
      fr->plane[p] = base + off[p];
    return fr;
  }

  void frame_put(const std::shared_ptr<DevFrame> &fr, unsigned long long producer, int frame, MvCudaTokenHeader &out)
  {
    const unsigned long long token = g_next_token.fetch_add(1);
    fr->seq = pipe_last_seq(); // the op just queued by this thread writes the frame
    fr->token = token;
    Held &hold = t_held[t_held_pos++ % HOLDS];
    hold.token = token;
    hold.fr = fr;
    {
      std::lock_guard<std::mutex> lk(g_mutex);
      g_lru.push_front(token);
      g_put_bytes += fr->bytes;
      const unsigned long long key = frame_key(producer, frame);
      g_frames[token] = Slot{ fr, g_lru.begin(), g_put_bytes, key };
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
          g_frames.erase(it); // the memory goes back to the pool when the last holder lets go
        }
        g_lru.pop_back();
        g_stats.evictions++;
      }
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
  // host <-> device copy of all planes through one pinned staging buffer, issued by the pipeline submitter. Upload: the
  // copy is queued in submission order (after the queued work that may still use the frame's buffer, before the work
  // queued later), no host wait; the staging buffer returns to its pool once the copy has run. Download: queued, then the
  // host waits for that copy (and the work queued before it). Without pinned memory: pageable copies, waited for.
  int copy_planes(mvc::DevFrame &fr, const void *const host_src[3], void *const host_dst[3], const int pitch_bytes[3])
  {
    const bool up = host_src != nullptr;
    struct Planes
    {
      void *dev[3];
      size_t pitch[3], h[3], off[3]; // off: offset in the frame buffer (frame_alloc), also in the staging buffer
      int n;
    } pl = {};
    pl.n = fr.f.planes;
    for (int p = 0; p < pl.n; p++)
    {
      pl.dev[p] = fr.plane[p];
      pl.pitch[p] = fr.pitch[p];
      pl.h[p] = (size_t)fr.h[p];
      pl.off[p] = (size_t)((unsigned char *)fr.plane[p] - (unsigned char *)fr.plane[0]);
    }
    void *const base = fr.plane[0];
    unsigned char *pin = (unsigned char *)mvc::pin_alloc(fr.bytes);
    if (!pin)
    {
      bool ok = true;
      mvc::pipe_call([&](cudaStream_t s) {
        for (int p = 0; p < pl.n && ok; p++)
          ok = (up ? cudaMemcpy2DAsync(pl.dev[p], pl.pitch[p], host_src[p], pitch_bytes[p], pl.pitch[p], pl.h[p], cudaMemcpyHostToDevice, s)
                   : cudaMemcpy2DAsync(host_dst[p], pitch_bytes[p], pl.dev[p], pl.pitch[p], pl.pitch[p], pl.h[p], cudaMemcpyDeviceToHost, s))
            == cudaSuccess;
        ok = cudaStreamSynchronize(s) == cudaSuccess && ok;
      });
      return ok ? 0 : -2;
    }
    const size_t total = fr.bytes;
    if (up)
    {
      for (int p = 0; p < pl.n; p++)
      {
        const unsigned char *s = (const unsigned char *)host_src[p];
        for (size_t y = 0; y < pl.h[p]; y++)
          std::memcpy(pin + pl.off[p] + y * pl.pitch[p], s + y * pitch_bytes[p], pl.pitch[p]);
      }
      if (fr.upload && mvc::up_copy(base, pin, total)) // the whole frame buffer, alignment gaps included
        return 0;
      mvc::pipe_submit([base, pin, total](cudaStream_t s) {
        cudaMemcpyAsync(base, pin, total, cudaMemcpyHostToDevice, s);
        mvc::pin_release_after(pin, total, s);
      });
      return 0;
    }
    // the copy runs on its own stream (a copy between the pipeline kernels costs an engine switch); the host waits for
    // this copy only
    // the copy waits for the op that made the frame only (mvc::mark_after), not for everything queued since
    cudaEvent_t after = mvc::event_get();
    bool ok = after != nullptr;
    if (ok)
    {
      const cudaStream_t cs = mvc::copy_stream();
      cudaEvent_t made = mvc::mark_after(fr.seq);
      ok = (!made || cudaStreamWaitEvent(cs, made, 0) == cudaSuccess)
        && cudaMemcpyAsync(pin, base, total, cudaMemcpyDeviceToHost, cs) == cudaSuccess && cudaEventRecord(after, cs) == cudaSuccess;
    }
    ok = ok && cudaEventSynchronize(after) == cudaSuccess;
    if (ok)
    {
      for (int p = 0; p < pl.n; p++)
      {
        unsigned char *d = (unsigned char *)host_dst[p];
        for (size_t y = 0; y < pl.h[p]; y++)
          std::memcpy(d + y * pitch_bytes[p], pin + pl.off[p] + y * pl.pitch[p], pl.pitch[p]);
      }
    }
    else
      cudaGetLastError();
    mvc::event_put(after);
    mvc::pin_free(pin, fr.bytes);
    return ok ? 0 : -2;
  }
}

namespace mvc
{
  // small host array (e.g. per-block tables) to pipeline memory, queued like a frame upload (no host wait)
  void *pipe_upload(const void *host, size_t bytes)
  {
    void *d = up_alloc(bytes);
    if (!d)
      return nullptr;
    if (void *pin = pin_alloc(bytes))
    {
      std::memcpy(pin, host, bytes);
      if (up_copy(d, pin, bytes))
        return d;
      pipe_submit([d, pin, bytes](cudaStream_t s) {
        cudaMemcpyAsync(d, pin, bytes, cudaMemcpyHostToDevice, s);
        pin_release_after(pin, bytes, s);
      });
      return d;
    }
    bool ok = true;
    pipe_call([&](cudaStream_t s) { ok = cudaMemcpyAsync(d, host, bytes, cudaMemcpyHostToDevice, s) == cudaSuccess && cudaStreamSynchronize(s) == cudaSuccess; });
    if (!ok)
    {
      cudaGetLastError();
      up_free(d, bytes);
      return nullptr;
    }
    return d;
  }
}
extern "C" int mvcuda_frame_upload(const MvCudaFrameFormat *f, const void *const src[3], const int src_pitch_bytes[3],
  unsigned long long producer, int frame, MvCudaTokenHeader *out)
{
  if (!f || !src || !src_pitch_bytes || !out)
    return -4;
  if (!mvc::use_device())
    return -1;
  auto fr = mvc::frame_alloc(*f, true);
  if (!fr)
    return -4;
  for (int p = 0; p < f->planes; p++)
    if (!src[p] || src_pitch_bytes[p] < (int)fr->pitch[p])
      return -4;
  const int rc = copy_planes(*fr, src, nullptr, src_pitch_bytes);
  if (rc)
    return rc;
  mvc::frame_put(fr, producer, frame, *out);
  std::lock_guard<std::mutex> lk(mvc::g_mutex);
  mvc::g_stats.uploads++;
  return 0;
}

extern "C" int mvcuda_frame_download(unsigned long long token, void *const dst[3], const int dst_pitch_bytes[3])
{
  if (!dst || !dst_pitch_bytes)
    return -4;
  if (!mvc::use_device())
    return -1;
  auto fr = mvc::frame_get(token);
  if (!fr)
    return MVCUDA_TOKEN_GONE;
  for (int p = 0; p < fr->f.planes; p++)
    if (!dst[p] || dst_pitch_bytes[p] < (int)fr->pitch[p])
      return -4;
  const int rc = copy_planes(*fr, nullptr, dst, dst_pitch_bytes);
  if (rc == 0)
  {
    std::lock_guard<std::mutex> lk(mvc::g_mutex);
    mvc::g_stats.downloads++;
  }
  return rc;
}

extern "C" unsigned long long mvcuda_producer_register(MvCudaRegenFn fn, void *user)
{
  if (!fn)
    return 0;
  const unsigned long long id = mvc::g_next_producer.fetch_add(1);
  std::lock_guard<std::mutex> lk(mvc::g_prod_mutex);
  mvc::g_producers[id] = mvc::Producer{ fn, user };
  return id;
}

extern "C" void mvcuda_producer_unregister(unsigned long long producer)
{
  std::lock_guard<std::mutex> lk(mvc::g_prod_mutex);
  mvc::g_producers.erase(producer);
}

extern "C" int mvcuda_token_regenerate(unsigned long long producer, int frame, void *env, MvCudaTokenHeader *out)
{
  if (!out)
    return -4;
  mvc::Producer pr;
  {
    std::lock_guard<std::mutex> lk(mvc::g_prod_mutex);
    auto it = mvc::g_producers.find(producer);
    if (it == mvc::g_producers.end())
      return -7;
    pr = it->second;
  }
  // called outside the lock: the producer makes the frame through the normal upload / kernel paths. A filter is not
  // destroyed while frames of its clip are still being requested, so the entry stays valid for the call.
  mvc::t_regen_depth++; // lookups inside a regeneration do not count for the budget (budget_on_miss)
  const int rc = pr.fn(pr.user, frame, env, out);
  mvc::t_regen_depth--;
  if (rc == 0)
  {
    std::lock_guard<std::mutex> lk(mvc::g_mutex);
    mvc::g_stats.regenerations++;
  }
  return rc;
}

extern "C" int mvcuda_token_pin(unsigned long long token)
{
  auto fr = mvc::frame_get(token);
  if (!fr)
    return MVCUDA_TOKEN_GONE;
  std::lock_guard<std::mutex> lk(mvc::g_mutex);
  auto it = mvc::g_pins.find(token);
  if (it != mvc::g_pins.end())
    it->second.count++;
  else
    mvc::g_pins[token] = mvc::Pin{ fr, 1 };
  return 0;
}

extern "C" void mvcuda_token_unpin(unsigned long long token)
{
  std::lock_guard<std::mutex> lk(mvc::g_mutex);
  auto it = mvc::g_pins.find(token);
  if (it != mvc::g_pins.end() && --it->second.count <= 0)
    mvc::g_pins.erase(it);
}

extern "C" void mvcuda_frame_stats(MvCudaFrameStats *s)
{
  if (!s)
    return;
  std::lock_guard<std::mutex> lk(mvc::g_mutex);
  *s = mvc::g_stats;
  s->bytes = mvc::g_bytes;
  s->budget = mvc::budget();
  s->frames = mvc::g_frames.size();
}
