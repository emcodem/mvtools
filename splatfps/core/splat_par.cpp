// splat_par: thread pool of the fast path; see splat_par.h.
//
// A job is published as a generation in one 64-bit atomic together with the next free index (generation in the high
// 32 bits): a thread claims index i of generation g by a compare-exchange from (g, i) to (g, i + 1), so it can never
// claim an index of a later job by mistake, and the job's function and count (two slots, by generation parity) are
// read only for a generation that is still current. A job ends when all its indices are done, not when every worker
// has looked at it: a worker that wakes late finds nothing left and waits for the next one. Workers that finished
// watch the counter for a short while (spinning) before they sleep on a condition variable, so the many short parallel
// phases of a frame start without a wake-up each time.
#include "splat_par.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#include <immintrin.h>
#define SPLAT_PAUSE() _mm_pause()
#else
#define SPLAT_PAUSE() std::this_thread::yield()
#endif

namespace splat
{
  // set while a thread executes pool work (nested run() calls then run inline)
  static thread_local bool t_in_pool = false;

  // pause iterations a thread waits actively for the next job / the end of the current one before it sleeps
  static const int SPIN = 4000;

  struct Pool::Impl
  {
    std::vector<std::thread> workers;
    std::mutex m, submit;
    std::condition_variable wake, done_cv;
    // job slots by generation parity
    std::atomic<const std::function<void(int)> *> job[2] = {};
    std::atomic<int> count[2] = {};
    std::atomic<uint64_t> ticket{ 0 }; // (generation << 32) | next index
    std::atomic<int> done{ 0 };        // indices of the current job finished
    std::atomic<int> sleepers{ 0 };
    std::atomic<bool> caller_sleeps{ false };
    std::atomic<bool> stop{ false };

    static uint32_t gen_of(uint64_t t) { return (uint32_t)(t >> 32); }

    // claims and runs indices of generation g until none is left
    void drain(uint32_t g)
    {
      uint64_t cur = ticket.load(std::memory_order_acquire);
      while (gen_of(cur) == g)
      {
        const int n = count[g & 1].load(std::memory_order_relaxed);
        const uint32_t i = (uint32_t)cur;
        if ((int)i >= n)
          return;
        if (!ticket.compare_exchange_weak(cur, cur + 1, std::memory_order_acq_rel, std::memory_order_acquire))
          continue; // cur reloaded
        (*job[g & 1].load(std::memory_order_relaxed))((int)i);
        // (sequentially consistent with the caller's flag store and count load: either it sees the last index done or
        // this sees it asleep)
        if (done.fetch_add(1) + 1 == n && caller_sleeps.load())
        {
          std::lock_guard<std::mutex> lk(m);
          done_cv.notify_one();
        }
        cur = ticket.load(std::memory_order_acquire);
      }
    }

    void worker()
    {
      t_in_pool = true;
      uint32_t seen = 0;
      for (;;)
      {
        uint32_t g = gen_of(ticket.load(std::memory_order_acquire));
        for (int s = 0; g == seen && !stop.load() && s < SPIN; s++)
        {
          SPLAT_PAUSE();
          g = gen_of(ticket.load(std::memory_order_acquire));
        }
        if (g == seen && !stop.load())
        {
          std::unique_lock<std::mutex> lk(m);
          sleepers.fetch_add(1);
          wake.wait(lk, [&] { return stop.load() || gen_of(ticket.load()) != seen; });
          sleepers.fetch_sub(1);
          g = gen_of(ticket.load(std::memory_order_acquire));
        }
        if (stop.load())
          return;
        seen = g;
        drain(g);
      }
    }
  };

  Pool::Pool(int threads) : impl_(new Impl)
  {
    int nt = threads > 0 ? threads : (int)std::thread::hardware_concurrency();
    nt = std::max(1, nt);
    for (int t = 1; t < nt; t++)
      impl_->workers.emplace_back([this] { impl_->worker(); });
  }

  Pool::~Pool()
  {
    {
      std::lock_guard<std::mutex> lk(impl_->m);
      impl_->stop.store(true);
    }
    impl_->wake.notify_all();
    for (auto &t : impl_->workers)
      t.join();
    delete impl_;
  }

  int Pool::threads() const { return (int)impl_->workers.size() + 1; }

  void Pool::run(int n, const std::function<void(int)> &fn)
  {
    if (n <= 0)
      return;
    if (n == 1 || t_in_pool || impl_->workers.empty())
    {
      for (int i = 0; i < n; i++)
        fn(i);
      return;
    }
    Impl &p = *impl_;
    std::lock_guard<std::mutex> one(p.submit);
    // the previous job is complete (every index done), so its slot and the counter are free
    const uint32_t g = Impl::gen_of(p.ticket.load()) + 1;
    p.job[g & 1].store(&fn, std::memory_order_relaxed);
    p.count[g & 1].store(n, std::memory_order_relaxed);
    p.done.store(0);
    // sequentially consistent with a worker's sleeper count and ticket load: it sees the job or is notified
    p.ticket.store((uint64_t)g << 32);
    if (p.sleepers.load() > 0)
    {
      std::lock_guard<std::mutex> lk(p.m);
      p.wake.notify_all();
    }
    t_in_pool = true;
    p.drain(g);
    t_in_pool = false;
    for (int s = 0; p.done.load(std::memory_order_acquire) != n && s < SPIN; s++)
      SPLAT_PAUSE();
    if (p.done.load(std::memory_order_acquire) != n)
    {
      std::unique_lock<std::mutex> lk(p.m);
      p.caller_sleeps.store(true);
      p.done_cv.wait(lk, [&] { return p.done.load() == n; });
      p.caller_sleeps.store(false);
    }
  }

  void Pool::serial(const std::function<void()> &fn)
  {
    const bool was = t_in_pool;
    t_in_pool = true;
    fn();
    t_in_pool = was;
  }

  void Pool::rows(int n, const std::function<void(int, int)> &fn, int min_chunk)
  {
    if (n <= 0)
      return;
    const int target = 4 * threads();
    const int chunk = std::max(std::max(1, min_chunk), (n + target - 1) / target);
    const int nchunks = (n + chunk - 1) / chunk;
    run(nchunks, [&](int i) { fn(i * chunk, std::min(n, (i + 1) * chunk)); });
  }
}
