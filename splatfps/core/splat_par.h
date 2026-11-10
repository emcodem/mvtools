// splat_par: the thread pool of SplatFps' fast path. A pool owns its worker threads (created in the constructor,
// joined in the destructor, so the owner decides their lifetime). run(n, fn) calls fn(i) for every i < n, each index
// once, spread over the workers and the calling thread: every thread takes the next free index, so faster cores take
// more of them. The fast path only parallelises work whose result does not depend on the split (each index writes
// its own outputs), so the results are the same with any thread count.
#pragma once

#include <functional>

namespace splat
{
  class Pool
  {
  public:
    // threads: total threads including the caller (0: all hardware threads)
    explicit Pool(int threads = 0);
    ~Pool();
    Pool(const Pool &) = delete;
    Pool &operator=(const Pool &) = delete;

    int threads() const;
    // fn(i) for 0 <= i < n; returns when all are done. A call from inside fn (nested) runs on the calling thread.
    void run(int n, const std::function<void(int)> &fn);
    // fn(y0, y1) over consecutive ranges of [0, n) of about n / (4 threads) items, at least min_chunk each
    void rows(int n, const std::function<void(int, int)> &fn, int min_chunk = 4);
    // fn() on the calling thread with the pool's run() / rows() calls inside it running inline (for work too small to
    // pay for waking the workers)
    void serial(const std::function<void()> &fn);

  private:
    struct Impl;
    Impl *impl_;
  };
}
