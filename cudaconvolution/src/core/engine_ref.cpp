#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

#include "core/engine.h"
#include "core/engine_rows.h"

#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace cudaconvolution {
namespace {

int border_index(int i, int n, Border b) {
  if (b == Border::Replicate) return std::clamp(i, 0, n - 1);
  if (n == 1) return 0;
  // mirror without edge repeat; kernels are far shorter than planes, one reflection is enough
  if (i < 0) i = -i;
  if (i > n - 1) i = 2 * (n - 1) - i;
  return std::clamp(i, 0, n - 1);
}

// Persistent worker pool (threads are created once and reused; starting threads per frame costs more than the work
// of a small frame). One job at a time; concurrent callers (AviSynth Prefetch) take turns.
class WorkerPool {
 public:
  explicit WorkerPool(int n) {
    for (int i = 0; i < n; ++i) workers_.emplace_back([this] { loop(); });
  }
  ~WorkerPool() {
    {
      std::lock_guard<std::mutex> lock(m_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
  }
  int size() const { return static_cast<int>(workers_.size()); }
  // runs job(i) for i in [0, chunks) on the pool and the calling thread; returns when all are done
  void run(int chunks, const std::function<void(int)>& job) {
    std::lock_guard<std::mutex> one_job(job_m_);
    {
      std::lock_guard<std::mutex> lock(m_);
      job_ = &job;
      chunks_ = chunks;
      next_ = 0;
      done_ = 0;
      ++generation_;
    }
    cv_.notify_all();
    work();
    std::unique_lock<std::mutex> lock(m_);
    done_cv_.wait(lock, [&] { return done_ == chunks_; });
    job_ = nullptr;
  }

 private:
  void work() {
    for (;;) {
      int i;
      const std::function<void(int)>* job;
      {
        std::lock_guard<std::mutex> lock(m_);
        if (!job_ || next_ >= chunks_) return;
        i = next_++;
        job = job_;
      }
      (*job)(i);
      std::lock_guard<std::mutex> lock(m_);
      if (++done_ == chunks_) done_cv_.notify_all();
    }
  }
  void loop() {
    unsigned seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [&] { return stop_ || generation_ != seen; });
        if (stop_) return;
        seen = generation_;
      }
      work();
    }
  }
  std::vector<std::thread> workers_;
  std::mutex m_, job_m_;
  std::condition_variable cv_, done_cv_;
  const std::function<void(int)>* job_ = nullptr;
  int chunks_ = 0, next_ = 0, done_ = 0;
  unsigned generation_ = 0;
  bool stop_ = false;
};

// The pool lives until process exit: it is never destroyed (joining threads while the DLL unloads can deadlock on the
// loader lock), and the module is pinned so its code stays mapped while the idle workers wait inside it.
WorkerPool* make_pool() {
#ifdef _WIN32
  HMODULE self = nullptr;
  GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                     reinterpret_cast<LPCWSTR>(&make_pool), &self);
#endif
  return new WorkerPool(std::max(1, static_cast<int>(std::thread::hardware_concurrency()) - 1));
}

// Runs fn(begin, end, worker) on contiguous chunks of [0, n) using up to `threads` workers (pool + caller).
template <class F> void parallel_for(int n, int threads, F&& fn) {
  threads = std::max(1, std::min(threads, n / 8));  // at least 8 rows per worker
  if (threads == 1) {
    fn(0, n, 0);
    return;
  }
  static WorkerPool* const pool = make_pool();
  threads = std::min(threads, pool->size() + 1);
  const std::function<void(int)> job = [&](int t) { fn(n * t / threads, n * (t + 1) / threads, t); };
  pool->run(threads, job);
}

bool detect_avx2_fma() {
#ifdef _MSC_VER
  int r[4];
  __cpuid(r, 0);
  if (r[0] < 7) return false;
  __cpuid(r, 1);
  const bool fma = r[2] & (1 << 12), osxsave = r[2] & (1 << 27), avx = r[2] & (1 << 28);
  if (!(fma && osxsave && avx) || (_xgetbv(0) & 6) != 6) return false;  // OS saves YMM state
  __cpuidex(r, 7, 0);
  return r[1] & (1 << 5);  // AVX2
#else
  return false;
#endif
}

}  // namespace

namespace rows {

void h_scalar(const float* row, float* out, int dw, int factor, const float* taps, int nh, float ilo, float ihi,
              float*) {
  for (int x = 0; x < dw; ++x) {
    const float* base = row + x * factor;
    float acc = 0.0f;
    for (int t = 0; t < nh; ++t) acc = std::fma(taps[t], base[t], acc);
    out[x] = std::clamp(acc, ilo, ihi);
  }
}

void v_scalar(const float* const* src, int nv, const float* taps, uint16_t* d, int dw, const Preset& p) {
  for (int x = 0; x < dw; ++x) {
    float acc = 0.0f;
    for (int t = 0; t < nv; ++t) acc = std::fma(taps[t], src[t][x], acc);
    d[x] = output_rule(acc, p);
  }
}

}  // namespace rows

bool cpu_has_avx2_fma() {
  static const bool has = detect_avx2_fma();
  return has;
}

void resample_plane(const ConstPlane& src, const Plane& dst, int row_first, int row_step, const Kernel& h,
                    const Kernel& v, const Preset& p, std::vector<float>& scratch, CpuPath path, int threads) {
  if (dst.width * p.factor_h != src.width || dst.height * p.factor_v != src.height)
    throw std::invalid_argument("resample_plane: plane sizes do not match the factors");
  for (const auto& e : h.left)
    if (e.src_first + static_cast<int>(e.taps.size()) > src.width)
      throw std::invalid_argument("resample_plane: left edge kernel reaches beyond the plane");
  for (const auto& e : h.right)
    if (src.width + e.src_first < 0) throw std::invalid_argument("resample_plane: right edge kernel reaches beyond the plane");
  if (path == CpuPath::Auto) path = cpu_has_avx2_fma() ? CpuPath::Avx2 : CpuPath::Scalar;
  if (path == CpuPath::Avx2 && !cpu_has_avx2_fma()) throw std::runtime_error("this CPU has no AVX2/FMA3");
  const auto h_row = path == CpuPath::Avx2 ? rows::h_avx2 : rows::h_scalar;
  const auto v_row = path == CpuPath::Avx2 ? rows::v_avx2 : rows::v_scalar;

  const int sw = src.width, sh = src.height, dw = dst.width;
  const int nh = static_cast<int>(h.taps.size()), nv = static_cast<int>(v.taps.size());
  const float off = static_cast<float>(p.offset);
  const float ilo = p.has_intermediate_clip ? static_cast<float>(p.intermediate_lo - p.offset) : -INFINITY;
  const float ihi = p.has_intermediate_clip ? static_cast<float>(p.intermediate_hi - p.offset) : INFINITY;

  // source rows needed by the requested output rows
  std::vector<char> needed(sh, 0);
  for (int y = row_first; y < dst.height; y += row_step)
    for (int t = 0; t < nv; ++t) needed[border_index(y * p.factor_v + v.first + t, sh, p.border)] = 1;

  // scratch: intermediate (sh x dw) + per worker: one padded source row + row work buffer (AVX2 phase split)
  const int pad_lo = std::min(0, h.first), pad_hi = std::max(0, h.first + nh - 1);
  const int padded_w = sw - pad_lo + pad_hi;
  const std::size_t work = static_cast<std::size_t>(padded_w) + 16 * p.factor_h;
  const int workers = std::max(1, threads);
  scratch.resize(static_cast<std::size_t>(sh) * dw + workers * (padded_w + work));
  float* inter = scratch.data();

  // H pass + intermediate clip; output x reads row[x * factor_h + (h.first - pad_lo) + t]. Rows are independent, so
  // splitting them over threads gives bit-identical results.
  parallel_for(sh, workers, [&](int r0, int r1, int worker) {
    float* row = inter + static_cast<std::size_t>(sh) * dw + worker * (padded_w + work);
    float* row_work = row + padded_w;
    for (int r = r0; r < r1; ++r) {
      if (!needed[r]) continue;
      const uint16_t* s = src.data + r * src.stride;
      for (int k = 0; k < padded_w; ++k) row[k] = static_cast<float>(s[border_index(k + pad_lo, sw, p.border)]) - off;
      float* out = inter + static_cast<std::size_t>(r) * dw;
      h_row(row + (h.first - pad_lo), out, dw, p.factor_h, h.taps.data(), nh, ilo, ihi, row_work);
      // edge columns: own kernels on in-plane sources (row[k - pad_lo] is source column k)
      for (int side = 0; side < 2; ++side) {
        const auto& list = side == 0 ? h.left : h.right;
        for (std::size_t i = 0; i < list.size() && static_cast<int>(i) < dw; ++i) {
          const EdgeKernel& e = list[i];
          const int x = side == 0 ? static_cast<int>(i) : dw - 1 - static_cast<int>(i);
          const float* es = row + ((side == 0 ? e.src_first : sw + e.src_first) - pad_lo);
          float acc = 0.0f;
          for (std::size_t t = 0; t < e.taps.size(); ++t) acc = std::fma(e.taps[t], es[t], acc);
          out[x] = std::clamp(acc, ilo, ihi);
        }
      }
    }
  });

  // V pass + output rule
  const int ny = (dst.height - row_first + row_step - 1) / row_step;
  parallel_for(ny, workers, [&](int j0, int j1, int) {
    std::vector<const float*> src_rows(nv);
    for (int j = j0; j < j1; ++j) {
      const int y = row_first + j * row_step;
      for (int t = 0; t < nv; ++t)
        src_rows[t] = inter + static_cast<std::size_t>(border_index(y * p.factor_v + v.first + t, sh, p.border)) * dw;
      v_row(src_rows.data(), nv, v.taps.data(), dst.data + y * dst.stride, dw, p);
    }
  });
}

void process_frame(const ConstFrame& first, const ConstFrame& second, const Frame& out, const Preset& p,
                   std::vector<float>& scratch, CpuPath path, int threads) {
  for (int i = 0; i < 3; ++i) {
    const Kernel& h = i == 0 ? p.luma_h : p.chroma_h;
    if (!p.interlaced) {
      resample_plane(first.plane[i], out.plane[i], 0, 1, h, p.v, p, scratch, path, threads);
      continue;
    }
    const int first_parity = p.tff ? 0 : 1;
    resample_plane(first.plane[i], out.plane[i], first_parity, 2, h, p.v, p, scratch, path, threads);
    resample_plane(second.plane[i], out.plane[i], 1 - first_parity, 2, h, p.v, p, scratch, path, threads);
  }
}

}  // namespace cudaconvolution
