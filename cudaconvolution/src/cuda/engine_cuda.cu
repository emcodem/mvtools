// CUDA kernels of the resampler. Compiled with --fmad=false: the only FMAs are the explicit __fmaf_rn per tap, in
// ascending tap order, exactly as std::fma in the CPU engine; all other float ops are explicit _rn intrinsics.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "cuda/engine_cuda.h"

namespace cudaconvolution {
namespace {

struct Taps {
  int first, n;
  float t[kMaxTaps];
};

// edge kernels of an H pass (see EdgeKernel); passed by value like Taps (kernel parameter space, < 4 KB in total)
struct EdgeTaps {
  int nl, nr;
  int first[2][kMaxEdgeCols], n[2][kMaxEdgeCols];
  float t[2][kMaxEdgeCols][kMaxEdgeTaps];
};

struct Params {
  float offset_f, ilo, ihi;
  int offset, olo, ohi;
  int trunc_rule;  // 1: trunc(o + 0.5) (measured reference), 0: floor(o + 0.5)
  int mirror;      // 0: replicate, 1: mirror without edge repeat
};

__device__ __forceinline__ int border_index(int i, int n, int mirror) {
  if (!mirror) return min(max(i, 0), n - 1);
  if (n == 1) return 0;
  if (i < 0) i = -i;
  if (i > n - 1) i = 2 * (n - 1) - i;
  return min(max(i, 0), n - 1);
}

// one thread per intermediate sample: row r of the source, output column x
__global__ void h_pass(const uint16_t* __restrict__ src, int sw, int sh, float* __restrict__ inter, int dw, int factor,
                       Taps h, EdgeTaps e, Params p) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
  if (x >= dw || r >= sh) return;
  const uint16_t* row = src + static_cast<size_t>(r) * sw;
  float acc = 0.0f;
  const int side = x < e.nl ? 0 : (dw - 1 - x < e.nr ? 1 : -1);
  if (side >= 0) {  // edge column: own kernel on in-plane sources
    const int i = side == 0 ? x : dw - 1 - x;
    const int first = side == 0 ? e.first[0][i] : sw + e.first[1][i];
    for (int t = 0; t < e.n[side][i]; ++t)
      acc = __fmaf_rn(e.t[side][i][t], __fsub_rn(__uint2float_rn(row[first + t]), p.offset_f), acc);
  } else {
    const int base = x * factor + h.first;
    for (int t = 0; t < h.n; ++t) {
      const float v = __fsub_rn(__uint2float_rn(row[border_index(base + t, sw, p.mirror)]), p.offset_f);
      acc = __fmaf_rn(h.t[t], v, acc);
    }
  }
  inter[static_cast<size_t>(r) * dw + x] = fminf(fmaxf(acc, p.ilo), p.ihi);
}

// one thread per output sample of rows row_first + j * row_step
__global__ void v_pass(const float* __restrict__ inter, int dw, int sh, uint16_t* __restrict__ dst, int row_first,
                       int row_step, int nrows, int factor_v, Taps v, Params p) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y;
  if (x >= dw || j >= nrows) return;
  const int y = row_first + j * row_step;
  const int base = y * factor_v + v.first;
  float acc = 0.0f;
  for (int t = 0; t < v.n; ++t)
    acc = __fmaf_rn(v.t[t], inter[static_cast<size_t>(border_index(base + t, sh, p.mirror)) * dw + x], acc);
  // o + 0.5 is exact in float32 for |o| < 2^22, so this equals the CPU output rule
  const float r = __fadd_rn(acc, 0.5f);
  const int q = static_cast<int>(p.trunc_rule ? truncf(r) : floorf(r));
  dst[static_cast<size_t>(y) * dw + x] = static_cast<uint16_t>(min(max(p.offset + q, p.olo), p.ohi));
}

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw CudaError(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(e));
}

Taps make_taps(const Kernel& k) {
  Taps t{};
  t.first = k.first;
  t.n = static_cast<int>(k.taps.size());
  std::memcpy(t.t, k.taps.data(), k.taps.size() * sizeof(float));
  return t;
}

EdgeTaps make_edges(const Kernel& k) {
  EdgeTaps e{};
  e.nl = static_cast<int>(k.left.size());
  e.nr = static_cast<int>(k.right.size());
  for (int side = 0; side < 2; ++side) {
    const auto& list = side == 0 ? k.left : k.right;
    for (std::size_t i = 0; i < list.size(); ++i) {
      e.first[side][i] = list[i].src_first;
      e.n[side][i] = static_cast<int>(list[i].taps.size());
      std::memcpy(e.t[side][i], list[i].taps.data(), list[i].taps.size() * sizeof(float));
    }
  }
  return e;
}

Params make_params(const Preset& p) {
  Params q{};
  q.offset_f = static_cast<float>(p.offset);
  q.ilo = p.has_intermediate_clip ? static_cast<float>(p.intermediate_lo - p.offset) : -INFINITY;
  q.ihi = p.has_intermediate_clip ? static_cast<float>(p.intermediate_hi - p.offset) : INFINITY;
  q.offset = p.offset;
  q.olo = p.output_lo;
  q.ohi = p.output_hi;
  q.trunc_rule = p.arith == Arith::F32FmaTrunc;
  q.mirror = p.border == Border::Mirror;
  return q;
}

std::size_t plane_samples(const ConstPlane& pl) { return static_cast<std::size_t>(pl.width) * pl.height; }
std::size_t plane_samples(const Plane& pl) { return static_cast<std::size_t>(pl.width) * pl.height; }

}  // namespace

// Per concurrent call: stream + device buffers for both source frames, the intermediate of one plane and the output.
struct CudaEngine::Context {
  cudaStream_t stream = nullptr;
  cudaEvent_t done[3] = {nullptr, nullptr, nullptr};  // per plane: download complete
  uint16_t* src[2] = {nullptr, nullptr};  // planes packed without padding
  float* inter = nullptr;
  uint16_t* dst = nullptr;
  std::size_t src_cap[2] = {0, 0}, inter_cap = 0, dst_cap = 0;  // in elements
  // pinned host staging: DMA from/to pinned memory is faster and truly asynchronous (PCIe bound, see bench_cuda.cu)
  uint16_t* host_src[2] = {nullptr, nullptr};
  uint16_t* host_dst = nullptr;
  std::size_t host_src_cap[2] = {0, 0}, host_dst_cap = 0;

  ~Context() {
    cudaFree(src[0]);
    cudaFree(src[1]);
    cudaFree(inter);
    cudaFree(dst);
    cudaFreeHost(host_src[0]);
    cudaFreeHost(host_src[1]);
    cudaFreeHost(host_dst);
    for (cudaEvent_t e : done)
      if (e) cudaEventDestroy(e);
    if (stream) cudaStreamDestroy(stream);
  }

  static void ensure_host(uint16_t*& ptr, std::size_t& cap, std::size_t need) {
    if (cap >= need) return;
    cudaFreeHost(ptr);
    ptr = nullptr;
    cap = 0;
    if (cudaError_t e = cudaHostAlloc(&ptr, need * sizeof(uint16_t), cudaHostAllocDefault); e != cudaSuccess)
      throw CudaError(std::string("cannot allocate ") + std::to_string(need * sizeof(uint16_t) >> 20) +
                      " MB of pinned host memory (" + cudaGetErrorString(e) + "); fewer Prefetch threads or cpu=true");
    cap = need;
  }

  template <class T> static void ensure(T*& ptr, std::size_t& cap, std::size_t need, const char* what) {
    if (cap >= need) return;
    cudaFree(ptr);
    ptr = nullptr;
    cap = 0;
    if (cudaError_t e = cudaMalloc(&ptr, need * sizeof(T)); e != cudaSuccess)
      throw CudaError(std::string("cannot allocate ") + std::to_string(need * sizeof(T) >> 20) + " MB of GPU memory for " +
                      what + " (" + cudaGetErrorString(e) + "); fewer Prefetch threads or cpu=true");
    cap = need;
  }
};

CudaEngine::CudaEngine(int device, Preset p) : device_(device), p_(std::move(p)) {
  check(cudaSetDevice(device_), "cudaSetDevice");
  // load the kernels now: with lazy module loading the first launch would load them, possibly on another library's
  // stream-submitting thread (zero-copy path), where a module load can stall all its work
  cudaFuncAttributes attr;
  check(cudaFuncGetAttributes(&attr, h_pass), "loading the kernels");
  check(cudaFuncGetAttributes(&attr, v_pass), "loading the kernels");
}

CudaEngine::~CudaEngine() {
  cudaSetDevice(device_);
  idle_.clear();
}

std::size_t CudaEngine::bytes_per_context(const ConstFrame& src, const Frame& dst) {
  std::size_t s = 0, d = 0, inter = 0;
  for (int i = 0; i < 3; ++i) {
    s += plane_samples(src.plane[i]);
    d += plane_samples(dst.plane[i]);
    inter = std::max(inter, static_cast<std::size_t>(src.plane[i].height) * dst.plane[i].width);
  }
  return 2 * s * sizeof(uint16_t) + inter * sizeof(float) + d * sizeof(uint16_t);
}

std::unique_ptr<CudaEngine::Context> CudaEngine::acquire() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!idle_.empty()) {
      auto ctx = std::move(idle_.back());
      idle_.pop_back();
      return ctx;
    }
  }
  auto ctx = std::make_unique<Context>();
  check(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking), "cudaStreamCreate");
  for (cudaEvent_t& e : ctx->done) check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "cudaEventCreate");
  return ctx;
}

void CudaEngine::release(std::unique_ptr<Context> ctx) {
  std::lock_guard<std::mutex> lock(mutex_);
  idle_.push_back(std::move(ctx));
}

void CudaEngine::process_frame(const ConstFrame& first, const ConstFrame& second, const Frame& out) {
  check(cudaSetDevice(device_), "cudaSetDevice");
  auto ctx = acquire();
  struct Releaser {
    CudaEngine* e;
    std::unique_ptr<Context>& c;
    ~Releaser() {
      if (c) e->release(std::move(c));
    }
  } releaser{this, ctx};

  std::size_t src_n = 0, dst_n = 0, inter_n = 0;
  for (int i = 0; i < 3; ++i) {
    const ConstPlane& s = first.plane[i];
    const Plane& d = out.plane[i];
    if (d.width * p_.factor_h != s.width || d.height * p_.factor_v != s.height)
      throw std::invalid_argument("CudaEngine: plane sizes do not match the factors");
    src_n += plane_samples(s);
    dst_n += plane_samples(d);
    inter_n = std::max(inter_n, static_cast<std::size_t>(s.height) * d.width);
  }
  for (int f = 0; f < (p_.interlaced ? 2 : 1); ++f) {
    Context::ensure(ctx->src[f], ctx->src_cap[f], src_n, "the source frames");
    Context::ensure_host(ctx->host_src[f], ctx->host_src_cap[f], src_n);
  }
  Context::ensure_host(ctx->host_dst, ctx->host_dst_cap, dst_n);
  Context::ensure(ctx->inter, ctx->inter_cap, inter_n, "the intermediate");
  Context::ensure(ctx->dst, ctx->dst_cap, dst_n, "the output frame");
  cudaStream_t st = ctx->stream;

  // Per plane: pack into pinned memory in chunks, each chunk's DMA overlaps packing the next one; the plane's kernels
  // run while the next plane is packed; its download is unpacked while later planes still compute.
  constexpr std::size_t kChunkSamples = std::size_t(1) << 21;  // 4 MB
  constexpr int kBlock = 128;
  const Params prm = make_params(p_);
  const Taps tv = make_taps(p_.v);
  const ConstFrame* srcs[2] = {&first, &second};
  const int frames = p_.interlaced ? 2 : 1;
  std::size_t src_off = 0, dst_off = 0;
  for (int i = 0; i < 3; ++i) {
    const ConstPlane& s0 = first.plane[i];
    const Plane& d = out.plane[i];
    for (int f = 0; f < frames; ++f) {
      const ConstPlane& s = srcs[f]->plane[i];
      uint16_t* h = ctx->host_src[f] + src_off;
      uint16_t* dev = ctx->src[f] + src_off;
      const int rows = std::max<int>(1, static_cast<int>(kChunkSamples / s.width));
      for (int y0 = 0; y0 < s.height; y0 += rows) {
        const int y1 = std::min(s.height, y0 + rows);
        for (int y = y0; y < y1; ++y)
          std::memcpy(h + std::size_t(y) * s.width, s.data + y * s.stride, s.width * sizeof(uint16_t));
        const std::size_t off = std::size_t(y0) * s.width, n = std::size_t(y1 - y0) * s.width;
        check(cudaMemcpyAsync(dev + off, h + off, n * sizeof(uint16_t), cudaMemcpyHostToDevice, st), "upload");
      }
    }
    const Kernel& kh = i == 0 ? p_.luma_h : p_.chroma_h;
    const Taps th = make_taps(kh);
    const EdgeTaps eh = make_edges(kh);
    for (int f = 0; f < frames; ++f) {
      // interlaced: frame f feeds the rows of parity (first field parity) xor f
      const int row_first = p_.interlaced ? ((p_.tff ? 0 : 1) ^ f) : 0;
      const int row_step = p_.interlaced ? 2 : 1;
      const int nrows = (d.height - row_first + row_step - 1) / row_step;
      const dim3 grid_h((d.width + kBlock - 1) / kBlock, s0.height);
      h_pass<<<grid_h, kBlock, 0, st>>>(ctx->src[f] + src_off, s0.width, s0.height, ctx->inter, d.width, p_.factor_h,
                                        th, eh, prm);
      const dim3 grid_v((d.width + kBlock - 1) / kBlock, nrows);
      v_pass<<<grid_v, kBlock, 0, st>>>(ctx->inter, d.width, s0.height, ctx->dst + dst_off, row_first, row_step, nrows,
                                        p_.factor_v, tv, prm);
    }
    check(cudaGetLastError(), "kernel launch");
    check(cudaMemcpyAsync(ctx->host_dst + dst_off, ctx->dst + dst_off, plane_samples(d) * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost, st),
          "download");
    check(cudaEventRecord(ctx->done[i], st), "cudaEventRecord");
    src_off += plane_samples(s0);
    dst_off += plane_samples(d);
  }

  // unpack each plane into the (pitched) output frame as soon as its download is complete
  dst_off = 0;
  for (int i = 0; i < 3; ++i) {
    const Plane& d = out.plane[i];
    check(cudaEventSynchronize(ctx->done[i]), "kernel execution");
    const uint16_t* h = ctx->host_dst + dst_off;
    for (int y = 0; y < d.height; ++y, h += d.width) std::memcpy(d.data + y * d.stride, h, d.width * sizeof(uint16_t));
    dst_off += plane_samples(d);
  }
}

struct CudaEngine::DeviceJob {
  CudaEngine* engine;
  std::unique_ptr<Context> ctx;
  int dw[3], dh[3], sw[3], sh[3];  // output / source plane sizes
  bool host_output;
};

void CudaEngine::DeviceJobDeleter::operator()(DeviceJob* j) const {
  if (j->ctx) j->engine->release(std::move(j->ctx));
  delete j;
}

CudaEngine::DeviceJobPtr CudaEngine::prepare_device(const Frame& out_shape, bool host_output) {
  check(cudaSetDevice(device_), "cudaSetDevice");
  DeviceJobPtr job(new DeviceJob{this, acquire(), {}, {}, {}, {}, host_output});
  std::size_t dst_n = 0, inter_n = 0;
  for (int i = 0; i < 3; ++i) {
    const Plane& d = out_shape.plane[i];
    job->dw[i] = d.width;
    job->dh[i] = d.height;
    job->sw[i] = d.width * p_.factor_h;
    job->sh[i] = d.height * p_.factor_v;
    dst_n += plane_samples(d);
    inter_n = std::max(inter_n, static_cast<std::size_t>(job->sh[i]) * d.width);
  }
  Context::ensure(job->ctx->inter, job->ctx->inter_cap, inter_n, "the intermediate");
  if (host_output) {
    Context::ensure(job->ctx->dst, job->ctx->dst_cap, dst_n, "the output frame");
    Context::ensure_host(job->ctx->host_dst, job->ctx->host_dst_cap, dst_n);
  }
  return job;
}

int CudaEngine::launch_device(DeviceJob& job, const DevicePlanes& first, const DevicePlanes& second,
                              uint16_t* const out_dev[3], void* stream) noexcept {
  // never switch the device here: the current context belongs to the calling library's thread
  int dev = -1;
  if (cudaGetDevice(&dev) != cudaSuccess || dev != device_) return -10;
  if (!out_dev && !job.host_output) return -11;
  const cudaStream_t st = static_cast<cudaStream_t>(stream);
  constexpr int kBlock = 128;
  const Params prm = make_params(p_);
  const Taps tv = make_taps(p_.v);
  const DevicePlanes* srcs[2] = {&first, &second};
  const int frames = p_.interlaced ? 2 : 1;
  std::size_t dst_off = 0;
  for (int i = 0; i < 3; ++i) {
    const Kernel& kh = i == 0 ? p_.luma_h : p_.chroma_h;
    const Taps th = make_taps(kh);
    const EdgeTaps eh = make_edges(kh);
    const int dw = job.dw[i], dh = job.dh[i], sw = job.sw[i], sh = job.sh[i];
    uint16_t* dst = out_dev ? out_dev[i] : job.ctx->dst + dst_off;
    for (int f = 0; f < frames; ++f) {
      const int row_first = p_.interlaced ? ((p_.tff ? 0 : 1) ^ f) : 0;
      const int row_step = p_.interlaced ? 2 : 1;
      const int nrows = (dh - row_first + row_step - 1) / row_step;
      const dim3 grid_h((dw + kBlock - 1) / kBlock, sh);
      h_pass<<<grid_h, kBlock, 0, st>>>(srcs[f]->plane[i], sw, sh, job.ctx->inter, dw, p_.factor_h, th, eh, prm);
      const dim3 grid_v((dw + kBlock - 1) / kBlock, nrows);
      v_pass<<<grid_v, kBlock, 0, st>>>(job.ctx->inter, dw, sh, dst, row_first, row_step, nrows, p_.factor_v, tv, prm);
    }
    const std::size_t n = static_cast<std::size_t>(dw) * dh;
    if (!out_dev &&
        cudaMemcpyAsync(job.ctx->host_dst + dst_off, dst, n * sizeof(uint16_t), cudaMemcpyDeviceToHost, st) != cudaSuccess)
      return -2;
    dst_off += n;
  }
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

void CudaEngine::finish_device(DeviceJob& job, const Frame& out) {
  const uint16_t* h = job.ctx->host_dst;
  for (int i = 0; i < 3; ++i) {
    const Plane& d = out.plane[i];
    for (int y = 0; y < d.height; ++y, h += d.width) std::memcpy(d.data + y * d.stride, h, d.width * sizeof(uint16_t));
  }
}

}  // namespace cudaconvolution
