// mvtools_vulkan: a small compute runtime on top of Vulkan 1.2 for the Vulkan build of the mvtools_cuda C API (mvcuda.h).
// The library exports the same functions as mvtools_cuda (same names, same behaviour, same results bit for bit), so
// mvtools2 and cudafilters load either one (dyn_lib.h open_gpu_backend: MVTOOLS_GPU_BACKEND=auto|cuda|vulkan).
//
// Model (simpler than the CUDA library's streams and submitter): every operation records one command buffer (Cmd), submits
// it and waits for it. Results are complete when an entry point returns, so device memory can be reused at once and there
// is no ordering between operations to track. Several threads run operations concurrently (one queue, submissions under a
// mutex). Device memory is addressed by 64-bit buffer device addresses (dptr): host code does pointer arithmetic on them as
// the CUDA code does on device pointers, and shaders read them through GL_EXT_buffer_reference.
//
// Shaders: Sources/vulkan/shaders/*.comp, compiled to SPIR-V at build time and embedded (shaders.cmake). A file named
// x.pix.comp is compiled twice: x_8 (MVC_PIX8 defined: 8-bit samples) and x_16. Kernel arguments: a C struct copied into
// a per-command argument buffer; the push constant is its address (shaders/common.glsl: ARGS(type)). Use only 32/64-bit
// integer, float and double members and dptr (uint64_t) for pointers, so that C layout == GLSL scalar layout.
#ifndef MVTOOLS_VKRT_H
#define MVTOOLS_VKRT_H

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <vector>

namespace vkrt
{
  typedef uint64_t dptr; // buffer device address (0 = null)

  // ---- device ----------------------------------------------------------------------------------------------------
  // true when a usable device is open (first call opens it: MVVK_DEVICE=n picks the n-th usable physical device; CPU
  // implementations such as lavapipe only with MVVK_ALLOW_CPU=1). Thread safe.
  bool ready();
  int device_count();                     // usable devices (0 when the loader or a driver is missing)
  const char *device_name();
  unsigned long long device_memory();     // device-local heap size in bytes
  unsigned long long device_free_memory();// estimate (VK_EXT_memory_budget when present, else heap size / 2)
  bool has_float64();                     // shaderFloat64 (analysis nLambda in double)
  bool log_enabled();                     // MVVK_LOG=1

  // ---- memory ----------------------------------------------------------------------------------------------------
  // Device-local storage buffers, pooled by exact size (allocation is slow). nullptr / 0 on failure.
  dptr dev_alloc(size_t bytes);
  void dev_free(dptr p, size_t bytes);
  void pool_trim();                       // frees all pooled (unused) buffers

  // owning handle for scratch buffers in host code
  struct DevBuf
  {
    dptr p = 0;
    size_t bytes = 0;
    DevBuf() = default;
    explicit DevBuf(size_t n) : p(dev_alloc(n)), bytes(p ? n : 0) {}
    ~DevBuf() { reset(); }
    DevBuf(const DevBuf &) = delete;
    DevBuf &operator=(const DevBuf &) = delete;
    DevBuf(DevBuf &&o) noexcept : p(o.p), bytes(o.bytes) { o.p = 0; o.bytes = 0; }
    DevBuf &operator=(DevBuf &&o) noexcept { if (this != &o) { reset(); p = o.p; bytes = o.bytes; o.p = 0; o.bytes = 0; } return *this; }
    void reset() { if (p) dev_free(p, bytes); p = 0; bytes = 0; }
    explicit operator bool() const { return p != 0; }
  };

  // ---- kernels ---------------------------------------------------------------------------------------------------
  struct Kernel;
  // The compute pipeline of the embedded shader `name` (file name without .comp; x.pix.comp gives x_8 and x_16), with the
  // specialization constants spec[i] at constant_id i (int). Created once and cached. nullptr when unknown or on failure.
  Kernel *kernel(const char *name, std::initializer_list<int> spec = {});
  Kernel *kernel(const char *name, const std::vector<int> &spec);

  // ---- command recording -----------------------------------------------------------------------------------------
  // One command buffer: record, then submit_wait(). Commands run in order only where barrier() separates them (Vulkan
  // runs dispatches of one command buffer concurrently otherwise). Every method returns false on failure (and submit_wait
  // then reports it); a Cmd that is destroyed without submit_wait() is discarded.
  class Cmd
  {
  public:
    Cmd();
    ~Cmd();
    Cmd(const Cmd &) = delete;
    Cmd &operator=(const Cmd &) = delete;

    bool ok() const;

    // groups gx * gy * gz of k; args (bytes copied now) is the kernel's argument struct (ARGS in common.glsl)
    bool dispatch(Kernel *k, uint32_t gx, uint32_t gy, uint32_t gz, const void *args, size_t args_bytes);
    template <class A> bool dispatch(Kernel *k, uint32_t gx, uint32_t gy, uint32_t gz, const A &args)
    {
      return dispatch(k, gx, gy, gz, &args, sizeof(A));
    }
    // makes the writes of everything recorded so far visible to everything recorded later (compute and transfer)
    void barrier();

    bool copy(dptr dst, dptr src, size_t bytes);           // device -> device
    bool fill(dptr dst, size_t bytes, uint32_t value);     // bytes and dst multiple of 4
    // host -> device: the host bytes are copied into a staging buffer now (host may reuse them on return)
    bool upload(dptr dst, const void *host, size_t bytes);
    // 2D host -> device: rows of row_bytes, host pitch hpitch, device pitch dpitch
    bool upload2d(dptr dst, size_t dpitch, const void *host, size_t hpitch, size_t row_bytes, size_t rows);
    // device -> host: copied into host after the GPU finished, inside submit_wait()
    bool download(void *host, dptr src, size_t bytes);
    bool download2d(void *host, size_t hpitch, dptr src, size_t dpitch, size_t row_bytes, size_t rows);

    // submits and waits; 0, or -2 on any Vulkan error (also errors while recording)
    int submit_wait();

  private:
    struct Impl;
    Impl *d;
  };

  // pointers of a device frame / object kept alive until a Cmd finished: not needed (submit_wait waits).
}

#endif
