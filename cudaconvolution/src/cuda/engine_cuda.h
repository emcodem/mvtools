#pragma once
// GPU engine: same arithmetic contract as the CPU engine (core/engine.h, resample_plane), bit-identical output.
// No CUDA headers here, so the AviSynth wrapper and the tests can include it from plain C++.
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/engine.h"
#include "core/preset.h"

namespace cudaconvolution {

struct CudaError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// 8-bit frames: the engine reads and writes uint8_t samples and runs the 10-bit arithmetic of the preset (adapted to
// 10 bit): samples x 4 in, every output sample rounded to 8 bit (to_8bit), then limited to lo..hi (8-bit codes).
// Bit-identical to from_8bit -> 10-bit engine -> to_8bit -> clamp(lo, hi).
struct EightBit {
  int lo = 0, hi = 255;
};

class CudaEngine {
 public:
  // device must have passed cuda_status(device). eight_bit: the engine works on 8-bit frames (Frame8 / ConstFrame8 and
  // 8-bit device planes), otherwise on uint16_t frames. Throws CudaError.
  CudaEngine(int device, Preset p, std::optional<EightBit> eight_bit = std::nullopt);
  ~CudaEngine();
  CudaEngine(const CudaEngine&) = delete;
  CudaEngine& operator=(const CudaEngine&) = delete;

  // Same semantics as cudaconvolution::process_frame. Thread-safe: concurrent calls use separate streams and buffers.
  // The overload must match the engine's sample type (throws std::logic_error otherwise).
  void process_frame(const ConstFrame& first, const ConstFrame& second, const Frame& out);
  void process_frame(const ConstFrame8& first, const ConstFrame8& second, const Frame8& out);
  bool eight_bit() const { return eight_.has_value(); }

  // Device memory one concurrent call needs for planes of this size (for messages).
  static std::size_t bytes_per_context(const ConstFrame& src, const Frame& dst);

  // Zero-copy path: the source frames are already in device memory (packed planes, stride = width), e.g. frames of
  // another library that runs our work on its own stream. Three steps:
  //   prepare_device  on the calling thread: takes a context, allocates what the call needs (may throw CudaError)
  //   launch_device   where the other library issues stream work: only kernel launches / async copies on `stream`,
  //                   no allocation, no waits, never throws; returns 0 or a negative code. out_dev: output planes
  //                   (packed) in device memory, or nullptr = copy the result into the job's host buffer
  //   finish_device   after `stream` has completed: unpacks the host buffer into `out` (host output only)
  // The job returns its context to the engine when it is destroyed.
  // Device planes hold the engine's sample type (uint16_t, or uint8_t for an 8-bit engine).
  struct DevicePlanes {
    const void* plane[3];
  };
  struct DeviceJob;
  struct DeviceJobDeleter {
    void operator()(DeviceJob* j) const;
  };
  using DeviceJobPtr = std::unique_ptr<DeviceJob, DeviceJobDeleter>;
  // out_shape: plane sizes of the output (data / stride unused); finish_device then fills `out`
  DeviceJobPtr prepare_device(const Frame& out_shape, bool host_output);
  DeviceJobPtr prepare_device(const Frame8& out_shape, bool host_output);
  int launch_device(DeviceJob& job, const DevicePlanes& first, const DevicePlanes& second, void* const out_dev[3],
                    void* stream) noexcept;
  void finish_device(DeviceJob& job, const Frame& out);
  void finish_device(DeviceJob& job, const Frame8& out);
  int device() const { return device_; }

 private:
  struct Context;
  std::unique_ptr<Context> acquire();
  void release(std::unique_ptr<Context> ctx);
  template <class T> void process(const BasicFrame<const T>& first, const BasicFrame<const T>& second,
                                  const BasicFrame<T>& out);
  template <class T> DeviceJobPtr prepare(const BasicFrame<T>& out_shape, bool host_output);
  template <class T> void finish(DeviceJob& job, const BasicFrame<T>& out);
  template <class T> void require_type() const;
  std::size_t sample_bytes() const { return eight_ ? 1 : 2; }

  int device_;
  Preset p_;
  std::optional<EightBit> eight_;
  std::mutex mutex_;
  std::vector<std::unique_ptr<Context>> idle_;
};

}  // namespace cudaconvolution
