#pragma once
// GPU engine: same arithmetic contract as the CPU engine (core/engine.h, resample_plane), bit-identical output.
// No CUDA headers here, so the AviSynth wrapper and the tests can include it from plain C++.
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/engine.h"
#include "core/preset.h"

namespace cudaconvolution {

struct CudaError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class CudaEngine {
 public:
  // device must have passed cuda_status(device). Throws CudaError.
  CudaEngine(int device, Preset p);
  ~CudaEngine();
  CudaEngine(const CudaEngine&) = delete;
  CudaEngine& operator=(const CudaEngine&) = delete;

  // Same semantics as cudaconvolution::process_frame. Thread-safe: concurrent calls use separate streams and buffers.
  void process_frame(const ConstFrame& first, const ConstFrame& second, const Frame& out);

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
  struct DevicePlanes {
    const uint16_t* plane[3];
  };
  struct DeviceJob;
  struct DeviceJobDeleter {
    void operator()(DeviceJob* j) const;
  };
  using DeviceJobPtr = std::unique_ptr<DeviceJob, DeviceJobDeleter>;
  DeviceJobPtr prepare_device(const Frame& out_shape, bool host_output);
  int launch_device(DeviceJob& job, const DevicePlanes& first, const DevicePlanes& second, uint16_t* const out_dev[3],
                    void* stream) noexcept;
  void finish_device(DeviceJob& job, const Frame& out);
  int device() const { return device_; }

 private:
  struct Context;
  std::unique_ptr<Context> acquire();
  void release(std::unique_ptr<Context> ctx);

  int device_;
  Preset p_;
  std::mutex mutex_;
  std::vector<std::unique_ptr<Context>> idle_;
};

}  // namespace cudaconvolution
