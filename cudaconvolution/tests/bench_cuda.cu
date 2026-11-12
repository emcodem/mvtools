// Stage timing of the GPU path on a UHD frame (not a test: run "cudaconvolution_tests [.bench]").
#include <cuda_runtime.h>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdio>
#include <vector>

#include "core/preset.h"
#include "core/presets.h"
#include "cuda/device.h"
#include "cuda/engine_cuda.h"
#include "test_frame.h"

using namespace cudaconvolution;
using cudaconvolution::test::TestFrame;

TEST_CASE("bench: GPU stages UHD -> 1080i", "[.bench]") {
  if (!cuda_status(0).ok) SKIP("no GPU");
  const Preset p = parse_preset(load_preset("balanced"), 2, 2);
  TestFrame a(3840, 2160), b(3840, 2160), o(1920, 1080);
  a.fill([](int, int y, int x) { return (x * 3 + y) % 1024; });
  b.fill([](int, int y, int x) { return (x + y * 5) % 1024; });
  CudaEngine e(0, p);
  e.process_frame(a.cref(), b.cref(), o.ref());  // warm-up, allocation
  const int n = 50;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < n; ++i) e.process_frame(a.cref(), b.cref(), o.ref());
  auto t1 = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / n;

  // raw transfer rates for comparison: 66 MB pageable vs pinned upload, 8 MB download
  const size_t up = 2 * a.pl[0].size() * 2 * 2, down = o.pl[0].size() * 2 * 2;
  std::vector<uint8_t> host(up);
  uint8_t *pinned = nullptr, *dev = nullptr;
  cudaMallocHost(&pinned, up);
  cudaMalloc(&dev, up);
  auto time_copy = [&](void* dst, const void* src, size_t bytes, cudaMemcpyKind k) {
    cudaDeviceSynchronize();
    auto s = std::chrono::steady_clock::now();
    for (int i = 0; i < 10; ++i) cudaMemcpy(dst, src, bytes, k);
    cudaDeviceSynchronize();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count() / 10;
  };
  const double up_pageable = time_copy(dev, host.data(), up, cudaMemcpyHostToDevice);
  const double up_pinned = time_copy(dev, pinned, up, cudaMemcpyHostToDevice);
  const double down_pageable = time_copy(host.data(), dev, down, cudaMemcpyDeviceToHost);
  cudaFreeHost(pinned);
  cudaFree(dev);
  std::printf("process_frame: %.2f ms/frame = %.1f fps\n", ms, 1000 / ms);
  std::printf("upload %zu MB: pageable %.2f ms (%.1f GB/s), pinned %.2f ms (%.1f GB/s); download %zu MB pageable %.2f ms\n",
              up >> 20, up_pageable, up / up_pageable / 1e6, up_pinned, up / up_pinned / 1e6, down >> 20, down_pageable);
}
