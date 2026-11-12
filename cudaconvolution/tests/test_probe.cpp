// Golden tests on the measurement probe. Needs CUDACONVOLUTION_DATA = folder with pin.yuv / pref.yuv (see tools/pio.py).
// CUDACONVOLUTION_PROBE_FRAMES limits the number of output frames (default 100).
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/engine.h"
#include "core/md5.h"
#include "core/preset.h"
#include "core/presets.h"
#ifdef CUDACONVOLUTION_HAVE_CUDA
#include "cuda/device.h"
#include "cuda/engine_cuda.h"
#endif

using namespace cudaconvolution;

namespace {

constexpr int SW = 3840, SH = 2160, DW = 1920, DH = 1080;
constexpr std::size_t SRC_SAMPLES = std::size_t(SW) * SH * 2, DST_SAMPLES = std::size_t(DW) * DH * 2;

std::vector<uint16_t> read_frame(const std::string& path, std::size_t samples, int index) {
  std::vector<uint16_t> buf(samples);
  FILE* f = std::fopen(path.c_str(), "rb");
  REQUIRE(f);
  _fseeki64(f, static_cast<long long>(samples) * 2 * index, SEEK_SET);
  const std::size_t n = std::fread(buf.data(), 2, samples, f);
  std::fclose(f);
  REQUIRE(n == samples);
  return buf;
}

// yuv422p10le frame in one buffer: Y (w x h), U, V (w/2 x h)
template <class F, class T> F planes(T* d, int w, int h) {
  F f;
  const std::size_t y = std::size_t(w) * h, c = std::size_t(w / 2) * h;
  f.plane[0] = {d, w, w, h};
  f.plane[1] = {d + y, w / 2, w / 2, h};
  f.plane[2] = {d + y + c, w / 2, w / 2, h};
  return f;
}

}  // namespace

TEST_CASE("probe: luma bit-exact to the reference, full output = python model", "[probe]") {
  const char* data = std::getenv("CUDACONVOLUTION_DATA");
  if (!data) SKIP("CUDACONVOLUTION_DATA not set");
  const std::string dir = data;
  const char* nf = std::getenv("CUDACONVOLUTION_PROBE_FRAMES");
  const int frames = nf ? std::atoi(nf) : 100;

  std::vector<std::string> model;
  std::ifstream md5s(CUDACONVOLUTION_SOURCE_DIR "/tests/data/model_balanced_uhd.md5");
  for (std::string line; std::getline(md5s, line);) model.push_back(line);
  REQUIRE(model.size() >= std::size_t(frames));

  Preset p = parse_preset(load_preset("balanced"), 2, 2);
  p.output_lo = 1;  // the reference's output clip: CudaConvolution(clip_min=1, clip_max=1020)
  p.output_hi = 1020;
  std::vector<uint16_t> out(DST_SAMPLES);
  std::vector<float> scratch;
  long long luma_mismatch = 0;
  int md5_mismatch = 0, gpu_md5_mismatch = 0;
#ifdef CUDACONVOLUTION_HAVE_CUDA
  std::unique_ptr<CudaEngine> gpu;
  if (cuda_status(0).ok) gpu = std::make_unique<CudaEngine>(0, p);
  else WARN("GPU part skipped: " << cuda_status(0).error);
#endif
  for (int k = 0; k < frames; ++k) {
    const auto a = read_frame(dir + "/pin.yuv", SRC_SAMPLES, 2 * k);
    const auto b = read_frame(dir + "/pin.yuv", SRC_SAMPLES, 2 * k + 1);
    const auto ref = read_frame(dir + "/pref.yuv", DST_SAMPLES, k);
    process_frame(planes<ConstFrame>(a.data(), SW, SH), planes<ConstFrame>(b.data(), SW, SH),
                  planes<Frame>(out.data(), DW, DH), p, scratch);
    for (std::size_t i = 0; i < std::size_t(DW) * DH; ++i) luma_mismatch += out[i] != ref[i];
    Md5 m;
    m.update(out.data(), out.size() * 2);
    const std::string h = m.hex();
    if (h != model[k]) {
      ++md5_mismatch;
      UNSCOPED_INFO("frame " << k << ": md5 " << h << " != model " << model[k]);
    }
#ifdef CUDACONVOLUTION_HAVE_CUDA
    if (gpu) {
      gpu->process_frame(planes<ConstFrame>(a.data(), SW, SH), planes<ConstFrame>(b.data(), SW, SH),
                         planes<Frame>(out.data(), DW, DH));
      Md5 g;
      g.update(out.data(), out.size() * 2);
      const std::string gh = g.hex();
      if (gh != model[k]) {
        ++gpu_md5_mismatch;
        UNSCOPED_INFO("frame " << k << ": GPU md5 " << gh << " != model " << model[k]);
      }
    }
#endif
  }
  INFO(frames << " output frames, " << luma_mismatch << " luma mismatches vs the reference");
  CHECK(luma_mismatch == 0);
  CHECK(md5_mismatch == 0);
  CHECK(gpu_md5_mismatch == 0);
}
