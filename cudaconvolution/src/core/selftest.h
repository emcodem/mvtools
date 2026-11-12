#pragma once
// Built-in test vectors: small deterministic frames per preset x format, expected output MD5 embedded in the DLL
// (selftest/vectors.json). Used by CudaConvolution_SelfTest() and the unit tests.
#include <functional>
#include <string>
#include <vector>

#include "core/engine.h"
#include "core/format.h"
#include "core/preset.h"

namespace cudaconvolution {

struct SelftestCase {
  std::string preset;          // built-in preset
  int factor_h = 1, factor_v = 1;
  SampleFormat format;
  int width = 0, height = 0;  // source size (small, the engine is size independent)
  std::string md5;            // expected output (planes in order, uint16 little endian); empty when generating
  std::string label() const;  // "balanced/h2v2 YUV422P10" (key in selftest/vectors.json)
};

// The case list per embedded preset: factors 1x1 and 2x2 in every supported format, factors 3x3, 4x4, 8x8, 2x1, 1x2,
// 3x2 in YUV422P10. Expected MD5 from the embedded vectors file.
std::vector<SelftestCase> selftest_cases();

// Runs one case through `run` (an engine) and returns the MD5 of its output.
using EngineFn = std::function<void(const ConstFrame& first, const ConstFrame& second, const Frame& out, const Preset& p)>;
std::string selftest_md5(const SelftestCase& c, const EngineFn& run);

// Deterministic input sample: xorshift32 stream per (frame, plane) in raster order, value = (state >> 8) mod 2^bits.
// tools/selftest_check.py reproduces it.
void selftest_fill(uint16_t* data, int width, int height, int frame, int plane, int bits);

}  // namespace cudaconvolution
