#pragma once
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace cudaconvolution {

constexpr int kMaxFactor = 8;      // per axis
constexpr int kMaxTaps = 128;      // per kernel (factor 8 luma H: 121)
constexpr int kMaxEdgeCols = 8;    // per side of an H kernel
constexpr int kMaxEdgeTaps = 32;   // per edge kernel

// Special kernel for one output column at a plane edge: taps[i] applies to source column src_first + i (left edge:
// absolute; right edge: relative to the plane width, i.e. width + src_first + i). All sources lie inside the plane,
// so no border rule is involved.
struct EdgeKernel {
  int src_first = 0;
  std::vector<float> taps;
};

// One 1D filter: taps[i] applies to source offset (first + i), accumulated in ascending i.
struct Kernel {
  int first = 0;
  std::vector<float> taps;  // integer taps / divisor (exact in float32 for divisor 2^n) or float32 values
  // H kernels only: left[i] replaces the kernel for output column i, right[i] for output column (width - 1 - i)
  std::vector<EdgeKernel> left, right;
};

enum class Arith {
  F32FmaTrunc,      // output = offset + trunc(o + 0.5)  (C cast: "+1 below offset" on flat fields), measured reference
  RoundHalfUp,      // output = offset + floor(o + 0.5)
};

enum class Border {
  Replicate,  // edge sample repeated (measured reference)
  Mirror,     // reflect without repeating the edge sample
};

// Validated, typed filter setup for one (factor_h, factor_v). All levels in codes of the 10-bit source.
struct Preset {
  std::string name;  // "<preset>/h<n>v<m>"
  int factor_h = 1, factor_v = 1;
  bool interlaced = true;
  bool tff = true;
  Arith arith = Arith::F32FmaTrunc;
  int offset = 512;
  bool has_intermediate_clip = false;
  int intermediate_lo = 0, intermediate_hi = 1023;
  int output_lo = 0, output_hi = 1023;  // storage range by default; CudaConvolution(clip_min, clip_max) narrows it
  Border border = Border::Replicate;
  Kernel luma_h, chroma_h, v;
};

// Preset JSON:
//   {"name", "target": {"scan", "field_order"},
//    "profiles": {"<profile>": {"arith", "offset_code", "intermediate_clip", "border"}, ...},
//    (the intermediate clip is part of the measured arithmetic; output clipping is not a preset property)
//    "h": {"<factor>": {"profile", "luma": kernel, "chroma": kernel}, ...},
//    "v": {"<factor>": {"profile", "kernel": kernel}, ...}}
// kernel = {"first", "taps", "divisor" (default 1), H only: "edges": {"left": [...], "right": [...]}}.
// The H and V entries for the requested factors are combined; the arithmetic profile is "measured" only if both
// entries use it (the measured reference was measured as one H+V filter), otherwise "designed".
// Throws std::runtime_error naming the offending key or listing the available factors.
Preset parse_preset(const nlohmann::json& j, int factor_h, int factor_v);

// Factors (1..kMaxFactor) the preset has kernels for, per axis.
std::vector<int> preset_factors(const nlohmann::json& j, const char* axis);

}  // namespace cudaconvolution
