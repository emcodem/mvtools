#pragma once
#include <algorithm>
#include <cstdint>
#include <string>

#include "core/preset.h"

namespace cudaconvolution {

enum class Layout {
  YUV422,  // chroma planes half width: chroma_h taps
  YUV444,  // chroma planes full width: luma_h taps
  RGB,     // planar RGB, all planes luma_h taps
};

struct SampleFormat {
  Layout layout = Layout::YUV422;
  int bits = 10;  // 10, 12 or 16
};

std::string format_name(SampleFormat f);  // "YUV422P10", "RGBP16", ...

// Preset for this input format (presets are written in 10-bit codes):
// - levels scaled by 2^(bits-10): offset and lower clip bounds x s, upper bounds (hi + 1) x s - 1
//   (e.g. 10-bit 4..1019 -> 16-bit 256..65279; the default output range 0..1023 -> 0..65535);
// - YUV444 / RGB: chroma planes use the luma H filter;
// - RGB: no offset, no intermediate clip, round_half_up.
// The measured reference arithmetic applies to YUV422P10; everything else is the same contract, not measured.
Preset adapt_preset(Preset p, SampleFormat f);

// 8-bit input runs the 10-bit arithmetic: samples x 4 in, out rounded to 8 bit = ConvertBits(10) ... ConvertBits(8)
// (no dither). Native 8-bit arithmetic would scale the trunc rule's "+1 below offset" to a full 8-bit code.
constexpr uint16_t from_8bit(uint8_t v) { return static_cast<uint16_t>(v << 2); }
constexpr uint8_t to_8bit(uint16_t v) { return static_cast<uint8_t>(std::min((v + 2) >> 2, 255)); }

}  // namespace cudaconvolution
