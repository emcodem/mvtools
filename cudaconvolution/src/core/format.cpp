#include "core/format.h"

#include <stdexcept>

namespace cudaconvolution {

std::string format_name(SampleFormat f) {
  const char* l = f.layout == Layout::YUV422 ? "YUV422P" : f.layout == Layout::YUV444 ? "YUV444P" : "RGBP";
  return l + std::to_string(f.bits);
}

Preset adapt_preset(Preset p, SampleFormat f) {
  if (f.bits != 10 && f.bits != 12 && f.bits != 16) throw std::invalid_argument("adapt_preset: bits must be 10, 12 or 16");
  if (f.layout != Layout::YUV422) p.chroma_h = p.luma_h;
  if (f.layout == Layout::RGB) {
    p.arith = Arith::RoundHalfUp;
    p.offset = 0;
    p.has_intermediate_clip = false;
    p.output_lo = 0;
    p.output_hi = (1 << f.bits) - 1;
    return p;
  }
  const int s = 1 << (f.bits - 10);
  p.offset *= s;
  p.intermediate_lo *= s;
  p.intermediate_hi = (p.intermediate_hi + 1) * s - 1;
  p.output_lo *= s;
  p.output_hi = (p.output_hi + 1) * s - 1;
  return p;
}

}  // namespace cudaconvolution
