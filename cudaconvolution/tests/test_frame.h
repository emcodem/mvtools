#pragma once
// Small 4:2:2 test frames for the engine tests.
#include <cstdint>
#include <vector>

#include "core/engine.h"

namespace cudaconvolution::test {

// 4:2:2 (default) or full-width-chroma (4:4:4 / RGB) test frame, planes filled with a function of (plane, y, x)
struct TestFrame {
  int w, h;
  bool half_chroma;
  std::vector<uint16_t> pl[3];
  TestFrame(int w_, int h_, bool half_chroma_ = true) : w(w_), h(h_), half_chroma(half_chroma_) {
    for (int i = 0; i < 3; ++i) pl[i].assign(std::size_t(pw(i)) * h, 0);
  }
  int pw(int i) const { return i && half_chroma ? w / 2 : w; }
  template <class F> void fill(F f) {
    for (int i = 0; i < 3; ++i)
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < pw(i); ++x) pl[i][std::size_t(y) * pw(i) + x] = uint16_t(f(i, y, x));
  }
  uint16_t at(int i, int y, int x) const { return pl[i][std::size_t(y) * pw(i) + x]; }
  ConstFrame cref() const {
    ConstFrame f;
    for (int i = 0; i < 3; ++i) f.plane[i] = {pl[i].data(), pw(i), pw(i), h};
    return f;
  }
  Frame ref() {
    Frame f;
    for (int i = 0; i < 3; ++i) f.plane[i] = {pl[i].data(), pw(i), pw(i), h};
    return f;
  }
};

}  // namespace cudaconvolution::test
