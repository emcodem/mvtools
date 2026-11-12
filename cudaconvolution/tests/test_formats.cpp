// Input formats (bit depths, 4:4:4, RGB) and the hd mode.
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

#include "core/engine.h"
#include "core/format.h"
#include "core/preset.h"
#include "core/presets.h"
#include "test_frame.h"

using namespace cudaconvolution;
using cudaconvolution::test::TestFrame;

namespace {

Preset preset(const char* mode) {  // "uhd": factor 2 (measured), "hd": factor 1
  const int f = std::string(mode) == "uhd" ? 2 : 1;
  return parse_preset(load_preset("balanced"), f, f);
}

TestFrame run(const Preset& p, const TestFrame& a, const TestFrame& b, CpuPath path = CpuPath::Scalar) {
  TestFrame out(a.w / p.factor_h, a.h / p.factor_v, a.half_chroma);
  std::vector<float> scratch;
  process_frame(a.cref(), b.cref(), out.ref(), p, scratch, path);
  return out;
}

}  // namespace

TEST_CASE("adapt_preset: level scaling and plane filters") {
  const Preset base = preset("uhd");
  const Preset p16 = adapt_preset(base, {Layout::YUV422, 16});
  CHECK(p16.offset == 32768);
  CHECK(p16.intermediate_lo == 0);
  CHECK(p16.intermediate_hi == 1021 * 64 - 1);
  CHECK(p16.output_lo == 0);
  CHECK(p16.output_hi == 65535);
  CHECK(p16.chroma_h.taps == base.chroma_h.taps);
  const Preset p444 = adapt_preset(base, {Layout::YUV444, 12});
  CHECK(p444.offset == 2048);
  CHECK(p444.chroma_h.taps == base.luma_h.taps);
  const Preset rgb = adapt_preset(base, {Layout::RGB, 10});
  CHECK(rgb.offset == 0);
  CHECK(rgb.arith == Arith::RoundHalfUp);
  CHECK_FALSE(rgb.has_intermediate_clip);
  CHECK(rgb.output_lo == 0);
  CHECK(rgb.output_hi == 1023);
  CHECK(rgb.chroma_h.taps == base.luma_h.taps);
  CHECK(format_name({Layout::RGB, 16}) == "RGBP16");
  CHECK_THROWS(adapt_preset(base, {Layout::YUV422, 8}));
}

TEST_CASE("hd mode: flat fields pass unchanged at every level and bit depth (no clipping by default)") {
  for (int bits : {10, 12, 16}) {
    const Preset p = adapt_preset(preset("hd"), {Layout::YUV422, bits});
    const int step = bits == 16 ? 61 : 1;
    for (int v = 0; v < (1024 << (bits - 10)); v += step) {
      TestFrame f(32, 16);
      f.fill([&](int, int, int) { return v; });
      const TestFrame o = run(p, f, f);
      INFO("bits " << bits << " level " << v);
      REQUIRE(o.at(0, 7, 5) == v);
      REQUIRE(o.at(1, 8, 3) == v);
    }
  }
}

TEST_CASE("8-bit input: 10-bit arithmetic, rounded like ConvertBits(8); flat fields unchanged in both modes") {
  CHECK(from_8bit(255) == 1020);
  CHECK(to_8bit(1021) == 255);
  CHECK(to_8bit(1022) == 255);  // (1022 + 2) >> 2 = 256 is clamped
  CHECK(to_8bit(5) == 1);
  CHECK(to_8bit(6) == 2);  // .5 rounds up
  for (const char* mode : {"uhd", "hd"}) {
    const Preset p = adapt_preset(preset(mode), {Layout::YUV422, 10});
    for (int v = 0; v < 256; ++v) {
      TestFrame f(96, 40);  // wide enough for the factor-2 chroma edge kernels
      f.fill([&](int, int, int) { return from_8bit(static_cast<uint8_t>(v)); });
      const TestFrame o = run(p, f, f);
      INFO(mode << " level " << v);
      REQUIRE(to_8bit(o.at(0, 5, 7)) == v);
      REQUIRE(to_8bit(o.at(2, 4, 3)) == v);
    }
  }
}

TEST_CASE("output limits (clip_min / clip_max) apply to all planes") {
  Preset p = preset("hd");
  p.output_lo = 64;
  p.output_hi = 940;
  for (int v : {0, 63, 64, 500, 940, 941, 1023}) {
    TestFrame f(32, 16);
    f.fill([&](int, int, int) { return v; });
    for (CpuPath path : {CpuPath::Scalar, CpuPath::Avx2}) {
      if (path == CpuPath::Avx2 && !cpu_has_avx2_fma()) continue;
      const TestFrame o = run(p, f, f, path);
      INFO("level " << v);
      for (int i = 0; i < 3; ++i) REQUIRE(o.at(i, 5, 3) == std::clamp(v, 64, 940));
    }
  }
}

TEST_CASE("hd mode: line impulse gives the designed taps, fields from the right frame") {
  const Preset p = preset("hd");
  TestFrame a(64, 32), b(64, 32);
  a.fill([](int, int y, int) { return y == 16 ? 912 : 512; });  // +400 on line 16 of the first frame
  b.fill([](int, int, int) { return 512; });
  const TestFrame o = run(p, a, b);
  const int taps[9] = {43, -195, -765, 4211, 9796, 4211, -765, -195, 43};
  for (int y = 12; y <= 20; ++y) {
    // TFF: even lines from frame a (impulse), odd lines from frame b (flat)
    const int expect = y % 2 == 0 ? 512 + int(std::floor(400.0 * taps[16 - y + 4] / 16384 + 0.5)) : 512;
    INFO("line " << y);
    CHECK(o.at(0, y, 10) == expect);
  }
}

TEST_CASE("all presets x formats: AVX2 == scalar, levels stay in range") {
  if (!cpu_has_avx2_fma()) SKIP("CPU has no AVX2/FMA3");
  uint32_t r = 99;
  for (const char* name : {"uhd", "hd"})
    for (Layout l : {Layout::YUV422, Layout::YUV444, Layout::RGB})
      for (int bits : {10, 12, 16}) {
        const Preset p = adapt_preset(preset(name), {l, bits});
        TestFrame a(96, 40, l == Layout::YUV422), b(96, 40, l == Layout::YUV422);
        auto rnd = [&] { return int(((r = r * 1664525u + 1013904223u) >> 4) % (1u << bits)); };
        a.fill([&](int, int, int) { return rnd(); });
        b.fill([&](int, int, int) { return rnd(); });
        const TestFrame o1 = run(p, a, b, CpuPath::Scalar), o2 = run(p, a, b, CpuPath::Avx2);
        INFO(name << " " << format_name({l, bits}));
        for (int i = 0; i < 3; ++i) {
          REQUIRE(o1.pl[i] == o2.pl[i]);
          for (uint16_t v : o1.pl[i]) REQUIRE((v >= p.output_lo && v <= p.output_hi));
        }
      }
}
