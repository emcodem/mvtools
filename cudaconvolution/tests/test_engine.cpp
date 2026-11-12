#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "core/engine.h"
#include "core/md5.h"
#include "core/preset.h"
#include "core/presets.h"
#include "test_frame.h"

using namespace cudaconvolution;
using cudaconvolution::test::TestFrame;

namespace {

Preset uhd() { return parse_preset(load_preset("balanced"), 2, 2); }

TestFrame run(const Preset& p, const TestFrame& a, const TestFrame& b) {
  TestFrame out(a.w / p.factor_h, a.h / p.factor_v);
  std::vector<float> scratch;
  process_frame(a.cref(), b.cref(), out.ref(), p, scratch);
  return out;
}

}  // namespace

TEST_CASE("md5 known vectors") {
  Md5 a;
  CHECK(a.hex() == "d41d8cd98f00b204e9800998ecf8427e");
  Md5 b;
  b.update("abc", 3);
  CHECK(b.hex() == "900150983cd24fb0d6963f7d28e17f72");
  Md5 c;  // fed byte by byte
  const std::string s = "The quick brown fox jumps over the lazy dog";
  for (char ch : s) c.update(&ch, 1);
  CHECK(c.hex() == "9e107d9d372bb6826bd81d3542a419d6");
}

TEST_CASE("flat fields: measured reference level transfer (+1 below 512, intermediate clip at 1020)") {
  const Preset p = uhd();
  for (int v = 0; v < 1024; ++v) {
    TestFrame src(64, 32);
    src.fill([&](int, int, int) { return v; });
    const TestFrame out = run(p, src, src);
    const int expect = v < 512 ? v + 1 : std::min(v, 1020);
    for (int i = 0; i < 3; ++i)
      for (int y = 0; y < out.h; ++y)
        for (int x = 0; x < out.pw(i); ++x)
          if (out.at(i, y, x) != expect)
            FAIL("level " << v << " plane " << i << " (" << y << "," << x << "): " << out.at(i, y, x) << " != " << expect);
  }
}

TEST_CASE("impulse: output = separable tap product") {
  const Preset p = uhd();
  TestFrame src(128, 64);
  src.fill([](int i, int y, int x) { return (i == 0 && y == 32 && x == 64) ? 912 : 512; });
  const TestFrame out = run(p, src, src);
  const double h0 = 8107 / 16384.0;
  // output (16, 32) is centred on source (32, 64): centre taps
  CHECK(out.at(0, 16, 32) == 512 + int(std::trunc(400 * h0 * (4905 / 16384.0) + 0.5)));
  // output row 15 <- source row 30: impulse at V offset +2 (tap 2099)
  CHECK(out.at(0, 15, 32) == 512 + int(std::trunc(400 * h0 * (2099 / 16384.0) + 0.5)));
  for (int y = 0; y < out.h; ++y)
    for (int x = 0; x < out.pw(1); ++x) REQUIRE(out.at(1, y, x) == 512);  // chroma untouched
}

TEST_CASE("field assembly: TFF / BFF take the right source frame") {
  Preset p = uhd();
  TestFrame a(64, 32), b(64, 32);
  a.fill([](int, int, int) { return 100; });
  b.fill([](int, int, int) { return 700; });
  for (bool tff : {true, false}) {
    p.tff = tff;
    const TestFrame out = run(p, a, b);
    for (int y = 0; y < out.h; ++y) {
      const bool from_a = (y % 2 == 0) == tff;  // TFF: even (top) rows from the first frame
      INFO("tff=" << tff << " row " << y);
      CHECK(out.at(0, y, 5) == (from_a ? 101 : 700));
    }
  }
}

TEST_CASE("borders: replicate vs mirror differ only near the edges") {
  Preset p = uhd();
  TestFrame src(128, 64);
  uint32_t r = 12345;
  src.fill([&](int, int, int) { return int(((r = r * 1103515245u + 12345u) >> 16) % 1024); });
  const TestFrame rep = run(p, src, src);
  p.border = Border::Mirror;
  const TestFrame mir = run(p, src, src);
  int edge_diff = 0;
  for (int y = 0; y < rep.h; ++y)
    for (int x = 0; x < rep.w; ++x) {
      const bool interior = y >= 4 && y < rep.h - 4 && x >= 8 && x < rep.w - 8;  // V reach 8 src rows, H 15 src cols
      if (interior) REQUIRE(rep.at(0, y, x) == mir.at(0, y, x));
      else edge_diff += rep.at(0, y, x) != mir.at(0, y, x);
    }
  CHECK(edge_diff > 0);
}

TEST_CASE("parse_preset: speaking errors") {
  auto j = load_preset("balanced");
  j["v"]["2"]["kernel"].erase("taps");
  try {
    parse_preset(j, 2, 2);
    FAIL("no exception");
  } catch (const std::runtime_error& e) {
    CHECK(std::string(e.what()) == "preset key 'v.2.kernel.taps': missing");
  }
  CHECK_THROWS_WITH(parse_preset(load_preset("balanced"), 9, 1), "scale factors must be 1..8 per axis, got 9 x 1");
  auto k = load_preset("balanced");
  k["h"].erase("5");
  CHECK_THROWS_WITH(parse_preset(k, 5, 5),
                    "preset 'balanced' has no horizontal filter for factor 5 (available: 1, 2, 3, 4, 6, 7, 8)");
}

TEST_CASE("AVX2 path is bit-identical to the scalar reference") {
  if (!cpu_has_avx2_fma()) SKIP("CPU has no AVX2/FMA3");
  uint32_t r = 987654321;
  auto rnd = [&] { return ((r = r * 1664525u + 1013904223u) >> 8) % 1024; };
  for (int fh : {1, 2})
    for (Border b : {Border::Replicate, Border::Mirror})
      for (Arith a : {Arith::F32FmaTrunc, Arith::RoundHalfUp})
        for (int w : {64, 244, 300}) {  // output widths 61 / 75 / 122 / 150: vector tails
          Preset p = uhd();
          p.factor_h = fh;
          p.border = b;
          p.arith = a;
          TestFrame src(w, 48);
          src.fill([&](int, int, int) { return int(rnd()); });
          TestFrame o1(w / fh, 24), o2(w / fh, 24);
          std::vector<float> s1, s2;
          process_frame(src.cref(), src.cref(), o1.ref(), p, s1, CpuPath::Scalar);
          process_frame(src.cref(), src.cref(), o2.ref(), p, s2, CpuPath::Avx2);
          INFO("fh=" << fh << " border=" << int(b) << " arith=" << int(a) << " w=" << w);
          for (int i = 0; i < 3; ++i) CHECK(o1.pl[i] == o2.pl[i]);
        }
}

TEST_CASE("edge kernels replace the H kernel at the outermost output columns") {
  Preset p = uhd();
  p.luma_h.left = {{0, {1.0f}}, {5, {0.5f, 0.5f}}};  // col 0 <- src 0, col 1 <- avg(src 5, 6)
  p.luma_h.right = {{-1, {1.0f}}};                    // last col <- last src
  p.border = Border::Replicate;
  TestFrame src(64, 32);
  src.fill([](int, int y, int x) { return y < 0 ? 0 : 100 + 10 * x; });  // ramp, constant down the columns
  const TestFrame out = run(p, src, src);
  for (int y = 0; y < out.h; ++y) {
    INFO("row " << y);
    CHECK(out.at(0, y, 0) == 101);              // +1: "below 512" rule, value 100
    CHECK(out.at(0, y, 1) == 156);              // (150 + 160) / 2 = 155, +1
    CHECK(out.at(0, y, out.w - 1) == 730);      // src 63: 730 (>= 512: no +1)
  }
  // the AVX2 path and the CUDA engine use the same edge handling (test_cuda.cpp: all modes)
  if (cpu_has_avx2_fma()) {
    TestFrame o2(32, 16);
    std::vector<float> s2;
    process_frame(src.cref(), src.cref(), o2.ref(), p, s2, CpuPath::Avx2);
    for (int i = 0; i < 3; ++i) CHECK(o2.pl[i] == out.pl[i]);
  }
}

TEST_CASE("preset: float taps and edge kernel validation") {
  auto j = load_preset("balanced");
  const Preset p = parse_preset(j, 2, 2);
  CHECK(p.chroma_h.left.size() == 4);
  CHECK(p.chroma_h.right.size() == 4);
  CHECK(p.chroma_h.right[0].src_first + int(p.chroma_h.right[0].taps.size()) == 0);
  CHECK(p.chroma_h.taps[14] == 0.480875999f);  // float32 value round-trips from JSON
  auto bad = j;
  bad["h"]["2"]["chroma"]["edges"]["right"][0]["src_first"] = -2;  // 18 taps would end beyond the edge
  CHECK_THROWS_WITH(parse_preset(bad, 2, 2),
                    "preset key 'h.2.chroma.edges.right[0]': taps must end at or before the right edge "
                    "(src_first + count <= 0)");
  auto vedge = j;
  vedge["v"]["2"]["kernel"]["edges"] = {{"left", nlohmann::json::array()}};
  CHECK_THROWS(parse_preset(vedge, 2, 2));
}

TEST_CASE("CPU worker threads give bit-identical results") {
  uint32_t r = 2024;
  for (auto [fh, fv] : {std::pair{2, 2}, {1, 1}, {3, 2}}) {
    Preset p = parse_preset(load_preset("balanced"), fh, fv);
    TestFrame a(256 * fh, 96 * fv), b(256 * fh, 96 * fv);
    a.fill([&](int, int, int) { return int(((r = r * 1664525u + 1013904223u) >> 4) % 1024u); });
    b.fill([&](int, int, int) { return int(((r = r * 1664525u + 1013904223u) >> 4) % 1024u); });
    TestFrame o1(256, 96), o2(256, 96);
    std::vector<float> s1, s2;
    process_frame(a.cref(), b.cref(), o1.ref(), p, s1, CpuPath::Auto, 1);
    process_frame(a.cref(), b.cref(), o2.ref(), p, s2, CpuPath::Auto, 7);
    INFO(fh << "x" << fv);
    for (int i = 0; i < 3; ++i) CHECK(o1.pl[i] == o2.pl[i]);
  }
}
