#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "core/engine.h"
#include "core/preset.h"
#include "core/presets.h"
#include "test_frame.h"

using namespace cudaconvolution;
using cudaconvolution::test::TestFrame;

TEST_CASE("built-in preset 'balanced': kernels for factors 1..8 on both axes") {
  auto names = preset_names();
  REQUIRE(std::find(names.begin(), names.end(), "balanced") != names.end());
  const auto j = load_preset("balanced");
  const std::vector<int> all{1, 2, 3, 4, 5, 6, 7, 8};
  CHECK(preset_factors(j, "h") == all);
  CHECK(preset_factors(j, "v") == all);
  for (int fh : all)
    for (int fv : all) CHECK_NOTHROW(parse_preset(j, fh, fv));
}

TEST_CASE("arithmetic profile: measured only for the measured H+V combination") {
  const auto j = load_preset("balanced");
  const Preset m = parse_preset(j, 2, 2);
  CHECK(m.name == "balanced/h2v2");
  CHECK(m.arith == Arith::F32FmaTrunc);
  CHECK(m.output_lo == 0);  // no output clipping by default (CudaConvolution clip_min / clip_max)
  CHECK(m.output_hi == 1023);
  CHECK(m.has_intermediate_clip);  // part of the measured arithmetic
  CHECK(m.intermediate_lo == 0);
  CHECK(m.intermediate_hi == 1020);
  for (auto [fh, fv] : {std::pair{1, 1}, {2, 1}, {1, 2}, {3, 3}, {2, 4}}) {
    INFO(fh << "x" << fv);
    const Preset d = parse_preset(j, fh, fv);
    CHECK(d.arith == Arith::RoundHalfUp);
    CHECK(d.output_lo == 0);
    CHECK(d.output_hi == 1023);
    CHECK_FALSE(d.has_intermediate_clip);
  }
}

TEST_CASE("measured factor-2 taps: support and DC gain") {
  const auto j = load_preset("balanced");
  const auto& luma = j["h"]["2"]["luma"];
  const auto& v = j["v"]["2"]["kernel"];
  for (const auto* k : {&luma, &v}) {
    std::vector<int> taps = (*k)["taps"];
    CHECK((*k)["divisor"] == 16384);
    CHECK(int((*k)["first"]) == -static_cast<int>(taps.size() / 2));
    CHECK(std::accumulate(taps.begin(), taps.end(), 0) == 16384);
  }
  CHECK(luma["taps"].size() == 31);
  CHECK(v["taps"].size() == 17);
  const auto& c = j["h"]["2"]["chroma"];  // float32 taps, divisor 1
  CHECK(c["first"] == -14);
  double sum = 0;
  for (double x : c["taps"].get<std::vector<double>>()) sum += x;
  CHECK(std::abs(sum - 1.0) < 1e-6);
}

TEST_CASE("designed factors 3..8: symmetric, DC gain exactly 1, length grows with the factor") {
  const auto j = load_preset("balanced");
  std::size_t prev = 0;
  for (int f = 3; f <= 8; ++f) {
    INFO("factor " << f);
    for (const auto* k : {&j["h"][std::to_string(f)]["luma"], &j["h"][std::to_string(f)]["chroma"],
                          &j["v"][std::to_string(f)]["kernel"]}) {
      std::vector<int> taps = (*k)["taps"];
      const int div = (*k)["divisor"];
      CHECK(std::accumulate(taps.begin(), taps.end(), 0) == div);
      CHECK(int((*k)["first"]) == -static_cast<int>(taps.size() / 2));
      CHECK(std::equal(taps.begin(), taps.end(), taps.rbegin()));
    }
    const std::size_t n = j["h"][std::to_string(f)]["luma"]["taps"].size();
    CHECK(n > prev);
    prev = n;
  }
}

TEST_CASE("designed factors: flat fields pass unchanged, levels stay legal") {
  const auto j = load_preset("balanced");
  for (auto [fh, fv] : {std::pair{3, 3}, {4, 2}, {8, 8}, {5, 7}}) {
    const Preset p = parse_preset(j, fh, fv);
    for (int v : {4, 64, 300, 511, 512, 700, 940, 1019}) {
      TestFrame src(128 * fh, 48 * fv);
      src.fill([&](int, int, int) { return v; });
      TestFrame out(128, 48);
      std::vector<float> scratch;
      process_frame(src.cref(), src.cref(), out.ref(), p, scratch);
      INFO(fh << "x" << fv << " level " << v);
      for (int i = 0; i < 3; ++i) CHECK(std::all_of(out.pl[i].begin(), out.pl[i].end(), [&](uint16_t x) { return x == v; }));
    }
  }
}

TEST_CASE("output_clip in a preset file is rejected with a pointer to the parameters") {
  auto j = load_preset("balanced");
  j["profiles"]["designed"]["output_clip"] = {4, 1019};
  CHECK_THROWS_WITH(parse_preset(j, 1, 1), Catch::Matchers::ContainsSubstring("clip_min / clip_max"));
}

TEST_CASE("unknown preset throws") { CHECK_THROWS(load_preset("nope")); }
