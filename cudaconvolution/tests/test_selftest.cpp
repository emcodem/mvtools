// Embedded self-test vectors: generation and verification on the CPU paths (GPU: test_cuda.cpp).
#include <catch2/catch_test_macros.hpp>
#include <fstream>

#include "core/selftest.h"

using namespace cudaconvolution;

namespace {

EngineFn cpu(CpuPath path) {
  return [path](const ConstFrame& a, const ConstFrame& b, const Frame& o, const Preset& p) {
    std::vector<float> scratch;
    process_frame(a, b, o, p, scratch, path);
  };
}

}  // namespace

// Regenerate after an intended change of the arithmetic or the presets: cudaconvolution_tests "[.gen-selftest]"
TEST_CASE("generate selftest/vectors.json from the scalar reference", "[.gen-selftest]") {
  nlohmann::json j = nlohmann::json::object();
  for (const SelftestCase& c : selftest_cases()) j[c.label()] = selftest_md5(c, cpu(CpuPath::Scalar));
  std::ofstream(CUDACONVOLUTION_SOURCE_DIR "/selftest/vectors.json") << j.dump(2) << "\n";
}

TEST_CASE("self-test vectors: scalar and AVX2 reproduce the embedded MD5") {
  const auto cases = selftest_cases();
  REQUIRE(cases.size() == 30);
  for (const SelftestCase& c : cases) {
    INFO(c.label() << " (regenerate with [.gen-selftest] after intended changes)");
    REQUIRE_FALSE(c.md5.empty());
    CHECK(selftest_md5(c, cpu(CpuPath::Scalar)) == c.md5);
    if (cpu_has_avx2_fma()) CHECK(selftest_md5(c, cpu(CpuPath::Avx2)) == c.md5);
  }
}
