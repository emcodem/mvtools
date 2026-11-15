// GPU engine == CPU engine, bit for bit.
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <thread>
#include <vector>

#include "core/engine.h"
#include "core/format.h"
#include "core/selftest.h"
#include "core/preset.h"
#include "core/presets.h"
#include "cuda/device.h"
#include "cuda/engine_cuda.h"
#include "test_frame.h"

using namespace cudaconvolution;
using cudaconvolution::test::TestFrame;

namespace {

Preset uhd() { return parse_preset(load_preset("balanced"), 2, 2); }

void require_gpu() {
  const CudaStatus& st = cuda_status(0);
  if (!st.ok) SKIP("no usable GPU: " << st.error);
}

void compare(const Preset& p, const TestFrame& a, const TestFrame& b) {
  TestFrame cpu(a.w / p.factor_h, a.h / p.factor_v, a.half_chroma), gpu(a.w / p.factor_h, a.h / p.factor_v, a.half_chroma);
  std::vector<float> scratch;
  process_frame(a.cref(), b.cref(), cpu.ref(), p, scratch, CpuPath::Scalar);
  CudaEngine engine(0, p);
  engine.process_frame(a.cref(), b.cref(), gpu.ref());
  for (int i = 0; i < 3; ++i) REQUIRE(cpu.pl[i] == gpu.pl[i]);
}

}  // namespace

TEST_CASE("GPU == CPU: random data, all modes, odd sizes", "[gpu]") {
  require_gpu();
  uint32_t r = 424242;
  auto rnd = [&] { return int(((r = r * 1664525u + 1013904223u) >> 8) % 1024); };
  for (int fh : {1, 2})
    for (bool interlaced : {true, false})
      for (bool tff : {true, false})
        for (Border b : {Border::Replicate, Border::Mirror})
          for (Arith ar : {Arith::F32FmaTrunc, Arith::RoundHalfUp})
            for (int w : {64, 244, 300}) {
              Preset p = uhd();
              p.factor_h = fh;
              p.interlaced = interlaced;
              p.tff = tff;
              p.border = b;
              p.arith = ar;
              TestFrame a(w, 52), c(w, 52);
              a.fill([&](int, int, int) { return rnd(); });
              c.fill([&](int, int, int) { return rnd(); });
              INFO("fh=" << fh << " interlaced=" << interlaced << " tff=" << tff << " border=" << int(b)
                          << " arith=" << int(ar) << " w=" << w);
              compare(p, a, c);
            }
}

TEST_CASE("GPU == CPU: extreme levels and every flat level", "[gpu]") {
  require_gpu();
  const Preset p = uhd();
  TestFrame a(128, 64), b(128, 64);
  a.fill([](int, int y, int x) { return ((x / 3 + y) % 2) ? 1023 : 0; });  // full-swing checkerboard: clipping
  b.fill([](int i, int y, int x) { return (x * 7 + y * 13 + i * 101) % 1024; });
  compare(p, a, b);
  for (int v = 0; v < 1024; v += 7) {
    TestFrame f(64, 32);
    f.fill([&](int, int, int) { return v; });
    compare(p, f, f);
  }
}

TEST_CASE("GPU engine: concurrent calls (AviSynth MT) give identical results", "[gpu]") {
  require_gpu();
  const Preset p = uhd();
  TestFrame a(640, 360), b(640, 360);
  uint32_t r = 7;
  a.fill([&](int, int, int) { return int(((r = r * 1103515245u + 12345u) >> 16) % 1024); });
  b.fill([&](int, int, int) { return int(((r = r * 1103515245u + 12345u) >> 16) % 1024); });
  TestFrame ref(320, 180);
  std::vector<float> scratch;
  process_frame(a.cref(), b.cref(), ref.ref(), p, scratch, CpuPath::Scalar);

  CudaEngine engine(0, p);
  constexpr int kThreads = 8, kRepeats = 10;
  std::vector<TestFrame> outs(kThreads, TestFrame(320, 180));
  std::vector<int> bad(kThreads, 0);
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t)
    threads.emplace_back([&, t] {
      for (int k = 0; k < kRepeats; ++k) {
        engine.process_frame(a.cref(), b.cref(), outs[t].ref());
        for (int i = 0; i < 3; ++i) bad[t] += outs[t].pl[i] != ref.pl[i];
      }
    });
  for (auto& th : threads) th.join();
  for (int t = 0; t < kThreads; ++t) CHECK(bad[t] == 0);
}

TEST_CASE("GPU == CPU: all presets x formats", "[gpu]") {
  require_gpu();
  uint32_t r = 4711;
  for (const char* name : {"uhd", "hd"})
    for (Layout l : {Layout::YUV422, Layout::YUV444, Layout::RGB})
      for (int bits : {10, 12, 16}) {
        const int f = std::string(name) == "uhd" ? 2 : 1;
        const Preset p = adapt_preset(parse_preset(load_preset("balanced"), f, f), {l, bits});
        TestFrame a(132, 44, l == Layout::YUV422), b(132, 44, l == Layout::YUV422);
        auto rnd = [&] { return int(((r = r * 1664525u + 1013904223u) >> 4) % (1u << bits)); };
        a.fill([&](int, int, int) { return rnd(); });
        b.fill([&](int, int, int) { return rnd(); });
        INFO(name << " " << format_name({l, bits}));
        compare(p, a, b);
      }
}

TEST_CASE("self-test vectors: GPU reproduces the embedded MD5", "[gpu]") {
  require_gpu();
  for (const SelftestCase& c : selftest_cases()) {
    INFO(c.label());
    const std::string md5 = selftest_md5(c, [](const ConstFrame& a, const ConstFrame& b, const Frame& o, const Preset& p) {
      CudaEngine(0, p).process_frame(a, b, o);
    });
    CHECK(md5 == c.md5);
    if (c.format.bits != 8) continue;
    const std::string md5_native = selftest_md5(
        c, {}, [](const ConstFrame8& a, const ConstFrame8& b, const Frame8& o, const Preset& p) {
          CudaEngine(0, p, EightBit{}).process_frame(a, b, o);
        });
    CHECK(md5_native == c.md5);
  }
}

namespace {

// 8-bit test frame (4:2:2 or full-width chroma)
struct TestFrame8 {
  int w, h;
  bool half_chroma;
  std::vector<uint8_t> pl[3];
  TestFrame8(int w_, int h_, bool half_chroma_ = true) : w(w_), h(h_), half_chroma(half_chroma_) {
    for (int i = 0; i < 3; ++i) pl[i].assign(std::size_t(pw(i)) * h, 0);
  }
  int pw(int i) const { return i && half_chroma ? w / 2 : w; }
  ConstFrame8 cref() const {
    ConstFrame8 f;
    for (int i = 0; i < 3; ++i) f.plane[i] = {pl[i].data(), pw(i), pw(i), h};
    return f;
  }
  Frame8 ref() {
    Frame8 f;
    for (int i = 0; i < 3; ++i) f.plane[i] = {pl[i].data(), pw(i), pw(i), h};
    return f;
  }
  TestFrame widened() const {  // the CPU route's input: from_8bit
    TestFrame t(w, h, half_chroma);
    for (int i = 0; i < 3; ++i)
      for (std::size_t k = 0; k < pl[i].size(); ++k) t.pl[i][k] = from_8bit(pl[i][k]);
    return t;
  }
};

// GPU on 8-bit frames == CPU scalar on from_8bit frames, rounded with to_8bit and limited to lo..hi
void compare8(const Preset& p, const TestFrame8& a, const TestFrame8& b, EightBit lim) {
  const int ow = a.w / p.factor_h, oh = a.h / p.factor_v;
  TestFrame cpu(ow, oh, a.half_chroma);
  std::vector<float> scratch;
  process_frame(a.widened().cref(), b.widened().cref(), cpu.ref(), p, scratch, CpuPath::Scalar);
  TestFrame8 gpu(ow, oh, a.half_chroma);
  CudaEngine(0, p, lim).process_frame(a.cref(), b.cref(), gpu.ref());
  for (int i = 0; i < 3; ++i)
    for (std::size_t k = 0; k < gpu.pl[i].size(); ++k) {
      const uint8_t want = std::clamp<uint8_t>(to_8bit(cpu.pl[i][k]), uint8_t(lim.lo), uint8_t(lim.hi));
      if (gpu.pl[i][k] != want) {
        INFO("plane " << i << " sample " << k << " gpu " << int(gpu.pl[i][k]) << " want " << int(want));
        REQUIRE(gpu.pl[i][k] == want);
      }
    }
}

}  // namespace

TEST_CASE("GPU 8 bit: native 8-bit frames == CPU 10-bit route, all modes, formats, limits", "[gpu]") {
  require_gpu();
  uint32_t r = 8080;
  auto rnd = [&] { return int(((r = r * 1664525u + 1013904223u) >> 8) % 256u); };
  for (int fh : {1, 2, 3})
    for (Layout l : {Layout::YUV422, Layout::YUV444, Layout::RGB})
      for (bool interlaced : {true, false})
        for (bool tff : {true, false})
          for (EightBit lim : {EightBit{}, EightBit{16, 235}, EightBit{1, 254}})
            for (int w : {64 * fh, 244 * fh}) {
              Preset p = adapt_preset(parse_preset(load_preset("balanced"), fh, fh), {l, 10});
              p.interlaced = interlaced;
              p.tff = tff;
              TestFrame8 a(w, 52 * fh, l == Layout::YUV422), b(w, 52 * fh, l == Layout::YUV422);
              for (auto* f : {&a, &b})
                for (auto& pl : f->pl)
                  for (uint8_t& v : pl) v = uint8_t(rnd());
              INFO("fh=" << fh << " " << format_name({l, 8}) << " interlaced=" << interlaced << " tff=" << tff
                         << " lim=" << lim.lo << ".." << lim.hi << " w=" << w);
              compare8(p, a, b, lim);
            }
}

TEST_CASE("GPU 8 bit: extremes and every flat level", "[gpu]") {
  require_gpu();
  const Preset p = adapt_preset(uhd(), {Layout::YUV422, 10});
  TestFrame8 a(128, 64), b(128, 64);
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < int(a.pl[i].size()); ++k) {
      a.pl[i][k] = ((k / 3 + k / a.pw(i)) % 2) ? 255 : 0;  // full-swing pattern: clipping both ends
      b.pl[i][k] = uint8_t((k * 7 + i * 101) % 256);
    }
  compare8(p, a, b, {});
  for (int v = 0; v < 256; ++v) {
    TestFrame8 f(64, 32);
    for (auto& pl : f.pl) pl.assign(pl.size(), uint8_t(v));
    compare8(p, f, f, {});
  }
}

TEST_CASE("GPU engine: sample type must match the engine", "[gpu]") {
  require_gpu();
  const Preset p = uhd();
  TestFrame a(64, 32), o(32, 16);
  TestFrame8 a8(64, 32), o8(32, 16);
  CHECK_THROWS_AS(CudaEngine(0, p, EightBit{}).process_frame(a.cref(), a.cref(), o.ref()), std::logic_error);
  CHECK_THROWS_AS(CudaEngine(0, p).process_frame(a8.cref(), a8.cref(), o8.ref()), std::logic_error);
}

TEST_CASE("GPU == CPU: designed factors 3..8, mixed factors", "[gpu]") {
  require_gpu();
  uint32_t r = 31337;
  for (auto [fh, fv] : {std::pair{3, 3}, {4, 4}, {8, 8}, {5, 3}, {2, 7}, {6, 1}, {1, 6}}) {
    Preset p = parse_preset(load_preset("balanced"), fh, fv);
    TestFrame a(132 * fh, 44 * fv), b(132 * fh, 44 * fv);
    a.fill([&](int, int, int) { return int(((r = r * 1664525u + 1013904223u) >> 4) % 1024u); });
    b.fill([&](int, int, int) { return int(((r = r * 1664525u + 1013904223u) >> 4) % 1024u); });
    INFO(fh << "x" << fv);
    compare(p, a, b);
  }
}
