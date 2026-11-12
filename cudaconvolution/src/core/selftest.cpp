#include "core/selftest.h"

#include <stdexcept>

#include "core/md5.h"
#include "core/presets.h"

namespace cudaconvolution {

const EmbeddedPreset* embedded_selftest();  // generated from selftest/*.json

namespace {


std::string md5_of(const nlohmann::json& vectors, const std::string& key) {
  return vectors.contains(key) ? vectors[key].get<std::string>() : std::string();
}

}  // namespace

void selftest_fill(uint16_t* data, int width, int height, int frame, int plane, int bits) {
  uint32_t x = 0x9E3779B9u ^ (static_cast<uint32_t>(frame * 3 + plane + 1) * 0x85EBCA6Bu);
  const uint32_t mask = (1u << bits) - 1;
  for (int i = 0; i < width * height; ++i) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    data[i] = static_cast<uint16_t>((x >> 8) & mask);
  }
}

std::string SelftestCase::label() const {
  return preset + "/h" + std::to_string(factor_h) + "v" + std::to_string(factor_v) + " " + format_name(format);
}

std::vector<SelftestCase> selftest_cases() {
  nlohmann::json vectors = nlohmann::json::object();
  for (const EmbeddedPreset* e = embedded_selftest(); e->name; ++e)
    vectors.update(nlohmann::json::parse(e->json, e->json + e->size));
  std::vector<SelftestCase> cases;
  auto add = [&](const std::string& name, int fh, int fv, SampleFormat f) {
    SelftestCase c;
    c.preset = name;
    c.factor_h = fh;
    c.factor_v = fv;
    c.format = f;
    c.width = 128 * fh;  // 4:2:2 chroma width 64 * fh: divisible by the factor; output 128 x 48
    c.height = 48 * fv;
    c.md5 = md5_of(vectors, c.label());
    cases.push_back(c);
  };
  for (const std::string& name : preset_names()) {
    for (int f : {1, 2})
      for (Layout l : {Layout::YUV422, Layout::YUV444, Layout::RGB})
        for (int bits : {8, 10, 12, 16}) add(name, f, f, {l, bits});
    for (auto [fh, fv] : {std::pair{3, 3}, {4, 4}, {8, 8}, {2, 1}, {1, 2}, {3, 2}}) add(name, fh, fv, {Layout::YUV422, 10});
  }
  return cases;
}

std::string selftest_md5(const SelftestCase& c, const EngineFn& run) {
  const bool eight = c.format.bits == 8;  // as the plugin: 10-bit arithmetic, from_8bit in, to_8bit out
  const Preset p = adapt_preset(parse_preset(load_preset(c.preset), c.factor_h, c.factor_v),
                                {c.format.layout, eight ? 10 : c.format.bits});
  const int cw = c.format.layout == Layout::YUV422 ? c.width / 2 : c.width;
  std::vector<uint16_t> src[2][3], dst[3];
  ConstFrame in[2];
  Frame out;
  for (int f = 0; f < 2; ++f)
    for (int i = 0; i < 3; ++i) {
      const int w = i ? cw : c.width;
      src[f][i].resize(static_cast<std::size_t>(w) * c.height);
      selftest_fill(src[f][i].data(), w, c.height, f, i, c.format.bits);
      if (eight)
        for (uint16_t& v : src[f][i]) v = from_8bit(static_cast<uint8_t>(v));
      in[f].plane[i] = {src[f][i].data(), w, w, c.height};
    }
  for (int i = 0; i < 3; ++i) {
    const int w = (i ? cw : c.width) / p.factor_h, h = c.height / p.factor_v;
    dst[i].assign(static_cast<std::size_t>(w) * h, 0);
    out.plane[i] = {dst[i].data(), w, w, h};
  }
  run(in[0], in[1], out, p);
  if (eight)
    for (auto& d : dst)
      for (uint16_t& v : d) v = to_8bit(v);
  Md5 m;
  for (const auto& d : dst) m.update(d.data(), d.size() * sizeof(uint16_t));
  return m.hex();
}

}  // namespace cudaconvolution
