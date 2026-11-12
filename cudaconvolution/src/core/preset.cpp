#include "core/preset.h"

#include <stdexcept>

namespace cudaconvolution {
namespace {

using nlohmann::json;

[[noreturn]] void fail(const std::string& key, const std::string& what) {
  throw std::runtime_error("preset key '" + key + "': " + what);
}

const json& need(const json& j, const char* key, const std::string& path) {
  if (!j.is_object() || !j.contains(key)) fail(path + key, "missing");
  return j.at(key);
}

int get_int(const json& j, const char* key, const std::string& path) {
  const json& v = need(j, key, path);
  if (!v.is_number_integer()) fail(path + key, "must be an integer");
  return v.get<int>();
}

std::pair<int, int> get_range(const json& j, const char* key, const std::string& path) {
  const json& v = need(j, key, path);
  if (!v.is_array() || v.size() != 2 || !v[0].is_number_integer() || !v[1].is_number_integer() || v[0] > v[1])
    fail(path + key, "must be [lo, hi] with integers lo <= hi");
  return {v[0].get<int>(), v[1].get<int>()};
}

// kernel object at `path`: {"first", "taps", "divisor" (default 1), "edges" (H only)}
Kernel get_kernel(const json& k, const std::string& path, bool allow_edges) {
  if (!k.is_object()) fail(path, "must be an object");
  Kernel out;
  out.first = get_int(k, "first", path + ".");
  int divisor = 1;
  if (k.contains("divisor")) {
    divisor = get_int(k, "divisor", path + ".");
    if (divisor <= 0) fail(path + ".divisor", "must be > 0");
  }
  // integers / divisor, or real values with divisor 1 (written with 9 significant digits they round-trip to float32)
  auto values = [&](const json& arr, const std::string& where, std::size_t max) {
    if (!arr.is_array() || arr.empty() || arr.size() > max)
      fail(where, "must be an array of 1.." + std::to_string(max) + " taps");
    std::vector<float> v;
    for (const json& x : arr) {
      if (!x.is_number()) fail(where, "must contain numbers");
      v.push_back(static_cast<float>(x.get<double>() / divisor));
    }
    return v;
  };
  out.taps = values(need(k, "taps", path + "."), path + ".taps", kMaxTaps);
  if (k.contains("edges")) {
    if (!allow_edges) fail(path + ".edges", "edge kernels are only supported for the H kernels");
    const json& e = k["edges"];
    for (const char* side : {"left", "right"}) {
      if (!e.contains(side)) continue;
      const std::string sp = path + ".edges." + side;
      if (!e[side].is_array() || e[side].size() > kMaxEdgeCols)
        fail(sp, "must be an array of at most " + std::to_string(kMaxEdgeCols) + " edge kernels");
      const bool left = std::string(side) == "left";
      auto& list = left ? out.left : out.right;
      for (std::size_t i = 0; i < e[side].size(); ++i) {
        const std::string ip = sp + "[" + std::to_string(i) + "]";
        EdgeKernel ek;
        ek.src_first = get_int(e[side][i], "src_first", ip + ".");
        ek.taps = values(need(e[side][i], "taps", ip + "."), ip + ".taps", kMaxEdgeTaps);
        if (left ? ek.src_first < 0 : ek.src_first + static_cast<int>(ek.taps.size()) > 0)
          fail(ip, left ? "src_first must be >= 0" : "taps must end at or before the right edge (src_first + count <= 0)");
        list.push_back(std::move(ek));
      }
    }
  }
  return out;
}

const json& factor_entry(const json& j, const char* axis, int factor) {
  const json& table = need(j, axis, "");
  if (!table.is_object()) fail(axis, "must be an object keyed by factor");
  const std::string key = std::to_string(factor);
  if (!table.contains(key)) {
    std::string have;
    for (int f : preset_factors(j, axis)) have += (have.empty() ? "" : ", ") + std::to_string(f);
    throw std::runtime_error("preset '" + j.value("name", std::string("(preset file)")) + "' has no " +
                             (std::string(axis) == "h" ? "horizontal" : "vertical") + " filter for factor " + key +
                             " (available: " + have + ")");
  }
  return table[key];
}

void apply_profile(Preset& p, const json& prof, const std::string& path) {
  const std::string arith = need(prof, "arith", path + ".").get<std::string>();
  if (arith == "f32_fma_trunc") p.arith = Arith::F32FmaTrunc;
  else if (arith == "round_half_up") p.arith = Arith::RoundHalfUp;
  else fail(path + ".arith", "must be 'f32_fma_trunc' or 'round_half_up'");
  p.offset = get_int(prof, "offset_code", path + ".");
  p.has_intermediate_clip = prof.contains("intermediate_clip") && !prof["intermediate_clip"].is_null();
  if (p.has_intermediate_clip)
    std::tie(p.intermediate_lo, p.intermediate_hi) = get_range(prof, "intermediate_clip", path + ".");
  if (prof.contains("output_clip")) fail(path + ".output_clip", "not supported: output clipping is set by the clip_min / "
                                                                "clip_max parameters of CudaConvolution");
  const std::string border = prof.value("border", std::string("replicate"));
  if (border == "replicate") p.border = Border::Replicate;
  else if (border == "mirror") p.border = Border::Mirror;
  else fail(path + ".border", "must be 'replicate' or 'mirror'");
}

}  // namespace

std::vector<int> preset_factors(const json& j, const char* axis) {
  std::vector<int> out;
  if (j.contains(axis) && j[axis].is_object())
    for (int f = 1; f <= kMaxFactor; ++f)
      if (j[axis].contains(std::to_string(f))) out.push_back(f);
  return out;
}

Preset parse_preset(const json& j, int factor_h, int factor_v) {
  if (factor_h < 1 || factor_v < 1 || factor_h > kMaxFactor || factor_v > kMaxFactor)
    throw std::runtime_error("scale factors must be 1.." + std::to_string(kMaxFactor) + " per axis, got " +
                             std::to_string(factor_h) + " x " + std::to_string(factor_v));
  Preset p;
  p.name = j.value("name", std::string("(preset file)")) + "/h" + std::to_string(factor_h) + "v" + std::to_string(factor_v);
  p.factor_h = factor_h;
  p.factor_v = factor_v;

  const json target = j.value("target", json::object());
  const std::string scan = target.value("scan", std::string("interlaced"));
  if (scan != "interlaced" && scan != "progressive") fail("target.scan", "must be 'interlaced' or 'progressive'");
  p.interlaced = scan == "interlaced";
  const std::string order = target.value("field_order", std::string("tff"));
  if (order != "tff" && order != "bff") fail("target.field_order", "must be 'tff' or 'bff'");
  p.tff = order == "tff";

  const std::string hk = "h." + std::to_string(factor_h), vk = "v." + std::to_string(factor_v);
  const json& h = factor_entry(j, "h", factor_h);
  const json& v = factor_entry(j, "v", factor_v);
  p.luma_h = get_kernel(need(h, "luma", hk + "."), hk + ".luma", true);
  p.chroma_h = get_kernel(need(h, "chroma", hk + "."), hk + ".chroma", true);
  p.v = get_kernel(need(v, "kernel", vk + "."), vk + ".kernel", false);

  // the measured arithmetic belongs to the measured H+V combination only
  const std::string hp = need(h, "profile", hk + ".").get<std::string>();
  const std::string vp = need(v, "profile", vk + ".").get<std::string>();
  const std::string profile = hp == vp ? hp : "designed";
  const json& profiles = need(j, "profiles", "");
  if (!profiles.contains(profile)) fail("profiles." + profile, "missing");
  apply_profile(p, profiles[profile], "profiles." + profile);
  return p;
}

}  // namespace cudaconvolution
