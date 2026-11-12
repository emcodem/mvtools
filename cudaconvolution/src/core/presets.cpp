#include "core/presets.h"

#include <stdexcept>

namespace cudaconvolution {

std::vector<std::string> preset_names() {
  std::vector<std::string> names;
  for (const EmbeddedPreset* p = embedded_presets(); p->name; ++p) names.emplace_back(p->name);
  return names;
}

nlohmann::json load_preset(const std::string& name) {
  for (const EmbeddedPreset* p = embedded_presets(); p->name; ++p) {
    if (name != p->name) continue;
    try {
      return nlohmann::json::parse(p->json, p->json + p->size);
    } catch (const nlohmann::json::exception& e) {
      throw std::runtime_error("embedded preset '" + name + "' is malformed: " + e.what());
    }
  }
  std::string known;
  for (const auto& n : preset_names()) known += (known.empty() ? "" : ", ") + n;
  throw std::runtime_error("unknown preset '" + name + "' (known: " + known + ")");
}

}  // namespace cudaconvolution
