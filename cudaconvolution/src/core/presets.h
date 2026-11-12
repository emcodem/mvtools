#pragma once
#include <cstddef>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace cudaconvolution {

struct EmbeddedPreset {
  const char* name;
  const char* json;
  std::size_t size;
};

// Null-terminated table (name == nullptr ends it), generated from presets/*.json at build time.
const EmbeddedPreset* embedded_presets();

std::vector<std::string> preset_names();

// Parsed preset by name; throws std::runtime_error if unknown or malformed.
nlohmann::json load_preset(const std::string& name);

}  // namespace cudaconvolution
