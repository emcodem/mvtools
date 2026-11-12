#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace cudaconvolution {

// Minimal MD5 (RFC 1321) for test vectors: per-frame checksums of the output.
class Md5 {
 public:
  Md5();
  void update(const void* data, std::size_t size);
  std::string hex();  // finalises; call once

 private:
  void block(const uint8_t* p);
  uint32_t s_[4];
  uint64_t bytes_ = 0;
  uint8_t buf_[64];
  std::size_t used_ = 0;
};

}  // namespace cudaconvolution
