#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/preset.h"

namespace cudaconvolution {

struct ConstPlane {
  const uint16_t* data;
  std::ptrdiff_t stride;  // in samples
  int width, height;
};

struct Plane {
  uint16_t* data;
  std::ptrdiff_t stride;  // in samples
  int width, height;
};

enum class CpuPath {
  Auto,    // AVX2/FMA3 if the CPU has it, else scalar
  Scalar,  // reference: std::fma per tap
  Avx2,    // 8 samples per step, bit-identical to Scalar
};

bool cpu_has_avx2_fma();

// CPU engine. Writes dst rows row_first, row_first + row_step, ... (< dst.height) from one source frame:
// output (y, x) is centred on source (y * factor_v, x * factor_h). H pass, intermediate clip, V pass, output rule,
// each tap accumulated with a float32 FMA in ascending tap order (no other contraction or reassociation): the
// arithmetic every engine (scalar, AVX2, CUDA) reproduces bit for bit.
// scratch is reused between calls to avoid allocations. threads > 1 splits the rows over worker threads (results are
// bit-identical for any thread count).
void resample_plane(const ConstPlane& src, const Plane& dst, int row_first, int row_step, const Kernel& h,
                    const Kernel& v, const Preset& p, std::vector<float>& scratch, CpuPath path = CpuPath::Auto,
                    int threads = 1);


// Planar YUV frame (plane 0 luma, 1/2 chroma).
struct ConstFrame {
  ConstPlane plane[3];
};
struct Frame {
  Plane plane[3];
};

// One output frame. Interlaced: the temporally first field (top for TFF, bottom for BFF) is computed from `first`,
// the other one from `second`. Progressive: all rows from `first` (`second` unused).
void process_frame(const ConstFrame& first, const ConstFrame& second, const Frame& out, const Preset& p,
                   std::vector<float>& scratch, CpuPath path = CpuPath::Auto, int threads = 1);

}  // namespace cudaconvolution
