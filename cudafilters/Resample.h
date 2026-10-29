// AviSynth+ resampling programs on the host (integer form), for the GPU resamplers of mvtools_cuda.dll.
// Ported from AviSynth+ 3.7.5 (avs_core\filters\resample_functions.cpp: ResamplingFunction::GetResamplingProgram and the
// filter kernels; resample.cpp: resize_prepare_coeffs, GetCenterShiftForResizers), GPL v2 or later, Copyright 2002 Ben
// Rudiak-Gould et al. and the AviSynth+ authors. Only the allocation was changed (std::vector instead of env->Allocate);
// the arithmetic must stay identical, because the GPU result is compared bit for bit with the CPU filter.
#pragma once

#include <vector>

namespace cf
{
  // one plane: for each output position the first source position and taps coefficients (zero padded), fixed point
  // 14 bits for 8-bit data and 13 bits for 10..16-bit data
  struct ResampleProgram
  {
    int source_size = 0, target_size = 0, taps = 0;
    std::vector<int> offset;
    std::vector<short> coeff; // target_size * taps
  };

  struct ResampleKernel
  {
    virtual double f(double x) const = 0;
    virtual double support() const = 0;
    virtual ~ResampleKernel() = default;
  };

  // BicubicResize (b, c)
  struct MitchellNetravaliKernel : ResampleKernel
  {
    MitchellNetravaliKernel(double b, double c);
    double f(double x) const override;
    double support() const override { return 2.0; }
    double p0, p2, p3, q0, q1, q2, q3;
  };

  // BilinearResize (TriangleFilter)
  struct TriangleKernel : ResampleKernel
  {
    double f(double x) const override;
    double support() const override { return 1.0; }
  };

  // GaussResize (p, b, s); AviSynth defaults p = 30, b = 2, s = 4
  struct GaussianKernel : ResampleKernel
  {
    GaussianKernel(double p, double b, double s);
    double f(double x) const override;
    double support() const override { return s; }
    double param, b, s;
  };

  // GetResamplingProgram followed by resize_prepare_coeffs; center_src / center_dst as AviSynth passes them (0.5 for
  // luma and for chroma with the default chroma placement)
  ResampleProgram make_program(const ResampleKernel &k, int source_size, double crop_start, double crop_size, int target_size,
    int bits_per_pixel, double center_src, double center_dst);
}
