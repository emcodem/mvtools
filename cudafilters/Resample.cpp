// See Resample.h: AviSynth+ 3.7.5 resampling programs, ported (GPL v2 or later).

#include "Resample.h"

#include <algorithm>
#include <cmath>

namespace cf
{
  namespace
  {
    template <typename T>
    T clamp(T v, T lo, T hi)
    {
      return v < lo ? lo : v > hi ? hi : v;
    }
  }

  MitchellNetravaliKernel::MitchellNetravaliKernel(double b, double c)
  {
    p0 = (   6. -  2.*b            ) / 6.;
    p2 = ( -18. + 12.*b +  6.*c    ) / 6.;
    p3 = (  12. -  9.*b -  6.*c    ) / 6.;
    q0 = (            8.*b + 24.*c ) / 6.;
    q1 = (         - 12.*b - 48.*c ) / 6.;
    q2 = (            6.*b + 30.*c ) / 6.;
    q3 = (      -     b -  6.*c    ) / 6.;
  }

  double MitchellNetravaliKernel::f(double x) const
  {
    x = fabs(x);
    return (x < 1) ? (p0 + x * x * (p2 + x * p3)) : (x < 2) ? (q0 + x * (q1 + x * (q2 + x * q3))) : 0.0;
  }

  double TriangleKernel::f(double x) const
  {
    x = fabs(x);
    return (x < 1.0) ? 1.0 - x : 0.0;
  }

  GaussianKernel::GaussianKernel(double p, double _b, double _s)
  {
    param = clamp(p, 0.01, 100.0);
    b = clamp(_b, 1.5, 3.5);
    s = _s;
    if (_s == 0)
      s = sqrt(4.6 / ((param * 0.1) * log(b)));
    s = clamp(s, 0.1, 150.0);
  }

  double GaussianKernel::f(double value) const
  {
    double p = param * 0.1;
    return pow(b, -p * value * value);
  }

  ResampleProgram make_program(const ResampleKernel &k, int source_size, double crop_start, double crop_size, int target_size,
    int bits_per_pixel, double center_pos_src, double center_pos_dst)
  {
    const int FPScale = 1 << 14, FPScale16 = 1 << 13;
    // ---- ResamplingFunction::GetResamplingProgram (integer branch)
    double src_step = crop_size / double(target_size);
    double zc_size = std::max(src_step, 1.0) / 1.0;
    double imp_step = 1.0 / zc_size;
    double filter_support = k.support() * zc_size;
    int fir_filter_size = std::max(int(std::ceil(filter_support * 2)), 1);
    int max_kernel_size = 0;
    const int last_line = source_size - 1;

    std::vector<int> pixel_offset(target_size);
    std::vector<short> kernel_sizes(target_size);
    std::vector<short> pixel_coefficient((size_t)target_size * fir_filter_size, 0);

    double pos = crop_start;
    if (filter_support > 0)
      pos += src_step * center_pos_dst - 1 * center_pos_src;
    else
      filter_support = 0.0001;

    const int current_FPScale = (bits_per_pixel > 8 && bits_per_pixel <= 16) ? FPScale16 : FPScale;

    std::vector<double> coef_tmp;
    for (int i = 0; i < target_size; ++i)
    {
      coef_tmp.clear();
      int start_pos = (int)(pos + filter_support) - fir_filter_size + 1;
      pixel_offset[i] = clamp(start_pos, 0, last_line);

      double total = 0.0;
      for (int kk = 0; kk < fir_filter_size; ++kk)
      {
        const int p = start_pos + kk;
        double val = k.f((pos - p) * imp_step);
        coef_tmp.push_back(val);
        total += val;
      }
      if (total == 0.0)
        total = 1.0;

      const int coeff_arr_base_index = i * fir_filter_size;
      double accu = 0.0;
      double prev_value = 0.0;
      int kernel_size = 0;
      for (int kk = 0; kk < fir_filter_size; ++kk)
      {
        const int p = start_pos + kk;
        double val = coef_tmp[kk];
        accu += val;
        if (p >= 0 && p <= last_line)
        {
          double new_value = prev_value + accu / total;
          pixel_coefficient[coeff_arr_base_index + kernel_size] = (short)((int)(new_value * current_FPScale + 0.5) - int(prev_value * current_FPScale + 0.5));
          prev_value = new_value;
          ++kernel_size;
          accu = 0;
        }
      }
      if (accu != 0)
      {
        if (kernel_size > 0)
        {
          double new_value = prev_value + accu / total;
          pixel_coefficient[coeff_arr_base_index + kernel_size - 1] += (short)((int)(new_value * current_FPScale + 0.5) - int(prev_value * current_FPScale + 0.5));
        }
        else
        {
          pixel_coefficient[coeff_arr_base_index + kernel_size] = (short)((int)(accu / total * current_FPScale + 0.5));
          ++kernel_size;
        }
      }
      if (kernel_size == 0)
      {
        pixel_coefficient[coeff_arr_base_index + kernel_size] = (short)((int)(1.0 * current_FPScale + 0.5));
        ++kernel_size;
      }
      kernel_sizes[i] = (short)kernel_size;
      if (kernel_size > max_kernel_size)
        max_kernel_size = kernel_size;
      pos += src_step;
    }
    const int filter_size_real = max_kernel_size;

    // ---- resize_prepare_coeffs: right-align the coefficients of rows whose window would pass the last source line,
    // and give every row filter_size_real taps (zero padded)
    ResampleProgram prog;
    prog.source_size = source_size;
    prog.target_size = target_size;
    prog.taps = filter_size_real;
    prog.offset.resize(target_size);
    prog.coeff.assign((size_t)target_size * filter_size_real, 0);
    for (int i = 0; i < target_size; i++)
    {
      const int kernel_size = kernel_sizes[i];
      const int offset = pixel_offset[i];
      const int last_coeff_index = offset + filter_size_real - 1;
      const int shift_needed = last_coeff_index > last_line ? filter_size_real - kernel_size : 0;
      for (int j = 0; j < kernel_size; j++)
        prog.coeff[(size_t)i * filter_size_real + j + shift_needed] = pixel_coefficient[(size_t)i * fir_filter_size + j];
      prog.offset[i] = offset - shift_needed;
    }
    return prog;
  }
}
