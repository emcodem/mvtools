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

  namespace
  {
    const double PI = 3.14159265358979323846; // M_PI as resample_functions.h defines it

    double sinc(double value) // the sinc of LanczosFilter / SincFilter / SincLin2Filter (value >= 0)
    {
      if (value > 0.000001)
      {
        value *= PI;
        return sin(value) / value;
      }
      return 1.0;
    }
  }

  double PointKernel::f(double) const
  {
    return 1.0;
  }

  LanczosKernel::LanczosKernel(int _taps)
  {
    taps = (double)clamp(_taps, 1, 100);
  }

  double LanczosKernel::f(double value) const
  {
    value = fabs(value);
    return value < taps ? sinc(value) * sinc(value / taps) : 0.0;
  }

  BlackmanKernel::BlackmanKernel(int _taps)
  {
    taps = (double)clamp(_taps, 1, 100);
    rtaps = 1.0 / taps;
  }

  double BlackmanKernel::f(double value) const
  {
    value = fabs(value);
    if (value < taps)
    {
      if (value > 0.000001)
      {
        value *= PI;
        return (sin(value) / value) * (0.42 + 0.5 * cos(value * rtaps) + 0.08 * cos(2 * value * rtaps));
      }
      return 1.0;
    }
    return 0.0;
  }

  double Spline16Kernel::f(double value) const
  {
    value = fabs(value);
    if (value < 1.0)
      return ((value - 9.0 / 5.0) * value - 1.0 / 5.0) * value + 1.0;
    else if (value < 2.0)
      return ((-1.0 / 3.0 * (value - 1.0) + 4.0 / 5.0) * (value - 1.0) - 7.0 / 15.0) * (value - 1.0);
    return 0.0;
  }

  double Spline36Kernel::f(double value) const
  {
    value = fabs(value);
    if (value < 1.0)
      return ((13.0 / 11.0 * (value) - 453.0 / 209.0) * (value) - 3.0 / 209.0) * (value) + 1.0;
    else if (value < 2.0)
      return ((-6.0 / 11.0 * (value - 1.0) + 270.0 / 209.0) * (value - 1.0) - 156.0 / 209.0) * (value - 1.0);
    else if (value < 3.0)
      return ((1.0 / 11.0 * (value - 2.0) - 45.0 / 209.0) * (value - 2.0) + 26.0 / 209.0) * (value - 2.0);
    return 0.0;
  }

  double Spline64Kernel::f(double value) const
  {
    value = fabs(value);
    if (value < 1.0)
      return ((49.0 / 41.0 * (value) - 6387.0 / 2911.0) * (value) - 3.0 / 2911.0) * (value) + 1.0;
    else if (value < 2.0)
      return ((-24.0 / 41.0 * (value - 1.0) + 4032.0 / 2911.0) * (value - 1.0) - 2328.0 / 2911.0) * (value - 1.0);
    else if (value < 3.0)
      return ((6.0 / 41.0 * (value - 2.0) - 1008.0 / 2911.0) * (value - 2.0) + 582.0 / 2911.0) * (value - 2.0);
    else if (value < 4.0)
      return ((-1.0 / 41.0 * (value - 3.0) + 168.0 / 2911.0) * (value - 3.0) - 97.0 / 2911.0) * (value - 3.0);
    return 0.0;
  }

  SincKernel::SincKernel(int _taps)
  {
    taps = (double)clamp(_taps, 1, 150);
  }

  double SincKernel::f(double value) const
  {
    return sinc(fabs(value));
  }

  SinPowerKernel::SinPowerKernel(double p)
  {
    param = clamp(p, 1.0, 10.0);
  }

  double SinPowerKernel::f(double value) const
  {
    value = fabs(value);
    value *= PI / param;
    if (value < (PI / 2))
      return pow(cos(value), 1.8);
    if (value < PI)
      return -(cos(value) * cos(value)) / (0.9 * value);
    return 0;
  }

  SincLin2Kernel::SincLin2Kernel(int _taps)
  {
    taps = (double)clamp(_taps, 1, 30);
  }

  double SincLin2Kernel::f(double value) const
  {
    value = fabs(value);
    if (value < (taps / 2.0))
      return sinc(value);
    return sinc(value) * ((2.0 - (2.0 * value / taps)));
  }

  UserDefined2Kernel::UserDefined2Kernel(double _b, double _c, double _s)
  {
    a = 1.0;
    b = clamp(_b, -50.0, 250.0);
    c = clamp(_c, -50.0, 250.0);
    b = (b - 16.0) / 219.0;
    c = (c - 16.0) / 219.0;
    s = clamp(_s, 1.5, 15.0);
  }

  double UserDefined2Kernel::f(double x) const
  {
    // UserDefined2Filter::sinc takes fabs itself: its arguments x - 1 / x - 2 can be negative
    auto sincs = [](double v) {
      if (fabs(v) > 0.000001)
      {
        v *= PI;
        return sin(v) / v;
      }
      return 1.0;
    };
    x = fabs(x);
    return c * sincs(x + 2) + b * sincs(x + 1) + a * sincs(x) + b * sincs(x - 1) + c * sincs(x - 2);
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
