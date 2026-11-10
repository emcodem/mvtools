// The filter arguments of splat::Config (splatfps/core/splat_config.h): one table gives the AviSynth parameter string,
// the parsing and the names in the error messages, for SplatFps, CudaSplatFps and SplatAnalyse (VapourSynth gets the
// same names through mvtools_vsbridge). Every argument is optional; an argument not given keeps the default of the
// clip's size (config_for_size).
#pragma once

#include "avisynth.h"
#include "core/splat_config.h"

#include <cstddef>
#include <string>

namespace splat
{
  // type: i int, f float, d double, b bool (stored as 0 / 1), I int array, D double array
  struct CfgArg { const char *name; char type; size_t off; int count; };

#define SPLAT_ARG(n, t, m, c) { n, t, offsetof(Config, m), c }
  static const CfgArg CFG_ARGS[] = {
    SPLAT_ARG("me_scale", 'i', scale, 1),
    SPLAT_ARG("pad_x", 'i', pad_x, 1),
    SPLAT_ARG("pad_y", 'i', pad_y, 1),
    SPLAT_ARG("pyramid_levels", 'i', levels, 1),
    SPLAT_ARG("field_stride_max", 'i', stride_max, 1),
    SPLAT_ARG("me_filter_generated", 'b', fir_generated, 1),
    SPLAT_ARG("me_filter_a_cutoff", 'f', fir_a_cutoff, 1),
    SPLAT_ARG("me_filter_b_cutoff", 'f', fir_b_cutoff, 1),
    SPLAT_ARG("me_filter_a_taps", 'i', fir_a_taps, 1),
    SPLAT_ARG("me_filter_b_taps", 'i', fir_b_taps, 1),
    SPLAT_ARG("me_filter_beta", 'f', fir_beta, 1),
    SPLAT_ARG("block_size", 'i', block_size, 1),
    SPLAT_ARG("block_peaks", 'I', peaks, CFG_NSTAGE),
    SPLAT_ARG("nearest_block", 'I', nearest, CFG_NSTAGE),
    SPLAT_ARG("block_taper", 'f', block_taper, 1),
    SPLAT_ARG("highpass_sigma", 'f', highpass_sigma, 1),
    SPLAT_ARG("noise_weight", 'f', noise_weight, 1),
    SPLAT_ARG("peak_ratio", 'f', peak_ratio, 1),
    SPLAT_ARG("global_candidate", 'i', global_candidate, 1),
    SPLAT_ARG("secondary_peak_cost", 'f', secondary_peak_cost, 1),
    SPLAT_ARG("refine_err_scale", 'f', refine_err_scale, 1),
    SPLAT_ARG("refine_reg", 'f', refine_reg, 1),
    SPLAT_ARG("refine_box", 'i', refine_box, 1),
    SPLAT_ARG("error_box", 'i', error_box, 1),
    SPLAT_ARG("error_edge_weight", 'f', error_edge_weight, 1),
    SPLAT_ARG("error_edge_dist", 'i', error_edge_dist, 1),
    SPLAT_ARG("error_edge_mode", 'i', error_edge_mode, 1),
    SPLAT_ARG("speed_change_cost", 'f', speed_change_cost, 1),
    SPLAT_ARG("vector_length_cost", 'f', vector_length_cost, 1),
    SPLAT_ARG("occlusion_error", 'f', occlusion_error, 1),
    SPLAT_ARG("occlusion_error_factor", 'f', occlusion_error_factor, 1),
    SPLAT_ARG("global_cost", 'f', global_cost, 1),
    SPLAT_ARG("zero_cost", 'f', zero_cost, 1),
    SPLAT_ARG("fallback_box", 'i', fallback_box, 1),
    SPLAT_ARG("fallback_dead_zone", 'f', fallback_dead_zone, 1),
    SPLAT_ARG("fallback_range", 'f', fallback_range, 1),
    SPLAT_ARG("fallback_strength", 'f', fallback_strength, 1),
    SPLAT_ARG("fallback_cap", 'f', fallback_cap, 1),
    SPLAT_ARG("fallback_err_lo", 'f', fallback_err_lo, 1),
    SPLAT_ARG("fallback_err_range", 'f', fallback_err_range, 1),
    SPLAT_ARG("fallback_err_gain", 'f', fallback_err_gain, 1),
    SPLAT_ARG("copy_snap", 'i', copy_snap, 1),
    SPLAT_ARG("pull_strength", 'd', pull_strength, 1),
    SPLAT_ARG("pull_error_lo", 'd', pull_error_lo, 1),
    SPLAT_ARG("pull_error_hi", 'd', pull_error_hi, 1),
    SPLAT_ARG("error_window_div", 'i', error_window_div, 1),
    SPLAT_ARG("error_window_cap", 'f', error_window_cap, 1),
    SPLAT_ARG("scene_cut_error", 'f', scene_cut_error, 1),
    SPLAT_ARG("splat_err_scale", 'f', splat_err_scale, 1),
    SPLAT_ARG("splat_min_weight", 'f', splat_min_weight, 1),
    SPLAT_ARG("splat_gain", 'f', splat_gain, 1),
    SPLAT_ARG("still_vector", 'f', still_vector, 1),
    SPLAT_ARG("hole_weight", 'f', hole_weight, 1),
    SPLAT_ARG("pyramid_sigma", 'f', pyr_sigma, 1),
  };
#undef SPLAT_ARG
  constexpr int CFG_NARGS = (int)(sizeof(CFG_ARGS) / sizeof(CFG_ARGS[0]));

  // the AviSynth parameter string of the table (appended to a filter's own parameters)
  inline std::string config_arg_string()
  {
    std::string s;
    for (const CfgArg &a : CFG_ARGS)
    {
      s += '[';
      s += a.name;
      s += ']';
      switch (a.type)
      {
      case 'i': s += 'i'; break;
      case 'b': s += 'b'; break;
      case 'f':
      case 'd': s += 'f'; break;
      case 'I': s += "i*"; break;
      default: s += "f*"; break; // D
      }
    }
    return s;
  }

  // Config of a w x h clip from args[first .. first + CFG_NARGS) (the table order): defaults, the given arguments,
  // the per-size rules for what is left, the checks (ThrowError with fname on a bad value)
  inline Config config_from_args(const AVSValue &args, int first, int w, int h, IScriptEnvironment *env, const char *fname)
  {
    Config c = config_defaults();
    for (int k = 0; k < CFG_NARGS; k++)
    {
      const CfgArg &a = CFG_ARGS[k];
      const AVSValue &v = args[first + k];
      if (!v.Defined())
        continue;
      unsigned char *p = reinterpret_cast<unsigned char *>(&c) + a.off;
      switch (a.type)
      {
      case 'i': *reinterpret_cast<int *>(p) = v.AsInt(); break;
      case 'b': *reinterpret_cast<int *>(p) = v.AsBool() ? 1 : 0; break;
      case 'f': *reinterpret_cast<float *>(p) = (float)v.AsFloat(); break;
      case 'd': *reinterpret_cast<double *>(p) = v.AsFloat(); break;
      default:
      {
        const int n = v.IsArray() ? v.ArraySize() : 1;
        if (n == 0)
          break;
        if (n != a.count)
          env->ThrowError("%s: %s takes %d values", fname, a.name, a.count);
        for (int i = 0; i < n; i++)
        {
          const AVSValue &e = v.IsArray() ? v[i] : v;
          if (a.type == 'I')
            reinterpret_cast<int *>(p)[i] = e.AsInt();
          else
            reinterpret_cast<double *>(p)[i] = e.AsFloat();
        }
      }
      }
    }
    config_resolve(c, w, h);
    char why[512];
    if (!config_check(c, w, h, why, sizeof(why)))
      env->ThrowError("%s: %s", fname, why);
    return c;
  }
}
