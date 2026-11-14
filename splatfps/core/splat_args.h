// The named settings of splat::Config: one table gives the names, the types and the places in the struct. The filters
// parse their arguments with it (config_args.h); the stage tests set the same names (config_set_arg).
#pragma once

#include "splat_config.h"

#include <cstddef>
#include <cstring>

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
    SPLAT_ARG("blend_sigma", 'd', blend_sigma, 1),
    SPLAT_ARG("blend_levels", 'i', blend_levels, 1),
    SPLAT_ARG("blend_err_lo", 'd', blend_err_lo, 1),
    SPLAT_ARG("blend_err_hi", 'd', blend_err_hi, 1),
    SPLAT_ARG("error_window_div", 'i', error_window_div, 1),
    SPLAT_ARG("error_window_cap", 'f', error_window_cap, 1),
    SPLAT_ARG("scene_cut_error", 'f', scene_cut_error, 1),
    SPLAT_ARG("splat_err_scale", 'f', splat_err_scale, 1),
    SPLAT_ARG("splat_min_weight", 'f', splat_min_weight, 1),
    SPLAT_ARG("splat_gain", 'f', splat_gain, 1),
    SPLAT_ARG("still_vector", 'f', still_vector, 1),
    SPLAT_ARG("hole_weight", 'f', hole_weight, 1),
    SPLAT_ARG("pyramid_sigma", 'f', pyr_sigma, 1),
    SPLAT_ARG("fallback_edge", 'i', fallback_edge, 1),
    SPLAT_ARG("fallback_edge_margin", 'f', fallback_edge_margin, 1),
  };
#undef SPLAT_ARG
  constexpr int CFG_NARGS = (int)(sizeof(CFG_ARGS) / sizeof(CFG_ARGS[0]));

  // Sets the named setting of c to vals[0 .. n - 1] (one value for a scalar, the table's count for an array; ints
  // and bools truncate the value). Returns 0, -1 for an unknown name, -2 for a wrong number of values. The caller
  // resolves and checks c afterwards (config_resolve, config_check).
  inline int config_set_arg(Config &c, const char *name, const double *vals, int n)
  {
    for (const CfgArg &a : CFG_ARGS)
    {
      if (std::strcmp(a.name, name) != 0)
        continue;
      if (n != a.count)
        return -2;
      unsigned char *p = reinterpret_cast<unsigned char *>(&c) + a.off;
      for (int i = 0; i < n; i++)
        switch (a.type)
        {
        case 'i':
        case 'I': reinterpret_cast<int *>(p)[i] = (int)vals[i]; break;
        case 'b': *reinterpret_cast<int *>(p) = vals[i] != 0.0 ? 1 : 0; break;
        case 'f': *reinterpret_cast<float *>(p) = (float)vals[i]; break;
        default: reinterpret_cast<double *>(p)[i] = vals[i]; break;
        }
      return 0;
    }
    return -1;
  }

  // The values of the named setting of c into vals (room for the table's count). Returns the count, -1 for an unknown
  // name.
  inline int config_get_arg(const Config &c, const char *name, double *vals)
  {
    for (const CfgArg &a : CFG_ARGS)
    {
      if (std::strcmp(a.name, name) != 0)
        continue;
      const unsigned char *p = reinterpret_cast<const unsigned char *>(&c) + a.off;
      for (int i = 0; i < a.count; i++)
        switch (a.type)
        {
        case 'i':
        case 'I':
        case 'b': vals[i] = reinterpret_cast<const int *>(p)[i]; break;
        case 'f': vals[i] = *reinterpret_cast<const float *>(p); break;
        default: vals[i] = reinterpret_cast<const double *>(p)[i]; break;
        }
      return a.count;
    }
    return -1;
  }
}
