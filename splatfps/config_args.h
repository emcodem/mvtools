// The filter arguments of splat::Config (splatfps/core/splat_config.h): the table of core/splat_args.h gives the
// AviSynth parameter string, the parsing and the names in the error messages, for SplatFps, CudaSplatFps and
// SplatAnalyse (VapourSynth gets the same names through mvtools_vsbridge). Every argument is optional; an argument not
// given keeps the default of the clip's size (config_for_size).
#pragma once

#include "avisynth.h"
#include "core/splat_args.h"

#include <cstddef>
#include <string>

namespace splat
{

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
