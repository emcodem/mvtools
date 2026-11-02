// VapourSynth entry point of mvtools2 (namespace mvpf). The filters run unchanged through mvtools_vsbridge, a small
// AviSynth+ host for VapourSynth that lives in its own library next to this one:
// this file only opens the bridge and hands it AvisynthPluginInit3. Under AviSynth+ this function is never called.
//
// The bridge is loaded lazily and checked like every optional library (dyn_lib.h): when it is missing or broken the
// namespace stays empty and the reason goes to stderr; VapourSynth itself keeps working.

#include "avisynth.h"
#include "cuda/dyn_lib.h"
#include <VapourSynth4.h>

#include <cstdio>

extern "C" __declspec(dllexport) const char *__stdcall AvisynthPluginInit3(IScriptEnvironment *env,
                                                                           const AVS_Linkage *const vectors);

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi)
{
  vspapi->configPlugin("com.pinterf.mvtools2", "mvpf", "MVTools2 (pinterf) for VapourSynth via mvtools_vsbridge",
                       VS_MAKE_VERSION(2, 754), VAPOURSYNTH_API_VERSION, 0, plugin);

  const std::filesystem::path lib_path =
      dynlib::module_dir(reinterpret_cast<const void *>(&VapourSynthPluginInit2)) / dynlib::vsbridge_lib_name;
  void *lib = dynlib::open(lib_path);
  if (!lib)
  {
    fprintf(stderr, "mvtools2: VapourSynth namespace mvpf is empty: cannot load %s (%s)\n",
            lib_path.string().c_str(), dynlib::last_error().c_str());
    return;
  }
  typedef int (*RegisterFn)(int abi, VSPlugin *plugin, const VSPLUGINAPI *vspapi, void *init3, char *err, int errlen);
  RegisterFn reg = reinterpret_cast<RegisterFn>(dynlib::symbol(lib, "mvbridge_register_functions"));
  if (!reg)
  {
    fprintf(stderr, "mvtools2: VapourSynth namespace mvpf is empty: %s has no mvbridge_register_functions\n",
            lib_path.string().c_str());
    return;
  }
  char err[1024] = {};
  if (reg(1, plugin, vspapi, reinterpret_cast<void *>(&AvisynthPluginInit3), err, (int)sizeof(err)))
    fprintf(stderr, "mvtools2: VapourSynth namespace mvpf: %s\n", err);
}
