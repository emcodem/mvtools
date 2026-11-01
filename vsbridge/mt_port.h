// mvtools_vsbridge: masktools2 filters the bridge computes itself: mt_lut, mt_lutxy, mt_adddiff (VAPOURSYNTH_PLAN.md,
// step 4, masktools tables).
//
// cudafilters build their GPU lookup tables by running mt_lutxy / mt_lut + mt_adddiff on a test pattern
// (cudafilters/CudaKobs.cpp, lut_from_cpu), and use the same filters as their CPU fallback. VapourSynth has no masktools,
// so the bridge serves these names itself, computing what masktools2 2.2.30 computes under AviSynth+ 3.7.5:
//   - masktools' own evaluator (double precision, round half up) for use_expr 0, for 1 at 8 bit and for 2 without
//     realtime;
//   - AviSynth+ Expr (float32, scalef / scaleb with the chroma rule on U / V, store trunc(v + 0.5f)) where masktools hands
//     the expression to Expr (use_expr 1 above 8 bit, 2 with realtime, 3).
// Only a subset of both expression languages is ported (arithmetic, comparisons, ?, logic, abs / min / max / clip /
// round / floor / ceil / trunc, dup / swap, the bit-depth constants and scalef / scaleb). Transcendental functions,
// pow and % are refused: their results depend on the C library or on AviSynth+'s SSE / AVX2 code path, so they could
// not be bit-identical. Anything outside the port is a clear error at creation.

#ifndef MV_VSBRIDGE_MT_PORT_H
#define MV_VSBRIDGE_MT_PORT_H

#include "bridge_core.h"

#include <string>

// The ported filter for `lname` (lower case: mt_lut, mt_lutxy, mt_adddiff), or a null PClip when the name
// is not ported. Unsupported arguments, formats or expression tokens: env->ThrowError.
PClip mt_port_invoke(const std::string &lname, const AVSValue &args, const char *const *arg_names,
                     IScriptEnvironment *env);

#endif
