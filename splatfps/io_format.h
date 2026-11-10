// The clip formats of the splatfps filters (planar YUV 8 .. 16 bit, 4:4:4 / 4:2:2 / 4:2:0) and the chroma-siting
// check of their frames. Shared by SplatFps.cpp and SplatAnalyse.cpp.
#pragma once

#include "avisynth.h"
#include "core/splat_core.h"

#include <cstdint>

namespace splat
{
  // the format of a clip, or an error naming what is accepted
  inline IoFormat io_format(const VideoInfo &vi, IScriptEnvironment *env, const char *fname)
  {
    const int bits = vi.BitsPerComponent();
    const bool ok_type = vi.IsPlanar() && vi.IsYUV() && !vi.IsYUVA() && !vi.IsY() && bits >= 8 && bits <= 16 &&
      (vi.Is444() || vi.Is422() || vi.Is420());
    if (!ok_type)
      env->ThrowError("%s: the clip must be planar YUV 4:4:4, 4:2:2 or 4:2:0 with 8 to 16 bit integer samples (no "
        "alpha, no float)", fname);
    IoFormat f;
    f.bits = bits;
    f.sub_w = vi.Is444() ? 0 : 1;
    f.sub_h = vi.Is420() ? 1 : 0;
    return f;
  }

  // _ChromaLocation of a source frame -> IoFormat::centre (0 left, 1 centre). Absent or left (0, MPEG-2) and centre
  // (1, JPEG / MPEG-1) are converted; 4:2:2 also accepts top left / bottom left (2, 4: left horizontally) and top /
  // bottom (3, 5: centre horizontally). 4:2:0 with chroma on the rows (2 .. 5) is an error, never converted.
  inline int siting_of(const PVideoFrame &f, const IoFormat &fmt, int n, IScriptEnvironment *env, const char *fname)
  {
    if (!fmt.sub_w)
      return 0;
    int err = 0;
    const int64_t loc = env->propGetInt(env->getFramePropsRO(f), "_ChromaLocation", 0, &err);
    if (err || loc == 0 || (!fmt.sub_h && (loc == 2 || loc == 4)))
      return 0;
    if (loc == 1 || (!fmt.sub_h && (loc == 3 || loc == 5)))
      return 1;
    static const char *const names[6] = { "left", "center", "top_left", "top", "bottom_left", "bottom" };
    env->ThrowError("%s: frame %d has _ChromaLocation %d (%s); %s supports left (0) and centre (1) chroma siting "
      "for 4:2:0. Convert the chroma first.", fname, n, (int)loc, loc >= 0 && loc < 6 ? names[loc] : "unknown", fname);
    return 0;
  }

  // the siting of a frame against the clip's (frame 0): a change within the clip is an error
  inline void check_siting(const PVideoFrame &f, const IoFormat &fmt, int n, IScriptEnvironment *env, const char *fname)
  {
    if (siting_of(f, fmt, n, env, fname) != fmt.centre)
      env->ThrowError("%s: frame %d has another chroma siting (_ChromaLocation) than frame 0", fname, n);
  }
}
