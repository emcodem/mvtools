// Candidate clips: the link between a vector source (SplatAnalyse, MSplatVectors in mvtools2, ...) and SplatFps.
//
// One frame per source picture c holds the candidate vector fields SplatFps refines and assigns for c:
// a forward set (vectors from c into c+1) and, depending on SplatBwd, a backward set. A set is n fields of the ME grid
// ((width + 2 pad_x) / scale x (height + 2 pad_y) / scale for a width x height source, splat::Config; at 1080p
// (1920 + 32) / 4 x (1080 + 40) / 4), each a vx and a vy plane in ME pixels (1 ME pixel = scale picture pixels),
// stacked top to bottom in one Y32 (float) frame:
//   rows [0, 2n mh): forward set vx0, vy0, vx1, vy1, ...;  rows [2n mh, 4n mh): backward set (SplatBwd 1 or 2).
// Unused rows are zero. Frame properties (all integers unless noted):
//   SplatCand    layout version (1)
//   SplatN       candidates per set (the zero vector is added by SplatFps, not stored)
//   SplatGridW/H ME grid size
//   SplatScale   ME scale; SplatPadX / SplatPadY the border of the internal grid (optional: when present, SplatFps
//                checks them against its own configuration)
//   SplatFwd     1: the forward set is present (0 on the last picture)
//   SplatBwd     0: none (first picture); 1: in this frame, vectors from c into c-1; 2: in this frame, vectors from
//                c-1 into c; 3: the forward set of frame c-1 (vectors from c-1 into c), as phase correlation gives
//                them: one measurement per pair, stored once
//   SplatRefine  1: refine the candidates against the pictures; 0: take them as they are (assignment only)
//   SplatBias    float array, n + 1 values: per-candidate cost added in the assignment (zero vector last); it
//                belongs to the source (how much each candidate rank is trusted)
#pragma once

#include "avisynth.h"
#include "core/splat_core.h"

#include <cstring>
#include <vector>

namespace splat
{
  constexpr int CAND_LAYOUT = 1;

  struct CandInfo
  {
    int n = 0, gw = 0, gh = 0;
    int scale = -1, pad_x = -1, pad_y = -1; // -1: not stored
    bool fwd = false;
    int bwd = 0; // SplatBwd
    bool refine = true;
    std::vector<float> bias; // n + 1
  };

  // the Y32 clip of candidate frames for a source clip of width x height (rows for 2 sets when two_sets)
  inline VideoInfo cand_clip_info(const VideoInfo &src, int n, bool two_sets, const Config &cfg)
  {
    const Geometry g(src.width, src.height, cfg);
    VideoInfo vi = src;
    vi.pixel_type = VideoInfo::CS_Y32;
    vi.width = g.MW;
    vi.height = (two_sets ? 4 : 2) * n * g.MH;
    return vi;
  }

  inline void cand_write_props(IScriptEnvironment *env, PVideoFrame &f, const CandInfo &ci)
  {
    AVSMap *m = env->getFramePropsRW(f);
    env->propSetInt(m, "SplatCand", CAND_LAYOUT, 0);
    env->propSetInt(m, "SplatN", ci.n, 0);
    env->propSetInt(m, "SplatGridW", ci.gw, 0);
    env->propSetInt(m, "SplatGridH", ci.gh, 0);
    if (ci.scale > 0)
    {
      env->propSetInt(m, "SplatScale", ci.scale, 0);
      env->propSetInt(m, "SplatPadX", ci.pad_x, 0);
      env->propSetInt(m, "SplatPadY", ci.pad_y, 0);
    }
    env->propSetInt(m, "SplatFwd", ci.fwd ? 1 : 0, 0);
    env->propSetInt(m, "SplatBwd", ci.bwd, 0);
    env->propSetInt(m, "SplatRefine", ci.refine ? 1 : 0, 0);
    std::vector<double> b(ci.bias.begin(), ci.bias.end());
    env->propSetFloatArray(m, "SplatBias", b.data(), (int)b.size());
  }

  // false when f is no candidate frame of a known layout (why says what is wrong)
  inline bool cand_read_props(IScriptEnvironment *env, const PVideoFrame &f, CandInfo &ci, const char *&why)
  {
    const AVSMap *m = env->getFramePropsRO(f);
    int err = 0;
    const auto geti = [&](const char *k) { return err ? 0 : (int)env->propGetInt(m, k, 0, &err); };
    if (geti("SplatCand") != CAND_LAYOUT || err)
    {
      why = "the vectors clip is not a candidate clip of this version (SplatAnalyse or MSplatVectors)";
      return false;
    }
    ci.n = geti("SplatN");
    ci.gw = geti("SplatGridW");
    ci.gh = geti("SplatGridH");
    ci.fwd = geti("SplatFwd") != 0;
    ci.bwd = geti("SplatBwd");
    ci.refine = geti("SplatRefine") != 0;
    if (!err)
    {
      int e2 = 0;
      const int sc = (int)env->propGetInt(m, "SplatScale", 0, &e2);
      if (!e2)
      {
        ci.scale = sc;
        ci.pad_x = (int)env->propGetInt(m, "SplatPadX", 0, &e2);
        ci.pad_y = e2 ? -1 : (int)env->propGetInt(m, "SplatPadY", 0, &e2);
        if (e2)
          ci.pad_x = ci.pad_y = -1;
      }
    }
    const double *b = err ? nullptr : env->propGetFloatArray(m, "SplatBias", &err);
    if (err || !b || env->propNumElements(m, "SplatBias") != ci.n + 1 || ci.n < 1 || ci.bwd < 0 || ci.bwd > 3)
    {
      why = "the vectors clip has incomplete or inconsistent candidate properties";
      return false;
    }
    ci.bias.resize((size_t)ci.n + 1);
    for (int i = 0; i <= ci.n; i++)
      ci.bias[i] = (float)b[i];
    return true;
  }

  // set s (0 forward, 1 backward) of a candidate frame <-> n (vx, vy) planes
  inline void cand_store(PVideoFrame &f, int s, const std::vector<Plane> &set)
  {
    const int pitch = f->GetPitch() / (int)sizeof(float);
    float *base = reinterpret_cast<float *>(f->GetWritePtr());
    if (set.empty())
      return;
    const int gh = set[0].h, gw = set[0].w;
    for (size_t i = 0; i < set.size(); i++)
      for (int y = 0; y < gh; y++)
        memcpy(base + (size_t)((s * set.size() + i) * gh + y) * pitch, &set[i].d[(size_t)y * gw], sizeof(float) * gw);
  }

  inline std::vector<Plane> cand_load(const PVideoFrame &f, int s, const CandInfo &ci)
  {
    const int pitch = f->GetPitch() / (int)sizeof(float);
    const float *base = reinterpret_cast<const float *>(f->GetReadPtr());
    std::vector<Plane> set(2 * (size_t)ci.n, Plane(ci.gw, ci.gh));
    for (size_t i = 0; i < set.size(); i++)
      for (int y = 0; y < ci.gh; y++)
        memcpy(&set[i].d[(size_t)y * ci.gw], base + (size_t)((s * set.size() + i) * ci.gh + y) * pitch,
          sizeof(float) * ci.gw);
    return set;
  }
}
