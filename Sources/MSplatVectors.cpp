// MSplatVectors: MVTools block vectors as a candidate clip for SplatFps (splatfps.dll, SplatFps(vectors=...)).
//
//   MSplatVectors(mvbw, mvfw [, refine = true, thSCD1 = 400, thSCD2 = 130, me_scale, pad_x, pad_y, pyramid_levels])
//
// mvbw / mvfw are MAnalyse(isb = true / false, delta = 1) clips of the clip SplatFps converts, as for MFlowFps. Frame c
// of the result holds, on SplatFps' ME grid (one sample per scale x scale pixels of the picture padded by pad_x / pad_y
// pixels; the defaults of the picture size as in SplatFps, e.g. 4 x 4 and 16 / 20 at 1080p; me_scale, pad_x, pad_y and
// pyramid_levels must match the SplatFps arguments when those are given there), the
// forward candidate (the mvbw vector of frame c: from c into c+1) and the backward candidate (the mvfw vector of frame
// c: from c into c-1), one candidate each, taken from the block whose centre is nearest to the ME sample. SplatFps adds
// the zero vector, refines both against the pictures (refine = false: assigns them as they are) and handles occlusions
// and unreliable areas as it does for its own phase-correlation candidates. A frame whose vectors fail MVTools' scene
// change test (thSCD1 / thSCD2, as in MFlowFps) gives zero candidates. Layout and frame properties: cand_clip.h.
// MVTools internals stay in this DLL: splatfps.dll reads only the candidate clip.

#include "MVClip.h"
#include "../splatfps/cand_clip.h"

#include "avisynth.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace
{
  class MSplatVectors : public GenericVideoFilter
  {
    MVClip mvB_, mvF_;
    bool refine_;
    VideoInfo pic_vi_; // the analysed picture size (what SplatFps converts)
    splat::Config cfg_; // geometry of SplatFps' ME grid

    // one MVTools frame -> (vx, vy) planes of the ME grid; false when the frame is not usable (scene change)
    bool expand(MVClip &clip, int n, std::vector<splat::Plane> &set, IScriptEnvironment *env)
    {
      PVideoFrame f = clip.GetFrame(n, env);
      clip.Update(f, env);
      if (!clip.IsUsable())
        return false;
      const int bx = clip.GetBlkX(), by = clip.GetBlkY();
      const int sx = clip.GetBlkSizeX() - clip.GetOverlapX(), sy = clip.GetBlkSizeY() - clip.GetOverlapY();
      const double cx = (clip.GetBlkSizeX() - 1) / 2.0, cy = (clip.GetBlkSizeY() - 1) / 2.0;
      const int s = cfg_.scale;
      const float scale = 1.0f / ((float)s * clip.GetPel()); // pel units of the picture -> ME pixels
      const double half = (s - 1) / 2.0;
      const int gw = set[0].w, gh = set[0].h;
      for (int j = 0; j < gh; j++)
      {
        const double py = (double)s * j + half - cfg_.pad_y; // centre of ME sample j in picture coordinates
        const int b_y = std::min(by - 1, std::max(0, (int)std::lround((py - cy) / sy)));
        for (int i = 0; i < gw; i++)
        {
          const double px = (double)s * i + half - cfg_.pad_x;
          const int b_x = std::min(bx - 1, std::max(0, (int)std::lround((px - cx) / sx)));
          const VECTOR v = clip.GetBlock(0, b_y * bx + b_x).GetMV();
          set[0].at(i, j) = v.x * scale;
          set[1].at(i, j) = v.y * scale;
        }
      }
      return true;
    }

  public:
    MSplatVectors(PClip mvbw, PClip mvfw, bool refine, int scd1, int scd2, const int geo[4], IScriptEnvironment *env)
      : GenericVideoFilter(mvbw), mvB_(mvbw, scd1, scd2, env, 1, 0), mvF_(mvfw, scd1, scd2, env, 1, 0), refine_(refine)
    {
      if (mvB_.GetDeltaFrame() != 1 || mvF_.GetDeltaFrame() != 1)
        env->ThrowError("MSplatVectors: the vector clips must come from MAnalyse with delta = 1");
      if (mvB_.GetWidth() != mvF_.GetWidth() || mvB_.GetHeight() != mvF_.GetHeight())
        env->ThrowError("MSplatVectors: mvbw and mvfw are analyses of different picture sizes");
      pic_vi_ = vi;
      pic_vi_.width = mvB_.GetWidth();
      pic_vi_.height = mvB_.GetHeight();
      // the ME grid of SplatFps for this picture size: its defaults, with the geometry arguments given here
      cfg_ = splat::config_defaults();
      if (geo[0] > 0)
        cfg_.scale = geo[0];
      if (geo[1] >= 0)
        cfg_.pad_x = geo[1];
      if (geo[2] >= 0)
        cfg_.pad_y = geo[2];
      if (geo[3] > 0)
        cfg_.levels = geo[3];
      splat::config_resolve(cfg_, pic_vi_.width, pic_vi_.height);
      char why[512];
      if (!splat::config_check(cfg_, pic_vi_.width, pic_vi_.height, why, sizeof(why)))
        env->ThrowError("MSplatVectors: %s", why);
      vi = splat::cand_clip_info(pic_vi_, 1, true, cfg_);
    }

    int __stdcall SetCacheHints(int cachehints, int) override
    {
      return cachehints == CACHE_GET_MTMODE ? MT_SERIALIZED : 0;
    }

    PVideoFrame __stdcall GetFrame(int c, IScriptEnvironment *env) override
    {
      c = c < 0 ? 0 : (c >= vi.num_frames ? vi.num_frames - 1 : c);
      PVideoFrame out = env->NewVideoFrameP(vi, nullptr);
      memset(out->GetWritePtr(), 0, (size_t)out->GetPitch() * vi.height);
      const splat::Geometry g(pic_vi_.width, pic_vi_.height, cfg_);
      splat::CandInfo ci;
      ci.n = 1;
      ci.gw = g.MW;
      ci.gh = g.MH;
      ci.scale = cfg_.scale;
      ci.pad_x = cfg_.pad_x;
      ci.pad_y = cfg_.pad_y;
      ci.fwd = c + 1 < vi.num_frames;
      ci.bwd = c > 0 ? 1 : 0;
      ci.refine = refine_;
      ci.bias = { 0.0f, 1.0f }; // the zero vector costs a little more than the measured one
      std::vector<splat::Plane> set(2, splat::Plane(g.MW, g.MH));
      if (ci.fwd)
      {
        // read here, not in the constructor: the backward flag is only reliable once a frame was read
        if (expand(mvB_, c, set, env))
          splat::cand_store(out, 0, set);
        if (!mvB_.IsBackward())
          env->ThrowError("MSplatVectors: mvbw must be a backward analysis (MAnalyse isb = true)");
      }
      if (ci.bwd)
      {
        set.assign(2, splat::Plane(g.MW, g.MH));
        if (expand(mvF_, c, set, env))
          splat::cand_store(out, 1, set);
        if (mvF_.IsBackward())
          env->ThrowError("MSplatVectors: mvfw must be a forward analysis (MAnalyse isb = false)");
      }
      splat::cand_write_props(env, out, ci);
      return out;
    }
  };
}

AVSValue __cdecl Create_MSplatVectors(AVSValue args, void *, IScriptEnvironment *env)
{
  const int geo[4] = { args[5].AsInt(0), args[6].AsInt(-1), args[7].AsInt(-1), args[8].AsInt(0) };
  return new MSplatVectors(args[0].AsClip(), args[1].AsClip(), args[2].AsBool(true), args[3].AsInt(400),
    args[4].AsInt(130), geo, env);
}
