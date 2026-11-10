// SplatAnalyse: phase-correlation motion measurement of a clip as a candidate clip (cand_clip.h) for
// SplatFps(vectors=...). Frame c holds the candidates of the pair (c, c+1): vectors from c into c+1, which are the
// forward set of c and, read from frame c, the backward set of c+1 (SplatBwd = 3). It runs the exact CPU path.
//
// Phase correlation (G. A. Thomas, "Television motion measurement for DATV and other applications", BBC Research
// Department Report 1987/11, https://downloads.bbc.co.uk/rd/pubs/reports/1987-11.pdf): the normalised cross-power
// spectrum of two picture blocks has an inverse transform with peaks at the dominant displacements; the strongest
// peaks of several block sizes plus a global measurement become the candidate vectors of every pixel, which the
// later stages refine and assign per pixel.

#include "avisynth.h"
#include "cand_clip.h"
#include "config_args.h"
#include "io_format.h"
#include "core/splat_const.h"
#include "core/splat_core.h"
#include "core/splat_me.h"
#include "core/splat_pcorr.h"

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>

using namespace splat;

namespace
{
  class SplatAnalyse : public GenericVideoFilter
  {
    VideoInfo src_vi_;
    IoFormat fmt_;
    Config cfg_;
    int n_ = 0;               // candidates per pair
    std::vector<float> costs_; // n + 1
    std::map<int, std::shared_ptr<PcorrPicture>> pics_;

    // luma -> ME picture B -> spectra (cached: each picture serves two pairs)
    std::shared_ptr<PcorrPicture> picture(int q, IScriptEnvironment *env)
    {
      auto it = pics_.find(q);
      if (it != pics_.end())
        return it->second;
      PVideoFrame f = child->GetFrame(q, env);
      const int W = src_vi_.width, H = src_vi_.height;
      Plane y(W, H);
      const unsigned char *p = f->GetReadPtr(PLANAR_Y);
      const int pitch = f->GetPitch(PLANAR_Y), b = fmt_.bits;
      for (int r = 0; r < H; r++)
      {
        const unsigned char *row = p + (size_t)r * pitch;
        for (int x = 0; x < W; x++)
          y.at(x, r) = to_val(b == 8 ? row[x] : reinterpret_cast<const uint16_t *>(row)[x], b);
      }
      auto s = std::make_shared<PcorrPicture>(pcorr_ingest(me_picture(y, MeFir::B, cfg_), cfg_));
      for (auto i = pics_.begin(); i != pics_.end();)
        i = (i->first < q - 2 || i->first > q + 2) ? pics_.erase(i) : std::next(i);
      pics_[q] = s;
      return s;
    }

  public:
    SplatAnalyse(PClip c, const Config &cfg, IScriptEnvironment *env)
      : GenericVideoFilter(c), src_vi_(c->GetVideoInfo()), fmt_(io_format(src_vi_, env, "SplatAnalyse")), cfg_(cfg)
    {
      const Geometry g(src_vi_.width, src_vi_.height, cfg_);
      n_ = pcorr_ncand(cfg_, g.MW, g.MH);
      float b[CFG_MAX_COSTS];
      costs_.assign(b, b + pcorr_costs(cfg_, g.MW, g.MH, b));
      vi = cand_clip_info(src_vi_, n_, false, cfg_);
      (void)env;
    }

    int __stdcall SetCacheHints(int cachehints, int) override
    {
      return cachehints == CACHE_GET_MTMODE ? MT_SERIALIZED : 0;
    }

    PVideoFrame __stdcall GetFrame(int c, IScriptEnvironment *env) override
    {
      c = c < 0 ? 0 : (c >= vi.num_frames ? vi.num_frames - 1 : c);
      PVideoFrame out = env->NewVideoFrameP(vi, nullptr);
      const Geometry g(src_vi_.width, src_vi_.height, cfg_);
      CandInfo ci;
      ci.n = n_;
      ci.gw = g.MW;
      ci.gh = g.MH;
      ci.scale = cfg_.scale;
      ci.pad_x = cfg_.pad_x;
      ci.pad_y = cfg_.pad_y;
      ci.fwd = c + 1 < vi.num_frames;
      ci.bwd = c > 0 ? 3 : 0;
      ci.refine = true;
      ci.bias = costs_;
      if (ci.fwd)
        cand_store(out, 0, pcorr_candidates(*picture(c, env), *picture(c + 1, env), cfg_));
      else
        memset(out->GetWritePtr(), 0, (size_t)out->GetPitch() * vi.height);
      cand_write_props(env, out, ci);
      return out;
    }
  };

  AVSValue __cdecl Create_SplatAnalyse(AVSValue args, void *, IScriptEnvironment *env)
  {
    const PClip c = args[0].AsClip();
    const VideoInfo &vi = c->GetVideoInfo();
    io_format(vi, env, "SplatAnalyse");
    return new SplatAnalyse(c, config_from_args(args, 1, vi.width, vi.height, env, "SplatAnalyse"), env);
  }
}

void splat_register_analyse(IScriptEnvironment *env)
{
  static const std::string sig = "c" + config_arg_string();
  env->AddFunction("SplatAnalyse", sig.c_str(), Create_SplatAnalyse, 0);
}
