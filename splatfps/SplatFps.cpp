// splatfps.dll: SplatFps, motion-compensated frame-rate conversion (any progressive rate to any other).
//
// Output frame k sits at source time k * rate_in / rate_out. When that time is within 1 / copy_snap of a source
// picture (1/128 by default), the output is that picture (put through the output rounding); otherwise it is built
// from the two source pictures around it (P before, N after): both are splatted along their motion vectors into a
// Laplacian pyramid and blended, with fallbacks where the vectors are unreliable. The stages are in core/ (CPU) and
// in mvtools_cuda (GPU).
//
// Per source picture: input conversion and ME pictures (source()). Per centre picture c: phase-correlation
// candidates of the pairs (c, c+1) and (c-1, c), Lucas-Kanade refinement, match errors, forward / backward
// assignment and local fallback, giving the forward and backward field of c (fields()). Per built frame: the pair
// statistic (scene cut, temporal fallback), Laplacian-pyramid forward splatting and output conversion (built()). The
// schedule is made once in the constructor.
//
// Settings: every tuned constant and the size-dependent geometry (ME scale, borders, pyramid, block stages) are a
// splat::Config (core/splat_config.h), the defaults of the clip's size unless named arguments override them
// (config_args.h).
//
// Formats: planar YUV 4:4:4, 4:2:2 or 4:2:0, 8 .. 16 bit, left chroma siting (io_format.h), progressive, any size
// whose ME picture holds the largest phase-correlation block (about 480 x 480 at 1080p settings); the output has the
// format of the input, and frames that are source pictures are passed on unchanged. Internally every picture is
// 4:4:4 float32 (input / output conversion in core/splat_core.h). exact = false (default): the fast path of
// core/splat_fast.h (all cores, AVX2 + FMA kernels where the CPU has them), the same stages with the same arithmetic
// except the transcendental functions, which can differ in the last bit in rare places; the phase-correlation
// candidates of a pair are kept for the next centre picture instead of being computed twice. exact = true: the exact
// path, the deterministic scalar path: float32 in a fixed evaluation order, no contraction, IEEE division and sqrt,
// exp through double; the fast CPU path and CUDA (exact=true) are checked bit for bit against it. It runs one thread
// for most stages (about a second per built frame).

#include "avisynth.h"
#include "dyn_lib.h"
#include "mvcuda.h"
#include "core/splat_core.h"
#include "core/splat_me.h"
#include "core/splat_motion.h"
#include "core/splat_pcorr.h"
#include "core/splat_const.h"
#include "core/splat_fast.h"
#include "cand_clip.h"
#include "config_args.h"
#include "io_format.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

using namespace splat;

namespace
{
  // greatest common divisor for reducing rates
  int64_t gcd64(int64_t a, int64_t b)
  {
    while (b)
    {
      const int64_t t = a % b;
      a = b;
      b = t;
    }
    return a < 0 ? -a : a;
  }

  class SplatFps : public GenericVideoFilter
  {
  protected:
    Config cfg_;                // resolved and checked settings
    IoFormat fmt_;              // the clip's sample format (input and output)
    const char *name_;          // filter name for errors
    std::vector<float> costs_;  // costs of the phase-correlation candidates (n + 1, zero vector last)
    std::vector<PlanEntry> plan_;
    int n_src_ = 0;
    double strength_ = 1.0;
    PClip vectors_; // candidate clip (cand_clip.h) or null: the built-in phase correlation
    // per source picture: samples (4:4:4), ME picture A and the phase-correlation state, the picture-build pyramid
    // (fast path, made when a built frame first needs it); and its fields once made
    struct Source
    {
      Pic3 pic;
      Plane A;
      PcorrPicture pcorr;
      Pyramid pyr;
    };
    struct Fields
    {
      Field4 fwd, bwd;
    };
    std::map<int, std::shared_ptr<Source>> src_;
    std::map<int, std::shared_ptr<Fields>> fields_;
    // fast path (exact = false): the engine, and the phase-correlation candidates of the pair (q, q + 1) by q
    std::unique_ptr<FastEngine> fast_;
    std::map<int, std::shared_ptr<VecSet>> pairs_;
    Pic3 fast_out_; // the last built picture (its memory is reused)
    std::vector<std::shared_ptr<Source>> spare_src_; // dropped sources and pairs whose memory is reused
    std::vector<std::shared_ptr<VecSet>> spare_pairs_;

  public:
    SplatFps(PClip c, int num, int den, PClip vectors, const Config &cfg, IScriptEnvironment *env, bool exact = true,
      const char *name = "SplatFps")
      : GenericVideoFilter(c), cfg_(cfg), name_(name), vectors_(vectors)
    {
      if (!exact)
        fast_ = std::make_unique<FastEngine>();
      fmt_ = io_format(vi, env, name_);
      // the siting of frame 0 already here: a script error at load time (some hosts read frame 0 while opening the
      // script and do not handle an error thrown there); every source frame is checked again when it is used
      if (vi.num_frames > 0)
        fmt_.centre = siting_of(child->GetFrame(0, env), fmt_, 0, env, name_);
      {
        const Geometry g(vi.width, vi.height, cfg_);
        float b[CFG_MAX_COSTS];
        costs_.assign(b, b + pcorr_costs(cfg_, g.MW, g.MH, b));
      }
      if (num <= 0 || den <= 0)
        env->ThrowError("SplatFps: num and den must be positive");
      if (vi.fps_numerator <= 0 || vi.fps_denominator <= 0)
        env->ThrowError("SplatFps: the clip has no valid frame rate");
      int64_t on = num, od = den, g = gcd64(on, od);
      on /= g;
      od /= g;
      int64_t in = vi.fps_numerator, id = vi.fps_denominator;
      g = gcd64(in, id);
      in /= g;
      id /= g;
      n_src_ = vi.num_frames;
      plan_ = plan(n_src_, in, id, on, od, cfg_.copy_snap);
      strength_ = phase_pull_strength(in, id, on, od, cfg_);
      vi.num_frames = (int)plan_.size();
      vi.SetFPS((unsigned)on, (unsigned)od);
      if (vectors_)
      {
        const VideoInfo &v = vectors_->GetVideoInfo();
        const Geometry g(vi.width, vi.height, cfg_);
        if (v.pixel_type != VideoInfo::CS_Y32 || v.width != g.MW)
          env->ThrowError("SplatFps: vectors must be a candidate clip of this clip (SplatAnalyse or MSplatVectors)");
        if (v.num_frames != n_src_)
          env->ThrowError("SplatFps: vectors has %d frames, the clip %d", v.num_frames, n_src_);
      }
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) override
    {
      return cachehints == CACHE_GET_MTMODE ? MT_SERIALIZED : 0;
    }

    // source picture q, its chroma siting checked
    PVideoFrame src_frame(int q, IScriptEnvironment *env)
    {
      PVideoFrame f = child->GetFrame(q, env);
      check_siting(f, fmt_, q, env, name_);
      return f;
    }

    // plane pointers and pitches of a frame (Y, U, V)
    static void planes_of(const PVideoFrame &f, const void *ptr[3], int pitch[3])
    {
      const int pl[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
      for (int i = 0; i < 3; i++)
      {
        ptr[i] = f->GetReadPtr(pl[i]);
        pitch[i] = f->GetPitch(pl[i]);
      }
    }
    static void planes_of(PVideoFrame &f, void *ptr[3], int pitch[3])
    {
      const int pl[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
      for (int i = 0; i < 3; i++)
      {
        ptr[i] = f->GetWritePtr(pl[i]);
        pitch[i] = f->GetPitch(pl[i]);
      }
    }

    // keeps the entries near q (a frame needs pictures p-1 .. p+2; output order is mostly increasing)
    template <class M> static void trim(M &m, int q)
    {
      for (auto it = m.begin(); it != m.end();)
        it = (it->first < q - 4 || it->first > q + 4) ? m.erase(it) : std::next(it);
    }

    // trim for the fast path: dropped entries nobody else holds go to spare (up to 2), so their memory is reused
    template <class M, class S> static void trim_keep(M &m, int q, S &spare)
    {
      for (auto it = m.begin(); it != m.end();)
      {
        if (it->first < q - 4 || it->first > q + 4)
        {
          if (it->second.use_count() == 1 && spare.size() < 2)
            spare.push_back(it->second);
          it = m.erase(it);
        }
        else
          it = std::next(it);
      }
    }

    // input conversion, ME pictures and the phase-correlation ingest of source picture q (cached; pictures far behind are dropped)
    std::shared_ptr<Source> source(int q, IScriptEnvironment *env)
    {
      auto it = src_.find(q);
      if (it != src_.end())
        return it->second;
      PVideoFrame f = src_frame(q, env);
      const int W = vi.width, H = vi.height;
      const void *ptr[3];
      int pitch[3];
      planes_of(f, ptr, pitch);
      if (fast_)
      {
        std::shared_ptr<Source> s;
        if (!spare_src_.empty())
        {
          s = spare_src_.back();
          spare_src_.pop_back();
        }
        else
          s = std::make_shared<Source>();
        for (int c = 0; c < 3; c++)
          if (s->pic.c[c].w != W || s->pic.c[c].h != H)
            s->pic.c[c] = Plane(W, H);
        // the same conversion, rows spread over the engine's threads
        fast_->rows(H, [&](int y0, int y1) { input_rows(ptr, pitch, fmt_, y0, y1, s->pic); });
        s->pyr.valid = false; // (a reused entry holds the pyramid of another picture)
        s->A = fast_->me_picture(s->pic.c[0], MeFir::A, cfg_);
        if (!vectors_)
          fast_->pcorr_ingest_into(fast_->me_picture(s->pic.c[0], MeFir::B, cfg_), cfg_, s->pcorr);
        src_[q] = s;
        trim_keep(src_, q, spare_src_);
        return s;
      }
      auto s = std::make_shared<Source>();
      for (int c = 0; c < 3; c++)
        s->pic.c[c] = Plane(W, H);
      input_rows(ptr, pitch, fmt_, 0, H, s->pic);
      s->A = me_picture(s->pic.c[0], MeFir::A, cfg_);
      if (!vectors_)
        s->pcorr = pcorr_ingest(me_picture(s->pic.c[0], MeFir::B, cfg_), cfg_);
      src_[q] = s;
      trim(src_, q);
      return s;
    }

    // forward field of c (vectors into c+1) and backward field (into c-1); picture 0 has no backward side
    std::shared_ptr<Fields> fields(int c, IScriptEnvironment *env)
    {
      auto it = fields_.find(c);
      if (it != fields_.end())
        return it->second;
      if (vectors_)
        return fields_from_clip(c, env);
      const bool single = c == 0;
      std::shared_ptr<Source> prv = single ? nullptr : source(c - 1, env);
      std::shared_ptr<Source> cur = source(c, env), nxt = source(c + 1, env);
      if (fast_)
      {
        // candidates of the pairs (c, c + 1) and (c - 1, c), each made once and kept for the neighbouring picture
        auto pair = [&](int q, const Source &a, const Source &b) {
          auto p = pairs_.find(q);
          if (p != pairs_.end())
            return p->second;
          std::shared_ptr<VecSet> v;
          if (!spare_pairs_.empty())
          {
            v = spare_pairs_.back();
            spare_pairs_.pop_back();
          }
          else
            v = std::make_shared<VecSet>();
          fast_->pcorr_candidates_into(a.pcorr, b.pcorr, cfg_, *v);
          pairs_[q] = v;
          trim_keep(pairs_, q, spare_pairs_);
          return v;
        };
        const std::shared_ptr<VecSet> pf = pair(c, *cur, *nxt), pb = single ? pf : pair(c - 1, *prv, *cur);
        CandSet sf, sb;
        sf.vecs = pf.get();
        sf.toward = true;
        sb.vecs = pb.get();
        sb.toward = false;
        auto r = std::make_shared<Fields>();
        fast_->fields(single ? nullptr : &prv->A, cur->A, nxt->A, sf, sb, costs_.data(), true, cfg_, r->fwd, r->bwd);
        fields_[c] = r;
        trim(fields_, c);
        return r;
      }
      const VecSet cf = pcorr_candidates(cur->pcorr, nxt->pcorr, cfg_);
      VecSet cb;
      if (!single)
        cb = pcorr_candidates(prv->pcorr, cur->pcorr, cfg_);
      CandSet sf, sb;
      sf.vecs = &cf;
      sf.toward = true;
      sb.vecs = single ? &cf : &cb;
      sb.toward = false; // the pair (c-1, c) points into c; the refinement turns it around
      auto r = std::make_shared<Fields>();
      splat::fields(single ? nullptr : &prv->A, cur->A, nxt->A, sf, sb, costs_.data(), true, cfg_, r->fwd, r->bwd);
      fields_[c] = r;
      trim(fields_, c);
      return r;
    }

    CandInfo cand_info(const PVideoFrame &f, int c, IScriptEnvironment *env)
    {
      CandInfo ci;
      const char *why = "";
      if (!cand_read_props(env, f, ci, why))
        env->ThrowError("SplatFps: frame %d of vectors: %s", c, why);
      const Geometry g(vi.width, vi.height, cfg_);
      if (ci.gw != g.MW || ci.gh != g.MH)
        env->ThrowError("SplatFps: frame %d of vectors is for another picture size or ME grid (me_scale, pad_x, pad_y "
          "must match)", c);
      if (ci.scale > 0 && (ci.scale != cfg_.scale || ci.pad_x != cfg_.pad_x || ci.pad_y != cfg_.pad_y))
        env->ThrowError("SplatFps: frame %d of vectors is for me_scale %d, pad %d x %d; this filter uses %d, %d x %d", c,
          ci.scale, ci.pad_x, ci.pad_y, cfg_.scale, cfg_.pad_x, cfg_.pad_y);
      if (ci.n + 1 > CFG_MAX_COSTS)
        env->ThrowError("SplatFps: frame %d of vectors has more than %d candidates", c, CFG_MAX_COSTS - 1);
      if (2 * ci.n * ci.gh > vectors_->GetVideoInfo().height / (ci.bwd == 1 || ci.bwd == 2 ? 2 : 1))
        env->ThrowError("SplatFps: frame %d of vectors holds fewer rows than its properties say", c);
      return ci;
    }

    // refinement to local fallback of picture c from the candidate clip: the forward set of frame c; the backward set as SplatBwd says (in
    // frame c, pointing from c or into c, or the forward set of frame c-1, which points into c)
    std::shared_ptr<Fields> fields_from_clip(int c, IScriptEnvironment *env)
    {
      PVideoFrame f = vectors_->GetFrame(c, env);
      const CandInfo ci = cand_info(f, c, env);
      if (!ci.fwd)
        env->ThrowError("SplatFps: frame %d of vectors has no forward candidates", c);
      const VecSet cf = cand_load(f, 0, ci);
      VecSet cb;
      CandSet sf, sb;
      sf.vecs = &cf;
      sf.toward = true;
      sf.refine = ci.refine;
      sb.refine = ci.refine;
      const bool single = c == 0 || ci.bwd == 0;
      if (!single && ci.bwd == 3)
      {
        PVideoFrame fp = vectors_->GetFrame(c - 1, env);
        const CandInfo cp = cand_info(fp, c - 1, env);
        if (cp.n != ci.n || !cp.fwd)
          env->ThrowError("SplatFps: frames %d and %d of vectors do not match", c - 1, c);
        cb = cand_load(fp, 0, cp);
        sb.toward = false;
      }
      else if (!single)
      {
        cb = cand_load(f, 1, ci);
        sb.toward = ci.bwd == 1;
      }
      sb.vecs = single ? &cf : &cb;
      std::shared_ptr<Source> prv = single ? nullptr : source(c - 1, env);
      std::shared_ptr<Source> cur = source(c, env), nxt = source(c + 1, env);
      auto r = std::make_shared<Fields>();
      if (fast_)
        fast_->fields(single ? nullptr : &prv->A, cur->A, nxt->A, sf, sb, ci.bias.data(), true, cfg_, r->fwd, r->bwd);
      else
        splat::fields(single ? nullptr : &prv->A, cur->A, nxt->A, sf, sb, ci.bias.data(), true, cfg_, r->fwd, r->bwd);
      fields_[c] = r;
      trim(fields_, c);
      return r;
    }

    // pair statistic, build and output conversion of an in-between frame: forward field of P, backward field of N, adjusted phase, build, codes
    virtual PVideoFrame built(const PlanEntry &e, IScriptEnvironment *env)
    {
      const std::shared_ptr<Fields> fp = fields(e.p, env), fn = fields(e.p + 1, env);
      const double d = phase_pull_adjust(fp->fwd, fn->bwd, (double)e.dnum / (double)e.dden, strength_, cfg_);
      const std::shared_ptr<Source> P = source(e.p, env), N = source(e.p + 1, env);
      Pic3 built_exact;
      if (fast_)
      {
        // the pyramids of P and N, each made once per source picture (they do not depend on the frame)
        for (Source *s : { P.get(), N.get() })
          if (!s->pyr.valid)
            fast_->pyramid_into(s->pic, cfg_, s->pyr);
        fast_->build_pyr_into(P->pyr, N->pyr, fp->fwd, fn->bwd, (float)d, cfg_, fast_out_);
      }
      else
        built_exact = build(P->pic, N->pic, fp->fwd, fn->bwd, (float)d, cfg_);
      const Pic3 &o = fast_ ? fast_out_ : built_exact;
      PVideoFrame src = src_frame(e.p, env);
      PVideoFrame dst = env->NewVideoFrameP(vi, &src);
      void *ptr[3];
      int pitch[3];
      planes_of(dst, ptr, pitch);
      if (fast_) // the same output rule, rows spread over the engine's threads
        fast_->rows(vi.height, [&](int y0, int y1) { output_rows(o, fmt_, y0, y1, ptr, pitch); });
      else
        output_rows(o, fmt_, 0, vi.height, ptr, pitch);
      return dst;
    }

    PVideoFrame __stdcall GetFrame(int k, IScriptEnvironment *env) override
    {
      k = k < 0 ? 0 : (k >= vi.num_frames ? vi.num_frames - 1 : k);
      const PlanEntry &e = plan_[k];
      switch (e.kind)
      {
      case PlanEntry::CopyP: // source pictures pass unchanged
        return src_frame(e.p, env);
      case PlanEntry::CopyN:
        return src_frame(e.p + 1, env);
      case PlanEntry::EosPlaceholder:
        // the end of the clip is not handled yet: the nearest source picture
        return src_frame(std::min(n_src_ - 1, e.p + (2 * e.dnum > e.dden ? 1 : 0)), env);
      default:
        break;
      }
      return built(e, env);
    }
  };

  // mvtools_cuda.dll next to this plugin (the same file mvtools2 and cudafilters load)
  struct CudaLib
  {
    decltype(&mvcuda_splat_ctx_create) create = nullptr;
    decltype(&mvcuda_splat_ctx_free) free = nullptr;
    decltype(&mvcuda_splat_ctx_has_source) has_source = nullptr;
    decltype(&mvcuda_splat_ctx_source) source = nullptr;
    decltype(&mvcuda_splat_ctx_frame_start) frame_start = nullptr;
    decltype(&mvcuda_splat_ctx_frame_finish) frame_finish = nullptr;
    decltype(&mvcuda_splat_ctx_external) external = nullptr;
    decltype(&mvcuda_splat_ctx_has_fields) has_fields = nullptr;
    decltype(&mvcuda_splat_ctx_candidates) candidates = nullptr;
    std::string why;
    bool ok = false;
    static CudaLib &get()
    {
      static CudaLib lib;
      return lib;
    }

  private:
    CudaLib()
    {
      if (std::getenv("MVTOOLS_GPU_DISABLE"))
      {
        why = "disabled by the environment variable MVTOOLS_GPU_DISABLE";
        return;
      }
      const std::filesystem::path dir = dynlib::module_dir(reinterpret_cast<const void *>(&CudaLib::get));
      void *lib = dir.empty() ? nullptr : dynlib::open(dir / dynlib::cuda_lib_name);
      if (!lib)
      {
        why = std::string(dynlib::cuda_lib_name) + " was not found or could not be loaded from the plugin folder";
        return;
      }
      auto version = reinterpret_cast<int (*)()>(dynlib::symbol(lib, "mvcuda_api_version"));
      if (!version || version() != MVCUDA_API_VERSION)
      {
        why = std::string(dynlib::cuda_lib_name) + " is from another build (library API version differs)";
        return;
      }
      create = reinterpret_cast<decltype(create)>(dynlib::symbol(lib, "mvcuda_splat_ctx_create"));
      free = reinterpret_cast<decltype(free)>(dynlib::symbol(lib, "mvcuda_splat_ctx_free"));
      has_source = reinterpret_cast<decltype(has_source)>(dynlib::symbol(lib, "mvcuda_splat_ctx_has_source"));
      source = reinterpret_cast<decltype(source)>(dynlib::symbol(lib, "mvcuda_splat_ctx_source"));
      frame_start = reinterpret_cast<decltype(frame_start)>(dynlib::symbol(lib, "mvcuda_splat_ctx_frame_start"));
      frame_finish = reinterpret_cast<decltype(frame_finish)>(dynlib::symbol(lib, "mvcuda_splat_ctx_frame_finish"));
      external = reinterpret_cast<decltype(external)>(dynlib::symbol(lib, "mvcuda_splat_ctx_external"));
      has_fields = reinterpret_cast<decltype(has_fields)>(dynlib::symbol(lib, "mvcuda_splat_ctx_has_fields"));
      candidates = reinterpret_cast<decltype(candidates)>(dynlib::symbol(lib, "mvcuda_splat_ctx_candidates"));
      ok = create && free && has_source && source && frame_start && frame_finish && external && has_fields && candidates;
      if (!ok)
        why = std::string(dynlib::cuda_lib_name) + " lacks the SplatFps entry points";
    }
  };

  // The GPU filter: the same frames as SplatFps (exact = true: bit-identical; false: fast arithmetic, a few codes +-1),
  // built in a device context of mvtools_cuda. Copy frames are the source frames, passed on.
  class CudaSplatFps : public SplatFps
  {
    void *ctx_ = nullptr;

  public:
    CudaSplatFps(PClip c, int num, int den, bool exact, PClip vectors, const Config &cfg, IScriptEnvironment *env)
      : SplatFps(c, num, den, vectors, cfg, env, true, "CudaSplatFps")
    {
      CudaLib &lib = CudaLib::get();
      if (!lib.ok)
        env->ThrowError("CudaSplatFps: %s (SplatFps is the CPU version)", lib.why.c_str());
      MvCudaSplatParams p = { vi.width, vi.height, exact ? 1 : 0, &cfg_, fmt_.bits, fmt_.sub_w, fmt_.sub_h,
        fmt_.centre };
      ctx_ = lib.create(&p);
      if (!ctx_)
        env->ThrowError("CudaSplatFps: no usable CUDA device, or not enough device memory (SplatFps is the CPU version)");
      if (vectors_)
        lib.external(ctx_);
    }

    ~CudaSplatFps() override
    {
      if (ctx_)
        CudaLib::get().free(ctx_);
    }

    void upload(int q, IScriptEnvironment *env)
    {
      CudaLib &lib = CudaLib::get();
      if (lib.has_source(ctx_, q))
        return;
      PVideoFrame f = src_frame(q, env);
      const void *ptr[3];
      int pitch[3];
      planes_of(f, ptr, pitch);
      if (lib.source(ctx_, q, ptr, pitch))
        env->ThrowError("CudaSplatFps: CUDA error while uploading source frame %d", q);
    }

    // the fields of picture c on the device from the candidate clip (sources c-1 .. c+1 must be there)
    void cand_fields(int c, IScriptEnvironment *env)
    {
      CudaLib &lib = CudaLib::get();
      if (lib.has_fields(ctx_, c))
        return;
      PVideoFrame f = vectors_->GetFrame(c, env);
      const CandInfo ci = cand_info(f, c, env);
      if (!ci.fwd)
        env->ThrowError("CudaSplatFps: frame %d of vectors has no forward candidates", c);
      PVideoFrame fp;
      const float *bwd = nullptr;
      int bwd_pitch = 0;
      bool toward = false;
      if (c > 0 && ci.bwd == 3)
      {
        fp = vectors_->GetFrame(c - 1, env);
        const CandInfo cp = cand_info(fp, c - 1, env);
        if (cp.n != ci.n || !cp.fwd)
          env->ThrowError("CudaSplatFps: frames %d and %d of vectors do not match", c - 1, c);
        bwd = reinterpret_cast<const float *>(fp->GetReadPtr());
        bwd_pitch = fp->GetPitch();
      }
      else if (c > 0 && ci.bwd != 0)
      {
        bwd = reinterpret_cast<const float *>(f->GetReadPtr() + (size_t)2 * ci.n * ci.gh * f->GetPitch());
        bwd_pitch = f->GetPitch();
        toward = ci.bwd == 1;
      }
      const int rc = lib.candidates(ctx_, c, reinterpret_cast<const float *>(f->GetReadPtr()), f->GetPitch(), bwd,
        bwd_pitch, ci.n, toward ? 1 : 0, ci.refine ? 1 : 0, ci.bias.data());
      if (rc)
        env->ThrowError("CudaSplatFps: CUDA error %d with the candidates of picture %d", rc, c);
    }

    // the sources and fields frame e needs, then its start on the device
    void start(const PlanEntry &e, IScriptEnvironment *env)
    {
      CudaLib &lib = CudaLib::get();
      for (int q = std::max(0, e.p - 1); q <= e.p + 2; q++)
        upload(q, env);
      if (vectors_)
      {
        cand_fields(e.p, env);
        cand_fields(e.p + 1, env);
      }
      const int rc = lib.frame_start(ctx_, e.p, (double)e.dnum / (double)e.dden, strength_, nullptr);
      if (rc)
        env->ThrowError("CudaSplatFps: CUDA error %d while building frame %d", rc, e.k);
    }

    std::deque<int> started_; // output frames started on the device and not finished yet, oldest first (at most 2)

    // Frame k is usually started already (by the call for an earlier frame). Before it is handed on, the next built
    // frame (up to 3 frames on) is started, so the device builds it while the host copies frame k out and the caller
    // works on it; then the next source picture is decoded and staged (its fields follow on the device).
    PVideoFrame built(const PlanEntry &e, IScriptEnvironment *env) override
    {
      CudaLib &lib = CudaLib::get();
      if (started_.empty() || started_.front() != e.k)
      {
        // out of order (a seek) or after an error: the frames started ahead are dropped
        for (; !started_.empty(); started_.pop_front())
          lib.frame_finish(ctx_, nullptr, nullptr);
        start(e, env);
        started_.push_back(e.k);
      }
      const PlanEntry *last = &e;
      for (int j = e.k + 1; j <= std::min(e.k + 3, vi.num_frames - 1); j++)
        if (plan_[j].kind == PlanEntry::Built)
        {
          if (started_.size() < 2)
          {
            start(plan_[j], env);
            started_.push_back(j);
          }
          last = &plan_[j];
          break;
        }
      if (last->p + 3 < n_src_)
      {
        upload(last->p + 3, env);
        if (vectors_)
          cand_fields(last->p + 2, env); // the next frames' fields, queued behind the started frames
      }
      PVideoFrame src = src_frame(e.p, env);
      PVideoFrame dst = env->NewVideoFrameP(vi, &src);
      void *ptr[3];
      int pitch[3];
      planes_of(dst, ptr, pitch);
      // the oldest started frame is e (frames are finished in the order they were started)
      started_.pop_front();
      const int rc = lib.frame_finish(ctx_, ptr, pitch);
      if (rc)
        env->ThrowError("CudaSplatFps: CUDA error %d while building frame %d", rc, e.k);
      return dst;
    }
  };

  // the filter arguments 0 .. 4, then the Config arguments (config_args.h)
  AVSValue __cdecl Create_CudaSplatFps(AVSValue args, void *, IScriptEnvironment *env)
  {
    const PClip c = args[0].AsClip();
    const VideoInfo &vi = c->GetVideoInfo();
    io_format(vi, env, "CudaSplatFps");
    const Config cfg = config_from_args(args, 5, vi.width, vi.height, env, "CudaSplatFps");
    return new CudaSplatFps(c, args[1].AsInt(), args[2].AsInt(), args[3].AsBool(false),
      args[4].Defined() ? args[4].AsClip() : nullptr, cfg, env);
  }

  AVSValue __cdecl Create_SplatFps(AVSValue args, void *, IScriptEnvironment *env)
  {
    const PClip c = args[0].AsClip();
    const VideoInfo &vi = c->GetVideoInfo();
    io_format(vi, env, "SplatFps");
    const Config cfg = config_from_args(args, 5, vi.width, vi.height, env, "SplatFps");
    return new SplatFps(c, args[1].AsInt(), args[2].AsInt(), args[3].Defined() ? args[3].AsClip() : nullptr, cfg, env,
      args[4].AsBool(false));
  }
}

void splat_register_analyse(IScriptEnvironment *env); // SplatAnalyse.cpp

const AVS_Linkage *AVS_linkage = 0;

extern "C" __declspec(dllexport) const char *__stdcall AvisynthPluginInit3(IScriptEnvironment *env, const AVS_Linkage *const vectors)
{
  AVS_linkage = vectors;
  // NewVideoFrameP (frame properties) is interface V8 (AviSynth+ 3.6)
  try { env->CheckVersion(8); }
  catch (const AvisynthError &) { env->ThrowError("splatfps: needs AviSynth+ 3.6 or newer (interface V8)"); }
  static const std::string sig_cpu = "cii[vectors]c[exact]b" + config_arg_string();
  static const std::string sig_gpu = "cii[exact]b[vectors]c" + config_arg_string();
  env->AddFunction("SplatFps", sig_cpu.c_str(), Create_SplatFps, 0);
  env->AddFunction("CudaSplatFps", sig_gpu.c_str(), Create_CudaSplatFps, 0);
  splat_register_analyse(env);
  return "splatfps: motion-compensated frame-rate conversion (SplatFps, CudaSplatFps, SplatAnalyse)";
}
