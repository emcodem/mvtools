// splat_fast: the fast CPU path of SplatFps (SplatFps(exact=false)): the stages of the exact path (ME pictures to
// local fallback, picture build) spread over all cores and, on CPUs with AVX2 and FMA, computed 8 floats (or 4
// doubles) at a time. The exact path is the deterministic scalar path: float32 in a fixed evaluation order, no
// contraction, IEEE division and sqrt, exp through double; the fast CPU path and CUDA (exact=true) are checked bit for
// bit against it.
//
// Same algorithm, same constants (splat_const.h, splat::Config) and, wherever it costs nothing, the same float32
// operations in the same order as the exact path, so most stages give the exact path's bits:
// - The ME pictures, the refinement, match errors, assignment, local fallback and the picture build evaluate every
//   sum in the exact path's order. The splats of the build run in parallel by bands of output rows: every source sample is filed under the output row it lands on (stable, in raster
//   order), so each accumulator cell still receives its terms side by side, corner by corner, in raster order.
// - The phase-correlation FFTs are pocketfft's passes computed for 4 (forward, double) or 8 (inverse, float)
//   lines at once, with the same operations per line, so they also give pocketfft's bits.
// - The transcendental functions (exp of the refinement weights, atan2 of the spectrum phases, sin / cos of the
//   cross-power spectrum) are double-precision polynomials rounded to float32 (see splat_fast_avx2.h): the result
//   equals the exact path's except where the double value falls within about 1e-15 of a float32 rounding boundary,
//   i.e. for a few values in 10^8. Such a difference can flip a phase-correlation peak where two surface values tie
//   to the last bit, so the candidate vectors (and through them a few output samples) may differ in rare places.
// Without AVX2 / FMA (or with simd = false) the stages without a portable parallel version run the exact functions,
// the others (picture build, sample conversion) run their portable code on all threads; the results are then those of
// the exact path.
#pragma once

#include "splat_core.h"
#include "splat_me.h"
#include "splat_motion.h"
#include "splat_pcorr.h"

#include <functional>
#include <memory>
#include <vector>

namespace splat
{
  // One engine serves one caller thread at a time (it keeps buffers between calls).
  // The Laplacian pyramid the build makes from one source picture: band[i] for the levels 0 .. L (band[L] = the low-pass
  // rest), padded as the build pads. It depends on the picture and the settings only, so a picture that is P or N of
  // several frames needs it once (FastEngine::pyramid_into, then build_pyr_into).
  struct Pyramid
  {
    std::vector<Pic3> band;
    bool valid = false;
  };

  class FastEngine
  {
  public:
    // threads: total threads (0: all hardware threads); simd: use the AVX2 kernels when the CPU has AVX2 and FMA
    explicit FastEngine(int threads = 0, bool simd = true);
    ~FastEngine();
    FastEngine(const FastEngine &) = delete;
    FastEngine &operator=(const FastEngine &) = delete;

    // AVX2 and FMA usable (CPU and operating system)
    static bool cpu_has_avx2_fma();
    bool simd() const;
    int threads() const;
    // fn(y0, y1) over row ranges of [0, n) on the engine's threads
    void rows(int n, const std::function<void(int, int)> &fn);

    // the stages, with the interfaces of the exact functions of the same name
    Plane me_picture(const Plane &y, MeFir fir, const Config &cfg);                              // ME picture
    PcorrPicture pcorr_ingest(const Plane &B, const Config &cfg);                                // phase correlation
    std::vector<Plane> pcorr_candidates(const PcorrPicture &prev, const PcorrPicture &cur, const Config &cfg,
      PcorrGlobal *glob = nullptr);
    // the same into o / out, reusing their memory where the sizes match
    void pcorr_ingest_into(const Plane &B, const Config &cfg, PcorrPicture &o);
    void pcorr_candidates_into(const PcorrPicture &prev, const PcorrPicture &cur, const Config &cfg,
      std::vector<Plane> &out, PcorrGlobal *glob = nullptr);
    VecSet refine(const Plane &cur, const Plane &other, const VecSet &cands, bool neg, const Config &cfg); // refinement
    std::vector<Plane> errors(const Plane &cur, const Plane &other, const VecSet &vecs, const Config &cfg); // match errors
    void assign(const VecSet &rf, const std::vector<Plane> &ef, const VecSet &rb, const std::vector<Plane> &eb,
      const float *bias, float lam, float thresh, float ratio, float lenc, Field4 &fwd, Field4 &bwd);                    // assignment
    Plane disorder_weight(const Plane &vx, const Plane &vy, const Plane *err, const Config &cfg);
    void fields(const Plane *prv, const Plane &cur, const Plane &nxt, const CandSet &cf, const CandSet &cb,
      const float *bias, bool local_fb, const Config &cfg, Field4 &fwd, Field4 &bwd);           // refinement to local fallback
    Pic3 build(const Pic3 &P, const Pic3 &N, const Field4 &MEP, const Field4 &MEN, float d, const Config &cfg); // build
    // build into out, reusing its memory when it already has the size
    // the same in two steps: the pyramid of a picture, then the build from the pyramids of P and N
    void pyramid_into(const Pic3 &pic, const Config &cfg, Pyramid &out);
    void build_pyr_into(const Pyramid &P, const Pyramid &N, const Field4 &MEP, const Field4 &MEN, float d,
      const Config &cfg, Pic3 &out);
    void build_into(const Pic3 &P, const Pic3 &N, const Field4 &MEP, const Field4 &MEN, float d, const Config &cfg,
      Pic3 &out);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
  };
}
