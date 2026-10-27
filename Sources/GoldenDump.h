// Golden-data dump for the GPU port (GPU_PLAN.md, milestone M1).
//
// Compiled in only when MV_GOLDEN_DUMP is defined; without it every hook below is an empty macro, so the
// normal build is unchanged. At run time nothing is written unless the environment variable MVDUMP_DIR is set.
//
//   MVDUMP_DIR     output directory (must exist)
//   MVDUMP_FRAMES  comma separated source frame numbers to dump, e.g. "10,50" (default: none)
//   MVDUMP_PLANES  1 = also write the raw super planes (large: about 150 MB per UHD frame and level set),
//                  otherwise only an FNV-1a hash per plane is logged
//
// One directory per call: <MVDUMP_DIR>\<MAnalyse|MRecalculate>_n<frame>_<b|f>\ with log.txt (parameters, per-level
// values, hashes), vec_in.bin (MRecalculate only), vec_out.bin (the complete vector frame payload as the filter
// returns it) and, with MVDUMP_PLANES=1, src_L<level>_<Y|U|V>_<subpel>.bin / ref_...
// Reference runs must use mt=false (single slice) so the result does not depend on the thread count.

#ifndef __MV_GOLDEN_DUMP__
#define __MV_GOLDEN_DUMP__

#ifdef MV_GOLDEN_DUMP

#include <cstddef>

class MVGroupOfFrames;

namespace golden
{
  bool enabled();
  bool want_frame(int nsrc);

  // starts a dump context for this thread; returns false (and logs nothing later) when the frame is not selected
  bool begin(const char *filter, int nsrc, bool backward);
  void end();
  bool active();

  void logf(const char *fmt, ...);
  void write_bin(const char *name, const void *data, size_t bytes);

  // levels 0..nlevels-1 of both groups, Y plane and, if chroma, U and V
  void dump_gof(const char *prefix, MVGroupOfFrames &gof, int nlevels, bool chroma);
}

#define GOLDEN_LOG(...) do { if (golden::active()) golden::logf(__VA_ARGS__); } while (0)

#else

#define GOLDEN_LOG(...) do { } while (0)

#endif // MV_GOLDEN_DUMP

#endif // __MV_GOLDEN_DUMP__
