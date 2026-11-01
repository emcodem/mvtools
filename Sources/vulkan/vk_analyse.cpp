// mvtools_vulkan: MAnalyse's hierarchical search on the device, port of Sources/cuda/mvcuda_analyse.cu (bit for bit).
//
// Per level, coarse to fine, in one command buffer: ana_global (global MV estimate from the coarser level, doubled),
// ana_interp (InterpolatePrediction into the level's vectors), then the search:
//   meander=true (or trymany): ana_search, one workgroup per CPU slice (slice_threads as the CPU's thread count); a block
//     that would take the CPU's "bad vector" wide search sets a flag and the call returns MVCUDA_NEEDS_CPU, as CUDA's
//     k_search;
//   meander=false: ana_wave, the wavefront of CUDA's k_wave (CPU result of meander=false, mt=false, bad vectors included),
//     one dispatch per level. Its inter-workgroup waits are bounded (MVVK_SPIN_MAX polls); a timeout (no forward progress
//     on this device) makes the call return -4 and the caller uses the CPU. MVVK_WAVE=0 forces ana_search for it
//     (comparisons only: bad blocks then give MVCUDA_NEEDS_CPU). MVVK_WAVE_GROUPS caps the workgroups of a level.
//
// nLambda (the CPU's double arithmetic) is computed exactly with integers in the shaders, so shaderFloat64 is not needed.

#include "vk_internal.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace
{
  // shaders/ana_level.glsl Args
  struct LevelArgs
  {
    uint64_t ref[3][MVCUDA_MAX_SUB];
    uint64_t src[3];
    uint64_t vec, glob, state, nlam;
    int ew[3], hpad[3], vpad[3];
    int n_blk_x, n_blk_y, pel, log_pel, log_scale, smallest;
    int bs_x, bs_y, ov_x, ov_y, cw, ch, log_x_uv, log_y_uv;
    int ext_w, ext_h, lambda_level, lsad, pnew, pzero, pglobal, bits, chroma_scale, radius;
    int meander, n_slices, search_type, try_many, bad_sad, bad_range, spin_max, pad_;
    int chunk_rows, chunk, pad2_, pad3_;
  };

  struct InterpArgs
  {
    uint64_t coarse, vec, nlam;
    int n_blk_x, n_blk_y, c_blk_x, c_blk_y;
    int bs_x, bs_y, ov_x, ov_y, log_pel, log_pel_coarse;
    int lambda_level, lsad, n_slices, pad_;
  };

  struct GlobalArgs
  {
    uint64_t vec, freq, glob;
    int n_blk, use_global;
  };

  enum { S_ONETIME = 1, S_NSTEP = 2, S_LOGARITHMIC = 4, S_EXHAUSTIVE = 8, S_HEX2 = 16, S_UMH = 32, S_HORIZONTAL = 64, S_VERTICAL = 128 };

  bool supported_type(int t)
  {
    return t == S_ONETIME || t == S_NSTEP || t == S_LOGARITHMIC || t == S_EXHAUSTIVE || t == S_HEX2 || t == S_UMH
      || t == S_HORIZONTAL || t == S_VERTICAL;
  }
  int max_param(int t) { return t == S_EXHAUSTIVE ? 64 : 24; }

  int ilog2(int v)
  {
    int l = 0;
    while ((1 << l) < v)
      l++;
    return l;
  }

  int env_int(const char *name, int def)
  {
    const char *e = getenv(name);
    return (e && *e) ? atoi(e) : def;
  }

  const int FREQ_WORDS = 2 * 2 * 8192;

  int analyse_dev(const MvCudaSuperParams *sp, const MvCudaAnalyseParams *ap, const std::shared_ptr<mvk::DevFrame> &fs,
    const std::shared_ptr<mvk::DevFrame> &fr, int *payload, int payload_words)
  {
    if (!sp || !ap || !fs || !fr || !payload)
      return -4;
    const int esize = sp->bits_per_pixel <= 8 ? 1 : 2;
    const int user_type = ap->search_type ? ap->search_type : S_EXHAUSTIVE;
    if (sp->levels != ap->n_levels || ap->n_levels < 1 || ap->n_levels > MVCUDA_MAX_LEVELS
      || ap->search_param < 1 || ap->pel_search < 1 || ap->n_blk_x < 1 || !supported_type(user_type)
      || ap->n_blk_y < 1 || ap->slice_threads < 1 || ap->plevel < 0 || ap->plevel > 2 || ap->blk_x > 32)
      return -4;
    // the integer nLambda of the shaders assumes non-negative lambda and lsad (MAnalyse never makes others)
    if (ap->lambda < 0 || ap->lsad < 0)
      return -4;
    const bool horv = user_type == S_HORIZONTAL || user_type == S_VERTICAL;
    int lvl_type[MVCUDA_MAX_LEVELS];
    for (int l = 0; l < ap->n_levels; l++)
    {
      lvl_type[l] = (l == 0 || horv) ? user_type : S_EXHAUSTIVE;
      if ((l == 0 ? ap->pel_search : ap->search_param) > max_param(lvl_type[l]))
        return -4;
    }
    if (!vkrt::ready())
      return -1;

    const int nl = ap->n_levels;
    int nbx[MVCUDA_MAX_LEVELS], nby[MVCUDA_MAX_LEVELS];
    const int w_b = (ap->blk_x - ap->overlap_x) * ap->n_blk_x + ap->overlap_x;
    const int h_b = (ap->blk_y - ap->overlap_y) * ap->n_blk_y + ap->overlap_y;
    int words = 2;
    int offs[MVCUDA_MAX_LEVELS]; // word offset of the first triple of each level inside the payload
    for (int l = nl - 1; l >= 0; l--)
    {
      nbx[l] = ((w_b >> l) - ap->overlap_x) / (ap->blk_x - ap->overlap_x);
      nby[l] = ((h_b >> l) - ap->overlap_y) / (ap->blk_y - ap->overlap_y);
      if (nbx[l] < 1 || nby[l] < 1)
        return -4;
      offs[l] = words + 1;
      words += 1 + nbx[l] * nby[l] * 3;
    }
    if (payload_words < words)
      return -4;
    static const bool wave_allowed = env_int("MVVK_WAVE", 1) != 0;
    const bool wave = !ap->meander && wave_allowed && !ap->try_many;
    static const int spin_max = env_int("MVVK_SPIN_MAX", 1 << 22);
    static const int wave_groups = env_int("MVVK_WAVE_GROUPS", 256);
    static const int chunk_blocks = env_int("MVVK_CHUNK_BLOCKS", 1024);

    const char *sfx = esize == 1 ? "_8" : "_16";
    char kname[32];
    std::snprintf(kname, sizeof(kname), "%s%s", wave ? "ana_wave" : "ana_search", sfx);
    // compile-time row counts of the luma / chroma block (shaders/ana_common.glsl SC_BSY, SC_CHH)
    const int sc_ch = sp->chroma ? ap->blk_y / sp->y_ratio_uv : 0;
    const bool sc_ok = ap->blk_y <= 32 && sc_ch <= 32;
    vkrt::Kernel *k_search = vkrt::kernel(kname, { sc_ok ? ap->blk_y : 0, sc_ok ? sc_ch : 0 }), *k_interp = vkrt::kernel("ana_interp"), *k_global = vkrt::kernel("ana_global");
    if (!k_search || !k_interp || !k_global)
      return -2;

    const bool timing = getenv("MVCUDA_TIMING") != nullptr;
    auto t0 = std::chrono::steady_clock::now();

    std::shared_ptr<mvk::DevSuper> s_src, s_ref;
    int rc = mvk::frame_super(fs, *sp, s_src);
    if (!rc)
      rc = mvk::frame_super(fr, *sp, s_ref);
    if (rc)
      return rc;
    auto t1 = std::chrono::steady_clock::now();

    // one buffer: [payload][freq scratch][glob: 2 per level][per level state: bad, abort, ticket, progress per row, bad
    // blocks per row][per level nLambda per block]
    int st_off[MVCUDA_MAX_LEVELS], st_words = 0, nl_off[MVCUDA_MAX_LEVELS], nl_words = 0;
    for (int l = 0; l < nl; l++)
    {
      st_off[l] = st_words;
      st_words += 3 + 2 * nby[l];
      nl_off[l] = nl_words;
      nl_words += nbx[l] * nby[l];
    }
    const size_t w_freq = (size_t)words, w_glob = w_freq + FREQ_WORDS, w_state = w_glob + 2 * MVCUDA_MAX_LEVELS;
    const size_t w_nlam = w_state + st_words, total_words = w_nlam + nl_words;
    vkrt::DevBuf buf((total_words * sizeof(int) + 3) & ~(size_t)3);
    if (!buf)
      return -2;
    const vkrt::dptr d_pay = buf.p, d_freq = buf.p + w_freq * 4, d_glob = buf.p + w_glob * 4, d_state = buf.p + w_state * 4;

    vkrt::Cmd cmd;
    cmd.fill(buf.p, total_words * sizeof(int), 0);
    for (int l = nl - 1; l >= 0; l--)
    {
      LevelArgs a = {};
      a.n_blk_x = nbx[l];
      a.n_blk_y = nby[l];
      a.pel = (l == 0) ? sp->pel : 1;
      a.log_pel = ilog2(a.pel);
      a.log_scale = l;
      a.smallest = (l == nl - 1) ? 1 : 0;
      a.bs_x = ap->blk_x;
      a.bs_y = ap->blk_y;
      a.ov_x = ap->overlap_x;
      a.ov_y = ap->overlap_y;
      const int has_chroma = sp->chroma ? 1 : 0;
      a.search_type = lvl_type[l];
      a.log_x_uv = has_chroma ? ilog2(sp->x_ratio_uv) : 0;
      a.log_y_uv = has_chroma ? ilog2(sp->y_ratio_uv) : 0;
      a.cw = has_chroma ? a.bs_x >> a.log_x_uv : 0;
      a.ch = has_chroma ? a.bs_y >> a.log_y_uv : 0;
      for (int p = 0; p < (has_chroma ? 3 : 1); p++)
      {
        const mvk::DevPlane &dr = s_ref->pl[l * 3 + p];
        a.src[p] = s_src->pl[l * 3 + p].sub[0];
        for (int i = 0; i < MVCUDA_MAX_SUB; i++)
          a.ref[p][i] = dr.sub[i];
        a.ew[p] = dr.d.ew;
        a.hpad[p] = dr.d.hpad;
        a.vpad[p] = dr.d.vpad;
      }
      a.ext_w = s_ref->pl[l * 3].d.ew;
      a.ext_h = s_ref->pl[l * 3].d.eh;
      const int n_scale = 1 << l;
      a.lambda_level = ap->lambda / (a.pel * a.pel);
      if (ap->plevel == 1)
        a.lambda_level *= n_scale;
      else if (ap->plevel == 2)
        a.lambda_level *= n_scale * n_scale;
      a.lsad = ap->lsad;
      a.pnew = ap->pnew;
      a.pzero = ap->pzero;
      a.pglobal = ap->use_global ? ap->pglobal : ap->pzero;
      a.bits = sp->bits_per_pixel;
      a.chroma_scale = ap->chroma_sad_scale;
      a.radius = (l == 0) ? ap->pel_search : ap->search_param;
      a.try_many = (ap->try_many && l > 0) ? 1 : 0; // GroupOfPlanes::SearchMVs: never at the finest level
      a.meander = ap->meander;
      a.bad_sad = ap->bad_sad;
      a.bad_range = ap->bad_range;
      a.spin_max = spin_max;
      a.glob = d_glob + 8 * l;
      a.vec = d_pay + 4 * (size_t)offs[l];
      a.state = d_state + 4 * (size_t)st_off[l];
      a.nlam = buf.p + 4 * (w_nlam + (size_t)nl_off[l]);
      int slices = ap->meander ? ap->slice_threads : 1; // meander=false: the single-slice result, as the wavefront
      if (slices > 64)
        slices = 64;
      if (slices > a.n_blk_y / 4)
        slices = a.n_blk_y / 4;
      a.n_slices = slices < 1 ? 1 : slices;

      if (l < nl - 1)
      {
        GlobalArgs g = {};
        g.vec = d_pay + 4 * (size_t)offs[l + 1];
        g.freq = d_freq;
        g.glob = a.glob;
        g.n_blk = nbx[l + 1] * nby[l + 1];
        g.use_global = ap->use_global;
        cmd.dispatch(k_global, 1, 1, 1, g);
        InterpArgs ia = {};
        ia.coarse = g.vec;
        ia.vec = a.vec;
        ia.n_blk_x = nbx[l];
        ia.n_blk_y = nby[l];
        ia.c_blk_x = nbx[l + 1];
        ia.c_blk_y = nby[l + 1];
        ia.bs_x = a.bs_x;
        ia.bs_y = a.bs_y;
        ia.ov_x = a.ov_x;
        ia.ov_y = a.ov_y;
        ia.log_pel = a.log_pel;
        ia.log_pel_coarse = 0;
        ia.nlam = a.nlam;
        ia.lambda_level = a.lambda_level;
        ia.lsad = a.lsad;
        ia.n_slices = a.n_slices;
        cmd.dispatch(k_interp, (uint32_t)((nbx[l] * nby[l] + 127) / 128), 1, 1, ia);
      }
      // (the smallest level: glob and vectors are zero from the fill)
      if (wave)
      {
        int groups = a.n_blk_y < wave_groups ? a.n_blk_y : wave_groups;
        if (groups < 1)
          groups = 1;
        cmd.dispatch(k_search, (uint32_t)groups, 1, 1, a);
      }
      else
      {
        // the slices in chunks of rows (about chunk_blocks blocks per slice and dispatch): a single-slice search of a big
        // frame is one long-running workgroup, which a GPU that preempts only between workgroups (and the OS watchdog)
        // does not tolerate
        const int max_rows = (a.n_blk_y + a.n_slices - 1) / a.n_slices;
        int rows = chunk_blocks / a.n_blk_x;
        rows = rows < 1 ? 1 : (rows > max_rows ? max_rows : rows);
        a.chunk_rows = rows;
        for (int k = 0; k * rows < max_rows; k++)
        {
          a.chunk = k;
          cmd.dispatch(k_search, (uint32_t)a.n_slices, 1, 1, a);
        }
      }
    }
    std::vector<int> state(st_words > 0 ? st_words : 1);
    cmd.download(payload, d_pay, (size_t)words * sizeof(int));
    cmd.download(state.data(), d_state, (size_t)st_words * sizeof(int));
    rc = cmd.submit_wait();
    if (rc)
      return rc;
    // the payload header
    payload[0] = words;
    payload[1] = 1;
    for (int l = 0; l < nl; l++)
      payload[offs[l] - 1] = nbx[l] * nby[l] * 3 + 1;
    bool bad = false, aborted = false;
    for (int l = 0; l < nl; l++)
    {
      bad = bad || state[st_off[l] + 0] != 0;
      aborted = aborted || state[st_off[l] + 1] != 0;
    }
    if (timing)
    {
      auto t2 = std::chrono::steady_clock::now();
      fprintf(stderr, "mvvk analyse: supers %7.2f ms, search %7.2f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count(),
        std::chrono::duration<double, std::milli>(t2 - t1).count());
    }
    rc = aborted ? -4 : (bad ? MVCUDA_NEEDS_CPU : 0);
    if (vkrt::log_enabled())
      fprintf(stderr, "mvvk analyse: %dx%d blocks, %d levels, %s, type %d, rc %d%s\n", ap->n_blk_x, ap->n_blk_y, nl,
        wave ? "wave" : "slices", user_type, rc, aborted ? " (wave wait timed out)" : "");
    return rc;
  }
}

extern "C" int mvcuda_analyse_host(const MvCudaSuperParams *sp, const MvCudaAnalyseParams *ap, const void *const src[3], const int src_pitch_bytes[3], const void *const ref[3], const int ref_pitch_bytes[3], int *payload, int payload_words)
{
  if (!sp || !ap || !src || !ref || !src_pitch_bytes || !ref_pitch_bytes || !payload)
    return -4;
  if (!vkrt::ready())
    return -1;
  const int esize = sp->bits_per_pixel <= 8 ? 1 : 2;
  auto fs = mvk::upload_picture(*sp, esize, src, src_pitch_bytes);
  auto fr = mvk::upload_picture(*sp, esize, ref, ref_pitch_bytes);
  if (!fs || !fr)
    return -4;
  return analyse_dev(sp, ap, fs, fr, payload, payload_words);
}

extern "C" int mvcuda_analyse_token(const MvCudaSuperParams *sp, const MvCudaAnalyseParams *ap, unsigned long long src_token, unsigned long long ref_token, int *payload, int payload_words)
{
  if (!sp || !ap || !payload)
    return -4;
  if (!vkrt::ready())
    return -1;
  auto fs = mvk::frame_get(src_token), fr = mvk::frame_get(ref_token);
  if (!fs || !fr)
    return MVCUDA_TOKEN_GONE;
  const int esize = sp->bits_per_pixel <= 8 ? 1 : 2;
  if (!mvk::super_fits(*fs, *sp, esize) || !mvk::super_fits(*fr, *sp, esize))
    return -4;
  return analyse_dev(sp, ap, fs, fr, payload, payload_words);
}
