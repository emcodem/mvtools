// splat_core: the exact CPU path of SplatFps (plain C++, float32): the sample I/O (input: codes -> 4:4:4 samples,
// output: 4:4:4 samples -> codes) and the picture build. The motion stages are in splat_pcorr (phase correlation),
// splat_me (ME pictures, match errors, local fallback, schedule, pair statistic) and splat_motion (refinement,
// assignment).
//
// The chain, per output frame between source pictures P and N at phase d (0 = P, 1 = N):
//   input (codes -> 4:4:4 samples), ME pictures, phase correlation (candidate vectors), refinement, match errors,
//   forward / backward assignment with occlusion handling, local fallback, schedule (which frames are copies and
//   which are built), pair statistic (scene cut and temporal fallback, may adjust d), build, output (samples ->
//   codes).
// Field contract between the motion stages and the pair statistic / build: per source picture c, a forward field
// (vectors from c into c+1) and a backward field (vectors from c into c-1), each 4 float32 planes on the ME grid: vx,
// vy (ME pixels), err (match error), local-fallback weight. Output frame (P, N, d) uses the forward field of P and the
// backward field of N. Keeping a forward and a backward field per input picture and projecting both to the output
// time follows the two-field method of US 6,011,596 (BBC, 1991, https://patents.google.com/patent/US6011596A/en).
//
// Exact path: the deterministic scalar path: float32 in a fixed evaluation order, no contraction, IEEE division and
// sqrt, exp through double; the fast CPU path and CUDA (exact=true) are checked bit for bit against it. Rules for this
// file and its .cpp: no fused multiply-add (the build disables contraction), sums accumulated left to right as
// written, and exp through double (exp32).
//
// Sample domain: float32 in 8-bit units, value = (code - 2^(b-1)) / 2^(b-8) for Y, Cb and Cr of a b-bit clip (10 bit:
// (code - 512) / 4). Pictures are planar 4:4:4.
//
// Picture build, in own words. It is forward splatting (motion-compensated projection): every sample of P is pushed
// forward to where it will be at the output time (x + d v_P), every sample of N backward (x + (1 - d) v_N), each
// spread bilinearly over the 4 nearest output samples with a confidence weight, and the output is the weighted mean
// of what landed. This is done separately for every band of a Laplacian pyramid (P. J. Burt, E. H. Adelson, "The
// Laplacian Pyramid as a Compact Image Code", IEEE Trans. Communications, 1983), with a vector field per level, and
// the built bands are summed back into one picture.
// - Pictures are placed in an internal grid with an edge-repeat border of pad_x / pad_y samples.
// - Laplacian pyramid with L levels: low[i+1] = pyr_reduce(low[i]) (separable 7-tap on even samples, vertical first, edge
//   clamp); band[i] = low[i] - pyr_expand(low[i+1]) (2x polyphase); low[L] is the last band.
// - Vector fields come at ME resolution (internal grid / scale), 4 channels: vx, vy (ME pixels), err, local-fallback
//   weight. Per side they become projection fields: vectors scaled by the side's distance to the output time (d for
//   P, 1 - d for N), splat weight = splat_gain * proximity (with splat_err_scale > 0:
//   splat_gain * max(splat_min_weight, exp(-err / splat_err_scale)) * proximity), fallback weight * proximity, where
//   the proximity is the temporal weight of the side (1 - d for P, d for N: the nearer picture counts more, as in the
//   usual motion-compensated interpolation); the ME border (the padding) takes the values of the nearest cell inside
//   the picture. Then up-sampled x scale bilinearly to full resolution (vectors * scale).
// - Level i uses the full-res field subsampled with stride 2^i while 2^i <= stride_max; coarser levels use a 3x3
//   [1 2 1] reduction of the previous coarse field (first one from the ME-res field), vectors / 2.
// - Per level: an accumulator starts with the fallback weights of P and N in place, i.e. the non-motion-compensated
//   blend of the two pictures, used where the local fallback says the field is unreliable (pixels with
//   |v| <= still_vector also put their whole splat weight there, since splatting them would land in place anyway);
//   every other pixel is splatted bilinearly at x + v with its weight; out = sum / weight. Holes (total weight below
//   hole_weight: nothing landed and no fallback weight, e.g. where vectors diverge) get the plain cross-fade
//   (1 - d) P + d N.
// - Reconstruction from coarse to fine: out[i] = built[i] + pyr_expand(out[i+1]), clamped to [-128, 127] at the finest level.
#pragma once

#include "splat_config.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace splat
{
  // Geometry and tuned scalars come from splat::Config (splat_config.h); the description above uses the 1080p
  // defaults (scale 4: ME grid = internal grid / 4, border 16 x 20, L = 4, stride_max = 4). The ME-resolution border of
  // the projection fields is pad_x / scale x pad_y / scale ME pixels (the padding area, where the samples stay in place
  // with zero vectors).

  struct Plane
  {
    int w = 0, h = 0;
    std::vector<float> d;
    Plane() = default;
    Plane(int w_, int h_) : w(w_), h(h_), d((size_t)w_ * h_, 0.0f) {}
    float &at(int x, int y) { return d[(size_t)y * w + x]; }
    float at(int x, int y) const { return d[(size_t)y * w + x]; }
    // edge-clamped read
    float clamped(int x, int y) const
    {
      x = x < 0 ? 0 : (x >= w ? w - 1 : x);
      y = y < 0 ? 0 : (y >= h ? h - 1 : y);
      return at(x, y);
    }
  };

  struct Pic3 { Plane c[3]; };    // Y, Cb, Cr (4:4:4)
  struct Field4 { Plane c[4]; };  // vx, vy, err, local-fallback weight

  // Internal-grid and ME sizes for a W x H picture.
  struct Geometry
  {
    int W, H, GW, GH, MW, MH;
    Geometry(int w, int h, const Config &c)
      : W(w), H(h), GW(w + 2 * c.pad_x), GH(h + 2 * c.pad_y), MW(GW / c.scale), MH(GH / c.scale) {}
  };

  float exp32(float x); // float32 exp, correctly rounded through double

  // ---- input / output: I/O in the sample domain
  float to_val(int code10); // (code10 - 512) / 4
  int to_code(float v); // 512 + trunc(4 v + 0.5), clamped to 0..1023
  // 4:2:2 -> 4:4:4 of one row (half-band chroma interpolation): even outputs keep the input, odd ones (half way
  // between two input samples) a 14-tap symmetric half-band FIR (edge clamp)
  void chroma_upsample_row(const float *in, int w2, float *out);

  // Clip formats: planar YUV, 8 .. 16 bit integer samples (8 bit: one byte, else two), chroma 4:4:4, 4:2:2 or 4:2:0
  // (left chroma siting: horizontally co-sited with the even luma samples; 4:2:0 vertically half way between two
  // luma rows, as in MPEG-2). The output has the format of the input.
  struct IoFormat
  {
    int bits = 10;  // bits per sample
    int sub_w = 1;  // chroma subsampling, log2: horizontal (0 or 1)
    int sub_h = 0;  // vertical (0 or 1; 1 needs sub_w = 1)
    int centre = 0; // subsampled chroma horizontally half way between two luma samples (_ChromaLocation centre,
                    // as JPEG / MPEG-1); 0: co-sited with the even luma samples (left, as MPEG-2)
  };
  // code of a b-bit sample <-> value in 8-bit units: (code - 2^(b-1)) / 2^(b-8). Output: 2^(b-1) + round(v 2^(b-8)),
  // clamped, round half up, so a picture that passes unchanged keeps its codes.
  float to_val(int code, int bits);
  int to_code(float v, int bits);
  // input of rows y0 .. y1 - 1 of a W x H source picture (planes Y, U, V with pitches in bytes) into out (3 planes
  // W x H, allocated by the caller): luma scaled; chroma 4:2:0 -> 4:2:2 vertically (CHROMA_420) and 4:2:2 -> 4:4:4
  // horizontally (chroma_upsample_row) as the format needs. Rows are independent (any split over threads).
  void input_rows(const void *const planes[3], const int pitch[3], const IoFormat &f, int y0, int y1, Pic3 &out);
  // output of luma rows y0 .. y1 - 1 (and the 4:2:0 chroma rows j with y0 <= 2 j < y1) of in (4:4:4) into the planes:
  // 4:2:2 chroma from the even samples; 4:2:0 also vertically half way between rows 2 j and 2 j + 1 (CHROMA_HB pairs).
  void output_rows(const Pic3 &in, const IoFormat &f, int y0, int y1, void *const planes[3], const int pitch[3]);

  // ---- picture build
  // src placed at (px, py) in a gw x gh plane, the border filled by repeating the nearest edge sample
  Plane pad_edge(const Plane &src, int gw, int gh, int px, int py);
  // pyramid reduce: separable 7-tap low-pass (vertical pass first, edge clamp), keeping the even rows / columns
  Plane pyr_reduce(const Plane &src, const float *taps); // -> ceil(w / 2) x ceil(h / 2); taps: Config::pyr_reduce
  // pyramid expand: 2x polyphase interpolation, one 2-D kernel per output phase (even / odd row x even / odd column)
  Plane pyr_expand(const Plane &low, int ow, int oh);    // 2x up, cropped to ow x oh

  // ME field of one side -> its projection field (ME resolution): vectors * distance to the output time, splat weight
  // from the match error, fallback weight, both times the side's temporal proximity; zero vectors on the border
  Field4 border_weight(const Field4 &F, float d, bool n_side, const Config &cfg);
  // x scale bilinear up-sampling of a field to full resolution (vectors * scale, weights unchanged), ow x oh samples
  // (0: scale w x scale h); positions past scale w / scale h (internal grid not a multiple of the scale) use the last
  // ME pixel, as edge clamp. scale: a power of two
  Field4 field_upsample(const Field4 &G, int scale, int ow = 0, int oh = 0);
  // 2x reduction of a field for the coarse pyramid levels: [1 2 1] x [1 2 1] / 16 at even positions, vectors / 2
  Field4 field_reduce2(const Field4 &F);

  struct LevelDebug
  {
    Plane acc_init[4], acc_splat[4];  // 3 sample channels + weight, after the in-place weights / after the splats
  };
  // One pyramid level of the build: in-place weights, bilinear forward splats of both sides, normalisation, hole
  // cross-fade (see the description at the top).
  // P, N: one pyramid band (w x h); FP, FN: full-res fields read with stride, or the level's own field (stride 1)
  Pic3 build_level(const Pic3 &P, const Pic3 &N, const Field4 &FP, const Field4 &FN, int stride, float d,
    const Config &cfg, LevelDebug *dbg = nullptr);

  struct BuildDebug
  {
    std::vector<Pic3> lowP, lowN, bandP, bandN, built, recon;
    Field4 projP, projN, fullP, fullN;
    std::vector<Field4> fieldP, fieldN; // coarse levels only (empty Field4 where the full-res field is used)
    std::vector<int> stride;            // per level: stride into the full-res field, 0 = own coarse field
    std::vector<LevelDebug> levels;
  };
  // The whole build: pad, pyramids of P and N, projection fields, every level built, then reconstructed from
  // coarse to fine and cropped. d is the phase after the temporal fallback.
  // P, N: source pictures W x H; MEP / MEN: ME fields (MW x MH) of P (forward) and N (backward). Returns W x H.
  // Levels, stride_max, border and scale from cfg.
  Pic3 build(const Pic3 &P, const Pic3 &N, const Field4 &MEP, const Field4 &MEN, float d, const Config &cfg,
    BuildDebug *dbg = nullptr);
}
