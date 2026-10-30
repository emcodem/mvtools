// Shared parts of the cudafilters plugin: the mvtools_cuda.dll loader and the token-frame helpers (see CudaFilters.cpp).
#pragma once

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "avisynth.h"
#include "mvcuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace cf
{
  // mvtools_cuda.dll next to this plugin; the same file mvtools2.dll loads, so both share one frame registry
  struct CudaLib
  {
    bool ok = false;
    decltype(&mvcuda_frame_upload) upload = nullptr;
    decltype(&mvcuda_frame_download) download = nullptr;
    decltype(&mvcuda_producer_register) producer_register = nullptr;
    decltype(&mvcuda_producer_unregister) producer_unregister = nullptr;
    decltype(&mvcuda_token_regenerate) regenerate = nullptr;
    decltype(&mvcuda_frame_stats) stats = nullptr;
    decltype(&mvcuda_resample_program_create) resample_program_create = nullptr;
    decltype(&mvcuda_resample_program_free) resample_program_free = nullptr;
    decltype(&mvcuda_resample_v) resample_v = nullptr;
    decltype(&mvcuda_resample_2d) resample_2d = nullptr;
    decltype(&mvcuda_temporal_soften) temporal_soften = nullptr;
    decltype(&mvcuda_merge) merge = nullptr;
    decltype(&mvcuda_lut2_create) lut2_create = nullptr;
    decltype(&mvcuda_lut2_create_bits) lut2_create_bits = nullptr;
    decltype(&mvcuda_lut2_free) lut2_free = nullptr;
    decltype(&mvcuda_kobs) kobs = nullptr;
    decltype(&mvcuda_removegrain) removegrain = nullptr;
    decltype(&mvcuda_token_pin) pin = nullptr;
    decltype(&mvcuda_token_unpin) unpin = nullptr;
    decltype(&mvcuda_nnedi3_create) nnedi3_create = nullptr;
    decltype(&mvcuda_nnedi3_free) nnedi3_free = nullptr;
    decltype(&mvcuda_nnedi3) nnedi3 = nullptr;
    decltype(&mvcuda_mt) mt = nullptr;
    decltype(&mvcuda_qtgmc_sharp) qtgmc_sharp = nullptr;
    decltype(&mvcuda_qtgmc_backblend) qtgmc_backblend = nullptr;
    decltype(&mvcuda_temporal_binomial) temporal_binomial = nullptr;

    static CudaLib &get();

  private:
    CudaLib();
  };

  inline bool log_enabled()
  {
    static const bool on = std::getenv("CUDAFILTERS_LOG") != nullptr;
    return on;
  }

  // the formats the device frames support: planar Y / YUV, 8..16 bit integer, chroma subsampling 1 or 2, and the
  // header must fit into row 0 of the Y plane
  bool make_format(const VideoInfo &vi, MvCudaFrameFormat &f);

  // The input of a filter's CPU fallback (the exact CPU filter used when the GPU cannot make a frame): c behind
  // CudaDownload, which passes ordinary frames through and turns token frames into pixels (making them again if evicted).
  // A CPU filter reading a token frame directly would compute on the header bytes.
  PClip cpu_input(PClip c, IScriptEnvironment *env);

  // the table of mt_lutxy(x, y, expr, use_expr=...) as the installed masktools computes it for bits 8 or 10 (an n x n
  // pattern through the CPU filter, n = 1 << bits), uploaded; 0 on failure or for other bit depths
  unsigned long long lut_from_cpu(const char *expr, int use_expr, IScriptEnvironment *env, int bits = 8);

  // n x n 4:4:4 8 / 10-bit clip (one frame, n = 1 << bits) whose samples in every plane are their column (by_row = false:
  // the x of a table) or their row (true: y)
  PClip pattern_clip(bool by_row, int bits = 8);
  // uploads the tables read from an n x n frame made from the two pattern clips (table[x * n + y]; luma from Y, chroma from
  // U when the frame has it); 0 on failure
  unsigned long long lut_from_frame(const PVideoFrame &f, int bits = 8);

  // a masktools integer parameter given in 8-bit units (paramscale "i8", the default) at bits: 255 stays the max value,
  // others are shifted up (masktools ScaleParam)
  inline int scale_i8(int v, int bits) { return v >= 255 ? (1 << bits) - 1 : v << (bits - 8); }

  const int g_planes[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };

  bool read_token(const PVideoFrame &f, const VideoInfo &vi, MvCudaTokenHeader &h);
  void write_token(PVideoFrame &dst, const MvCudaTokenHeader &h);
  bool same_format(const MvCudaFrameFormat &a, const MvCudaFrameFormat &b);
  void print_stats(const char *who);

  // The token of src (a frame of format f): src's own token if it is a token frame, else src uploaded now (a
  // transient device frame without producer). 0 on success.
  int input_token(const PVideoFrame &src, const VideoInfo &vi, const MvCudaFrameFormat &f, unsigned long long &token);

  // Runs op(tokens) on the device frames of the input frames: frames that are not tokens are uploaded; when op reports
  // MVCUDA_TOKEN_GONE (an input was evicted from the registry), the token inputs are made again by their producers, the
  // uploaded ones uploaded again, all pinned, and op runs once more (the frames a thread just made stay reachable for it,
  // see mvcuda_frames.cu). Returns op's code.
  template <class Op>
  int with_inputs(const PVideoFrame *frames, int count, const VideoInfo &vi, const MvCudaFrameFormat &f, IScriptEnvironment *env, Op op)
  {
    unsigned long long tok[64];
    MvCudaTokenHeader hdr[64];
    bool is_tok[64];
    if (count > 64)
      return -4;
    for (int i = 0; i < count; i++)
    {
      is_tok[i] = read_token(frames[i], vi, hdr[i]);
      int rc = 0;
      if (is_tok[i])
      {
        if (!same_format(hdr[i].format, f))
          return -4;
        tok[i] = hdr[i].token;
      }
      else
        rc = input_token(frames[i], vi, f, tok[i]);
      if (rc)
        return rc;
    }
    int rc = op(tok);
    if (rc != MVCUDA_TOKEN_GONE)
      return rc;
    // Some input was evicted: pin the inputs still there and make the others again (pinning each right away), so that
    // making one input cannot evict another under a small budget; unpin after the operation.
    CudaLib &lib = CudaLib::get();
    bool pinned[64] = {};
    rc = 0;
    for (int i = 0; i < count && rc == 0; i++)
      if (is_tok[i])
      {
        if (lib.pin(tok[i]) == 0)
        {
          pinned[i] = true;
          continue;
        }
        MvCudaTokenHeader h;
        int rr = -100, pr = -100;
        if (!hdr[i].producer || (rr = lib.regenerate(hdr[i].producer, hdr[i].frame, env, &h)) != 0 || (pr = lib.pin(h.token)) != 0)
        {
          if (log_enabled())
            std::fprintf(stderr, "cudafilters: input %d of %d (producer %llu, frame %d) not made again: regenerate %d, pin %d\n", i, count,
              hdr[i].producer, hdr[i].frame, rr, pr);
          rc = MVCUDA_TOKEN_GONE;
        }
        else
        {
          tok[i] = h.token;
          pinned[i] = true;
        }
      }
    // ordinary frames uploaded above (transient device frames without a producer) may have been evicted by these
    // regenerations: keep them, or upload them again, now, after the token inputs
    for (int i = 0; i < count && rc == 0; i++)
      if (!is_tok[i])
      {
        if (lib.pin(tok[i]) != 0)
        {
          rc = input_token(frames[i], vi, f, tok[i]);
          if (rc == 0 && lib.pin(tok[i]) != 0)
            rc = MVCUDA_TOKEN_GONE;
        }
        pinned[i] = rc == 0;
      }
    if (rc == 0)
    {
      rc = op(tok);
      if (rc == MVCUDA_TOKEN_GONE && log_enabled())
        std::fprintf(stderr, "cudafilters: still gone after making the inputs again (%d inputs)\n", count);
    }
    for (int i = 0; i < count; i++)
      if (pinned[i])
        lib.unpin(tok[i]);
    return rc;
  }
}
