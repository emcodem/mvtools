// NVIDIA video encoder (NVENC) in motion-estimation-only mode: the session pool behind mvcuda_nvenc_token
// (mvcuda_nvenc_vec.cu, the nvencmv plugin's CudaNVENCAnalyse). Video Codec SDK header third_party/nvenc/nvEncodeAPI.h
// (MIT); nvEncodeAPI64.dll (Linux: libnvidia-encode.so.1) of the driver is loaded at run time.
// SDK: https://developer.nvidia.com/video-codec-sdk
#ifndef MVCUDA_NVENC_H
#define MVCUDA_NVENC_H

#include <cstddef>
#include <cstdint>
#include <string>

namespace mvc_nvenc
{
  // What makes two sessions interchangeable (the encoder is configured once per session).
  struct Config
  {
    int width = 0, height = 0; // ME picture size (even)
    int hevc = 0;              // 0 H.264, 1 HEVC
    int preset = 4;            // P1..P7
    int qp = -1;               // -1: the preset's rate control, else constant QP (the encoder's lambda for vector bits)
    int no_intra = 0;          // H.264: disableIntraSearch
    int p16 = 0;               // H.264: 16x16 partitions only
    int hints = 0;             // external hint candidates per 16x16 macroblock (H.264), 0: none
    int slots = 2;             // NV12 picture buffers of the session
    bool operator==(const Config &o) const
    {
      return width == o.width && height == o.height && hevc == o.hevc && preset == o.preset && qp == o.qp
        && no_intra == o.no_intra && p16 == o.p16 && hints == o.hints && slots == o.slots;
    }
  };

  // One motion vector of the 8x8 grid, quarter pel of the ME picture, from the current into the reference picture.
  struct Mv
  {
    int16_t x, y;
  };

  // An encoder session with `slots` NV12 picture buffers in device memory (luma rows then interleaved chroma rows, pitch
  // bytes per row; the chroma is set to 128 once). Kernels write the luma of slot i at pic[i]; run() reads two slots.
  struct Session
  {
    Config cfg;
    uint8_t *pic[9] = {}; // slots, then the flat picture of reset runs
    size_t pitch = 0;
    int gw = 0, gh = 0; // the 8x8 grid of the output
    void *priv = nullptr;
  };

  // A free session of that configuration (made on first use; at most `pool_cap` sessions open). Call from a worker
  // thread with the CUDA device current, never from the submitter thread (NVENC calls block). nullptr + err on failure.
  Session *acquire(const Config &c, std::string &err);
  // Motion estimation of slot cur into slot ref; the device writes to the slots must have finished. The encoder keeps
  // the previous field as predictors, which makes the result depend on the call order; mode 1: a new encoder for this
  // run; 2: a run on a flat picture first (resets the predictors, keeps the encoder); 0: neither. hints: Config::hints
  // candidates (x, y in integer pel of the ME picture) per 16x16 macroblock in raster order, or null. out: gw x gh
  // vectors. Blocks; 0 or an error (err set).
  int run(Session *s, int cur, int ref, int mode, const Mv *hints, Mv *out, std::string &err);
  void release(Session *s);
  // 1 when the library loads, the driver supports the header's API version and a small ME-only session of that codec
  // can be made, else 0 (reason in err). Device current.
  int available(int hevc, std::string &err);
}

#endif
