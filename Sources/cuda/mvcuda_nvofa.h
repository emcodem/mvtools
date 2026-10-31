// Experiment (opticalflow.md): MAnalyse's block search replaced by NVIDIA's optical flow accelerator (NVOFA, Optical
// Flow SDK through nvofapi64.dll of the driver). Off unless MVCUDA_NVOFA=fast|medium|slow; NOT bit-exact with the CPU
// search (other algorithm), so the golden / md5 tests only hold with it off.
#ifndef MVCUDA_NVOFA_H
#define MVCUDA_NVOFA_H

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace mvc_nvofa
{
  // NVOFA perf level from MVCUDA_NVOFA (read once): 0 = off, else NV_OF_PERF_LEVEL_FAST / MEDIUM / SLOW.
  int perf_level();

  // One NVOFA session with its buffers for one picture size. in[0] = input frame (MVTools source), in[1] = reference
  // frame; 8-bit grayscale, in_pitch bytes per row. flow: one int16 (x, y) pair per grid x grid cell, S10.5 (1/32 pixel),
  // pointing from the input frame into the reference frame; flow_pitch in bytes; gw x gh cells.
  struct Session
  {
    uint8_t *in[2];
    size_t in_pitch;
    const short2 *flow;
    size_t flow_pitch;
    int width, height, grid, gw, gh;
    void *priv;
  };

  // A free session of that size from the pool (made on first use). Call from a thread with the CUDA device current.
  // nullptr when NVOFA is not available (message on stderr once).
  Session *acquire(int width, int height);
  // Queues the flow of in[0] -> in[1] on stream st (after whatever is queued on st before). 0 or an error.
  int execute(Session *s, cudaStream_t st);
  // Back to the pool; the caller has waited for the stream of execute.
  void release(Session *s);
}

#endif
