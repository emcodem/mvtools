# Why the Vulkan backend is a dead end (development stopped 2026-11-01)

Branch: `gpu_portable` (archive, pushed 2026-11-01). Base: `gpu_work` 4843b29. The non-Vulkan parts are on `gpu_work`
since f9553c5 (563ad05 + f9553c5).

## Decision

Maintaining two GPU backends (CUDA and Vulkan) is not wanted. Vulkan would only have replaced CUDA if it reached about
CUDA's speed on NVIDIA. It does not: on an RTX 5090 it is at about **55 % of CUDA's speed** for QTGMC, and no measured
cause of the gap can be fixed for a large gain. Development of `mvtools_vulkan` is stopped. The branch stays as an archive;
the parts that do not depend on Vulkan are listed at the end for porting to `gpu_work`.

## What the Vulkan backend achieved

`mvtools_vulkan` exports the full C API of `mvtools_cuda` (API 18) except NVOFA, and its results are bit-identical to CUDA
and the CPU (status in `Sources/vulkan/PORTING.md`):

- golden tests (super, analyse, recalc, flow): 85/85 on the RTX 5090 and on an Intel iGPU (no float64)
- C API cross-checks against mvtools_cuda (about 520 random configurations): identical
- cudafilters regression with every stage on Vulkan: 65/65
- QTGMC Fast / Slower, 1500 PAL frames: md5 = CUDA = CPU fallback
- VapourSynth vs AviSynth+: 44/44, Windows and Linux

So the port is functionally complete. Speed is the only reason to stop.

## Speed

Test: QTGMC Fast, 1500 frames PAL 720x576 TFF 4:2:2 MPEG-2 (`C:\temp\mxf.js\mpeg2_allintra_50m_pal_60s_hard.mxf`),
`profile\vk_int\tb_Fast\gpu_all.avs`, ffmpeg `-f null`, `rtime`.

| Backend | Time | Speed vs CPU (6.9 s, 24 threads) | Speed vs CUDA |
|---|---|---|---|
| CUDA, RTX 5090 | 1.88 s | ~365 % | 100 % |
| Vulkan, RTX 5090 (with the fix of experiment 2 below) | ~3.4 s | ~205 % | ~55 % |
| Vulkan, RTX 5090 (committed state) | 3.53 s | ~195 % | ~53 % |
| Vulkan, Intel iGPU (Device 1) | 123.8 s | ~6 % | ~1.5 % |

The Intel iGPU is 18x slower than the CPU path: integrated GPUs are useless for this workload, so a Vulkan backend would
only matter for discrete AMD / Intel Arc cards (never tested, no hardware) or for NVIDIA without CUDA.

## Where the time goes

Measured with Nsight Systems (CUDA) and with GPU timestamps built into the Vulkan runtime (`MVVK_GPUTIME=1`, see Tools;
Nsight's Vulkan trace needs admin rights to register its layer and fails on this machine).

**GPU busy share, Prefetch 12:** CUDA 88 %, Vulkan 92 %. The GPU is not idle on Vulkan, so the synchronous call model
(every Vulkan call waits for its fence; PORTING.md's planned fix was timeline semaphores) is **not** the cause of the gap.

**GPU time per stage, Prefetch 1 (no concurrency), 300 frames:**

| Stage | CUDA | Vulkan | Ratio |
|---|---|---|---|
| MAnalyse wavefront search (`k_wave` / `ana_wave` + `ana_global` + `ana_interp`) | 1257 ms (1.05 ms per analysis) | 2561 ms (2.13 ms per analysis) | **2.0x** |
| all other stages together | about 200 ms | about 250 ms | 1.25x |

The whole gap is the MAnalyse wavefront kernel (`meander=false`). If it matched `k_wave`, Vulkan QTGMC Fast would take
about 1.9-2.1 s.

## Experiments on the wavefront kernel

All on the RTX 5090, output checked byte-identical (golden tests, plus QTGMC framemd5 against the reference).

| # | Change | ana_wave per dispatch (Prefetch 1) | QTGMC Fast P12 | Result |
|---|---|---|---|---|
| 0 | committed state | 345 us | 3.53 s | |
| 1 | inter-workgroup reads with GLSL `coherent volatile` loads instead of `atomicOr(x, 0)` | - | (all waits time out) | **broken**, see Incident |
| 2 | Vulkan memory model: `atomicLoad` acquire / `atomicStore` release (CUDA's `ld.acquire` / `st.release`), abort flag polled every 64 polls | **309 us (-10 %)** | 3.32-3.48 s | kept (in the last commit of this branch) |
| 3 | 2 + SAD column sums with `subgroupClusteredAdd` (16 / 8) instead of one shared `atomicAdd` per column (`ana_common.glsl` `eval_batch`) | 379 us (+23 %) | 3.95-4.12 s | **slower**, and wrong on the Intel iGPU; reverted |

Intel iGPU, experiment 2: 2547 -> 2479 us per dispatch (-3 %).

Notes:

- Experiment 1: in GLSL without the Vulkan memory model, `coherent volatile` does not guarantee that a load sees another
  workgroup's store; on NVIDIA the progress counters were never seen, every wait ran into `MVVK_SPIN_MAX`, and the frame
  went to the CPU. The output was still correct (exact CPU fallback), only 2x slower: a silent fallback hides a broken
  GPU path. Since then a timed-out wait is always printed to stderr (`vk_analyse.cpp`).
- Experiment 3: NVIDIA's shared atomics with 16 lanes per address are cheap; the clustered add makes every lane run
  4 shuffle steps twice. On Intel the clustered add assumes invocation i is lane i mod size of a fixed-size subgroup,
  which Intel's driver only guarantees with `VK_EXT_subgroup_size_control` (required size, full subgroups); without it
  the columns of different candidates were summed together (golden: analyse rc=1).

## Why the remaining steps would not close the gap

| Candidate | Evidence | Expected gain |
|---|---|---|
| Asynchronous submission (timeline semaphores) | GPU already 92 % busy | small |
| Atomic hand-off between rows | experiment 2 | 10 % (done) |
| Shared-atomic SAD sums | experiment 3 | none (worse) |
| One dispatch per analysis instead of 6 (one per level, plus 5 x global + interp) | launch / barrier cost of about 11 dispatches out of 1.9 ms | 5-8 % (estimate) |
| Packed SADs (`__vsadu4` equivalent) and next-block prefetch | on CUDA, packed 8-bit SADs gave QTGMC Fast's single k_wave call +2 % (1021 -> 1037 us, GPU_PIPELINE_PLAN.md); GPU_PLAN.md: prefetch / unrolling tricks about 6 % (the kernel is latency-bound on its serial block chain) | small for Fast |

Together these reach roughly 1.6-1.7x of CUDA, not the 1.2x that would justify a second backend. The rest of the gap is
in the per-block instruction stream of the search itself (GLSL -> SPIR-V -> driver code generation, barriers instead of
warp-level operations). Locating it would need a per-phase cycle profile of `ana_wave` with `VK_KHR_shader_clock`, the
counterpart of CUDA's `-DMVC_PHASE_PROF` (about half a day); that diagnoses, it does not fix, and was not done.

## Incident: GPU driver reset (2026-11-01 20:09:50)

The first full-length run of experiment 1 made the wavefront waits time out. With the default `MVVK_SPIN_MAX` of 4 million
polls, a timed-out wait keeps the GPU busy for seconds, longer than the Windows watchdog (TDR, 2 s): the driver reset the
GPU (System log nvlddmkm event 153, 12 times) and the screens went black for a moment. The rest of the run continued on
the CPU (device lost -> CPU fallback, as designed).

Lessons, should anyone pick this up again:

- Every spin wait between workgroups must be bounded in time well below 2 s, not by a poll count: the default
  `MVVK_SPIN_MAX` (1 << 22 polls) can reset the display driver on any GPU where a wait ever fails.
- Test any change to GPU waits with `MVVK_SPIN_MAX=20000` on a 60-frame clip first (`profile\out\vkprof\short_*.avs`);
  20000 polls gave no false timeouts with the committed or the experiment-2 kernel at Prefetch 1 and 12.
- Check `MVVK_LOG=1` / the timeout message, not only the md5: the CPU fallback makes a broken GPU path "pass".

## Tools and files

- `MVVK_GPUTIME=1` (and `MVVK_GPUTIME_CSV=<path>`): GPU timestamps per command buffer and per command in
  `Sources/vulkan/vkrt.cpp`, report at exit (GPU busy union, per-kernel sums, host submit-to-fence time). About 1 %
  overhead. Uncommitted on this branch.
- Results and scripts: `C:\dev\mvtools\profile\out\vkprof` (logs, framemd5 files, nsys reports `nsys_Fast_cuda.*`,
  `nsys_p1_cuda.*`), test DLL folders `profile\vk_gt` (experiment 2) and `profile\vk_base` (committed shader, with timestamps).
- CUDA analysis: `profile\nsys_busy.py` (GPU busy union) on an nsys sqlite export.

## Worth keeping from this branch (not Vulkan-specific)

See the commit list in the answer that accompanied this file, or `git log gpu_work..gpu_portable`:

- `3c2cc87` mvtools_cuda for compute capability 6.0+ (Pascal / GTX 10xx, Volta, Turing) with CUDA 12.x
- `a29cf12` CMake for Windows (MSVC + Ninja) and Linux for all plugins, `build_windows.ps1`, `build_linux.sh`, the
  MinGW toolchain file, vsbridge crash handler and test scripts (minus the `mvtools_vulkan` target)
- the Pascal / CMake parts of the docs commit `74d01ac`
