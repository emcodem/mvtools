# GPU plan: motion analysis on CUDA (branch `gpu-cuda`)

Status 2026-10-27. Supersedes the scoping in `C:\dev\mvtools\handoff_gpu.md` (that file assumed the pixel interpolation
was the cost and the search was hard to parallelise - both turned out different, see "What changed"). Measurements and the
tests to repeat afterwards: `C:\dev\mvtools\profile\RESULTS.md` (section 5), CPU baseline frozen in `profile\baseline_cpu\`.

## Goal

Cut the wall time of the UHD 10-bit `uc_FRC` chain (MSuper, MAnalyse, MRecalculate, MFlowFps) by moving the motion analysis
to the GPU, without making the picture worse and with an automatic CPU fallback. CPU-side wins already taken: threading
(`avstp-only-mt`) and the half-resolution coarse pass (v3, option `uc_frc_coarse_half`, about 45% less wall time).

## What changed since the handoff

1. **Profile:** analysis (`MAnalyse` about 99 CPU-s + `MRecalculate` about 85 CPU-s per 240 frames at 10-bit) is about 45% of the
   chain; interpolation plus upsizing is only about 18%. So the first GPU target is the analysis, not `FlowInter*`.
2. **The search is cheap per block.** The production configuration is `search=3, searchparam=2` = `ExhaustiveSearch` over a
   3x3 window around a predicted vector (`PlaneOfBlocks.cpp`), after `FetchPredictors` (left in scan direction, up,
   up-right / bottom-right, their median, the coarser level's vector, zero, global MV). Per block that is roughly a few
   dozen SADs of 32x32 pixels per level. The work is small; the problem is the dependency chain, not the compute.
3. **The dependency chain is a fixed graph.** Block (x,y) depends only on the predictor blocks above, and with `meander`
   the order is a snake scan. The dependency levels can be precomputed once per geometry (level(b) = 1 + max(level of its
   predictors)); all blocks of one level are independent. For UHD, 32x32 blocks, overlap 16: about 239 x 134 blocks, a few
   hundred dependency levels per search level. A bit-exact GPU port is therefore possible with a wavefront, not only an
   approximate one. (Parallelism per level is about 100 blocks, each block's SADs also parallel - enough for the GPU.)
4. **The CPU code is not bit-identical across thread counts**, by its own comment ("MAnalyze mt-inconsistency reason #1" in
   `FetchPredictors`: the sliced MT version drops the up predictors at every slice boundary). So the reference for
   bit-exact checks is the **single-thread CPU result** (`mt=false`); the production MT result is judged with the quality
   tests (R4-R7), not by byte comparison.
5. **Hardware:** dev machine has an RTX 5090 (cc 12.0, 32 GB), the render node an RTX 5060 Ti (also Blackwell, cc 12.0); the
   same `sm_120` build targets both. Dev CPU is 24 logical cores (the handoff assumed 32).

## Architecture

- `mvtools_cuda.dll`, built with `nvcc` (`Sources/cuda/build.bat`), static CUDA runtime, found next to `mvtools2.dll`
  the way `avstp.dll` is (see `AvstpFinder.cpp` / `AvstpWrapper`). C API in `Sources/cuda/mvcuda.h`; no CUDA types in
  the plugin. The MSBuild CUDA integration is not installed for the VS 18 Build Tools, so the DLL is a separate build step.
- `mvtools2.dll` gets an opt-in `gpu` parameter on `MAnalyse` and `MRecalculate` (default false). With it set, the filter
  calls the DLL and still returns the normal MVTools vector frames, so `MFlowFps`, `MCompensate`, `MDegrain*` stay on the CPU
  unchanged.
- **Fallback is part of the design from the first integration step:** library missing, no device, API version mismatch,
  unsupported format or any CUDA error -> the filter runs the existing CPU code, no error. Checked at filter creation and
  per frame.
- The GPU builds its own super pyramid from the uploaded raw frame (planes at all levels, pel=2 sub-pel planes stay on the
  device), so only the raw frame crosses PCIe (about 25 MB per UHD 10-bit 4:2:2 frame), plus the vectors back (small).
- First supported formats: the production ones - 16-bit-container pixels (10-bit data), 4:2:2 and 4:4:4, chroma on;
  everything else falls back to the CPU. 8-bit and float come later.
- Several filter instances (one per `Prefetch` thread, forward and backward) share one device context and per-instance
  streams.

## Milestones

| # | Work | Exit criterion |
|---|---|---|
| M0 | Toolchain: nvcc build, DLL, block-SAD kernel, self-test (**done**, uncommitted) | `mvcuda_selftest()` = 0, matches the CPU |
| M1 | Golden data: instrumented CPU build (`mt=false`) dumps, for a few frames of the real segments, the effective parameters (lambda, lsad, pnew, pzero, pglobal, plevel, global, levels, overlap...), the super planes at each level, the vectors after each level, the `MRecalculate` result. Loader for the dumps | dumps reproducible; parameter set for the production call written down. **Done 2026-10-27** (seg1 frames 10 and 50; two runs identical; see `profile\golden\README.md`) |
| M2 | GPU `MSuper`: pyramid reduction (`rfilter=4`) and pel=2 sub-pel planes | planes bit-exact vs the golden data. **Done 2026-10-28** for the production case (16-bit container, 10-bit 4:2:2, pel 2, Wiener, rfilter 4): `Sources\cuda\mvcuda_super.cu`, `mvcuda_super_host()`; 30/30 planes (7 levels x Y/U/V + all sub-pel planes) exact for `src` and `ref` of 2 real frames, forward and backward, plus a clipping stress frame; the 12-plane single-level super used by MRecalculate is exact too (`profile\golden\test_super.py`). Not done: pel 4, 8-bit, float, other `rfilter`/`sharp` (all return -4 = CPU fallback), device-resident output (the function copies everything back to host for the test) |
| M3 | GPU search for one level in reference order: predictors, `MotionDistorsion` + SAD + `penaltyNew` + chroma cost with the CPU's tie rule (first strictly lower cost wins), 3x3 exhaustive window, precomputed dependency levels. Then coarse-to-fine over all levels incl. `InterpolatePrediction` and the global MV estimate | vector fields bit-exact vs the single-thread CPU for the golden frames, 8 and 10-bit paths |
|   | **M3 status 2026-10-28: done for the production case** (10-bit 4:2:2, 32x32 / overlap 16, 7 levels, exhaustive radius 2, global MV, meander): `Sources\cuda\mvcuda_analyse.cu`, `mvcuda_analyse_host()`. Bit-exact payload (all 7 levels) vs the **single-thread** CPU with `slice_threads=1`, and vs the **24-thread** CPU with `slice_threads=24`, both directions, 2 frames incl. the clipping stress frame; negative controls differ (19210 blocks). The parallelism is the CPU's own slicing (see "Dependency chain" below); frames that would need the CPU's bad-vector wide search return `MVCUDA_NEEDS_CPU` (-5; verified with a tiny badSAD). Kernel time per frame and direction, 24 slices: 325 ms first version, 60 ms after unrolled SADs + shared source block, **25 ms** after parallel candidate fold and one-thread FP64 nLambda; per block about 14 us. Not done: 8-bit, pel 4, divide, dct, temporal, trymany, fieldShift != 0 |
| M4 | GPU `MRecalculate` (`RecalculateMVs`, `thSAD`, own search). M1 showed its blocks are independent (each starts from the interpolated old vector, no neighbour predictors), so it is plain data-parallel work and could be done **before** M3 | bit-exact vs golden. **Done 2026-10-28** for the production case (16x16 blocks, overlap 8, smooth 1, exhaustive radius 2, 10-bit 4:2:2, pel 2): `Sources\cuda\mvcuda_recalc.cu`, `mvcuda_recalc_host()`; 128851 of 128851 blocks exact for 4 real-frame dumps (forward/backward, 2 frames incl. the clipping stress frame); negative control (smooth=0) differs in 35214 blocks, and 17% of blocks are above thSAD so the search path is exercised (`profile\golden\test_recalc.py`). Cost: kernel 0.7 ms per frame; the two super builds with per-call cudaMalloc and pageable copies take 26 ms in steady state (first call 120-150 ms: context creation), so the integration needs persistent device buffers and pinned memory |
| M5 | Plugin integration: `gpu` parameter, DLL discovery, pinned host buffers, streams, fallback, thread-safety across `Prefetch` instances | R11 (fallback) and R1 pass; output of `gpu=true` equals the CPU `mt=false` output. **Done 2026-10-28** for the 16-bit path: output equals the CPU **multi-threaded** result with the same slice count (240 frames, `Prefetch(3)`), fallbacks identical; pooled memory, streams and a (clip, frame) super-frame cache instead of pinned buffers; see "Status" |
| M6 | Measure: full test suite R2-R12 from `RESULTS.md`, including 4 concurrent jobs and GPU contention | numbers in `RESULTS.md` |
| M7 (only if M6 says so) | Faster, non-bit-exact search order (e.g. start each block from the coarser level only), GPU interpolation/upsizing (`FlowInter*`, `SimpleResize`), fusing stages so data stays on the GPU | judged with R4-R7, not byte comparison |

## Risks and open points

- **Exactness details:** `nLambda` uses a double factor; keep it IEEE-exact on the device (no fused multiply-add contraction in
  those lines). Any SAD/cost path that is float in the CPU code needs the same care. The M1 dumps are what makes this checkable.
- **Hidden state in `MAnalyse`:** global MV estimation, `temporal`/`trymany`, `meander`, `dct`, `divide`, `pelsearch`.
  Only the production parameter set is supported at first; any other combination falls back to the CPU.
- **Per-level kernel launch overhead:** a few hundred dependency levels per search level per frame per direction; use one
  persistent cooperative kernel or CUDA graphs if per-level launches turn out to dominate. Measure before optimising.
- **Many instances on one GPU** (four jobs x `Prefetch(3)` x two directions): VRAM per instance (pel=2 UHD 16-bit super about 200 MB
  per frame pair), stream scheduling and PCIe sharing. R9 must be measured on the CPU first.
- **DLL deployment:** `mvtools_cuda.dll` next to `mvtools2.dll` on every render node; nodes without a capable NVIDIA GPU use the
  CPU path (never confirmed which nodes have one except the ingest node).
- **TDR / driver timeouts** are a new failure mode (R12 soak test).

## Scope gap: 8-bit is the default FRC path

`uc_FRC` runs at 8 bit unless `frc_8bit=false` (`UConvertClip`: `frc_8bit = Default(frc_8bit, true)`), so the production default is the
8-bit (uint8) path, which none of M2-M4 supports yet (they use the 16-bit container). The 8-bit kernels need a pixel-type template and
the 8-bit arithmetic details of the CPU code (`MotionDistorsion` shift 8 instead of 16 - bits, 32-bit penalty products, 32-bit
`InterpolatePrediction`). 4:2:0 and 4:4:4 are covered by the generic chroma ratios but untested.

## Decisions I have made without asking (change if you disagree)

1. Bit-exact reference-order GPU port first, fast order only later (M7).
2. Target is `MAnalyse` + `MRecalculate` first (largest measured cost), `MFlowFps` later.
3. CUDA code in a separate optional DLL, API surface = `gpu` parameter on the existing filters.
4. The bit-exact reference is the single-thread CPU result, because the MT result is already position-dependent.

## Dependency chain (corrects "What changed" item 3)

Item 3 above assumed a few hundred dependency levels per plane. With `meander=true` (default) that is wrong: block (x, y+1) depends on
its right neighbour (left in scan order) and on (x, y), so the true chain is a snake through the whole plane (32026 steps at level 0),
no wavefront parallelism. A bit-exact port of the *single-thread* result is therefore serial (measured: 3.1 s per frame and direction
with one slice). The multi-threaded CPU result is not that chain: the CPU cuts every plane into slices (`min(threads, 64, blkY / 4)`,
at least 1; `y_end = (t + 1) * blkY / n`) that are searched independently, dropping the up predictor / nLambda on the first row and the
bottom-right predictor on the last row of a slice. That result is deterministic for a given thread count (checked: two MT runs identical)
and is what production produces. The GPU reproduces exactly that, with the slice count as a parameter. Consequence for tests: R1 compares
against the CPU run with the same slice count; quality tests (R4-R7) are unaffected by the choice because the CPU already varies with
the core count (more slices = shorter chains = lower GPU latency, slightly more slice boundaries).

## CPU quirks found while porting (GPU must reproduce them)

- **`rfilter` only applies to the first reduction.** `MVGroupOfFrames::set_interp` configures level 0 only, so the level 0 -> 1
  reduction uses `rfilter` (4 = cubic 1 5 10 10 5 1) and every deeper level keeps the `MVPlane` constructor default
  `RB2BilinearFiltered` (1 3 3 1). Found because levels 2-6 were off by 1-2 until this was copied; documented in `mvcuda_super.cu`.
- Level 0 is reduced **before** it is padded (padding follows the reduction); levels >= 1 are padded right after their own
  reduction, so the odd-height step 135 -> 68 reads a padded row. Reproduced.
- The vertical Wiener SSE path writes past the row end for 16-bit (loop bound `nWidth*sizeof(pixel_t)`); the overwritten
  data is recomputed later, so the final planes equal the clean C semantics (checked bit-exact).

## Status 2026-10-28 (M1-M5 done for the production case)

**Done, all bit-exact against the CPU:** M1 golden data, M2 `MSuper`, M3 `MAnalyse` search, M4 `MRecalculate`, M5 plugin integration
(`gpu=true` on `MAnalyse` and `MRecalculate`, `GpuBackend`, per-frame CPU fallback). Scope: 8-bit and 16-bit container (9..16 bit data, both bit-exact, same tests with 8-bit golden dumps incl. a clipping stress frame),
chroma on, pel 1 or 2, `search=3` radius 1-3, block width <= 32, default `MSuper` settings (`rfilter=4`, `sharp=2`, no pelclip). Not covered
(silently CPU): pel 4 / HD, `divide`, `dct`, `temporal`, `trymany`, `outfile`, field shift,
and frames that need the CPU's bad-vector wide search (`MVCUDA_NEEDS_CPU`).

### Use

- Build `Sources\cuda\build.bat` -> `mvtools_cuda.dll` (CUDA 13.x toolkit; machine code for compute capability 8.0, 8.6, 8.9, 9.0, 12.0 plus PTX for 8.0, so RTX 30xx and newer; static runtime: only the driver is needed at run time; the PTX path was checked on the RTX 5090 with `CUDA_FORCE_PTX_JIT=1`, the 30/40 series machine code itself is untested on real hardware; `MVCUDA_DEVICE=n` picks a device, default the first one with cc >= 8.0) and
  the plugin as before; put `mvtools_cuda.dll` next to `mvtools2.dll`.
- `MAnalyse(..., gpu=true)` and `MRecalculate(..., gpu=true)`; `uconvert.avsi`: `global uc_frc_gpu = true` after the Import (default false).
- The vectors equal the CPU **multi-threaded** result for the same slice count: the GPU uses the avstp thread count (24 if mt is on but no thread
  pool exists, 1 if `mt=false`). Environment: `MVTOOLS_GPU_LOG=1` (log every call / fallback reason to stderr), `MVTOOLS_GPU_DISABLE=1`,
  `MVTOOLS_GPU_SLICES=n` (more slices = shorter GPU chains but a different vector field), `MVCUDA_TIMING=1`, `MVCUDA_CACHE_FRAMES=n` (device
  frame cache, default 24 super frames, about 150 MB each at UHD 10-bit).
- Tests: `profile\golden\test_super.py`, `test_recalc.py`, `test_analyse.py` (golden dumps), end to end `profile\avs\golden\e2e*.avs` +
  `profile\run_e2e.ps1` (framemd5 of the final `MFlowFps` frames, `gpu=false` vs `gpu=true`).

### Measurements (this PC, RTX 5090, 24 threads, seg1 UHD 10-bit 4:2:2, High priority)

| | CPU | GPU |
|---|---|---|
| Final frames, 240 frames, `Prefetch(3)` | reference | **identical** (also 60 frames, with/without prefetch; both fallbacks R11 identical to CPU) |
| Wall time, 240 frames, `Prefetch(3)` (3 runs each) | 15.5-16.7 s | 15.2-16.2 s |
| CPU time, same run | 169.6 CPU-s (10.9 cores) | **97.6 CPU-s** (6.4 cores) |
| `MAnalyse` search, one frame and direction | about 8-10 ms wall at 24 threads, 200 CPU-ms | kernels 22 ms (24 slices), 15-16 ms (33 slices); call about 27 ms incl. super |
| `MRecalculate`, one frame | about 85 CPU-s / 240 frames | kernel 0.8 ms, call 3-9 ms incl. uploads |

So the GPU path removes 42% of the CPU work but does **not** make this chain faster in wall time: with `Prefetch(3)` the wall time is the per-frame
latency of the remaining CPU stages (`MFlowFps` upsizing and interpolation, two CPU `MSuper`, decode) plus the GPU call latency; deeper
`Prefetch` makes both variants slower (the cloned filter chains duplicate work). The win is CPU headroom for other jobs (R9, 4 concurrent jobs, not
measured yet) and the base for M7.

### Speed work done and what is left (kernel `k_search` in `mvcuda_analyse.cu`)

325 ms -> 22 ms per frame and direction (24 slices). What worked: unrolled SAD loops with the source block in shared memory (325 -> 60), parallel
(cost, index) fold of the 24 ring candidates and one-thread FP64 `nLambda` (60 -> 25), compile-time row counts + async source copy one block ahead
+ L1 prefetch (25 -> 22). What did not: staging the reference windows in shared memory with plain loads (twice as slow, 46 ms; the staging loop
itself is expensive), more prefetching. Ablation (one frame, ms): without the reference loads 16.7 (so global loads are only about 25%), without
both candidate phases 6.7, floor with barriers and folds about 4.7. The serial chain is ~2300 blocks per frame (L0: 1334 per slice); a block costs
about 21k cycles = 10 us of which SAD evaluation is ~13k (shared-memory issue bound, about 350 instructions per candidate x 31 candidates). Ideas
not done: packed 16-bit SAD (`__vsadu2`, 2 pixels per instruction, needs 4-byte alignment handling), a thread-block cluster per slice (several SMs
share the candidates of one block), early-out of the chroma SAD like the CPU, evaluating bv and fv in one call (they are separate filter
instances), skipping the CPU `MSuper` pyramid when the GPU rebuilds it (about 12 of the 97 remaining CPU-s), sharing results between cloned
`Prefetch` chains (duplicate analysis), GPU `MFlowFps` (M7, the real wall-time limiter now), 8-bit kernels.

### 8-bit result (2026-10-28)

All three GPU stages are bit-exact in 8-bit (kernel files compiled twice, `mvcuda_*_8.cu` / `_16.cu`, shared type-independent core). End to end, 240 frames
`Prefetch(3)`, final frames identical to the CPU. Honest numbers: the 8-bit chain is already fast on the CPU (7.5 s wall, 113 CPU-s); with the GPU it is
8.5 s wall and 63 CPU-s, so again the saving is CPU time, not wall time. The coarsest levels use a plain (not cp.async) source copy because 8-bit plane rows
there are not 4-byte aligned.

Portability notes: thread-block clusters (several SMs per slice) need Hopper (cc 9.0) or newer and would not run on a 3060, so that idea is dropped; packed
SAD (`__vsadu2`) works on every supported card. A 3060 has 28 SMs (24 slices fit), 100 KB shared memory per SM (the kernel uses about 9 KB) and 64K registers
(the kernel is capped at 80 per thread for 768 threads).

### MFlowFps on the GPU (M7, 2026-10-28)

`MFlowFps(..., gpu=true)`: the mask=2 path (`mvcuda_flow.cu`, `Sources\MVFlowFps.cpp gpu_flow`): small vector fields and occlusion masks, the bilinear
upsizing evaluated on the fly (no 220 MB of full-size arrays per frame), and `FlowInterExtra` per pixel through the pel-interleaved layout of the cached super
frames. Bit-exact stage by stage against golden data from the instrumented CPU build (`profile\golden\test_flow.py`: small arrays, all upsized vectors / masks,
final frame) for 10-bit, 8-bit, 4:2:2, 4:2:0, 4:4:4. The kernel renders a UHD frame in about 6 ms. Other mask modes, `blend`, non-planar formats and frames whose
vectors are unusable (scene change) stay on the CPU (the vector-clip state is restored before the CPU path runs). The raw planes come from the super clip's frame
(so the result is right even if the super is not made from the source clip), usually a frame-cache hit on the device.

All stages were also tested in 8-bit, 4:2:0 (10 and 8 bit) and 4:4:4: `python -I profile\golden\run_all.py 10 f8 f420 f420b8 f444` runs 17 checks per format (super src/ref,
analysis single/24 slices, recalc single/24 slices, flow 3 frames): 85 of 85 exact. Plugin end to end (60 frames, `Prefetch(3)`, GPU analysis + recalc + flow vs CPU) identical
for 4:2:0, 4:4:4, 8-bit 4:2:0.

### Measurements, all stages on the GPU (this PC, RTX 5090, seg1 UHD, `Prefetch(3)`, final frames identical to the CPU; CORRECTION: the clip has only 149 source frames, so these runs render about 111-119 output frames, not 240: roughly 138 ms/frame CPU vs 84 ms/frame GPU)

| | CPU | GPU analysis only | **GPU analysis + MFlowFps** |
|---|---|---|---|
| 10-bit (16-bit path) wall / CPU time | 15.0-15.6 s / 168-179 CPU-s | 14.7 s / 98 CPU-s | **9.3-9.5 s / 40-41 CPU-s** |
| 8-bit wall / CPU time | 7.4-7.6 s / 111-116 CPU-s | 8.5 s / 63 CPU-s | **6.7-7.1 s / 26-29 CPU-s** |

What got the last 2 s: the CUDA context in blocking-sync mode (`MVCUDA_SPIN=1` restores spinning): waiting threads no longer burn a core, 11.7 s -> 9.6 s and 61 -> 43 CPU-s.
Per output frame the GPU side costs about 83 ms of latency per prefetch thread (analysis 2 x 27 ms, recalc 2 x 11 ms mostly frame uploads, flow 6 ms); with three
threads that is the wall time. More `Prefetch` threads make it worse (the cloned filter chains duplicate work and CPU time grows: 4 threads 11.0 s, 5 threads 11.5 s, 8 threads 20 s).
What is still on the CPU (40 CPU-s): decode, `RemoveGrain`, the two `MSuper` (about 12-18 CPU-s; the GPU rebuilds the pyramid anyway), vector clip updates and the
host-side vector copies. The profile showed the CPU pyramid + sub-pel build was 36% of the thread time, which led to the lite MSuper below.

### Lite MSuper (`MSuper(gpu=true)`, 2026-10-28)

When the GPU filters downstream can do everything from the plain picture (pel 1/2, sharp 2, rfilter 4, planar YUV with chroma, 8..16 bit, no pelclip, GPU backend usable) MSuper only copies the picture
into a frame of height `nHeight + 2*vpad` and sets bit 6 of the spare byte in the pseudo-audio params (`MV_SUPER_IS_LITE`). MAnalyse / MRecalculate / MFlowFps use the picture on the GPU; whenever
they fall back to the CPU (bad-vector UMH search, field shift, unsupported option, no GPU) they rebuild the full frame with `MVSuper::CompleteLite` (same GroupOfFrames code, bit-identical).
`MVFinest` completes lite frames for pel 2 (and copies the rows for pel 1) when called from MFlowFps; MCompensate, MDegrain*, MBlockFps, MFlow, MFlowBlur, MFlowInter, MShow throw a clear error on a lite super.
`MVTOOLS_GPU_FORCE_FALLBACK=1` forces every GPU call to report "use the CPU" (tests the completion paths).
Checked: final frames identical to the CPU (10-bit 4:2:2 and 8-bit, GPU and forced fallback) and through `UConvertClip` with `uc_frc_gpu` true/false. All-GPU run, 10-bit: 8.6 s vs 10.2-11.4 s without lite on the same day
(8-bit: 7.3 s vs 8.1 s); the machine was slower than at the earlier 9.3 s measurement.

### Prefetch and concurrent jobs with lite MSuper (2026-10-28, ~115 output frames, 10-bit, High priority)

With the CPU pyramid gone, more prefetch threads now help (before lite they made things worse): Prefetch 2/3/4/5/6/8/10/12 = 9.6 / 8.5 / 7.7 / 7.4 / 7.05 / 7.6 / 8.5 / 9.5 s.
Best is about 6. Measured with `-benchmark`: CPU variant Prefetch(3) 16.0 s / 180 CPU-s; all-GPU lite Prefetch(3) 7.8 s / 27 CPU-s; all-GPU lite Prefetch(6) 6.5 s / 32 CPU-s.
Four ffmpeg jobs at once (R9): GPU lite Prefetch(3) 22.6 s total (one job alone 8.5 s), CPU variant 41.9 s total, GPU lite Prefetch(6) 20.7 s. So concurrent throughput is about 1.9x the CPU, less than the single-job gain: separate processes
have separate CUDA contexts and the GPU time-slices between them (no MPS on Windows), so kernels of different jobs do not overlap. VRAM used by 4 jobs: about 3 GB.
Not done, with reasons: more slices (changes the vectors relative to the CPU result at its thread count), packed SAD (the analysis kernel is latency-bound on its serial block chain, earlier unrolling / prefetch tricks gave only about 6%),
concurrent bv/fv analysis (Prefetch already overlaps frames), recalc upload cut (recalc kernel is 1 ms, uploads are cached in the device cache and average a few ms).

### Where the time goes now (2026-10-28 night, quiet machine, 10-bit UHD, ~115 output frames, all-GPU lite chain)

- Decode alone (LWLibav, 149 frames) is about 1.9 s of ffmpeg wall time; the full chain is 4.9-5.4 s at Prefetch(6). Half of the current time is not reachable by optimising our code.
- **LWLibavVideoSource default decoder threads burn CPU**: `threads=4` gives the same wall time with 15.5 instead of 36 CPU-s (Prefetch(6): 4.85-5.2 s vs 6.3-6.4 s). It is set where the source is opened (production scripts, not uconvert.avsi).
- Nsight Systems (`nsys`, no admin needed; Nsight Compute counters are NOT available without admin): k_search is 96% of GPU kernel time; kernels from different streams overlap rarely in the real run (union busy 4.5-5 s of 6.3-7.4 s span), although a microbenchmark (`profile/golden/bench_analyse.py`) shows 4 concurrent analyses give 2.3x throughput. The threads simply do not ask for the GPU at the same time.
- k_search per block at the finest level: about 24k cycles (start + predictors 4.7k, phase A 7 candidates 5.3k, fold A 1.3k, phase B 24 ring candidates 11.5k, fold B 1.4k). Build `Sources/cuda/build_prof.bat` (`-DMVC_PHASE_PROF`) prints these. With the reference loads removed it is still 15k, so about 9k is memory and 15k is instructions / barriers. Tried and no gain: nLambda (FP64 division) moved off the critical path, next-block vector loads by one thread into shared memory, loads moved later (the stall just moves to the next phase). Left: packed 16-bit SAD (`__vsadu2`) and wide loads, expected 20-30% of the kernel, a few percent to 20% of the run.
- Pinned upload staging is in (`MVCUDA_NOPIN=1` disables it); A/B on a quiet machine: median 5.47 s pinned vs 5.72 s pageable, within noise. Upload of one frame (memcpy into pinned) is DRAM-bound, 5-8 ms per 33 MB.
- `MVCUDA_TIMING=1` prints per-call phases, `MVTOOLS_GPU_PROFILE=1` prints thread-time per filter step.

### Portability

The build targets compute capability 8.0 / 8.6 / 8.9 / 9.0 / 12.0 plus PTX (RTX 30xx and newer; older cards report "no usable device", `MVCUDA_DEVICE` selects a device). The PTX path
(`CUDA_FORCE_PTX_JIT=1`) passes the exactness tests on the 5090; the 30/40-series machine code itself has not been run on such a card.

## Next step

1. M6 measurements still open: R9 (4 concurrent jobs, GPU contention, memory: each prefetch thread keeps frames in the 24-entry device cache, about 150 MB per UHD 10-bit super frame), R2-R8 matrix for the GPU variant, R10 (side-by-side by you), R12 soak.
2. Speed ideas left (GPU_PLAN "Speed work done"): the analysis kernel (2 x 23 ms per frame), recalc frame uploads (pageable copies), packed SAD, sharing results between cloned prefetch chains, (lite MSuper is done).
3. Deployment: `mvtools_cuda.dll` next to `mvtools2.dll`, `global uc_frc_gpu = true` (now also switches `MFlowFps`).
