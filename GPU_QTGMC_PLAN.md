# GPU plan: QTGMC support (branch `gpu_work`)

Started 2026-10-28. Continues `GPU_PLAN.md` (read that first; the golden-data method, bit-exact rule and CPU fallback rule are unchanged).
Source of truth for what QTGMC calls: `C:\FFAStrans\Processors\avs_plugins\avsi\QTGMC.avsi` (1338 lines, read 2026-10-28).

## What QTGMC passes to mvtools (from the preset table, lines 318-338, and the call sites)

| Preset (pNum 1..11) | Search | SearchParam | SubPel / PelSearch | Chroma | Blocksize | Overlap |
|---|---|---|---|---|---|---|
| Placebo | 5 UMH | 2 | 2 / 2 | true | bs | bs/2 |
| Very Slow, Slower, Slow | 4 hex | 2 | 2 / 2 (Slow: chroma false) | true, true, false | bs | bs/2 |
| Medium, Fast | 4 hex | 2 | 1 / 1 | false | bs | bs/2 |
| Faster | 4 hex | 2 | 1 / 1 | false | bs2 | bs/2 |
| Very Fast | 4 hex | 1 | 1 / 1 | false | bs2 | bs/4 |
| Ultra Fast, Super Fast, Draft | 0 onetime | 1 | 1 / 1 | false | bs2 | bs/4 |

(pNum order as in the file: Placebo, Very Slow, Slower, Slow, Medium, Fast, Faster, Very Fast, Super Fast, Ultra Fast, Draft; the arrays have 11
entries, re-read lines 330-338 before relying on a single cell.) `ESearchP=true` changes `SearchParam` to 24 (presets < 4) or 16 (< 8) for search 4/5.
Always: `truemotion=false` (lambda 100*bs^2/64, lsad 400, pnew 25, plevel 0) unless the user overrides, `global=true`, `DCT=0` unless overridden,
`divide` not passed, `temporal`/`trymany` not passed. `MRecalculate` only when `RefineMotion=true` (default false): blksize/2, overlap/2, same search
type, `chroma=ChromaMotion`. Search supers: `MSuper(pel, sharp=2, hpad, vpad, chroma=ChromaMotion)` (all levels); every other `MSuper` has `levels=1`.

## What the GPU covers today vs what QTGMC needs

| Need | GPU now | Work |
|---|---|---|
| `search=3` radius 1-3 (exhaustive) | yes, bit-exact | - |
| `search=0` (OneTimeSearch), `4` (Hex2Search), `5` (UMHSearch), other types 1, 2, 6, 7 | no, CPU fallback | Q2: new refine kernels, bit-exact |
| `chroma=false` in MAnalyse / MRecalculate / MSuper | no, host gate requires chroma | Q1: luma-only paths |
| `MSuper(levels=1)` for compensation | MSuper gpu path builds all levels | Q3 (only if MCompensate / MDegrain go to the GPU) |
| `MCompensate`, `MDegrain1..6`, `MMask`, `MFlowBlur` | CPU only | Q4-Q6 |
| `DCT != 0`, `divide`, `temporal`, `trymany`, `outfile`, pel 4 | CPU fallback | stays on the CPU unless a QTGMC setting needs it (decide after Q2) |
| `gpu` parameter in QTGMC.avsi | none | Q7 |

## Search semantics that decide how to port them

CPU search (`PlaneOfBlocks::PseudoEPZSearch` -> `Refine`): predictors first (zero, global, predictor, 4 neighbour/coarse predictors), then
`Refine`. A candidate wins only if its cost is **strictly lower** than the running best (first wins on ties). Candidate batches whose positions are
relative to a *fixed* centre (hex ring step, UMH cross and hex-grid, expanding square) can be evaluated in parallel and folded in sequence order;
that is what the exhaustive kernel does already. What is sequential and data dependent:
- **OneTimeSearch** (`search=0`): per halving step `i = param, param/2, ...`: left/right at the current best, then *walk* in the winning direction while it
  keeps improving (each step centred on the moved point), then the same vertically. The walk is a serial chain of single-candidate evaluations.
- **Hex2Search** (`search=4`): hexagon around the best (6 candidates, parallel), then up to `param/2 - 1` half-hexagons (3 candidates each) centred on
  the moved point, stopping on no improvement or `!IsVectorOK`; then the 8-point square refine (parallel). `param=1` skips the hexagon (square only).
- **UMHSearch** (`search=5`): cross (fixed centre, parallel), `param/4` rings of 16 hex-grid points (fixed centre, parallel), then `Hex2Search`.
Also needed bit-exactly: `CheckMV` / `CheckMV0` / `CheckMV2` / `CheckMVdir` variants (they differ in which of cost, sad, direction they update, and in
the `IsVectorOK` clipping), `tryMany=false` path, the bad-vector wide search (`badSAD` / `badrange`, UMH at radius `badrange*pel`) which the GPU
reports as `MVCUDA_NEEDS_CPU` today, and `MAnalyse(search=4/5)` default `badrange` (24?) and `badSAD` (10000?): **check what defaults the plugin uses for
QTGMC calls, since QTGMC does not pass them. If the wide search triggers often the GPU result would keep falling back; measure it in Q2.**

## Milestones

| # | Work | Exit criterion |
|---|---|---|
| Q0 | Plan, golden scripts for QTGMC parameter sets (one per preset family, 8-bit 4:2:0 / 4:2:2 and 10-bit), CPU mt=false and sliced references | `profile\avs\golden\qtgmc_*.avs`, dumps regenerate |
| Q1 | `chroma=false` for MSuper / MAnalyse / MRecalculate (all existing search code, exhaustive only) | golden tests bit-exact, new `run_all.py` format keys |
| Q2 | Refine kernels `onetime`, `hex`, `umh` (+ `nstep`, `logarithmic`, `h`, `v` if cheap) for MAnalyse and MRecalculate, incl. the bad-vector wide search | bit-exact vs CPU, all QTGMC preset parameter sets, negative controls |
| Q3 | `MSuper(levels=1)` + lite rules for QTGMC's other supers | bit-exact picture planes |
| Q4 | `MCompensate` on the GPU | bit-exact vs CPU |
| Q5 | `MDegrain1..6` on the GPU (out16 / lsb variants as used) | bit-exact vs CPU |
| Q6 | `MMask` (kind 0, 1), `MFlowBlur` | bit-exact vs CPU |
| Q7 | `gpu` plumbing in a *copy* of QTGMC.avsi (the FFAStrans original is not edited): `QTGMC_gpu` global, passed to every mvtools call that has a GPU path | QTGMC output with GPU = CPU output, all presets |
| Q8 | Profile QTGMC CPU vs GPU per preset; decide what is worth more work | numbers in `profile\RESULTS.md` |

Order note: profile the CPU QTGMC chain (per filter CPU-s, per preset) **before Q4**, so MDegrain/MCompensate work is only done if they matter.
Q1 + Q2 come first because every QTGMC call is blocked by them.

## Decisions I have made without asking (change if you disagree)

1. The QTGMC.avsi in `C:\FFAStrans\...` is read-only for me; the GPU switch goes into a copy under `C:\dev\uconvert` or the profile folder.
2. "All modes" = every search type and chroma setting QTGMC can pass, plus every mvtools filter it calls. `DCT != 0`, `divide`, `temporal`,
   `trymany` stay CPU fallbacks (exact), and the user is told so rather than implemented blindly; DCT in particular needs an exact FFT/int-DCT port.
3. Same exactness rule as before: bit-exact against the CPU at the same slice count, silent per-frame CPU fallback.

## Status 2026-10-28 (late)

Done (uncommitted, branch `gpu_work`): Q1 chroma=false (MAnalyse, MRecalculate, MSuper path), Q2 all search types incl. the batch scheme of
`Sources\cuda\mvcuda_refine.cuh` (groups of candidates around a fixed centre are evaluated in parallel and folded in CPU order), rfilter 2 and 4,
non-aligned overlaps (plain source copy), `search_supported()` in GpuBackend.h, API version 5. Q7 `profile\qtgmc\QTGMC_gpu.avsi` (global `QTGMC_gpu`;
MSuper of srchClip, MAnalyse, MRecalculate, ShutterBlur MRecalculate get `gpu=`).

Tests: `profile\run_qtgmc_vec.ps1` compares the vector payload (header ignored: heap pointers) CPU mt vs GPU: 29 synthetic cases on two clips and 8 cases on
the real bobbed PAL clip, all identical, no fallbacks except where the filter is not supported (ESearchP radius 16/24 coarse levels, block size 64, DCT,
lite MSuper with chroma=false). `profile\run_qtgmc_full.ps1`: full QTGMC presets Medium / Slow / Slower / Placebo / Fast, output frames identical in the
runs that fell into the same outcome class (see below).

Findings: (1) QTGMC passes no rfilter (default 2); the GPU super refused it, so nothing ran on the GPU before rfilter 2 was added. (2) A field-based clip
(SeparateFields) gives fieldShift != 0 for pel 2 and the GPU path is skipped; QTGMC analyses a bobbed clip, so this does not matter there. (3) The whole
QTGMC chain under ffmpeg + Prefetch(6) is non-deterministic on the CPU too (two outcome classes, about half of the runs each; some runs also differ in a
few frames at the start); GPU runs fall into the same two classes. Cause not investigated (decoder / threaded filter upstream of mvtools?). (4) Speed:
the GPU analysis call takes 40-70 ms at 720x576 / 1920x540 against a few ms of 24-thread CPU wall time, so one job is slower in wall time (PAL 300 frames:
Medium 2.2 s CPU vs 5.1 s GPU, Slower 4.2 s vs 14.8 s) while CPU time drops (24 -> 16 CPU-s, 46 -> 22 CPU-s). Wall time improves with more Prefetch threads.

Not done: Q3-Q6 (MCompensate, MDegrain, MMask, MFlowBlur on the GPU), 4-concurrent-jobs QTGMC measurement, lite MSuper for chroma=false, kernel latency work.

## Profile: what MDegrain / MCompensate / MMask / MFlowBlur cost in QTGMC (2026-10-28 night)

Method: `profile\run_qtgmc_prof.ps1`, env `MVTOOLS_SKIP` (Sources\ProfSkip.h, profiling aid only): the named filter returns after fetching its inputs
without pixel work; drop in total CPU seconds (ffmpeg -benchmark, best of 2) = cost. Real PAL clip, 300 frames, Prefetch 6, analysis on the CPU.
Noise is about +-2 CPU-s.

| preset | CPU-s | degrain | compensate | mask | flowblur | all four together |
|---|---|---|---|---|---|---|
| Medium | 19.2 | within noise | within noise | within noise | 0.8 | 0.5 (3%) |
| Slower | 46.5 | 4.5 (10%) | 0.2 | 0.6 | 0.3 | 7.0 (15%) |
| Placebo | 81.4 | 4.0 (5%) | 4.1 (5%) | within noise | 0.5 | 3.0 (4%) |

Conclusion: the four families are 3-15% of QTGMC CPU time, so a GPU port would save at most that; not worth it now. Together with the analysis (about
33% Medium, 51% Slower, 67% Placebo of CPU-s, measured as CPU-only minus GPU-analysis runs) the rest, about 30-60%, is in non-mvtools filters (nnedi3,
dfttest / fft3dfilter, RemoveGrain / Repair, MaskTools, decode), not yet broken down.
