# Decisions

Decisions about the GPU work in this fork, newest first. Each entry: what was decided, why, and what it rules out. The
plans (`GPU_PLAN.md`, `GPU_QTGMC_PLAN.md`, `GPU_PIPELINE_PLAN.md`) describe how the work got done; this file is the short
list of choices that later work must respect or consciously revisit. Status: **decided**, **open** (needs a decision),
**superseded** (with a pointer).

## D7. The mvtools GPU code stays in `mvtools2.dll`, with the CPU code (2026-10-30, decided)

**Decision.** The GPU paths of MSuper, MAnalyse, MRecalculate, MCompensate, MDegrain1..6 and MFlowFps stay in this fork's
`mvtools2.dll`, built from the same source as the CPU filters. They are not moved into `cudafilters.dll` with a stock
mvtools2.dll as the CPU fallback (the model cudafilters uses for RgTools, MaskTools2, nnedi3 and the AviSynth core).

**Why.** The GPU path depends on mvtools internals that are not an interface: the vector clip layout and its header
(`MVAnalysisData`), block geometry and padding, the super clip layout (and the picture-only "lite" super, which stock
filters cannot read), overlap windows, the degrain weights, the scene-change rules, and how `mt` slices the search. With
a separate CPU DLL, cudafilters would carry a copy of those internals that must match one exact stock build, and a user
with another mvtools2 version would silently get frames from two algorithms in one clip (the GPU frames and the per-frame
CPU fallbacks). In one DLL, built from one source, the GPU path and its fallback cannot disagree. The cudafilters model
fits small, stable filters whose only contract is parameters in, pixels out; mvtools is not one of them.

**Consequences.**
- The fork's CPU code should stay as close to upstream as possible (the GPU parts as additions and small hooks), so that
  "this fork's CPU result" and "stock mvtools2's result" stay the same thing. Rebasing on current upstream (pinterf
  2.7.47) and proving CPU output = stock with the golden tests is the way to keep that true (see D8).
- The price: users switch their mvtools2.dll to this fork to use the GPU (FFAStrans ships its own copy).

## D8. How scripts call the mvtools GPU paths (2026-10-30, open)

The cudafilters style is one `Cuda*` function per CPU filter, same parameters. mvtools uses a `gpu=true` parameter.
Proposed: add `CudaMSuper`, `CudaMAnalyse`, `CudaMRecalculate`, `CudaMCompensate`, `CudaMDegrain1..6`, `CudaMFlowFps` to
`mvtools2.dll` (same filters with the GPU on), switch `uc_frc_gpu` and `make_qtgmc_gpu.py` to them, then keep `gpu=true`
as a compatibility switch or remove it. Open: whether `gpu=true` stays; whether to rebase on upstream first.

## D6. Usage rule for every GPU filter (2026-10-30, decided)

Token frames in, token frames out; normal frames in, normal frames out (computed on the GPU, uploaded and downloaded by
the filter). Existing scripts keep working unchanged. Implemented for cudafilters and, with Part M, for mvtools
(`GPU_PIPELINE_PLAN.md` section 25).

## D5. Bit-exact, silent fallback, opt-in (2026-10-28 onwards, decided)

Every integer GPU path gives exactly the CPU filter's result with the same arguments (for mvtools: the CPU result with
the same slice count). Anything the GPU does not cover runs on the CPU, per call or per frame, without an error and with
the same result. GPU use is off by default. Floating point stages (NNEDI3, later dfttest / FFT3DFilter) are compared with
a calibrated tolerance and the user's visual judgement, never PSNR alone (`GPU_PIPELINE_PLAN.md` section 2).

## D4. `meander=false` selects the wavefront search (2026-10-29, decided)

With `gpu=true`, `meander=false` runs the wavefront kernel (equal to the CPU with one slice, the fallback then searches in
one slice too); `meander=true` keeps the exact slice kernel. The user approved slightly different vectors when the
pictures look the same: the GPU QTGMC uses `meander=false` by default, `QTGMC_gpu_meander=true` gives stock results.
Switching that default to `meander=true` waits until the slice kernel is clearly faster than the CPU
(`GPU_PIPELINE_PLAN.md` sections 10, 11, 25).

## D3. Three DLLs, one build (2026-10-29, decided)

Kernels and the device frame registry in `mvtools_cuda.dll` (C API, version checked); mvtools filters in `mvtools2.dll`;
everything else in `cudafilters.dll`. All three come from one build (library API version check). Token frames carry
frames between plugins on plain AviSynth+ (no Neo). cudafilters uses only AviSynth+ interface V8 calls (users run 3.7.3)
(`GPU_PIPELINE_PLAN.md` section 2, START_HERE.md 2026-10-29).

## D2. Test with real material only (2026-10-28, decided)

Interlaced tests use real interlaced material (the PAL MXF clip), never fake-interlaced clips; QTGMC tests read it with
ffms2 (LWLibav returns wrong frames on random access in MXF).

## D1. Hardware target: NVIDIA compute capability 8.0 and newer (2026-10-28, decided)

RTX 30xx and newer; older cards count as "no usable device" and run on the CPU.
