# Decisions

Decisions about the GPU work in this fork, newest first. Each entry: what was decided, why, and what it rules out. The
plans (`GPU_PLAN.md`, `GPU_QTGMC_PLAN.md`, `GPU_PIPELINE_PLAN.md`) describe how the work got done; this file is the short
list of choices that later work must respect or consciously revisit. Status: **decided**, **open** (needs a decision),
**superseded** (with a pointer).

## D10. masktools tables under VapourSynth: the bridge ports masktools' arithmetic (2026-11-01, decided)

**Decision.** CudaKeepOnlyBobShimmerFixes, CudaLutxy, CudaQtgmcSharpLimit (smode 1 / 2) and CudaQtgmcAddNoise build
their GPU lookup table by running `mt_lutxy` / `mt_lut` + `mt_adddiff` on a test pattern. Under VapourSynth the bridge
computes these three masktools filters itself (`vsbridge/mt_port.cpp`), exactly as masktools2 2.2.30 under AviSynth+
3.7.5 does: masktools' own double-precision evaluator, or AviSynth+ Expr's float32 rules where masktools hands the
expression to Expr (`use_expr`). Only a subset of the expression language: pow, %, exp, log and trigonometric
functions are refused with an error.

**Why.** Tried before deciding: a translation to VapourSynth's `std.Expr` was not exact (half-to-even rounding, no
`scalef`: up to 74898 of 1M table entries off by one); a Python prototype of the two evaluators matched all 24 QTGMC
tables exactly. The alternative, an error under VapourSynth, would leave QTGMC_gpu's output side unusable there.

**Consequences.**
- VapourSynth output equals AviSynth+ with masktools2 2.2.30 / AviSynth+ 3.7.5; another masktools or AviSynth+ version
  that changes rounding or `scalef` would differ (AviSynth+ changed both in 3.7.1 / 3.7.2). `vsbridge/test/mt_tables.py`
  (94 cases, references made on Windows) is the check.
- The ported filters are also the exact CPU fallback of these cudafilters under VapourSynth.
- Excluded functions stay excluded: AviSynth+'s results for them depend on the SSE / AVX2 code path or the C library.

## D9. VapourSynth through an AviSynth+ host, not a port (2026-10-31, decided)

**Decision.** mvtools2 (and later cudafilters) run under VapourSynth unchanged, through `mvtools_vsbridge`: a small
AviSynth+ host on the VapourSynth API 4 that loads the AviSynth+ plugin. The plugins only gain a
`VapourSynthPluginInit2` entry that opens the bridge. Namespaces `mvpf` (mvtools2) and `cuda` (cudafilters), so
dubhater's port (`mv`) can be loaded next to it. Function and parameter names stay the AviSynth ones.

**Why.** A rewrite of the filters for VapourSynth (as dubhater/vapoursynth-mvtools) would be a second code base that
drifts from this one, GPU paths included. Through the bridge the same arithmetic runs, so VapourSynth output is
bit-identical with AviSynth+ (17 test chains, `VAPOURSYNTH_PLAN.md`). The bridge is its own library because it compiles
`avisynth.h` in core mode (the `AVS_Linkage` implementation), which must never share a binary with plugin-mode code.

**Consequences.**
- Every new filter works under VapourSynth without extra work, as long as it only uses the AviSynth+ API the bridge
  implements (an unimplemented environment call throws "not supported under VapourSynth").
- Rules the bridge must keep: frames of one format share one pitch (mvtools reads one frame with another's pitch);
  MT_MULTI_INSTANCE filters get one instance per concurrent call; never a crash or hang when the bridge is missing.
- Vector and super clips carry their parameters in the AviSynth `VideoInfo`, which only the bridge's node registry
  keeps: they are not usable through other VapourSynth filters (`std.Trim` on a vector clip).

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
