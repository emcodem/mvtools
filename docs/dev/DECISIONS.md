# Decisions

Decisions about the GPU work in this fork, newest first. Each entry: what was decided, why, and what it rules out. The
former plans (`GPU_PLAN.md`, `GPU_QTGMC_PLAN.md`, `GPU_PIPELINE_PLAN.md`, ...; in git at af2e51e) describe how the work got
done, `ARCHITECTURE.md` summarises the design; this file is the short
list of choices that later work must respect or consciously revisit. Status: **decided**, **open** (needs a decision),
**superseded** (with a pointer).

## D11. Hardware target: compute capability 6.0 and newer; mvtools_cuda is built with CUDA 12 only (2026-11-01, decided)

**Decision.** mvtools_cuda runs on compute capability 6.0 and newer (GTX 10xx / Pascal, Volta, RTX 20xx .. 50xx). The
instructions that need newer devices (cp.async: 8.0; ld.acquire / st.release, `__nanosleep`, `__grid_constant__`: 7.0)
have fallbacks that compute the same results (`MVC_ARCH_BELOW`, `mvcuda_internal.h`). The library is built with CUDA
12 only, 12.8 or newer (12.9 recommended): machine code for 6.1 .. 12.0 plus PTX 6.0 and 8.0. On Windows that means the
MSVC 14.4x host toolset (v143); `build_windows.ps1` selects it, CMake refuses other toolkits.

**Why.** CUDA 13 no longer compiles for Pascal or Volta, so one toolkit for every build keeps one DLL for every card;
two release variants (CUDA 13 for 7.5+, CUDA 12 for older cards) would double the testing. The sm_80+ code is the same
as before (bit-identical output on the RTX 5090). The fallbacks are bit-exact as PTX 6.1 on the RTX 5090 (golden 85/85,
QTGMC Fast framemd5, cudafilters regression), about 5 % slower there.

**Consequences.**
- Builds need CUDA 12.8 .. 12.x and, on Windows, the "MSVC v143" component next to VS 2026's own 14.5x toolset (CUDA
  12 does not compile with the VS 2026 STL); the whole CMake build then uses 14.4x.
- New device code must keep a `MVC_ARCH_BELOW(700)` / `(800)` path, and changes to it need a test of the
  `-CudaArch 61-virtual` build (Pascal code only, JIT-compiled for the local GPU).
- Not verified on real 6.x / 7.0 hardware yet.
- Supersedes D1.

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

## D8. How scripts call the GPU paths: Cuda* names, token out, explicit fallback (2026-11-02, decided; not implemented yet)

Asked 2026-10-30, decided by the user 2026-11-02. Supersedes D6 and the "silent fallback, opt-in" part of D5.
- **Names.** mvtools2 gets `CudaMSuper`, `CudaMAnalyse`, `CudaMRecalculate`, `CudaMCompensate`, `CudaMDegrain1..6` /
  `CudaMDegrainN`, `CudaMFlowFps` (same parameters as the CPU filter). The `gpu` parameter is **removed**: MSuper,
  MAnalyse, ... are pure CPU again, exactly as upstream. `make_qtgmc_gpu.py`, the bundle examples, uconvert's
  `uc_frc_gpu` and the docs switch to the new names.
- **One rule for every filter of this repo (mvtools2 and cudafilters):** a Cuda* filter accepts normal or token frames
  (normal ones are uploaded inside the filter) and always returns token frames; the script ends the GPU part with
  `CudaDownload`. `CudaUpload` stays (no longer needed). `CudaDownload` passes normal frames through.
- **Vector clips stay normal frames** (small host data; CPU consumers such as MMask, MFlowBlur, MBlockFps, MDepan,
  MStoreVect read them directly). The library keeps a device copy per vector frame so GPU consumers do not upload them
  again (measure the host round trip with nsys first).
- **Fallback is explicit:** `global cuda_cpu_fallback = true / false` (AviSynth+, read once when a filter is created, so
  it must be set before the first Cuda* call), else the environment variable `CUDA_CPU_FALLBACK=0/1` (VapourSynth has
  no script globals; the script global wins under AviSynth+). **Default false.** Optionally a per-call
  `cpu_fallback=` parameter overrides both.
  - false: a Cuda* filter that cannot run on the GPU at all (no device, library missing, unsupported settings) stops
    with an error at creation that names the reason and the alternatives (`cuda_cpu_fallback = true` or the M* / CPU
    filter).
  - true: such a filter runs its CPU code and returns normal frames (CudaDownload passes them through), so one script
    works on machines with and without a GPU.
  - Either way, single frames that need the CPU during a run (field-based clip, delta <= 0, bad-vector search with
    meander=true, a frame that cannot be regenerated) are computed on the CPU and uploaded: the output stays a token.
- Bit-exactness (D5) is unchanged. uconvert work (deploy, HD pel 4, LWLibav fallback) waits for this, so uconvert.avsi
  is edited once.

## D6. Usage rule for every GPU filter (2026-10-30, superseded by D8)

Token frames in, token frames out; normal frames in, normal frames out (computed on the GPU, uploaded and downloaded by
the filter). Existing scripts keep working unchanged. Implemented for cudafilters and, with Part M, for mvtools
(`GPU_PIPELINE_PLAN.md` section 25).

## D5. Bit-exact, silent fallback, opt-in (2026-10-28 onwards, decided; fallback and opt-in superseded by D8)

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

## D1. Hardware target: NVIDIA compute capability 8.0 and newer (2026-10-28, superseded by D11 on 2026-11-01)

RTX 30xx and newer; older cards count as "no usable device" and run on the CPU.
