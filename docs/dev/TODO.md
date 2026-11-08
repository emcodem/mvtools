# Open work

All open items of the GPU fork in one place (collected 2026-11-02 from the former plan files, whose last versions are in
git history at commit af2e51e: GPU_PIPELINE_PLAN.md, GPU_PLAN.md, GPU_QTGMC_PLAN.md, REBASE_PLAN.md, VAPOURSYNTH_PLAN.md,
NVOF_FILTER_PLAN.md). Items from DTL's doom9 comments (MRecalculate trymany, UMH / hex radius above 24, parent
predictors, sub-pel / chroma precision, area mode, hardware ME, the reference quality test) are in
[DTL_COMMENTS_TODO.md](DTL_COMMENTS_TODO.md). Design background: [ARCHITECTURE.md](ARCHITECTURE.md).

Mark an item done by deleting it here (the commit message says what was done). **User** = needs a decision by the user.

## Next

- uconvert.avsi (`C:\dev\uconvert`, D8 done): switch `uc_frc_gpu` to the Cuda* names (+ CudaDownload at the end of the
  FRC chain, cudafilters.dll loaded), deploy a new all-arch DLL set (replaces its mvtools2.dll too: 2.7.48 CPU result),
  uc_FRC HD branch with the whole pel-4 chain on the GPU (CudaMSuper is lite at pel 4 since D8; `run_frc_pel4.ps1`
  passes), the MxfSource -> LWLibav fallback (wrong frames on random access with Prefetch: use ffms2 or no Prefetch
  there). The `profile\avs\m\frc_ucdep_*` / `frc_uc_*` templates use uconvert.avsi and follow it.
- Crash: `MAnalyse(..., divide=2)` dies with an access violation (exit -1073741819) after a few frames, with and without
  Prefetch, build `profile\build_d8\bin` (CPU MAnalyse). Chain: Dancers.mxf SelectOdd -> 1920x1080 10-bit 4:2:2,
  RemoveGrain(22), MSuper of the half-size picture (pel 4, hpad/vpad 8, rfilter 4), MAnalyse blksize 8 overlap 4
  search 3 searchparam 2 truemotion=false chroma=true divide=2, MScaleVect(2), MRecalculate 8x8, MFlowFps. Script:
  `TESTFILES\reach_for_phc\01_mvtools_params\avs\V5_divide2.avs`. Not isolated yet (an MShow-based bisect failed on its own); check
  whether stock 2.7.48 crashes too, and 8-bit / 4:2:0 / full size.
- mc FRC fallback is too weak (2026-11-02, `TESTFILES\reach_for_phc`): no MAnalyse / MRecalculate / NVOFA setting keeps
  Dancers' fast leg whole (frames 15 and 37 at 50p); uc_FRCFallback (nearest source picture where the refined-vector
  SAD is high) is what keeps the leg today, and that is not good enough either. Design and test a better fallback
  (detection + replacement), see `TESTFILES\reach_for_phc\README.md`.
- D8 left over: vector clips' device copy cache (measure the host round trip with nsys first, DECISIONS.md D8).
- **User:** should the bundle's `QTGMC_gpu_on()` also set `cuda_cpu_fallback = true`? Today QTGMC above 10 bit, or on a
  CPU where nnedi3 does not take its AVX2 + FMA path (AVX-512, older CPUs), stops with the CudaNNEDI3 / lookup-table
  creation error unless the script sets it.

## Tests and release

- trymany / trymanyall / search_coarse cases for `vsbridge/test/compare.py` and the Linux test run (agreed 2026-11-02).
- Golden dumps (`profile\golden`) predate upstream #59: 11 failures per 4:2:0 / 4:2:2 tag (f444 passes). Regenerate with
  the merged CPU build.
- Next release: everything since gpu-2026-11-01c (2.7.48 rebase, search_coarse / trymanyall, fast trymany, D8 Cuda*
  names) is only in dev builds. Full matrix: Windows AviSynth+, Windows VapourSynth, Linux, all-arch build with register
  / spill check. D8: the VapourSynth bundle examples (`examples/*.vpy`, written in the release folder) and
  `profile\release_tools\vs_example_test.py` / `lin_ex.sh` still use `gpu=True` / `gpu=true`: switch to the Cuda*
  names (+ CudaDownload). The `compare/` experiment scripts stay as run (old DLL sets, `gpu=true`).
- No run on 30 / 40-series or older hardware yet (machine code for those archs never executed; DLL 47 MB, could be
  trimmed).
- Not done from the first plan: R2-R8 test matrix for the GPU variant, R10 side-by-side by the user, R12 soak test (VRAM
  soak, teardown with frames in flight, TDR / watchdog on a busy desktop GPU).

## Exact speed work

- **meander=true slice kernel critical path** (largest remaining exact gap): compute the predictor-independent candidates
  of the next block while the serial chain runs. PAL QTGMC Fast meander=true 3.88 s vs meander=false 1.85 s (before E1),
  HD 1.9 vs 0.9 s. Also the reason the GPU MAnalyse default is slower than the CPU on SD (500 frames 0.71 vs 0.49 s;
  MDegrain2 chain 1.99 vs 1.20 s), together with decode and per-frame host work.
- 16-bit SAD in `pk_row` with `__dp2a_lo(__vmaxu2(a,b) - __vminu2(a,b), 0x0101u, acc)` (4 instructions vs about 10 for
  emulated `__vsadu2` on sm_120); then neighbour-lane shuffles instead of the second 8-byte word in 10-bit k_wave.
  10-bit Slower 5.08 s vs 8-bit 4.25 s.
- Thread block clusters (sm_90+): k_wave row hand-over through distributed shared memory (today 2.5-5 us per wavefront
  step), or one meander slice spread over several SMs. Large refactor, gain unknown.
- CUDA Graphs for fixed per-frame launch sequences (super build, QTGMC output side): about 9 us per WDDM launch.
- Fewer launches: super build in one launch (5 fewer per frame), MCompensate / MDegrain tables without separate upload
  waits, fuse RemoveGrain + GaussResize + MergeLuma in the search clip (submitter about 70% busy, ~30 driver calls / frame).
- trymany: share the first (start-vector-only) batch of hex / UMH Refines like the exhaustive table (trymanyall level 0
  5.7 vs 4.0 ms / frame).
- Interleave bVec and fVec wavefronts in one CTA.
- Copy-engine copies that wait on kernels (recalc / flow / nvof_vec stage copies) may block uploads the way E1 did. Check
  with nsys. CudaDownload measured 2026-11-02: no such effect in uc_FRC UHD or QTGMC Fast (a UVA-writing kernel instead
  was slower for QTGMC); one 11.5 s uc_FRC series that looked like it did not reproduce.
- 4 concurrent uc_FRC jobs +6% since Part M (17.4 -> 18.5 s); not re-measured after E1. User: no need to optimise.
- VRAM: return free buffers (`cudaMallocAsync` with a release threshold; only the transient overshoot); sub-pel planes
  computed on the fly instead of stored (pel 2 frame = 4x; pel 4 via a shared-memory window, see DTL list).
- Decode floor: ffms2 `threads=`, Prefetch contention on the source (decode alone is slower than decode + Bob).
- Share analysis results between cloned Prefetch chains.
- NNEDI3 predictor on int8 tensor cores (only for high nns presets).

## Non-exact options (each behind its own switch, default off) - User

- X1 fully parallel block search (predictors only from the coarser level and the previous frame): expected 10-50x for the
  search; vector fields less coherent, needs visual judgement (e.g. `exact=false`).
- X2 approximate SADs for pruning, exact SAD for the best few (2-4x at large radii).
- X3 NVENC ME-only (see NVENCMEAnalyse.md, DTL list item 14).
- X4 tolerance-based instead of exact CPU fallback (fewer forced CPU frames, possible flicker).
- meander=false for uc_FRC (vectors differ from stock; FRC 8.71 -> 5.51 s at the time).
- Milestone F: FFT3DFilter (and dfttest, only if used anywhere) on the GPU with cuFFT; cannot be bit-exact.

## Features (each falls back to the CPU exactly today)

- pel 4 for CudaMCompensate / CudaMDegrain (`mvcuda_comp.cu` refuses pel other than 1 / 2; CudaMSuper is lite at pel 4
  already, the CPU MCompensate / MDegrain complete it).
- Lookup tables above 10 bit and at 9 bit; SharpLimit at 16 bit (Merge rounds differently); CudaNNEDI3 above 10 bit
  (other float order) and at 9 bit.
- QTGMC 4:2:0 + ChromaMotion: the GPU search clip differs from the CPU (Slower, Very Slow, Placebo); stage not found
  (suspects: CudaBob / GaussResize chroma placement, a chroma lut). Excluded in `qtgmc/make_qtgmc_gpu.py`.
- QTGMC options not on the GPU: SLMode 1 / 3 / 4, Sbb with SVThin, Rep1 > 0, source match, lossless, lsb / n16, ChromaEdi,
  EdiExt, the noise branch (FFT3D).
- MMask, MFlowBlur, MBlockFps on the GPU (judged not worth it: 3-15% of CPU time); float samples, `divide`, `dct`,
  `temporal`, `outfile`, fieldShift != 0, other rfilter / sharp values.
- CudaNVOFAnalyse: HD (pel 4) path not verified (uc_FRC's HD branch keeps the normal search); visual defects AB15
  (frame 53), AF13-16 (frame 42), Z13 (frame 50), suspect: per-block median of blocks straddling two motions.
- Not done on purpose: `_gpu_mutex` over the MCompensate / MDegrain CPU path (would race with regeneration).

## VapourSynth / bridge

- MDepan under VS not verified (upstream now zeroes the reserved header field: byte-for-byte test possible); DePan /
  DePanEstimate through the shim (optional).
- `MVBRIDGE_FALLBACK=nearest` measured per mapping against AviSynth+ (postponed by the user 2026-10-31).
- Optional: nearest mappings to VS plugins when loaded (zsmooth, znedi3); QTGMC_gpu.avsi as a whole under VS.
- Later: MRestoreVect of a stored file under VS (Gray8 clip); token frames <-> VS Vulkan GPU frames.

## Build / code hygiene

- Upstream's `mvtools.vcxproj` builds mvtools2.dll again (GPU backend + VapourSynth entry added 2026-11-02) but is not
  part of any test run; CMake stays the tested build.
- Some cudafilters arguments are unnamed in the `AddFunction` strings (positional only), though gpu.md names them:
  CudaTemporalSoften, CudaQtgmcSharpLimit, CudaQtgmcBackBlend, CudaQtgmcAddNoise, CudaGaussResize, CudaBilinearResize.
- Not tested: Clang, aarch64.
- User: report upstream that a74dcb6 built with its own CMake crashes in `AvisynthPluginInit3` on Linux (all symbols
  exported, AVS_linkage interposition); ask upstream to reserve bits 1-7 of `SuperParams64Bits::param`.
- Two different copies of mvtools_cuda.dll in one process (cudafilters and mvtools2 from different folders) hang instead
  of failing cleanly.

## Deployment / scripts - User

- Which render nodes have a GPU.
- LWLibav `threads=4` in the production scripts; higher Prefetch for Very Slow / Placebo (24: Placebo 12.3 -> 8.1 s).
- Decided 2026-11-02, no action: QTGMC's FFT3DFilter keeps stock `measure=true` (rare second md5 class in Very Slow /
  Placebo is FFTW, not the GPU); branch `gpu_portable` is kept; non-exact options stay parked.

## Unexplained (the CPU does it too)

- Stock QTGMC HD Slower differs run to run on the CPU (2-8 frames); Fast 1 frame in 1 of 6 runs.
- SD content resized to 768x576 gives two md5 classes run to run (frame 24), release DLL too.
- Yadifmod2 reads uninitialised memory at row ends (stock Ultra Fast not reproducible; third-party).
