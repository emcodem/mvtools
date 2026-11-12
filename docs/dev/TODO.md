# Open work

All open items of the GPU fork in one place (collected 2026-11-02 from the former plan files, whose last versions are in
git history at commit af2e51e: GPU_PIPELINE_PLAN.md, GPU_PLAN.md, GPU_QTGMC_PLAN.md, REBASE_PLAN.md, VAPOURSYNTH_PLAN.md,
NVOF_FILTER_PLAN.md). Items from DTL's doom9 comments (MRecalculate trymany, UMH / hex radius above 24, parent
predictors, sub-pel / chroma precision, area mode, hardware ME, the reference quality test) are in
[DTL_COMMENTS_TODO.md](DTL_COMMENTS_TODO.md). Design background: [ARCHITECTURE.md](ARCHITECTURE.md).

Mark an item done by deleting it here (the commit message says what was done). **User** = needs a decision by the user.

## Next

- uconvert.avsi (`uconvert/`, D8 done): switch `uc_frc_gpu` to the Cuda* names (+ CudaDownload at the end of the
  FRC chain, cudafilters.dll loaded), deploy a new all-arch DLL set (replaces its mvtools2.dll too: 2.7.48 CPU result),
  uc_FRC HD branch with the whole pel-4 chain on the GPU (CudaMSuper is lite at pel 4 since D8; `run_frc_pel4.ps1`
  passes), the MxfSource -> LWLibav fallback (wrong frames on random access with Prefetch: use ffms2 or no Prefetch
  there). The `profile\avs\m\frc_ucdep_*` / `frc_uc_*` templates use uconvert.avsi and follow it.
- **User:** report upstream (pinterf/mvtools) that MScaleVect crashes on `divide > 0` vector clips (stock 2.7.48 too):
  it added the 0xFFFFFFFF length marker of the divided level as a plane length. Fixed in this fork (074321e).
- mc FRC fallback is too weak: no MAnalyse / MRecalculate / NVOFA setting keeps Dancers' fast leg whole (frames 15 and
  37 at 50p). Detection: `TESTFILES\reach_for_phc\03_detection` D9 covers the limbs (open: false positives on pans,
  seams, cost of the CPU MFlowInter / MFlow at pel 4). Replacement: SplatFps inside the mask was tried
  (`20_mc_fallback_splat`, 2026-11-12) and does not help: SplatFps ghosts the same fast limbs. **User:** judge D9 +
  nearest picture (solid, 25p judder) against the production soft mask in motion.
- CudaMAnalyse `meander=true` (slice kernel) is still slower than the CPU at 1080p: 300 frames 10-bit 4:2:2, pel 2,
  blksize 16 overlap 8, MShow chain incl. 2.6 s decode: CPU 5.5 s, GPU meander=false 6.3 s, meander=true 10.5 s
  (release) -> 7.9 s (2026-11-12: hex / UMH ranges 2..3 and hex range <= 1 as fixed inline batches + the ring
  fold instead of mvc_refine, nLambda per row). The bad-vector CPU frames are not the cause (3 of 300). Per block at
  level 0 now ~11k cycles (was 24k); left: three dependent SAD rounds (phase A, hexagon, ring) plus barriers. Coarse
  levels cost ~45 % because they have fewer slices. The slice kernel uses ~32 of 170 SMs: the remaining lever is
  overlapping frames (an analysis lookahead of frame n+1 while n is consumed, exact, needs a design decision: async
  results per instance, token pins across threads, Prefetch behaviour) - **User**.
- D8 vector-clip device cache: measured 2026-11-12, recommend not doing it. uc_FRC-like 1080p chain (blksize 8 overlap
  4 refine, 4 vector fields per interpolated frame = 6.2 MB): staging + H2D 0.3 ms, host extraction (`grab` in
  MVFlowFps::gpu_flow) ~1.1 ms per interpolated frame, of ~22 ms per frame without Prefetch (~6 %), mostly hidden with
  Prefetch. (nsys 2026.3 traced only cudafilters' calls in that run, not mvtools_cuda's: per-call timers
  `MVTOOLS_GPU_PROFILE=1` + a micro-benchmark instead.)
- **User:** should the bundle's `QTGMC_gpu_on()` also set `cuda_cpu_fallback = true`? Today QTGMC above 10 bit, or on a
  CPU where nnedi3 does not take its AVX2 + FMA path (AVX-512, older CPUs), stops with the CudaNNEDI3 / lookup-table
  creation error unless the script sets it.

## SplatFps (branch splatfps, own DLL splatfps.dll)

1. Modular split (D12), left over: the NVOFA raw grid as a vector source; `CudaSplatAnalyse` with token candidate
   clips (needs float / 4-byte frames in the registry; low value, it duplicates the built-in path); overlap of the
   vector source with the build; tests in the suites, docs. `MSplatVectors(CudaMAnalyse(...))` works (vector clips are
   host frames, D8): 42 fps against 73 built-in at 1080p. Open (**User**): export of our fields as MVTools vector
   clips (lossy).
2. Fast CPU path: GCC / Clang full build untested; the build (~26 ms at 1080p) is the largest cost (memory-bound
   level-0 passes: fuse them, overlap source analysis with the build).
3. CUDA tuning (kernel shapes depend on the frame sizes): the phase-correlation FFTs and peak search are a large
   share of GPU time; host side still ~7 ms per frame at 1080p fast against ~3.7 ms GPU (profile with nsys), ~2.2 s
   startup per run.
- Comparison with stock mvtools (user, 2026-11-08): how much of SplatFps' result MAnalyse / MRecalculate vectors
  through MSplatVectors, MFlowFps / MBlockFps give on the same clips, stage by stage: which parts are new
  (phase-correlation candidates, assignment with occlusion switch, pyramid splat build, local / temporal fallbacks)
  and what each buys in the picture.
- Token frames per D8, tests in every suite (framemd5 GPU exact == CPU, token / no-token, `MVCUDA_CACHE_MB=1`), docs.
- Clip end (found 2026-11-09 on Dancers 25p -> 50p): the in-between frame(s) of the last source pair are never built.
  `plan()` marks them `EosPlaceholder` because the backward field of N needs source N + 1 (`core/splat_me.cpp`
  `p + 2 <= n - 1`), and `GetFrame` shows the nearest source picture, at phase 0.5 an exact copy of P (Dancers
  frame 347 == 346: a visible stutter on the last step), at every clip length. Fix: a one-sided field for the last
  picture (mirror of the `single` path at the clip start, no N + 1), mark the entry `Built`; CPU exact / fast and CUDA
  alike, plus a parity case on a short clip.
- CUDA fast is not deterministic run to run (found 2026-11-09): `CudaSplatFps(exact=false)` gives a different md5 for
  every built frame on each run (Dancers 1080p, 60 source frames, 3 runs). Size: 749 of 120 frames' samples
  (0.0002 %) differ, each by exactly 1 (10-bit). `exact=true` on the GPU and both CPU versions are identical in every
  run. Cause (confirmed 2026-11-11): the float atomicAdd splats of `k_splat` add in a varying order; the released
build behaves the same. Fix (fixed-point atomics, see todo_performance.md) or document; until then md5 checks
  of CUDA fast against an earlier run fail without a code change, so parity checks of that path need a tolerance
  (+-1) instead of md5.
- **User decision, postponed (2026-11-09):** the quality option `speed_change_cost=0.25, splat_err_scale=30` stays a
  documented option, not the default. It is better in every case measured so far on the test corpus (all 10 football
  cases); the HD acceptance encodes use it explicitly.
- **User decision, postponed (2026-11-09):** the temporal fallback (`pull_strength`, `pull_error_lo`,
  `pull_error_hi`) is off by default but still in the plugin: keep or remove. Measured against ground truth it costs
  (judder, frames at the wrong time), though single frames look cleaner with it (fewer artefacts on deforming cloth).
- Scene cuts: the cuts `scene_cut_error` (15) misses join very similar shots: a relative rule (a pair's error far
  above that of the pairs before and after) instead of the absolute threshold.
- UHD ME scale (later; HD first): the auto scale at UHD is 4. On a fast 8K->UHD football graphics scene (corpus
  `fb_graphics`) scale 4 is clearly worse than scale 8 (more artefacts); on other UHD corpus cases (fine detail,
  moderate motion) scale 8 is 10-20 % worse than 4. Cost is equal (UHD 300 frames: 6.3 s vs 6.5 s incl. ~2.9 s
  decode). Plan: `me_scale=-1` = dynamic: two internal engines (auto scale and 2x), the second only analyses where
  used; the choice from statistics the first engine has anyway (vector reach vs search range, match errors), taken
  over a window of pairs (independent of access order) and switched only at cuts or after sustained change. First
  UHD corpus cases from the 8K football source (fast / slow, graphics / players), check that a simple statistic
  separates "8 better" from "4 better", then CPU + CUDA, parity, corpus run.
- Prefetch does not help SplatFps / CudaSplatFps (MT_SERIALIZED; measured 2026-11-09, 1080p): Prefetch(4) equal to
  none, 8 / 24 up to 2x slower (decode threads compete with the CPU version's own 24 threads). Say so in the docs.

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
  EdiExt, the noise branch (FFT3D); from 3.39x: PhosphorSoft (search clip), xflateMaTo with Rep0 / Rep2, MEus's
  LanczosResize, temporal=true (MAnalyse), pscrn 0 / 1 (NNEDI3).
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
