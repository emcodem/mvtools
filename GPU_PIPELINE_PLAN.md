# GPU pipeline plan: QTGMC end to end on CUDA, plain AviSynth+ (branch `gpu_work`)

Written 2026-10-29. Builds on `GPU_PLAN.md` (FRC work, golden-data method), `GPU_QTGMC_PLAN.md` (analysis for QTGMC, measurements) and the
ideas read in nekopanda's `AviSynthCUDAFilters` (cloned read-only to `C:\dev\ext\AviSynthCUDAFilters`, `C:\dev\ext\NNEDI3`; KTGMC / NNEDI3 are GPL, KFM is MIT).

## 1. Why, and what the measurements say

On the real PAL clip (`C:\temp\mxf.js\mpeg2_allintra_50m_pal_60s_hard.mxf`, 720x576 TFF 4:2:2, 300 frames, Prefetch 6, CPU 24 threads):

| QTGMC preset | CPU-s | motion analysis | nnedi3 | MDegrain / MCompensate / MMask / MFlowBlur | rest (denoise, repair, merges, decode, ...) |
|---|---|---|---|---|---|
| Medium | 19.9 | about 33% | about 20% | 3% | about 45% |
| Slower | 47.2 | about 51% | about 15% | about 15% | about 20% |
| Placebo | 81.4 | about 67% | not measured | about 4% | about 30% |

The GPU analysis alone gives bit-identical vectors but is slower in wall time than the 24-thread CPU (one call 40-70 ms against a few ms), because the
bit-exact search is a serial block chain cut into only 8-9 slices at this frame size. It only saves CPU time. The FRC speedup came from moving the whole
chain (MSuper, MAnalyse, MFlowFps) to the GPU, not from the analysis. Goal here: do the same for QTGMC.

**Targets** (real PAL clip, one job, quiet machine, High priority): wall time at or below the CPU variant for Medium, Slower and Placebo; CPU seconds at
most one third of the CPU variant; four concurrent jobs finish faster than four CPU jobs; output equal to the CPU output (exactly where the CPU code is
integer, within a measured, documented tolerance where it is floating point FFT / NNEDI3), judged by the user by eye where the tolerance is used.

## 2. Architecture decisions

1. **Plain AviSynth+ only.** No Neo device frames. Frames stay on the GPU through *token frames* (section 4).
2. **One device library, several AviSynth plugins.** Kernels and the device frame registry live in `mvtools_cuda.dll` (C API, version checked, per-call
   fallback rules as today). `mvtools2.dll` keeps the mvtools filters. Everything that is not mvtools goes into a new plugin `cudafilters.dll` (GPL)
   that loads `mvtools_cuda.dll` the way mvtools2 does (`GpuBackend` pattern). A separate plugin keeps mvtools2 a drop-in for people who do not want it.
3. **Exactness rule.** Integer code (SAD search, RemoveGrain / Repair, merges, MaskTools expressions, resize with integer filters) must be bit-exact
   against the CPU filter. Floating point code (NNEDI3, dfttest, fft3dfilter) is compared with a tolerance that is *calibrated* against the difference
   between two CPU builds of the same filter, and signed off by the user by eye (see `benchmark-and-image-judgement-rules`: no PSNR-only claims).
4. **Every GPU filter keeps an exact CPU fallback** (library missing, no device, unsupported parameters, any CUDA error): silent per frame, result equal
   to the CPU filter. For the new plugin the fallback calls the original CPU plugin's function through the script (`cuda=false` default path).
5. **Opt-in per call, one switch for QTGMC.** `QTGMC_gpu` in the patched `QTGMC_gpu.avsi` (the FFAStrans original stays untouched) selects the GPU
   versions of every stage that has one and wires the `cuda_out` flags.
6. **Test with real interlaced material only** (the PAL clip). Never fake interlaced material.

## 3. Milestones

Effort in working sessions (S) of a few hours, rough.

| # | Work | Exit criterion | Effort |
|---|---|---|---|
| P0 | Make the validation trustworthy: find why a full QTGMC run through ffmpeg + Prefetch gives two different outputs on the CPU alone (two outcome classes, about half of the runs each; some runs differ in a few start frames). Bisect by dumping intermediate clips (decoder, bob, srchClip, vectors, denoised clip); suspects: LWLibav `threads=4` with all-intra MPEG-2, threaded filters, `FftThreads`. Record the real backend of `sneo_dfttest` / `sneo_FFT3D` in the FFAStrans setup (they try `neo_dfttest` / `neo_fft3d` and fall back to `dfttest` / `FFT3DFilter` inside `try`/`catch` in `Zs_RF_Shared.avsi`) | Cause known, a deterministic reference run exists (or the classes are explained); backends written down | 1-2 S |
| W | **Wavefront analysis** (section 5) | bit-exact vs CPU `meander=false, mt=false`; PAL analysis call below 10 ms | 3-4 S |
| D | **Device frame layer** (section 4) | round trip exact, eviction and regenerate paths exact, Prefetch safe, mvtools filters take and give tokens | 3-4 S |
| N | **NNEDI3** (section 6.1) | within the calibrated tolerance, user sign-off | 2-3 S |
| R | **RemoveGrain / Repair / Merge / MaskTools subset** (section 6.2) | bit-exact | 4-6 S |
| T | **Temporal and misc stages** (TemporalSoften, Blur, Weave / SeparateFields helpers, binomial soften, tweak search clip, bob shimmer fixes) | bit-exact | 2-3 S |
| F | **dfttest and fft3dfilter** (section 7) | within the calibrated tolerance, user sign-off | 6-8 S |
| Q | Full QTGMC wiring and measurements (all presets, noise presets, SourceMatch / Lossless / ShutterBlur as far as stages exist) | section 1 targets measured and written to `RESULTS.md`; R9 style 4 concurrent jobs | 2-3 S |

Order: P0 and W first (W is independent and fixes the wall-time problem), then D, then N / R / T in the order of measured CPU share, F last because it is the
biggest and least exact. **The user's daily preset is `Fast` (section 11): build and measure the complete Fast chain first.** Each stage is switched on in the QTGMC wiring only after its own test passes, so the chain can be GPU for a prefix of stages and
CPU after a `CudaDownload()`.

## 4. Device frame layer (D): keeping frames on the GPU between filters

**Token frames.** A GPU filter with `cuda_out=true` leaves its result in device memory and returns an ordinary AviSynth frame of the declared size whose
first bytes hold: magic (`MVCUDATK`), layout version, 64-bit token id, producer instance id, frame number, format info (width, height, planes, bit depth,
subsampling, device index). The rest of the frame is not written (no memset; allocation only) or filled with a recognisable pattern, so a filter that does
not understand tokens produces obvious garbage instead of a subtly wrong picture. With `cuda_out=false` (default) the filter returns a normal CPU frame as
today.

**Consumers detect tokens.** Every GPU filter looks at the magic of its input frame: token -> use the device buffer; normal frame -> upload (today's path,
through the pinned staging). No separate "cuda_in" parameter. Explicit converters: `CudaUpload(clip)` and `CudaDownload(clip)`; `CudaDownload` on a normal
frame is a no-op copy. The last GPU stage before a CPU filter (for example a denoiser that stays on the CPU) is set to `cuda_out=false`.

**Registry and lifetime.** AviSynth does not tell us when a frame is freed, so no reference counting. Device buffers live in an LRU registry with a byte
budget (env `MVCUDA_CACHE_MB`, default a fraction of free VRAM, per device). Every token records producer instance and frame number. A consumer whose
token is no longer in the registry asks the producing filter instance, through a process-wide table of live instances, to **regenerate** that frame
directly (bypassing AviSynth's cache), exactly the idea of `MVSuper::CompleteLite` and the (clip, frame) key of the existing super frame cache. Instances
are registered and unregistered in their constructor / destructor; regeneration while the producer is being destroyed returns "unavailable" and the consumer
reports a clear error (can only happen on teardown).

**Rules.** Prefetch: clones of a filter are separate instances; tokens are valid across instances in the same process (the registry is global);
regeneration goes to the instance that made the token. Memory: budget per device, eviction never frees a buffer in use by a running kernel (events /
stream fences). Formats: planar YUV and Y 8..16 bit, 4:2:0 / 4:2:2 / 4:4:4, no YUY2 / packed / RGB at first. Context: all kernels use the device's primary
context (shared by the plugin DLLs of one process), blocking-sync as today.

**Tests (all must pass before integration):** upload -> download round trip bit-exact for all formats; forced tiny budget (`MVCUDA_CACHE_MB=1`) gives
identical output to a large budget (regenerate path); token clip into a non-GPU filter shows the pattern; Prefetch 1 / 6 / 16 give identical output;
no VRAM growth over a long run (soak, R12); teardown with frames in flight.

**Integration with mvtools:** MSuper and MAnalyse accept tokens (the lite super idea extends: the super frame is built from the device picture without a
host round trip), MRecalculate, MFlowFps, MDegrain / MCompensate (not ported, they need an explicit `CudaDownload`).

## 5. Wavefront analysis (W)

Idea taken from `KTGMC\MVKernel.cu` (`kl_search`): one persistent CUDA block per block *column*, taken in order from an atomic ticket; it walks down the
rows and, before each block, waits until the left column's progress counter has passed this row (`while (prog[x-1] < y)` on a volatile global with
`__threadfence` on publish). The dependency graph of one level (left and up neighbours; the bottom-right / up-right predictors come from the coarser
level or are on the last row) is a wavefront, so the critical path is about (columns + rows) block steps instead of (rows per slice) x columns. For
720x576 with 16x16 blocks that is about 160 steps against about 510 serial blocks per slice today; for UHD 32x32 about 370 against about 2300.

**Semantics.** This is the **non-meander** scan with no slices: the result equals the CPU run with `meander=false, mt=false` bit for bit; it differs
slightly from the default (meander, multi-threaded) CPU result. Selection rule: `gpu=true` with `meander=false` uses the wavefront; `gpu=true` with
`meander=true` keeps today's slice kernel (equal to the CPU multi-threaded result). The patched `QTGMC_gpu.avsi` passes `meander=false` when `QTGMC_gpu`
is on. The user decides by eye whether the slightly different motion field is acceptable (documented, no hidden change of the default).

**Details to get right.** Deadlock freedom: tickets are handed out in column order and a column only waits for a lower column, so every waited-on column
has started; grid size is capped at the resident capacity; test on a small GPU emulation by limiting `MVCUDA_MAX_CTAS`. Per level kernel (coarse to fine as now),
global MV estimate and interpolation kernels unchanged, exhaustive coarse levels and all refine types reuse `mvcuda_refine.cuh` (batches evaluated by all
warps). Bad-vector wide search stays `MVCUDA_NEEDS_CPU` (exact CPU result for the frame). 8 and 16-bit paths, chroma on / off, all block sizes the slice kernel
supports. MRecalculate is already block-parallel. More than one frame pair per launch (nekopanda's `ANALYZE_MAX_BATCH`) is a later option: instances are
separate filters, so it needs a shared scheduler; Prefetch concurrency is the first lever.

**Tests:** new reference dumps from the instrumented CPU build with `meander=false, mt=false` for all QTGMC parameter sets (`run_qtgmc_vec.ps1` gets a
`-Meander false -CpuMt false` mode); negative controls; PAL and UHD clips; Prefetch concurrency; measured per-call time at 720x576, 1920x540 and UHD.
**Target:** analysis call below 10 ms at PAL, below 25 ms at UHD, and the QTGMC wall time below the CPU variant for Medium / Slower.

## 6. Stage ports (N, R, T)

Source material to read first: `KTGMC\Kernel.cu` (3328 lines; `kl_rg_clip`, `kl_repair_clip`, `kl_merge`, `kl_makediff`, `kl_logic1-3`, `kl_resample_h/v`,
`kl_box3x3_filter`, `kl_binomial_temporal_soften_*`, `kl_bobshimmerfixes_merge`, `kl_tweak_search_clip`, `kl_vertical_cleaner_median`, `kl_limit_over_sharpen`,
`kl_resharpen`, `kl_error_adjust`, `kl_lossless_proc`, `kl_weave`, ...), `NNEDI3\nnedi3\nnedi3_kernel.cu` (618 lines), `KTGMC\MVKernel.cu` (degrain / compensate:
only if the profile says so, they are 3-15% of CPU time). Their host code (Neo frames, `OnCUDA`) is not reused.

### 6.1 NNEDI3 (N)
QTGMC calls `NNEDI3(field=-2, nsize=NNSize, nns=NNeurons, qual=EdiQual, threads=EdiThreads, U=CEed, V=CEed)` (double-rate bob of the input fields). The CPU
plugin uses float weights with SSE / AVX / FMA code, so bit equality with the CPU is not expected. Steps: take the weights file the CPU plugin uses, port
the kernel (predictor network per pixel, `qual` 1 / 2, `nsize` 0..6, `nns` 0..4, `pscrn`, interlaced field handling, edge rows), compare against the CPU plugin
in all QTGMC settings, measure the difference distribution (max, count of pixels with |diff| > 1 per frame, local worst crops) and against the existing
OpenCL `NNEDI3CL` as a second reference.

**Findings 2026-10-29 (NNEDI3CL from `C:\dev\uconvert\dlls`, source `C:\dev\ext\AviSynthPlus-NNEDI3CL`, GPL, OpenCL float8 kernels with `native_divide`):**
- QTGMC accepts `EdiMode="NNEDI3CL"`. It failed for me at first only because `nnedi3_weights.bin` must sit next to `NNEDI3CL.dll` (the uconvert folder
  has the DLL but no weights file; tested with a scratch copy in `profile\nnedi3cl_test`, nothing in the user's folders changed).
- Speed is fine: QTGMC Slower 300 frames, CPU 47.2 s -> 38.1 CPU-s, Medium 19.9 -> 15.9 CPU-s, wall time unchanged (so nnedi3 is removed from the CPU).
- **Accuracy is not good enough** against the CPU nnedi3 (JPSDR build) with the QTGMC Slower settings (`nsize=1, nns=1, qual=1`, same field order, luma of 91 frames):
  6.1% of all pixels differ (about half of each frame is the untouched kept field, so about 12% of the interpolated pixels), 2.3% of all pixels by more than 1 level,
  maximum difference 60. Setting `pscrn=2` explicitly or `fapprox=0` on the CPU side does not change that. The CPU plugin's *own* C path (`opt=1`) against its
  default SIMD path differs in 0.514% of pixels, 0.0006% by more than 1, maximum 9: that is the natural spread. NNEDI3CL is about 100x outside it in the
  count of larger differences, so it is a different result, not rounding noise (probably prescreener decisions flipping through the native math, plus
  other detail differences; cause not analysed).
- Verdict: **do not rely on NNEDI3CL; write our own.** Acceptance bar for our CUDA NNEDI3: against the CPU C path (`opt=1`, deterministic) at most the
  CPU's own spread (differing pixels about 0.5% or less, more than 1 level in about 0.001% or less, max about 10), on the QTGMC settings of every preset
  (`nsize` 1 / 4 / 5, `nns` 0 / 1 / 2, `qual` 1, 2 (EdiQual), `pscrn`, `etype`) and on both chroma settings. To get there the kernel must follow the CPU
  evaluation order (float accumulation order of the dot products, the `exp` approximation of `fapprox`, the prescreener decision rule, rounding to the
  output range) instead of using fast-math intrinsics. The measuring scripts are the ones used on 2026-10-29: luma `Subtract` of the two outputs (zero
  level is 126, not 128), counts of differing pixels, pixels differing by more than 1, maximum; keep them in `profile\qtgmc\` when the work starts.

### 6.2 RemoveGrain, Repair, Merge, MaskTools (R)
Integer, bit-exact. Inventory from `QTGMC.avsi`: RemoveGrain modes 1, 2, 4, 12, 17, 20, 22 (+ `rgBlur` variants), Repair modes 12 and others,
Merge / MergeLuma / MergeChroma (weights), `mt_lutxy` (15 uses, expression strings), `mt_makediff`, `mt_adddiff`, `mt_logic`, `mt_inpand`, `mt_expand`,
`mt_merge`, `mt_lut`, `mt_clamp`, `mt_xxflate`. Plan: do **not** write a general expression engine first. Port exactly the expressions QTGMC uses as fused
kernels (the nekopanda kernels show which ones they fused), each with an exact CPU reference test on real frames and on random frames covering clamp / rounding edges;
an `mt_lutxy` expression compiler (NVRTC) is an option if the set of expressions turns out to be too large.

### 6.3 Temporal and misc (T)
TemporalSoften, Blur, binomial soften (`kl_binomial_temporal_soften`), SeparateFields / Weave / SelectEvery / Interleave plumbing as device-side index
changes where it saves copies (they are frame-order operations: with tokens they cost nothing and must simply pass tokens through, to be verified), bob
shimmer fixes, tweak search clip.

## 7. dfttest and fft3dfilter (F)

QTGMC denoises through `sneo_dfttest(U=.., V=.., sigma=Sigma*4, tbsize=noiseTD, threads=FftThreads, fft_threads=FftThreads, Dither=DftDither)` (Slower and
slower noise presets) and `sneo_FFT3D(u=.., v=.., sigma=Sigma, bt=noiseTD, ncpu=FftThreads)` (Fast / Faster noise presets); `noiseTD = Select(NoiseTR, 1, 3, 5)`. Both
wrappers try the Neo plugins and fall back to `dfttest` / `FFT3DFilter` (the exact backend in the FFAStrans setup is step 1, see P0). Both are FFTW float filters, so
the port uses **cuFFT** (static link, cc 8.0+ builds) and cannot be bit-exact; the acceptance bar is calibrated as described in section 2.

**Common method.** (1) Read the source of the build in use and write a short spec of exactly what it computes (window, block layout, overlap, edge
handling, sigma mapping, rounding, dithering) from the source, not from documentation. (2) Implement on the GPU with batched real-to-complex FFTs over all
blocks of a frame (and its temporal neighbours) at once, custom kernels for window / threshold / inverse scaling / overlap-add with weights / rounding,
no per-block launches. (3) Temporal window of `tbsize` / `bt` frames: the filter pulls neighbouring frames; with tokens the neighbours are device-resident
anyway. (4) Compare with the CPU filter on real frames: maximum difference, number of pixels differing by more than 1 per frame, local worst-case crops for
the user to look at, **and** the difference between the two CPU backends (FFTW build vs the other CPU plugin) as the reference spread the GPU result must stay inside.

**dfttest (tritical).** Per block: spatial `sbsize` x `sbsize` (default 16, overlap `sosize` default 12) and temporal `tbsize` frames window, 3D real DFT, `ftype` 0
(hard threshold at `sigma`; QTGMC uses the default ftype, `sigma = Sigma * 4`), `smode`, `tmode`, `swin` / `twin` window types, `zmean` option, per-frequency
`sigma2`..., inverse DFT, windowed overlap-add normalisation, dither (`Dither` parameter, ordered noise) when writing 8-bit output, `lsb` / 16-bit paths. Work items: window tables
and normalisation computed on the host in double and uploaded; `tbsize` 1, 3, 5 (QTGMC's three values); chroma planes optional; frame borders (mirror
padding) exactly as in the source; dither pattern reproduced deterministically (the CPU pattern is position based; reproduce it, else document).

**fft3dfilter (Fizick / pinterf).** Per plane blocks `bw` x `bh` (default 32), overlap `ow` / `oh` (default half), `bt` temporal 1..5 (QTGMC: 1 / 3 / 5), Wiener-like
filtering of the 3D spectrum with `sigma` (and `sigma2..4` per frequency band, `beta`, `kratio` for `bt` 0 / 1), optional sharpen / dehalo / degrid parts that QTGMC
does not use (reject them with the CPU fallback), `interlaced` / `pframe` / `px`, `measure`, window (`wintype`), edge handling, the `plane` mask (`y`, `u`, `v` 0..3).
The GPU version supports the parameter subset QTGMC uses and falls back (exactly) to the CPU plugin for everything else.

**Risks specific to F.** cuFFT vs FFTW rounding (accepted by the calibrated tolerance, not bit-exact); memory: a 3D batch of all blocks of `tbsize` frames at UHD is
large (stream it in bands); `sneo_*` wrappers may select a Neo plugin that does not exist in plain AviSynth+, so the CPU fallback target must be the plugin
that actually runs.

## 8. QTGMC wiring (Q)

The patched `QTGMC_gpu.avsi` is regenerated by `profile\qtgmc\make_qtgmc_gpu.py` (extend the script as stages appear; the original file is only read).
It sets `cuda_out=true` on every GPU stage that feeds another GPU stage, `CudaDownload()` before any stage that stays on the CPU and at the end, passes
`meander=false` to MAnalyse / MRecalculate (section 5), and keeps a per-stage switch (`QTGMC_gpu_stages`, a string list) so any stage can be forced back to
the CPU for bisecting. Measurements with `run_qtgmc_full.ps1` / `run_qtgmc_prof.ps1` on the real PAL clip, all presets, plus 4 concurrent jobs, High priority,
`nvidia-smi` quiet machine, best of 2-3; results into `profile\RESULTS.md`.

## 9. Risks and open points

- ~~QTGMC itself is non-deterministic on the CPU~~ Resolved by P0 (section 12): the LWLibav source returned wrong frames on random access; with ffms2
  the full chain is deterministic and full-run outputs can be compared byte for byte.
- **Wavefront changes the motion field** versus the default CPU result (meander=false, no slices): needs the user's visual sign-off; keep the slice kernel for
  exactness against the default.
- **Inter-block waiting in the wavefront kernel** needs resident-CTA discipline (section 5); TDR / watchdog on a busy desktop GPU (soak test R12).
- **Float stages** (NNEDI3, dfttest, fft3dfilter) are within a tolerance, not exact; user decides per stage.
- **VRAM:** token cache + super frames at UHD are large (about 150 MB per 10-bit UHD super frame today); budgeted, with regenerate on eviction. The render node has an
  RTX 5060 Ti (cc 12.0), the dev machine an RTX 5090; 30/40-series machine code is still untested on real hardware.
- **Licensing:** code taken from KTGMC / NNEDI3 is GPL (compatible with mvtools); KFM is MIT. Keep attribution in the file headers of every ported kernel.
- **Scope creep:** the presets' optional features (SourceMatch, Lossless, ShutterBlur, EZDenoise, ...) need more stages; wire them only when the stage exists and
  the CPU path is the fallback otherwise.
- **Windows build:** CUDA toolkit 13.x + VS 18 Build Tools as today (`Sources\cuda\build.bat`, about 2 min); the new plugin gets its own vcxproj; cuFFT static link adds size.

## 10. Decisions I made without asking (change if you disagree)

1. Wavefront is selected by `meander=false` with `gpu=true`; `meander=true` keeps the exact slice kernel (no silent change of results).
2. Kernels in `mvtools_cuda.dll`, non-mvtools filters in a new plugin `cudafilters.dll`, tokens as described (no Neo dependency).
3. Floats (NNEDI3, dfttest, fft3dfilter) accepted within a calibrated tolerance and the user's visual judgement; integer stages must be bit-exact.
4. Port only the expressions / modes QTGMC actually uses (inventory in 6.2) and fall back to the CPU for everything else.
5. Order W -> D -> N -> R -> T -> F, each stage wired into QTGMC only after it passes.

## 11. User's answers (2026-10-29) and what they change

- **FFAStrans is CPU only, nobody uses AviSynth Neo.** So the `sneo_dfttest` / `sneo_FFT3D` wrappers in `Zs_RF_Shared.avsi` always take their `catch` branch
  and call the plain `dfttest` / `FFT3DFilter` plugins; those are the CPU references (verify once which DLLs load, P0). The Neo question was only about which
  backend the wrappers pick; the plan (plain AviSynth+, token frames) does not depend on Neo at all. Every stage keeps an exact CPU fallback.
- **Preset in daily use: `Fast`** (all presets should work, but Fast comes first). QTGMC `Fast`: `EdiMode="NNEDI3"` with `NNSize=5, NNeurons=0, EdiQual=1`
  (the cheapest net, `nsize` 5 = 8x6? check against the plugin's table), `Search=4` hex, `SearchParam=2`, `PelSearch=1`, `SubPel=1`, `ChromaMotion=false`,
  `BlockSize=16` (tuning None), `Overlap=BlockSize/2`, `TR0=2, TR1=1, TR2=0`, `Rep0=3`, `ProgSADMask=0` (read from the table in `QTGMC.avsi` 2026-10-29: `Select(pNum, ...)`
  is 0-based and `Preset == "Fast"` gives pNum 5, i.e. the sixth column; the earlier `GPU_QTGMC_PLAN.md` table's "Medium, Fast" row is this configuration plus
  Medium). Default noise handling: `NoiseProcess` is not on by default; the default `NoisePreset="Fast"` means denoiser `FFT3DFilter`, `NoiseTR=1`, `bt=1`.
  Re-read the exact rows before writing tests.
- **Slightly different motion vectors are fine** if the pictures look the same: the wavefront (`meander=false`, no slices) is approved as the GPU default
  for QTGMC; keep the exact slice kernel for `meander=true`. Still validate bit-exact against CPU `meander=false, mt=false` and let the user look at pictures.

## 12. P0 result (2026-10-29): the non-determinism is the source filter, not QTGMC

**Cause.** `LWLibavVideoSource` (L-SMASH-Works r1310 from `C:\dev\uconvert\dlls`, used by the old test template) returns the wrong picture on any
non-sequential request on the PAL MXF: reading the first 60 frames in reverse order gives 38 wrong frames, each one an *earlier* source frame (frame
9 shows source 8, frame 2 shows source 1, ...). Independent of `threads=1/4`, `seek_mode=0, seek_threshold=0`, a fresh index, and of
`SetFilterMTMode(..., MT_SERIALIZED)`. Sequential reading (no Prefetch) is correct: equal to ffmpeg's own decode (`-f framemd5`) in all 60 frames.
Under Prefetch the request order changes from run to run, so every run gets a different set of duplicated input frames: the decoder alone with
Prefetch(6) gave 6 different outputs in 6 runs (28-35 of 60 frames wrong). The stock FFAStrans mvtools 2.7.46 (no avstp, none of our code) showed the
same behaviour, and so did the Fast preset, only less often (that is why "two classes" was seen earlier; there are many).

**Fix for all tests.** ffms2 5.0 (`C:\FFAStrans\Processors\avs_plugins\ffms2\x64\ffms2.dll`, the source filter FFAStrans actually ships) is correct:
reverse reading 0 wrong frames, Prefetch(6) 1 class in 5 runs, equal to ffmpeg. New template `profile\qtgmc\qtgmc_mxf_ffms.avs.tpl`; driver
`profile\run_p0.ps1` (N runs, outcome classes, differing frames, loaded DLLs). With it QTGMC Slower and Placebo are deterministic (1 class in 6 runs each),
and **CPU vs GPU analysis (slice kernel) is byte-identical on all 600 output frames for Fast, Medium, Slower and Placebo** (300 source frames, Prefetch 6):

| preset | CPU wall / CPU-s | GPU analysis wall / CPU-s | GPU calls |
|---|---|---|---|
| Fast | 1.69 s / 22.9 | 4.21 s / 13.4 | 1216 |
| Medium | 2.37 s / 26.5 | 5.26 s / 16.8 | 1214 |
| Slower | 4.30 s / 49.6 | 15.17 s / 25.0 | 2414 |
| Placebo | 8.53 s / 87.1 | 36.24 s / 38.5 | 3633 |

(GPU idle at start, High priority, one run each; Fast and Medium also have 6 lite `MSuper(gpu=true)` calls refusing `chroma=false`, known gap, D.)

**Backends actually loaded** (module list of the ffmpeg process, `run_p0.ps1 -Modules`): ffmpeg + `avisynth.dll` from `C:\FFAStrans\Processors\ffmpeg\x64`,
`fft3dfilter.dll` 2.12 (Fizick / pinterf) with `libfftw3f-3.dll` from the ffmpeg folder (only loaded when a denoiser runs), `plugins_JPSDR.dll` 4.1.0
(nnedi3), masktools2, RgTools, mvtools2. **No `neo_fft3d` / `neo_dfttest` exist on this machine and `C:\FFAStrans` has no `dfttest.dll` at all** (only the
older copy `C:\dev\FFAStrans` has one). Corrections to sections 7 and 11:
- `neo_fft3d` / `neo_dfttest` are ordinary AviSynth+ plugins (the "Neo" in the name is not AviSynth Neo); the `sneo_*` wrappers take the `catch` branch
  here because the DLLs are not installed, not because of the AviSynth flavour.
- In this setup a denoiser runs by default only in Placebo / Very Slow (`NoiseProcess=2`) or with `EZDenoise` / `EZKeepGrain`, and the default
  `NoisePreset="Fast"` selects **FFT3DFilter**. dfttest is only reached with `NoisePreset` Slower / Slow / Medium or `Denoiser="dfttest"`, and that
  would fail in the current `C:\FFAStrans` (missing DLL). So for F, FFT3DFilter comes first; dfttest only if the user confirms it is used anywhere.
- Open for the user: `uconvert.avsi` reads MXF through MxfSource and falls back to `LWLibavVideoSource` with `Prefetch`; if that fallback is ever
  taken for such files, production output would have duplicated frames. Not tested here.

## 13. W status (2026-10-29 night): wavefront kernel `k_wave` in `Sources\cuda\mvcuda_analyse.cu`

**Correction of section 1.** One isolated PAL analysis call (bob, 89x71 blocks, Fast/Medium parameters) took **8-10 ms** of kernel time with the
slice kernel, not 40-70 ms (that figure was measured under load); inside QTGMC Fast with Prefetch(6) the median was 9.4 ms, about 3 calls in
flight, 1215 calls per 300 frames. So the "<10 ms" target was already met and was the wrong target; to beat the CPU wall time a call has to be
around 1-2 ms.

**Design as built** (differs from section 5 in the details): one CTA of 4 warps per block *row* (not per column), rows taken by an atomic
ticket (deadlock-free for any grid size), per-row progress counters (`st.release` / `ld.acquire`); the left predictor stays in the CTA. A
candidate's SAD is computed by a group of GW lanes (GW = block width rounded to a power of two, lane = column, 8 groups per CTA for 16x16),
several rounds' reference loads in flight together; the fold is a CTA-wide argmin over (cost, index), exact whenever all cost parts are
non-negative (checked; the sequential fold runs otherwise); nLambda (FP64 on the CPU) is precomputed in `k_interp` for every level but the
smallest. `meander=false` with `gpu=true` selects it (`MVCUDA_WAVE=0` forces the slice kernel for tests); MAnalyse with `gpu=true,
meander=false` also runs its CPU path in one slice, so fallback frames have the same semantics.

**Exactness.** `run_qtgmc_vec.ps1 -NoMeander` (CPU `meander=false, mt=false` vs GPU `meander=false`): all 8 bobbed-PAL QTGMC parameter sets
identical. Negative controls: slice kernel with meander=false differs from CPU mt=false; the default meander path is still identical to the
CPU-mt result; the wavefront vectors differ from the default meander result (the expected small motion-field change, user approved).

**Speed per call (bob_medium = Fast's analysis parameters, nsys kernel times):** slice kernel 8.8 ms -> first wavefront 5.0 ms -> 2.84 ms ->
**0.94 ms** (level 0, 71 rows: 0.46 ms). What mattered, measured with clock64 phase counters (`-DMVC_PHASE_PROF`, ncu needs admin rights
here): (1) a noinline SAD evaluator made ptxas issue the 16 row loads one after another (about 5x slower per candidate); (2) the
`mvc_refine` path with 768-thread barriers cost 30k cycles per block; (3) one warp per row left each round latency-bound; four warps per row
and the parallel fold cut a coarse-level block from about 20k to 9k cycles and the row wait from 20k to 4k.

**Build time.** Inlining the SAD code into the about 20 call sites of `mvc_refine` in many kernel instances made cicc use 50 GB / 40 min
(killed). Now: ring-only kernels (exhaustive levels, hex2 with range <= 1; compile-time block sizes) separate from the general kernels
(runtime sizes, one `__noinline__` batch function per GW: same speed, measured). `Sources\cuda\build_dev.ps1` (sm_120 only, parallel units,
object cache) builds in about 80 s, `build.bat` (all architectures) in 6.5 min; the DLL grew from 9.5 to 47 MB (kernel instances x 6
architectures; trimming is possible).

**Bad vectors.** The CPU's wide search for blocks above badSAD (UMH with badrange * pel around (0, 0), or ESA rings for badrange < 0, then
sub-pel rings) depends on `badcount`, the number of earlier bad blocks in raster order (one slice). Ported exactly: a block above badSAD
(the threshold for badcount 0) waits until the rows above are finished, takes the per-row counts, decides as the CPU, and runs the search
through a noinline runtime-size checker (UMH candidate lists cut into batches in CPU order: for pel 2 they exceed one batch). API version
6 (`MvCudaAnalyseParams::bad_range`). Before: 12 of 174 HD calls fell back to the CPU; now none, identical.

**Final verification (release DLL `profile\build_w_rel`, 2026-10-29 morning).** Vectors identical: wavefront 8 PAL + 5 HD parameter sets (40 HD
frames), slice/meander path 29 synthetic + 8 field + 8 bob + 5 HD cases, golden FRC tests 85/85 (the ctypes structs in `profile\golden\test_*.py`
lacked `search_type` since API 5 and read past the struct; fixed). Full QTGMC (`run_qtgmc_full.ps1 -WaveRef`: GPU vs the same script with
`MVTOOLS_GPU_FORCE_FALLBACK=1`): Fast, Medium, Slower, Placebo PAL (600 output frames each) and Fast HD (500 frames) byte-identical.

**Per call (bob PAL, kernel time):** Ultra Fast 0.51 ms, Faster 0.76, Fast / Medium 0.94, Slower 1.6-1.8, Placebo 2.2-2.6 ms.

**End to end (PAL 300 frames, two repetitions, quiet machine; CPU = default QTGMC 24 threads, GPU = wavefront; best Prefetch of each):**

| preset | CPU wall / CPU-s | GPU wall / CPU-s | before W (GPU, Prefetch 6) |
|---|---|---|---|
| Fast | 1.33 s / 22-24 | 1.32 s / 17-22 | 4.21 s / 13.4 |
| Medium | 1.54 / 24-30 | 1.45 / 24 | 5.26 / 16.8 |
| Slower | 2.59-2.74 / 55 | 2.36-2.41 / 29 | 15.17 / 25.0 |
| Placebo | 4.70 / 86 | 4.58-4.87 / 47 | 36.24 / 38.5 |

HD Fast (250 frames): CPU 7.12 s / 123 CPU-s, GPU 6.64 s / 97 CPU-s. The GPU variant needs Prefetch 16 (each call blocks its frame thread);
the CPU variant also gains from 10-16 over 6.

## 14. Where the time goes now (2026-10-29): measured order for the next milestones

`profile\run_modprof.ps1` (module-level sampling profiler, `modprof.cs`: suspends the ffmpeg threads about every millisecond and counts the
DLL of the instruction pointer; no admin rights needed; ±5 points at a few hundred samples). QTGMC Fast, GPU analysis:

| module | PAL (21 CPU-s, 364 samples) | HD (97 CPU-s, 1602 samples) |
|---|---|---|
| masktools2 | 22% | 31% |
| avisynth.dll (core filters, caches, copies) | 24% | 28% |
| vcruntime140 (memcpy: frame copies) | 21% | 17.5% |
| mvtools2 (CPU MSuper, MDegrain, MCompensate, ...) | 15% | 14% |
| plugins_JPSDR (nnedi3) | 8.5% | 4.6% |
| RgTools | 3% | 3% |

Also measured: QTGMC Fast with `EdiMode="Bob"` (no nnedi3) saves only 1.5 of 20.6 CPU-s at PAL; decoding is 0.15 s. **Correction of section 11:
for Fast, NNEDI3 is not the next bottleneck.** New order: D (token frames: removes most of the copies) together with R (MaskTools
expressions QTGMC Fast uses) and T (the AviSynth core stages of the Fast chain), then the CPU mvtools filters (MSuper for MDegrain /
MCompensate, MDegrain1, MCompensate), then N, F last. Wall time only improves once a contiguous chain runs on the GPU: every
CPU stage in between costs upload / download and keeps the CPU threads busy.

## 15. The QTGMC Fast chain, operation by operation (read from QTGMC.avsi 2026-10-29; work list for D / R / T)

Fast on 8-bit 4:2:2 interlaced input: TR0=2, TR1=1, TR2=0, Rep0=3, Rep1=0, Rep2=4, RepChroma=true, SrchClipPP=2, SubPel=1, ChromaMotion=false,
SMode=2, SLMode=2 (temporal limit, SLRad=1), Sbb=0, SVThin=0, NoiseProcess=0, no SourceMatch / Lossless / ShutterBlur, Border=false.
maxTR = 1, so only bVec1 / fVec1 are analysed (2 MAnalyse calls per output frame). In order:

| # | stage | operation | plugin today |
|---|---|---|---|
| 1 | bobbed | `sh_Bob(0, 0.5)` (bicubic bob of the fields) | core |
| 2 | ts1, ts2 | `TemporalSoften(1 / 2, 255, 0, 28, 2)` | core |
| 3 | binomial0 | `ts1.MergeLuma(ts2, 0.357).MergeLuma(bobbed, 0.125)` | core |
| 4 | repair0 | `KeepOnlyBobShimmerFixes(binomial0, bobbed, 3, false)`: mt_makediff, mt_inpand / mt_expand vertical x2 each, mt_adddiff-style recombination (body at QTGMC.avsi 1129) | masktools |
| 5 | spatialBlur | `repair0.RemoveGrain(12, -1).GaussResize(w, h, 0, 0, w+eps, h+eps, p=2).MergeLuma(repair0, 0.1)` | rgtools + core |
| 6 | analysis | `MSuper` (srchClip), `MAnalyse` x2 (bVec1, fVec1) | **GPU (W)** |
| 7 | edi | `NNEDI3(field=-2, nsize=5, nns=0, qual=1)` | JPSDR |
| 8 | ediSuper | `edi.MSuper(levels=1)` | mvtools CPU |
| 9 | bComp1, fComp1 | `MCompensate` x2 | mvtools CPU |
| 10 | tMax, tMin | `mt_logic(max)` x2, `mt_logic(min)` x2 | masktools |
| 11 | degrain1 | `MDegrain1(ediSuper, bVec1, fVec1, thSAD=640)` | mvtools CPU |
| 12 | binomial1 | `degrain1.Merge(edi, 0.25)` | core |
| 13 | vresharp | `Merge(mt_expand(vertical), mt_inpand(vertical))` | masktools + core |
| 14 | resharp | `mt_lutxy(binomial1, vresharp.RemoveGrain(12), "x x y - sharpAdj * +")` | masktools + rgtools |
| 15 | sharpLimit1 | `mt_clamp(resharp, tMax, tMin, 0, 0)` | masktools |
| 16 | repair2 | `KeepOnlyBobShimmerFixes(sharpLimit1, edi, 4, true)` (ed 4: plus QTGMC_deflate / inflate) | masktools |

Every stage except 7 is integer, so the target is bit-exact. The GPU versions of 1-5 feed the GPU MSuper directly (token frames, D), and 8-16
consume the edi token. Steps for D/R/T: (a) exact specs from the sources of the installed builds (AviSynth+ core for Bob / TemporalSoften /
Merge / GaussResize; masktools2 for inpand / expand / logic / lutxy / clamp / makediff / adddiff and QTGMC_inflate / deflate; RgTools
RemoveGrain 12); (b) one CUDA kernel per stage (fused where QTGMC chains them, e.g. the whole KeepOnlyBobShimmerFixes), each tested
bit-exact against the CPU filter on real frames and random frames; (c) the token layer and `CudaUpload` / `CudaDownload`; (d) wire the
chain in `QTGMC_gpu.avsi`.

## 16. D status (2026-10-29 morning): token layer working, tested

- `Sources\cuda\mvcuda_frames.cu` (API version 7, `mvcuda.h` "device frames and tokens"): registry of device frames by 64-bit token, byte
  budget `MVCUDA_CACHE_MB` (default a quarter of free VRAM, max 4 GB), LRU eviction; holders keep a `shared_ptr`, so eviction never frees a
  buffer in use. Each thread keeps references to the last 8 frames it created (a thread can always read what it just made: without this a
  regenerated frame was evicted again by other threads before its download, found by the tiny-budget test). Producer table +
  `mvcuda_token_regenerate` for evicted tokens. Internal `mvc::frame_alloc / frame_put / frame_get` for the GPU stages.
- New plugin `cudafilters\` (`cudafilters.vcxproj`, `CudaFilters.cpp`, GPL): loads `mvtools_cuda.dll` from its own folder (must be the same
  folder as mvtools2.dll: one registry per library instance). `CudaUpload(clip)`, `CudaDownload(clip)`. Token frame = normal frame,
  `MvCudaTokenHeader` in Y row 0, every 8th luma row black (a non-GPU filter visibly shows stripes). Env: `CUDAFILTERS_STATS=1` prints
  registry statistics at teardown, `CUDAFILTERS_LOG=1`.
- Tests `profile\run_tokens.ps1` (test DLL set `profile\build_d`): 7 formats (8-bit 4:2:2 / 4:2:0 / 4:4:4 / Y, 10-bit 4:2:0 / 4:2:2, 16-bit
  4:4:4) x 6 cases (round trip, Prefetch 6, Prefetch 16 with 1 MB budget, token negative control, far reuse, same token twice with 1 MB
  budget = regeneration path, 1-9 regenerations per run): 42 of 42 pass, bit-exact.
- Not done yet in D: mvtools filters taking tokens (MSuper / MAnalyse reading the device frame instead of uploading), VRAM soak test,
  teardown with frames in flight.
- Reference sources for exact ports cloned read-only: `C:\dev\ext\AviSynthPlus-3.7.5` (core of the FFAStrans avisynth.dll 3.7.5),
  `masktools-2.2.30`, `RgTools-1.2` (the versions FFAStrans ships).

## 17. R / T status (2026-10-29 morning): first core stages on the GPU, bit-exact

New filters in `cudafilters.dll` (token output; each builds the CPU filter with the same arguments for its VideoInfo and as exact fallback;
kernels in `mvtools_cuda.dll`): `CudaBob(b, c, height)` (`mvcuda_resample.cu`; resampling programs made on the host by a port of AviSynth's
own code, `cudafilters\Resample.cpp`), `CudaTemporalSoften(radius, lt, ct, scenechange, mode)` (`mvcuda_temporal.cu`), `CudaMerge`,
`CudaMergeLuma`, `CudaMergeChroma` (`mvcuda_merge.cu`). Test driver `profile\run_cfcompare.ps1 -Cpu '<expr>' -Gpu '<expr>'` (framemd5 of
CPU vs GPU + CudaDownload over 8-bit 4:2:2 / 4:2:0 / 4:4:4 / Y, 10-bit 4:2:2 / 4:2:0, 16-bit 4:4:4, Prefetch 0 / 16; negative control =
the GPU expression without CudaDownload must differ).

Results: Bob identical in all formats, both field orders, other b / c and heights. TemporalSoften identical in all formats for QTGMC's
calls (radius 1 / 2, 255, 0, 28), chained after CudaBob, with chroma thresholds and with a scene-change limit low enough to disable
neighbours. Merge / MergeLuma / MergeChroma identical for 8 weights each (QTGMC's 0.357, 0.125, 0.1, 0.25, 0.5 and the 0 / 1 edges).

Exactness traps found (the GPU follows what the CPU actually runs on SSE4.1 / AVX2 machines, not the C reference):
- TemporalSoften 10-16 bit: `accumulate_line_16_sse41` divides in float (round to nearest even), the C code in fixed point; they
  differ at 16 bit (every frame). The GPU uses the float form with IEEE-exact `__fmul_rn` / `__fdiv_rn`.
- Merge 16 bit: the AVX2 / SSE2 body shifts samples to signed; because weight + invweight = 32767 this gives one more than the scalar
  formula, which the row tail (row size not a multiple of 16 bytes) still uses. Reproduced per sample (tested at width 718).
- Merge weights: SIMD 15-bit scale (`weight * 32767 + 0.5`), the C path uses 16 bit.
CPUs without SSE4.1 / AVX2 would give slightly different CPU results; not a target (documented in the kernels).

Later the same morning (all bit-exact, `profile\run_cf_regression.ps1` = every stage + the chained search clip + the 42 token tests):
- `CudaKeepOnlyBobShimmerFixes(input, ref, rep, chroma)` (`mvcuda_kobs.cu`, 8 bit; Rep without erosion 2 / 5 and over-dilation >= 3;
  else the CPU function). Its two `mt_lutxy` expressions are **not re-implemented**: the plugin runs the installed mt_lutxy once on a
  256 x 256 pattern of all (x, y) pairs and uploads the result as a table, exact by construction for any 8-bit two-clip expression.
  Tested: 8 Rep / chroma combinations x 4 formats x Prefetch 0 / 16.
- `CudaRemoveGrain(mode, modeU, modeV)` 8 bit, modes 0 / -1 (copy; the CPU leaves -1 planes undefined) / 12 / 20, as RgTools' SSE4.1
  code: mode 12 is a cascade of rounding averages with a saturating -1, mode 20 is (sum + 4) * 7282 >> 16, borders copied.
- `CudaGaussResize(...)` (`mvcuda_resample_2d`): both passes, order by area as CreateResize, identity passes skipped, chroma placement from
  frame 0's `_ChromaLocation` for "auto" (ffms2 sets it), keep_center. Identical for QTGMC's call, a downscale and an explicit placement.
- **The whole search-clip pre-filter of QTGMC Fast (stages 1-5) runs on the GPU**: identical luma to the CPU chain (120 frames,
  Prefetch 0 / 6 / 16, 4:2:2 and 4:2:0). Wired into `QTGMC_gpu.avsi` as `global QTGMC_gpu_search` (default false; needs cudafilters.dll
  loaded; only for the structure it reproduces: TR0 2, SrchClipPP 2, no chroma motion, not 4:2:0). Full QTGMC Fast output with it is
  byte-identical to the GPU-analysis-only run (600 frames).

**Pipeline stream (D, asynchronous).** All device-frame work is queued on one pipeline stream without host waits; the host waits only
for downloads (event after the copy) and its own uploads (own stream, ordered before / after the pipeline by events); pipeline buffers
come from their own pool (stream-ordered reuse); kernel parameters by value. Two measured pitfalls: (1) the registry's default budget of
a quarter of the VRAM (8 GB) meant frames almost never went back to the pool, so nearly every frame was a `cudaMalloc` (15,900 per run),
which synchronises the device: every thread's kernel launches then took 100 us (median) instead of 13 us. Budget now max 1 GB.
(2) The pool and the registry are static objects; the pool is now never destroyed (static destruction order crashed ffmpeg at exit).

QTGMC Fast PAL 300 frames, Prefetch 16, GPU analysis: without GPU search clip 1.31-1.39 s / 22.5 CPU-s; with it **1.42 s / 14.8 CPU-s**
(two runs each, identical output). Before the pipeline and budget fixes the search variant took 1.8 s. Remaining overhead: launch count
(KeepOnly alone is about 20 small kernels per frame: fuse), the CPU MSuper still built from the downloaded search clip (lite MSuper
refuses chroma=false; next: MSuper / MAnalyse taking tokens directly), and one download per frame.

**Consequences for the order of work:** W (wavefront analysis) and D (token layer) first as planned; then **N (NNEDI3, `nsize=5 nns=0` first, then the other nets)**,
then the Fast preset's other stages (R, T) so that a complete Fast chain runs on the GPU; **F (dfttest / fft3dfilter) moves last** because Fast does not denoise
by default (it matters for `NoisePreset` / `EZDenoise` users and the slower presets). Measure Fast end to end first (target section 1 for Fast), then widen to all presets.

## 18. Search clip stays on the GPU; one launch per analysis; measured stalls (2026-10-29, early morning)

**Token path into MSuper / MAnalyse (library API 9).** `QTGMC_gpu_srchclip` no longer ends in `CudaDownload` (global `QTGMC_gpu_tokens`,
default true; the download is kept when the search clip has other readers: ProgSADMask, ShutterBlur masks, FastMA bit-depth change).
- MSuper(gpu=true) "lite" is now also allowed for luma-only analysis (QTGMC's `chroma=false`). Given a token frame it writes only the
  header at the picture position. Every CPU consumer of a lite frame goes through `MVSuper::ResolveToken` (in `CompleteLite`, MFinest
  pel 1, the MFlowFps GPU path); the MRecalculate GPU path declines token pictures (its CPU path downloads). A token reaching a normal
  (non-lite) MSuper is downloaded first.
- MAnalyse detects the header at the picture start (`GpuBackend::read_token`) and calls `mvcuda_analyse_token`: the super frames are
  built from the device frames (`super_build_dev`), ordered after the pipeline stream by a pooled event. Mixed token / normal pairs -> CPU.
- `mvcuda_token_pin / unpin` (new): a pinned frame stays reachable after eviction. Consumers that regenerate inputs pin each one
  right after making it (cudafilters `with_inputs`, MAnalyse). Before, with a small budget (<= 64 MB) a 5-input TemporalSoften
  regeneration evicted its own inputs again; the resulting AviSynth error inside a Prefetch thread crashed ffmpeg (access violation in
  avisynth.dll). Now budgets of 8 / 16 / 64 MB give byte-identical output (8 MB: 42,000 regenerations, 4.3 s instead of 1.0 s).
- **Latent bug fixed:** each cudafilters stage used its CPU filter as fallback when the GPU could not make a frame, but that CPU filter
  read the *token* frames of its inputs (header bytes as pixels). The fallback filters now read their inputs through CudaDownload
  (`cf::cpu_input`), so the fallback is the exact CPU result.

**One launch per analysis call.** `k_wave` now takes all levels (`WaveCall`): one ticket sequence over the rows of all levels, coarse
first; the CTA that finishes a level's last row computes the next level's global MV (`glob_estimate`, the former k_global, with a
parallel first-maximum scan) and releases it; predictors are interpolated per row into shared memory (the former k_interp, no longer
stored). Per call: 1 memset + 1 launch + 1 copy (instead of 1 H2D copy, 3-4 memsets, 18 launches, 2 copies). One trap: ptxas capped the
restructured kernel at 128 registers with ~500 bytes of spills per thread (the old per-level kernel had 168, no spills), making each
block 30-40% slower; `__maxnreg__(255)` removes the spills: per call (PAL Fast, uncontended) L0 483 us, L1 267, L2 128, L3 63, L4 25 =
about 0.99 ms, the same as the old per-level kernels (0.98 ms incl. k_global / k_interp). Bit-exact: all 50 vector cases, golden 85/85.

**The large stall was a pageable copy.** nsys (12 threads): `cudaLaunchKernel` took 163 us on average (11 us single-threaded), and every
`cudaMemcpyAsync` of the vector payload went to pageable memory (synchronous inside the driver). The payload and the bad flag are now one
buffer copied into a pooled pinned staging buffer, then a blocking-sync wait: launches dropped to 29 us average and **QTGMC Fast with the
GPU search clip went from 8.1 s to 4.6 s** (1500 frames). Also: 300-frame runs understate the GPU path (the frame registry and super
cache fill with ~4,000 cudaMallocs at the start); use the full 1500-frame clip for decisions.

**Results (QTGMC Fast, Prefetch 12, best of 3, High priority, framemd5 identical across all GPU variants; the CPU differs only by
meander=true, as accepted):**

| clip | CPU | GPU analysis, CPU search clip | GPU analysis + GPU search clip (tokens) |
|---|---|---|---|
| PAL 720x576i 4:2:2, 1500 frames | 6.85 s / 126 CPU-s | 6.13 s / 90 CPU-s | **4.58 s / 70 CPU-s** (1.50x) |
| XDCAM HD 1080i 4:2:2, 400 frames | 7.18 s / 112 CPU-s | 6.95 s / 83 CPU-s | **4.51 s / 62 CPU-s** (1.59x) |
| 4 concurrent PAL jobs | 31.2 s / 646 CPU-s | | **20.6 s / 416 CPU-s** (1.52x), 4/4 identical |

Prefetch 12-16 is best for the GPU path (24 is slower). CPU time left in the token run (module samples): NNEDI3 (plugins_JPSDR) 23
CPU-s, masktools 18, memcpy (vcruntime, frame copies) 15, mvtools2 CPU filters (MDegrain / MCompensate / CPU MSuper) 14, AviSynth core
4, mvtools_cuda + nvcuda 4. The GPU analysis itself is no longer the limit (making it 30% faster did not change the wall time).

**Next (in this order):** N (own NNEDI3 port, the largest CPU block left), then the output-side Fast stages (MSuper levels=1 + MCompensate
/ MDegrain on the device, the masktools resharp / clamp / KeepOnly Rep), fusing the KeepOnly kernels (about 20 launches per frame), and
the remaining pageable copies (MRecalculate / MFlowFps payloads, CudaDownload into AviSynth frames).

## 19. N done: CudaNNEDI3, bit-exact port of plugins_JPSDR nnedi3 (2026-10-29 morning)

`CudaNNEDI3(clip, field, dh, Y, U, V, nsize, nns, qual, etype, pscrn, threads, opt, fapprox)` in cudafilters.dll, kernels in
`mvcuda_nnedi3.cu` (library API 10). Not NNEDI3CL and not a re-derivation: a port of exactly what plugins_JPSDR 4.1.0 (the build FFAStrans
ships, source cloned to `C:\dev\ext\plugins_JPSDR-4.1.0`) executes on this machine: opt 6 (AVX2 + FMA3, no AVX-512), new prescreener
(`computeNetwork0new_AVX2`, `processLine0_AVX2_ASM`), int16 predictor (`extract_m8_i16_AVX2`, `dotProd_m32/48_m16_i16_AVX2`,
`e0_m16_FMA3`, `weightedAvgElliottMul5_m16_FMA3`, `castScale_AVX`). Each float step follows the asm order with explicit `__f*_rn`
intrinsics (FMA only where the asm has vfmadd); integer dot products are exact in any order.
- **The trap:** the asm uses `rcpps` / `rcpss` (elliott, 1 / sum of weights, 1 / window size: `rcpss(64)` is not 1/64; Intel's
  `rcp(1.0)` = 0x3F7FF000) and `rsqrtss` (1 / stddev). Their bits are CPU specific. The plugin measures them on the running CPU at load:
  on the Ultra 9 285K the result depends only on the top 11 (rcp) / 10 (rsqrt, plus exponent parity) mantissa bits and the exponent,
  identical on P- and E-cores; 2 x 2048 entries go to the GPU. The filter re-checks that structure over all 2^23 mantissas at startup and
  uses the CPU plugin if it does not hold (as on an AVX-512 CPU, where the plugin takes other code paths).
- Weights: read from the loaded plugins_JPSDR.dll (resource 101) and prepared by the plugin's own constructor code (same double
  arithmetic), in plain order. Padding: the plugin's mirror (32 columns, 3 field lines) as a reflection in the kernels.
- Kernels: `k_nn_pre` (one thread per sample: kept lines copied, missing lines by the cubic kernel, prescreener per group of 4 aligned
  columns, queue of the pixels for the predictor), `k_nn_pred` (queue, window in shared memory). GPU time 0.15 ms per PAL output frame.

Exactness (framemd5, real PAL clip): QTGMC Fast's call in 4:2:2 / 4:2:0 / 4:4:4 / Y8 at Prefetch 0 and 16, defaults (nsize 6 nns 1),
nsize 0..6, nns 0..4, qual 2, pscrn 3 / 4, etype 1, field -2 / 0 / 1: **all identical**; negative control (nns 1 vs 0) differs on every
frame. With U / V = false the plugin leaves those planes unwritten (undefined), luma identical. Full QTGMC Fast with `QTGMC_gpu_edi`:
output identical (PAL and HD). `run_cf_regression.ps1` has four NNEDI3 cases.

Speed: PAL 1500 frames 4.42 s / 49 CPU-s (without the GPU NNEDI3 4.57 s / 72 CPU-s; CPU 6.96 s / 129 CPU-s). On the HD clip the plugin's
prescreener sends few pixels to the network (1.8 CPU-s of NNEDI3), so there it changes little. The wall time is now bound by per-frame
host latency, not by CPU or GPU throughput: with all CPU mvtools filters skipped (MVTOOLS_SKIP) it is still 3.5 s at 11 busy cores; the
search-clip chain costs 2.6 ms of host time per frame (about 30 launches at 48 us under driver-lock contention plus the upload) and each
analysis 1.7 ms. Next: fewer launches (fuse the super-frame build, KeepOnlyBobShimmerFixes, TemporalSoften), then the output side
(stages 8-16 of section 15: MCompensate / MDegrain1 on the device, mt_logic / mt_clamp / resharp, KeepOnly Rep 4).

## 20. The whole QTGMC Fast chain on the GPU (2026-10-29, morning; library API 11)

With `global QTGMC_gpu_out = true` (besides `QTGMC_gpu`, `QTGMC_gpu_search`, `QTGMC_gpu_edi`) the output side of the Fast structure runs
on the device too: the NNEDI3 clip stays a token clip and `QTGMC_gpu_outside` makes `repair2` (MSuper levels=1 -> MCompensate x2 ->
mt_logic max / min -> MDegrain1 -> Merge -> vertical expand / inpand -> Merge -> RemoveGrain -> mt_lutxy resharp -> mt_clamp ->
KeepOnlyBobShimmerFixes), downloaded once; everything after it (Crop, propSet, ...) is unchanged CPU code. Only for the structure it
reproduces (`QTGMC_gpuOut` in the patched QTGMC: TR1 1, TR2 0, Rep1 0, SMode 2 not Precise, SLMode 2 with SLRad <= 1, Sbb 0, SVThin 0,
no source match / lossless / noise restore, NNEDI3 without ChromaEdi, 8-bit YUV, SubPel 1, no lsb / n16).

New pieces, each bit-exact against its CPU filter (all in `run_cf_regression.ps1`, 24 cases):
- **MCompensate(gpu=true)** and **MDegrain1(gpu=true)** (mvtools2, kernels `mvcuda_comp.cu`, 8 bit, pel 1): the super is
  MSuper(gpu=true) of a token clip (lite, header only); a pel-1 one-level super is the picture with replicated edges, so the kernels
  read the device frame through clamped coordinates (no super build). The host makes the per-block table with mvtools' own code
  (vectors with `mv * time256 / 256`, SAD < thSAD; `DegrainWeight` + `norm_weights<1>` for luma and chroma), the kernels sum the at
  most 2 x 2 overlapping blocks per sample (16-bit sum of `(v * window + 32) >> 6`, then `min(255, (sum + 16) >> 5)` as Short2Bytes_sse2).
  Tested with scene-change detection off (QTGMC's thSCD on a raw bob makes every frame a scene change, which tests nothing), thSAD
  5 / 400 / 10000, thSADC, plane=0, 4:2:2 / 4:2:0 / 4:4:4 / Y8, negative controls. Regeneration through mvtools producers
  (`producer_register`, one lock per instance because these filters are MT_MULTI_INSTANCE).
- **CudaLogic / CudaClamp / CudaExpand / CudaInpand / CudaLutxy** (cudafilters `CudaMt.cpp`, `mvcuda_mt`): masktools semantics for
  plane modes 3 and 2; mt_lutxy through the table the installed mt_lutxy makes on a 256 x 256 pattern.
- Bug found and fixed on the way: MDegrain's CPU path read a token source frame as pixels when the super was not lite (Y8, where
  MSuper(gpu=true) cannot be lite); it now always resolves token frames.

**The launch lock is the limit, again.** First measurement of the whole chain: 5.82 s (worse than the 4.43 s with only search clip +
NNEDI3 on the GPU), 143,617 launches for 600 frames (120 per output frame) at 97 us each. Fixes: (1) uploads (frames, block tables)
go straight onto the pipeline stream from pinned staging buffers released by polled events, instead of a host wait for the whole
pipeline queue (every MCompensate / MDegrain call waited for all pipeline work of all threads); 5.45 s. (2) KeepOnlyBobShimmerFixes,
which was a dozen small kernels per plane (76 of the 120 launches per frame), is one tiled kernel per call (`k_kobs`: shared-memory
tile with a halo as large as the chain's reach, plane borders by the same rules; tested Rep 1 / 3 / 4 / 6 / 7 / 11 / 12 / 31 / 42 x
chroma x 4 formats): 51.5 launches per frame, **3.50 s**. Default device-frame budget raised to 2 GB (HD 4:2:2 regenerated a few
frames at 1 GB; PAL unchanged).

**Results (QTGMC Fast, Prefetch 12, High priority; output byte-identical to the GPU-analysis runs, which differ from the CPU only by
meander=false as accepted):**

| clip | CPU | GPU search + NNEDI3 | **whole chain on the GPU** |
|---|---|---|---|
| PAL 720x576i 4:2:2, 1500 frames | 6.78 s / 126 CPU-s | 4.29 s / 47 CPU-s | **3.50 s / 17.5 CPU-s (1.94x, 7x less CPU)** |
| XDCAM HD 1080i 4:2:2, 400 frames | 7.25 s / 117 CPU-s | 4.57 s / 56 CPU-s | **2.77 s / 11.8 CPU-s (2.6x)** |
| 4 concurrent PAL jobs | 31.2 s | 20.2 s | **16.4 s**, 4/4 identical |

Small budgets (MVCUDA_CACHE_MB=16 / 64) stay byte-identical (131,000 regenerations at 16 MB). Prefetch 12 is best (16-32 only add
lock contention). What limits it now: the wavefront analysis takes about 1 ms of GPU time per output frame (2 calls of ~1.1 ms,
latency-bound, overlap about 2.2), about the wall time per frame; and still ~50 launches + ~30 other API calls per output frame.
Next candidates: TemporalSoften (3 kernels) and the per-plane resample launches into single kernels; k_wave latency (the
per-block step of ~2 us) or two analyses (bVec / fVec of one frame) per launch.

## 21. One submitter thread, own-input waits, fused QTGMC stages (2026-10-29 night; library API 12)

**What limited section 20, measured.** (1) Driver calls from many threads contend: a kernel launch costs about 8-9 us from one thread on
this Windows / WDDM driver (any parameter size, 16 B to 4.4 KB), 20 us with 4 other threads launching, 56 us with 12 (microbenchmark
`launchbench` in the session scratchpad; event / copy calls from other threads hardly matter). (2) With launches moved to one thread,
the single in-order pipeline stream became the GPU-side bottleneck: 628 ms executing plus 498 ms of gaps between ops that were already
queued (engine switches around copies and memsets: 12-36 us each) in a 1544 ms span. (3) Every analysis waited for *everything* queued
on the pipeline (other frames' work), on the host and on the GPU. (4) Each `cudaEventSynchronize` spun: pooled events lacked
`cudaEventBlockingSync` (the device flag does not cover events).

**Changes (all in `Sources\cuda`, output byte-identical at every step):**
- *Submitter thread* (`mvcuda_frames.cu`): every pipeline-stream call is a closure queued with `pipe_submit` and issued in submission
  order by one thread (batches, no lock per op); `pipe_call` waits for its own op only.
- *Frames are one buffer* (planes at 256-byte aligned offsets): one copy per upload / download instead of one per plane.
- *Uploads ahead of the pipeline*: copied on an upload stream as soon as the worker has the data, into buffers from a pool that refills
  only after the pipeline stream has passed the buffer's last use (`up_alloc` / `up_free`); the pipeline stream only waits for an event.
- *Own-input waits*: a frame carries the sequence number of the op that made it (`DevFrame::seq`); the submitter records marks
  (events) right after such ops on demand (`mark_after`, `mark_request` / `mark_in_op`). Analysis and downloads (on a copy stream) wait
  for their frames' producers only.
- *Fewer ops*: no memsets (TemporalSoften scratch cleared by `k_decide`; NNEDI3 queue counters alternate), resample / RemoveGrain /
  TemporalSoften accumulate one launch for all planes; the analysis calls are issued by the submitter too; `k_wave` hands its result
  over itself (the last CTA copies the vectors and the bad flag into the pinned host buffer and clears its state; buffers come from a
  pool whose state part is zero), so no memset and no copy per analysis.
- *Fused QTGMC stages* (cudafilters `CudaQtgmc.cpp`, tests in `run_cf_regression.ps1` with negative controls):
  `CudaQtgmcSharpLimit(edi, degrain1, fComp1, bComp1, resharpExpr, rgBlur, Sovs)` = binomial1 Merge, vertical expand / inpand, Merge,
  RemoveGrain 12, resharp mt_lutxy, max / min of three, mt_clamp: eleven filters in one tiled kernel (`k_sharp_limit`, mvcuda_kobs.cu);
  `CudaQtgmcBinomial0(b)` = TemporalSoften radius 1 and 2 plus two MergeLuma: eight launches in three (the radius-1 SADs are those of
  radius 2 against the nearer neighbours). Their CPU fallbacks are the original filter chains. `make_qtgmc_gpu.py` uses both.
- *Blocking-sync pooled events*: CPU time halved, and two concurrent jobs no longer burn 20 CPU-s each in the driver.

**Results (QTGMC Fast, Prefetch 12, best of 3, High priority, quiet machine):**

| clip | CPU (24 threads) | section 20 | **now** |
|---|---|---|---|
| PAL 720x576i 4:2:2, 1500 frames | 6.82 s / 131 CPU-s | 3.50 s / 17.5 | **2.19 s / 8.3 CPU-s (3.1x)** |
| XDCAM HD 1080i 4:2:2, 250 frames (whole clip) | 6.97 s / 114 CPU-s | 2.77 s / 11.8 | **1.18 s / 4.1 CPU-s (5.9x)** |
| 4 concurrent PAL jobs | 31.2 s | 16.4 s | **9.9 s**, about 3 user-CPU-s per job |

Medium (GPU analysis + search clip + NNEDI3, CPU output side): 6.51 s / 64.5 CPU-s vs 7.71 s / 148.5.

**Correction (sections 18 and 20 too):** the XDCAM HD test clip has 250 frames; the runs labelled "400 frames" processed the whole
clip (250 source, 500 output frames). CPU and GPU always ran the same input, so the ratios stand.

**Exactness:** `run_cf_regression.ps1` 28 / 28 (now with Binomial0 and SharpLimit), `run_qtgmc_vec.ps1 -NoMeander` 50 / 50, golden 85 / 85,
every preset (Fast, Very Fast, Faster, Medium, Slower, Placebo) byte-identical across the five GPU variants of `run_tokbench.ps1`, HD
too, budgets 16 / 64 MB identical, and the whole GPU chain equals the CPU computation of the same algorithm
(`MVTOOLS_GPU_FORCE_FALLBACK=1` with GPU analysis only: same md5, PAL and HD). The only accepted difference to stock QTGMC remains
meander=false.

**Tried without gain:** analysis streams at high priority; a raw (pre-decoded) source; analysis issue through the submitter alone (wall
time equal, CPU lower; kept because it removed the remaining multi-thread launches).

**What limits now:** the submitter is about 70% busy (about 0.5 ms of driver calls per output frame at ~9 us each); per frame about 30
driver calls remain (analysis 2 x (wait, k_wave) + one super build of 6 launches, search clip 8, NNEDI3 2, output side 7 plus 3 table
waits, KeepOnly 2). Multi-process: CUDA contexts of different processes time-slice (no MPS on Windows), so N jobs take about N times one.
Next candidates: the super build in one launch (saves 5 per frame), MCompensate / MDegrain tables read without separate upload waits,
RemoveGrain + GaussResize + MergeLuma of the search clip fused, and the output side of the other presets (MDegrain2 / TR2 > 0).
## 22. Every preset on the GPU (2026-10-29, midday; library API 13)

**What each preset uses** (read from FFAStrans QTGMC.avsi, defaults, InputType 0; ProgSADMask is forced to 0 for InputType 0,
Rep1 = 0, SVThin = 0, SourceMatch = 0, Lossless = 0 for all; NoiseProcess 2 means NoisePreset "Fast": FFT3DFilter, NoiseTR 1,
no DenoiseMC / StabilizeNoise, NoiseDeint "" = DoubleWeave, GrainRestore 0.3, NoiseRestore 0.1):

| preset | TR0 | TR1 | TR2 (MDegrain) | Rep0 / Rep2 | SMode / SLMode / SLRad | Sbb | SubPel | Precise | ChromaMotion | SrchClipPP | EdiMode (nsize / nns) | NoiseProcess | blksize / overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ultra Fast | 1 | 1 | 0 | 0 / 3 | 2 / 0 / - | 0 | 1 | no | no | 1 | RepYadif | 0 | 32 / 8 |
| Super Fast | 1 | 1 | 0 | 0 / 3 | 2 / 0 / - | 0 | 1 | no | no | 1 | NNEDI3 (4 / 0) | 0 | 32 / 8 |
| Very Fast | 1 | 1 | 0 | 0 / 4 | 2 / 2 / 1 | 0 | 1 | no | no | 2 | NNEDI3 (4 / 0) | 0 | 32 / 8 |
| Faster | 1 | 1 | 0 | 0 / 4 | 2 / 2 / 1 | 0 | 1 | no | no | 2 | NNEDI3 (4 / 0) | 0 | 32 / 16 |
| Fast | 2 | 1 | 0 | 3 / 4 | 2 / 2 / 1 | 0 | 1 | no | no | 2 | NNEDI3 (5 / 0) | 0 | 16 / 8 |
| Medium | 2 | 1 | 1 (MDegrain1) | 3 / 4 | 2 / 2 / 1 | 0 | 1 | no | no | 3 | NNEDI3 (5 / 1) | 0 | 16 / 8 |
| Slow | 2 | 1 | 1 (MDegrain1) | 4 / 4 | 2 / 2 / 1 | 0 | 2 | no | no | 3 | NNEDI3 (1 / 1) | 0 | 16 / 8 |
| Slower | 2 | 2 | 1 (MDegrain1) | 4 / 4 | 2 / 2 / 1 | 1 | 2 | no | yes | 3 | NNEDI3 (1 / 1) | 0 | 16 / 8 |
| Very Slow | 2 | 2 | 2 (MDegrain2) | 4 / 4 | 2 / 2 / 1 | 1 | 2 | yes (rgBlur 11) | yes | 3 | NNEDI3 (1 / 2) | 2 | 16 / 8 |
| Placebo | 2 | 2 | 3 (MDegrain3) | 4 / 4 | 2 / 2 / 3 | 3 | 2 | yes (rgBlur 11) | yes | 3 | NNEDI3 (1 / 2) | 2 | 16 / 8 |

Output-side operations that were not on the GPU (API 12: only TR1 1, TR2 0, SMode 2, SLMode 2 / SLRad 1, Sbb 0, SubPel 1), by the
number of presets (Ultra Fast .. Placebo) that need them: final temporal smooth TR2 > 0 (MSuper + MDegrain1/2/3 of the sharpened
clip): 5; SubPel 2 (pel-2 MCompensate / MDegrain): 4; TR1 2 (second MDegrain1 at delta 2, three-way Merge): 3; Sbb back blend
(makediff, RemoveGrain 12, GaussResize p=5, makediff): 3; Precise (RemoveGrain 11, vresharp limit lutxy): 2; grain / noise restore
(mt_lut + mt_adddiff of the CPU noise clip): 2; SLMode 0 (no temporal limit): 2; SLRad 3 (MCompensate at delta 3, five-way max /
min): 1; MDegrain2 / MDegrain3: 1 each; a non-NNEDI3 interpolation (RepYadif): 1. Search side (only Fast had the GPU search clip):
SrchClipPP 3 (two mt_lutxy): 5; TR0 1 without Rep0: 4; ChromaMotion (Merge instead of MergeLuma, chroma KeepOnly / RemoveGrain /
lutxy, chroma analysis): 3; SrchClipPP 1 (BilinearResize half / RemoveGrain / BilinearResize back): 2.

**Correction of section 21 and of the session brief:** "the other presets get GPU analysis, search clip and NNEDI3" was not right.
The GPU search clip was only used by Fast (it needed TR0 2, SrchClipPP 2, Rep0 > 0, no chroma motion); Medium and slower ran the
CPU search clip (SrchClipPP 3), Faster / Very Fast the CPU search clip too (TR0 1) but already the GPU output side.

**New pieces (all bit-exact, in `run_cf_regression.ps1`, 59 cases, 7 of them negative controls that must differ):**
- *MCompensate / MDegrain1..6 gpu=true with pel 2* (`mvcuda_comp.cu`, `mvcuda_degrain` replaces `mvcuda_degrain1`): the three
  Wiener sub-pel planes of a device frame (padded picture, MSuper sharp=2 rules incl. the edge cases) are made once by one
  `k_subpel` launch the first time a pel-2 consumer needs them and kept with the frame (`DevFrame::subpel`, counted in the
  registry budget); blocks read plane (blx & 1) | (bly & 1) << 1 at (blx >> 1, bly >> 1) as `MVPlane::GetPointer`. MDegrainN's
  host table: vectors of every radius + `NORMWEIGHTS` weights (the 8-bit SIMD 16-bit sums cannot overflow: weights sum to 256).
- *CudaQtgmcSharpLimit* generalised (`mvcuda_qtgmc_sharp`): binomial1 for TR1 0 / 1 / 2, SMode 0 / 1 / 2, Precise (vresharp limit
  through a table made by the installed mt_lutxy), temporal limit over 0 / 3 / 5 references. RemoveGrain 11 = 12: RgTools 1.2 runs
  `rg_mode12_sse` for mode 11 on SSE4.1 CPUs (no scalar tail), so the GPU uses mode 12 there (only with SSE4.1, tested).
  With smode 0 / nref 0 it is QTGMC's lossed1.
- *CudaQtgmcBackBlend* (`mvcuda_qtgmc_backblend`): Sbb in three launches (makediff + RemoveGrain 12, first GaussResize pass,
  second pass fused with the final makediff and the optional temporal limit); the luma programs are made by the AviSynth port
  in `Resample.cpp`; only luma matters (the inner makediff has U=V=1, the outer copies chroma).
- *CudaQtgmcAddNoise*: grain / noise restore as one table lookup per sample; the table is the CPU chain (mt_lut + mt_adddiff with
  QTGMC's arguments) run on the 256 x 256 pattern. The noise clip itself (sh_Bob, FFT3DFilter, makediff, DoubleWeave) stays on
  the CPU and is uploaded: FFTW's float results cannot be matched bit for bit on the GPU (plan section 7).
- Search side: `CudaQtgmcBinomial0(chroma=true)` (Merge instead of MergeLuma), `CudaBilinearResize` (triangle kernel, same 2-D
  code as CudaGaussResize), `CudaRemoveGrain` mode 11; `QTGMC_gpu_srchclip` now covers TR0 0..2, Rep0, SrchClipPP 0..3 and
  ChromaMotion (where the CPU leaves chroma undefined, masktools U=V=1 / RemoveGrain -1, the GPU copies it; the luma-only analysis
  does not read it).
- `make_qtgmc_gpu.py`: `QTGMC_gpuOut` admits TR1 <= 2, TR2 <= 3, SMode <= 2 (incl. Precise), SLMode 0 / 2 (any SLRad), Sbb 0..3,
  SubPel 1 / 2 (sharp 2), any EdiMode (a CPU-made interpolation is uploaded with CudaUpload) and noise restore;
  `QTGMC_gpu_outside` now returns QTGMC's addNoise2 (lossed2 = sharpLimit2 = repair2 for the admitted structures). Not admitted
  (CPU, exact): SLMode 1 / 3 / 4 (Repair-based or after-smooth limits), Sbb with SVThin, Rep1 > 0, source match, lossless,
  lsb / n16, 16-bit input, ChromaEdi, EdiExt.

**Bug found and fixed:** under small device budgets a filter whose input was an ordinary (uploaded, producer-less) frame failed
when the regeneration of its other inputs evicted that upload: `with_inputs` now re-uploads and pins non-token inputs after the
regenerations. Found by the 16 MB test: Placebo crashed (the CPU fallback of MDegrain3 then threw on a lite token it could not
make again; an AviSynth error inside ffmpeg shows as an access violation). Also: a test loop that compared an old framemd5 file
after a crashed run reported "identical"; `run_presetcheck.ps1` relies on `run_tokbench.ps1` throwing on a non-zero exit code.

**Finding: stock Ultra Fast is not reproducible on the PAL clip, CPU included.** Yadifmod2 (FFAStrans yadif17\x64\yadifmod2.dll,
EdiMode RepYadif) returns different pixels in the first / last columns (x = 0, 1, 717..719) between identical single-threaded
runs: it reads uninitialised memory at the row ends, and not the source frame's padding (`AddBorders(8,0,8,0).Crop(8,0,-8,0)` in
front does not help). QTGMC spreads that into the picture. On the HD clip it happened to be stable. Ultra Fast's GPU path was
verified with `Extra=', EdiMode="Bob"'` (same structure, deterministic interpolation, also uploaded); a GPU yadif port would
have no exact target.

**Exactness (all byte-identical):** `run_presetcheck.ps1 -Fallback` (all five GPU variants of `run_tokbench.ps1` plus 'gpu no
search' with `MVTOOLS_GPU_FORCE_FALLBACK=1`, 100 source frames) for every preset on PAL and on HD (Ultra Fast PAL with EdiMode Bob,
see above); `run_gpucheck.ps1`: no run-time fallback in any preset; `run_qtgmc_vec.ps1 -NoMeander` 50 / 50; golden 85 / 85;
`MVCUDA_CACHE_MB=16` identical for Super Fast, Fast, Medium, Slow, Slower, Very Slow, Placebo (PAL; 140,000 regenerations in
Placebo) and Placebo / Medium HD; release set `profile\build_rel13` (build.bat, all archs) gives the same md5s.

**Results (Prefetch 12, best of 3, High priority, quiet machine; PAL 1500 source frames, XDCAM HD whole clip 250 frames):**

| preset | PAL CPU | PAL GPU | factor | HD CPU | HD GPU | factor |
|---|---|---|---|---|---|---|
| Ultra Fast | 2.47 s / 30.7 CPU-s | 1.32 s / 9.4 | 1.9x | 3.32 s / 40.3 | 0.73 s / 7.4 | 4.5x |
| Super Fast | 3.10 s / 55.9 | 1.34 s / 8.1 | 2.3x | 3.15 s / 40.9 | 0.61 s / 3.2 | 5.2x |
| Very Fast | 4.83 s / 82.7 | 1.45 s / 8.6 | 3.3x | 5.05 s / 69.0 | 0.61 s / 3.1 | 8.3x |
| Faster | 5.07 s / 91.2 | 1.57 s / 7.7 | 3.2x | 5.20 s / 73.2 | 0.67 s / 3.6 | 7.8x |
| Fast | 6.94 s / 130.5 | 2.12 s / 8.9 | 3.3x | 7.25 s / 116.2 | 1.18 s / 4.5 | 6.1x |
| Medium | 7.93 s / 140.0 | 2.61 s / 8.7 | 3.0x | 8.05 s / 119.5 | 1.26 s / 5.1 | 6.4x |
| Slow | 9.85 s / 183.3 | 3.04 s / 9.1 | 3.2x | 9.43 s / 139.7 | 1.45 s / 5.0 | 6.5x |
| Slower | 14.29 s / 281.6 | 5.06 s / 12.9 | 2.8x | 13.36 s / 227.7 | 3.14 s / 5.2 | 4.3x |
| Very Slow | 22.49 s / 342.6 | 9.47 s / 42.2 | 2.4x | 23.32 s / 275.0 | 7.19 s / 32.4 | 3.2x |
| Placebo | 27.17 s / 462.0 | 13.17 s / 41.2 | 2.1x | 36.43 s / 421.0 | 9.49 s / 30.8 | 3.8x |

Before this section (API 12): Medium PAL 6.51 s / 64.5 CPU-s on the GPU, Slow HD 58 fps = about 8.6 s; now 2.61 s / 8.7 and 1.45 s.
4 concurrent PAL jobs (1500 frames each, every job identical to its single run): Fast CPU 29.9 s / 668 CPU-s, GPU 10.9 s / 60;
Medium 33.8 s / 756 vs 13.1 s / 61; Slower 55.5 s / 1259 vs 24.7 s / 71. (Section 21 measured 9.9 s for Fast; today the API 12
DLL set gives the same 10.7-10.9 s as the new one, so that difference is the machine state, not a regression.)

**What limits now:** the GPU analysis. nsys, Slower PAL: `k_wave` is 91% of the kernel time (2.7 ms per call with pel 2 and chroma
SAD, 4 analyses per output frame); the new output-side kernels are small (pel-2 MDegrain 66 us, sub-pel build 44 us, sharp 11 us,
back blend about 30 us per frame). Very Slow / Placebo also spend about 30 CPU-s in the CPU noise branch (FFT3DFilter etc.) and
use 6 analyses per frame. Ultra Fast's interpolation (Yadifmod2 + Repair) is on the CPU. Next candidates: k_wave latency for pel 2
/ chroma (shared-memory reference tiles for the sub-pel planes, more blocks in flight per wave), analysing bVec / fVec pairs per
launch; an FFT3D port only with an accepted non-bit-exact result (decision for the user).
## 23. To do: VRAM use (measured 2026-10-29 evening, release 2026-10-29b)

Peak dedicated VRAM per process (Windows counter `\GPU Process Memory(pid_*)\Dedicated Usage`; nvidia-smi shows N/A per process
under WDDM), Prefetch 12, md5 identical in every cell; times are single runs with startup, only to show the regeneration cliff:

| | default (min(free/4, 2 GB)) | MVCUDA_CACHE_MB=512 | 256 | 128 |
|---|---|---|---|---|
| Fast PAL | 3362 MiB 3.6 s | 1396 3.5 s | 1066 3.5 s | 1006 3.5 s |
| Slower PAL | 4348 6.8 s | 2280 6.8 s | 1984 6.8 s | 2357 **96 s** |
| Placebo PAL | 4348 14.7 s | 2302 14.7 s | 2713 **206 s** | 2523 **232 s** |
| Fast HD | 3086 3.6 s | 1904 3.6 s | 1884 6.9 s | 1710 7.0 s |
| Slower HD | 4860 4.7 s | 4087 **32 s** | 3698 33 s | 3662 34 s |
| Placebo HD | 5276 12.5 s | 4153 **84 s** | 3973 89 s | 3921 88 s |

Fast HD, 128 MB: Prefetch 1 1030 MiB, 4 1178, 12 1710 (each thread keeps its last 8 frames outside the budget, `t_held`).
Where it goes: (1) the registry budget is far above the working set (Fast PAL needs about 256 MB, Slower / Placebo PAL about
512 MB, Slower / Placebo HD more than 512 MB); (2) the free pools (`pipe_pool`, `g_pool`, `g_up`) never give buffers back,
so a smaller budget does not lower the peak in proportion; (3) a fixed part of about 1 GB per process: context, modules and
probably the local-memory reservation for `k_wave` (cuobjdump: STACK 5248 bytes per thread for GEN=true, 2064 for GEN=false;
reserved for every resident thread: up to about 1.3 GB on the 170-SM 5090, about 0.2 GB on a 3060; not verified); (4) pel-2
sub-pel planes make a referenced frame 4x larger (`extra_acct`).

To do, in this order:
1. **Auto-scaled budget**: start small (about 256 MB) and grow when frames are regenerated after eviction (the registry counts
   regenerations), up to the current cap; replaces hand-set `MVCUDA_CACHE_MB` (env var, read once per process; keep it as an
   override). Goal: about the 512 column above with default settings, no regeneration cliff, also with 4 concurrent jobs
   (today a later job sizes its budget from the VRAM the others left).
2. Give free buffers back: stream-ordered allocator (`cudaMallocAsync` + memory pool with a release threshold) instead of the
   never-shrinking pools; check that the launch stalls of section 20 (`cudaMalloc` synchronising) do not come back.
3. Measure the `k_wave` local-memory reservation (`cudaDeviceGetLimit(cudaLimitStackSize)` / cudaMemGetInfo around the first
   launch) and shrink the stack frame if it is a large part of the fixed 1 GB.
4. Optional: sub-pel planes computed on the fly instead of cached (less VRAM, more GPU work on pel-2 presets).
Checks as usual: identical md5 (run_presetcheck -Fallback), MVCUDA_CACHE_MB=16 tests, 4 concurrent jobs, peak VRAM table again.
Status 2026-10-29 (late night): items 1 and 3 done, item 2 not done, item 4 open; see section 24.
## 24. k_wave latency and VRAM (2026-10-29 late night; library API 14)

**Where the time went (nsys, build of section 22/23, Prefetch 12, PAL 1500 / HD 250 source frames).** `k_wave` share of all
kernel time: Fast 82% (PAL) / 81% (HD), Slower 90% / 90%, Placebo 93% / 94%. Mean call 1.15 / 2.75 / 3.30 ms PAL, 3.0 / 7.0 /
8.3 ms HD. Section 22's 91% for Slower is confirmed. Correction: the whole-chain kernel time is not the wall limit by itself;
see "what limits" below.

**What was slow (clock64 phase counters, `-DMVC_PHASE_PROF`):** every preset with pel-search 2 (Slow .. Placebo) ran the
general kernel (`GEN`, mvc_refine with runtime-size SAD loops) on *every* level, because one level (0) needed Hex2Search(2) or
UMH(2): a refine batch cost about 6,000 cycles there against 2,400 on the ring path; level-0 block of Slower about 34,000
cycles (Fast 12,600), Placebo 56,000.

**Changes (all in `Sources\cuda\mvcuda_analyse.cu`, output byte-identical at every step):**
- *Fixed plans* (`wave_plan`): Hex2Search(2..3) = hexagon (CheckMVdir) + radius-1 ring around the moved centre, UMH(2..3) = cross
  +-1 and one hex grid ring, then the hexagon: run in the fast kernel as a stage loop with one inlined SAD call site. QTGMC never
  launches the GEN kernel now.
- *Packed 8-bit SADs* (`pk_load` / `pk_sum`): a lane takes 4 columns of every 4th row, two aligned word loads per row, funnel
  shift, `__vsadu4`; half the loads and registers of the byte form (integer sums, order-independent). `MVCUDA_WAVE_PACK=0`
  switches back for A/B tests.
- *Fused fold*: the group leaders add (cost << 8 | index) to a shared 64-bit atomicMin (and flag negative / overflowing cost
  parts, falling back to the sequential fold exactly like `wave_fold_par`): 1 barrier per batch instead of 3.
- Barrier after the phase-A fold removed, post-fold barrier skipped after the last stage; the block's vector is published by
  warp 1 after the end-of-block barrier (its release fence overlaps warp 0's wait / predictor fetch for the next block; 5%).
- R = 1 round for batches that fit one round (phase A, hexagon, radius-1 rings), R rounds only for larger batches (coarse
  radius-2 rings, UMH); register cap `MVC_WAVE_REGS` 96 with `MVC_WAVE_RSAMP` 24 (GEN kernels keep 255).
- Bad-vector search builds its candidate lists in shared memory (stack frame 2.3-2.6 KB -> 1.0-1.3 KB per thread for the QTGMC
  kernels; the device reserves stack x max resident threads x SMs of local memory: about 0.4-0.6 GB less on the 5090).

**Isolated latency per analysis call (Prefetch 1 under nsys, median k_wave, `profile\run_wavelat.ps1`):**
| preset | PAL before | PAL now | HD before | HD now |
|---|---|---|---|---|
| Fast | 1021 us | 1037 (+2%) | 2534 | 2524 |
| Slower | 2101 | **1669 (-21%)** | 5374 | **3933 (-27%)** |
| Placebo | 2890 | **2035 (-30%)** | 7059 | **4740 (-33%)** |

Fast's single call is not faster (its batches were already on the fast path; the 96-register cap costs it a little); Fast
gains end to end from the higher residency (more concurrent calls per SM).

**Tried and measured without gain (kept out or reverted):** compile-time sizes for the GEN kernels behind a noinline batch call
(255 registers, 2 CTAs/SM: slower); register caps 64 / 80 / 128 / 168 (96 is best end to end; 255 is worst); fewer CTAs per call
(40-70% of the rows: slower); L1 prefetch of the reference window around the block's predictor (slower); relaxed polling +
`fence.acq_rel` instead of `ld.acquire` (the fence on the critical path cost 5%); `__nanosleep` backoff in the row wait (no
change); `CUDA_DEVICE_MAX_CONNECTIONS=32` (the default 8 hardware queues do cap concurrency at 8 kernels, measured with a
micro-benchmark, but at Prefetch 12 only 6-7 analyses are in flight; with Prefetch 24 + 32 queues it got slower).

**What limits now (correction of the section-22 idea "more blocks in flight per wave"):** the GPU is not short of CTA slots.
About 6-7 analyses are in flight at Prefetch 12 (the 12 frame threads spend about 60% of their time waiting for analysis
results); per call about 75% of the duration is the wavefront's own critical path (isolated 1.04 ms Fast, 1.67 ms Slower vs
1.4 / 2.5 ms in the full run), and more concurrency (higher Prefetch) makes each call slower than it adds. Per-call latency
is the lever: per level about 2.5-5 us per wavefront step, and the coarse levels are about half of a call (they wait for each
other through the global-motion estimate). Ideas left: interleaving two wavefronts (bVec/fVec) in one CTA, shared-memory
reference windows (only saves round trips after phase A; the L1 prefetch result suggests little), sm_90 clusters for the row
hand-over (not on RTX 30xx).

**Very Slow / Placebo: the CPU noise branch.** Placebo PAL with NoiseProcess=0 takes 8.2 s instead of 12.3 s; with Prefetch 24
the normal Placebo takes 8.1 s (identical md5): the frame threads block on FFT3DFilter. Users of Very Slow / Placebo should use
a higher Prefetch (not changed in any script; reported numbers stay at Prefetch 12).

**Finding: FFT3DFilter is not deterministic under load.** About 1 in 50 Placebo runs gave a second, always the same, md5 class
(122 of 3000 frames, ~60 small clusters). FFT3DFilter alone: 2 of 10 runs differ when the CPU is busy, 10 of 10 identical with
`measure=false` (FFTW_MEASURE plans depend on timing). This is the stock CPU plugin (also a pure CPU QTGMC), not the GPU path;
`run_ab_avs.ps1` keeps mismatching framemd5 files. `measure=false` in QTGMC's noise call would fix it (decision for the user).

**VRAM (section 23):**
- *Item 1, auto-scaled budget* (`mvcuda_frames.cu`): `MVCUDA_CACHE_MB` still fixes it. Otherwise it starts at max(256 MB, 128
  frames of the first frame's size) and grows: (a) a consumer finding its frame near the LRU end (reuse distance > 80% of the
  budget) grows it to 1.5x that distance before evictions cost anything; (b) per 64 frames made, >= 8 lookups of evicted tokens
  (counted only outside regenerations) grow it to 1.25x their median reuse distance (evicted tokens are remembered as "ghosts"
  with the bytes put up to their last use), at least by half. Cap min(free / 3, 4 GB) (was min(free / 4, 2 GB)).
  What did not work, measured: growing per regeneration, by regeneration rate, or on single misses (late requests of frames
  evicted early through AviSynth's own caches have distances that grow with the run time and ran the budget to the cap).
- *Bug found and fixed (also in the released build): regeneration storms.* When a consumer finds token X evicted it regenerates
  frame (producer, n) as a new token Y, but AviSynth's frame cache keeps handing out the frame with X, so every later consumer
  made it again: storms that went on after the budget had grown. The fixed 2 GB default was at the edge for Placebo HD (one
  baseline run 51.8 s instead of 12 s); fixed 768 MB: Slower HD 160,000 regenerations, 8x the time. Now the registry keeps the
  newest live token per (producer, frame) and a lookup of an evicted token is served by it (`MvCudaFrameStats::redirects`,
  library API 14: all three DLLs together). Exact: a producer makes frame n bit-identically every time (the md5 checks below).
- *Item 3, local-memory reservation*: see the stack change above; QTGMC no longer launches the 5.2 KB-stack GEN kernel.
- *Item 2 not done*: with the budget sized to the chain the free pools only reach the size the chain needs, so releasing them
  would lower only transient overshoot, not the peak; left open (would need `cudaMallocAsync` and a check for launch stalls).

Peak per-process VRAM (run_vram.ps1, default settings, Prefetch 12; md5 identical in every cell; wall times single runs incl.
start-up):

| | before (fixed 2 GB, section 23) | now | budget reached |
|---|---|---|---|
| Fast PAL | 3334 MiB | 882-936 | 313-363 MB |
| Slower PAL | 4370 | 1130-1218 | 384-481 MB |
| Placebo PAL | 4362 | 1224-1440 | 466-612 MB |
| Fast HD | 3106 | 1650-1966 | 0.9-1.2 GB |
| Slower HD | 4802 | 3174-3586 | 1.4-1.9 GB |
| Placebo HD | 5276-6204 (storm-prone) | 3692-5954, no storms in 5 runs | 1.9-4.0 GB |

4 concurrent PAL jobs, total VRAM over idle (nvidia-smi): Fast 13.5 GB -> 3.7 GB, Slower 17.6 GB -> 4.4 GB.

**Exactness (all on build_e = API 14):** run_cf_regression 59/59 PASS; run_qtgmc_vec -NoMeander 50/50 IDENTICAL; golden 85/85;
run_presetcheck -Fallback ALL PASS for every preset on PAL (Ultra Fast with EdiMode="Bob") and HD; the same with
MVCUDA_CACHE_MB=16 (PAL Super Fast, Fast, Medium, Slow, Slower, Very Slow, Placebo; HD Fast, Medium, Placebo), same md5s;
full-length outputs (1500 PAL / 250 HD source frames) identical to the section-23 build for Fast / Slower / Placebo PAL and HD;
4 concurrent jobs 4/4 identical.

**Results (Prefetch 12, best of 3, High priority; 'before' = the section-22/23 build timed the same night):**

| preset | PAL CPU | PAL GPU before | **PAL GPU now** | HD CPU | HD GPU before | **HD GPU now** |
|---|---|---|---|---|---|---|
| Ultra Fast | 2.56 s / 33.3 CPU-s | 1.32 | **1.21 s / 9.2** | 3.37 / 38.6 | 0.74 | **0.74 / 7.8** |
| Super Fast | 3.15 / 57.4 | 1.33 | **1.24 / 7.3** | 3.25 / 42.9 | 0.60 | **0.62 / 3.8** |
| Very Fast | 4.99 / 82.5 | 1.53 | **1.37 / 8.3** | 5.16 / 70.5 | 0.63 | **0.64 / 3.7** |
| Faster | 5.31 / 93.0 | 1.51 | **1.42 / 8.5** | 5.22 / 73.8 | 0.69 | **0.63 / 3.4** |
| Fast | 7.11 / 128.4 | 2.21 | **1.79 / 8.2** | 7.42 / 117.5 | 1.22 | **0.93 / 4.4** |
| Medium | 8.00 / 146.8 | 2.69 | **2.29 / 8.2** | 8.42 / 119.6 | 1.29 | **1.01 / 5.4** |
| Slow | 9.99 / 183.2 | 3.13 | **2.80 / 9.8** | 10.15 / 147.5 | 1.49 | **1.18 / 5.4** |
| Slower | 14.38 / 283.1 | 5.21 | **4.33 / 12.2** | 14.42 / 243.2 | 3.14 | **2.12 / 6.5** |
| Very Slow | 22.52 / 343.1 | 9.62 | **9.07 / 41.4** | 28.07 / 297.8 | 7.32 | **7.04 / 34.4** |
| Placebo | 27.98 / 462.8 | 13.20 | **11.72 / 40.6** | 36.57 / 410.2 | 9.72 | **8.97 / 33.6** |

4 concurrent PAL jobs (1500 frames each): Fast CPU 30.4 s / GPU before 10.9 / **now 9.0 s**; Medium 34.3 / 13.1 / **11.4**; Slower
56.1 / 24.7 / **21.0** (all jobs identical to their single runs).

**Tools (profile\):** `run_nsys.ps1` (nsys kernel summary; hidden window), `nsys_timeline.py` (kernel union / k_wave
concurrency histogram / per-thread API time / launch-to-start delay from an nsys sqlite export), `run_wavelat.ps1` (isolated
per-call latency), `run_ab_avs.ps1` (A/B of DLL folders, keeps md5 mismatches against stored references), `make_qtgmc_avs.ps1`,
`run_repeat.ps1` (N runs, differing frames), `run_presettiming.ps1 -Dir`, `run_vram.ps1` (prints budget / regenerations with
`-Env2 CUDAFILTERS_STATS=1`), `run_concurrent.ps1` now passes `-nostdin` (four ffmpeg processes started from a PowerShell job
hung in PeekNamedPipe on their shared stdin; a harness artifact, not a GPU hang).
Test DLL set: `profile\build_e` (API 14; sm_120 dev build). The release DLL (build.bat, all architectures) was not rebuilt.
Released 2026-10-30: commit 01b702a, pre-release gpu-qtgmc-2026-10-30 (release set `profile\build_rel14`).

## 25. To do (agreed 2026-10-30): mvtools on the cudafilters model, then 10 bit everywhere

User's request: port the older mvtools GPU filters to the model the cudafilters use (both inside and in how scripts use them),
and support 10 bit wherever the CPU filter supports it. Rules unchanged: bit-exact with the CPU filter, silent fallback,
`gpu=true` stays opt-in.

**Where the mvtools filters stand (code read 2026-10-30).** MCompensate / MDegrain1..6 already follow the model (registry,
token output, regeneration, submitter). The older paths do not:
- MAnalyse host path (and the slice kernel, `meander=true`), MRecalculate and MFlowFps launch from the worker threads on leased
  streams, with a host wait per call (`cudaStreamSynchronize`, `mvcuda_analyse.cu:2677`, `mvcuda_recalc.cu:377`,
  `mvcuda_flow.cu:587`); only token + wave MAnalyse uses `pipe_call`.
- They keep their own super cache (`mvcuda_core.cu:192-331`): counted in frames (`MVCUDA_CACHE_FRAMES`), outside the registry
  budget, never shrinks. The key holds `levels` and `rfilter`, so the FRC chain uploads each source picture twice (levels=N for
  MAnalyse, levels=1 for MRecalculate / MFlowFps; MFlowFps hard-codes rfilter 4, `MVFlowFps.cpp:599-602`).
- `super_build` syncs once per plane only to give back the pinned staging buffer (`mvcuda_super.cu:329-339`).
- MRecalculate refuses token pictures and falls to the CPU (`MVRecalculate.cpp:719-722`); MFlowFps downloads tokens
  (`ResolveToken`, `MVFlowFps.cpp:571`) and uploads them again; MFlowFps output is always a host frame (pageable D2H + wait).
- Vector payloads go up with pageable copies (MRecalculate old vectors, 4 MFlowFps vector fields).
- MCompensate / MDegrain do not upload host input (`GpuBackend::with_tokens` has no upload branch, unlike `cf::with_inputs`).
- mvtools `write_token` writes no black stripes (`GpuBackend.cpp:162-165` vs `CudaFilters.cpp:101-108`).
- gpu.md errors in the 2026-10-30 release: line 11 says API 13 (14); line 50 claims MRecalculate / MFlowFps use token frames.
FRC chain today per source frame: about 2 full uploads, 4 small downloads, 1 output download per output frame, ~11 host waits.

**Usage rule (same as cudafilters):** token frames in -> token frames out; normal frames in -> normal frames out (computed on the
GPU, uploaded inside the filter). Existing scripts (uc_FRC, QTGMC_gpu.avsi) keep working unchanged.

**Part M: mvtools port (in this order; each step bit-exact, md5 = CPU fallback):**
- M1 One upload per source picture: host input is uploaded once, async on the upload stream, into a registry frame keyed by
  (super clip, n) with a producer that re-reads its child (as `CudaUpload::regen`); MAnalyse / MRecalculate / MFlowFps then all
  take the device path. No per-plane syncs (`pin_release_after`).
- M2 Super pyramids into the registry budget (accounted on the source frame like the pel-2 sub-pel planes, `frame_account_extra`);
  level 0 shared regardless of `levels` / `rfilter`; retire `MVCUDA_CACHE_FRAMES` (keep it read, ignored or as a cap).
- M3 MRecalculate and MFlowFps read token frames (`super_build_dev` ordered by marks), drop the refusal / ResolveToken.
- M4 Submitter for every call (slice-kernel MAnalyse, MRecalculate, MFlowFps); vector uploads through pinned `pipe_upload`;
  vector downloads wait on a blocking-sync event instead of a leased-stream sync.
- M5 MFlowFps token output when its input is token frames, producer + regeneration; host output otherwise.
- M6 MCompensate / MDegrain upload host input (upload + re-upload-after-regen branch in `with_tokens`), host out for host in.
- M7 Cleanup: MAnalyse uses `with_tokens` instead of its inline copy; black stripes in mvtools tokens; `_gpu_mutex` not held over
  the CPU path; gpu.md (API, readers list, usage rule).
Measure: uc_FRC chain (time, CPU-s, 4 concurrent jobs, VRAM), QTGMC presets unchanged or faster.

**Part B: 10 bit (stored in 16-bit containers; every CPU filter involved supports it):**
- B1 MCompensate / MDegrain 16-bit (`mvcuda_comp.cu`): template on pixel type; 16-bit overlap arithmetic differs from 8 bit
  (int accumulate of val*win, then `min((acc + 1024) >> 11, maxv)`, `overlap.cpp:449-463`); Wiener sub-pel clamped to maxv;
  thSAD scaling is host-side already. Gates: `MVCompensate.cpp:243`, `MVDegrain3.cpp:551`, `mvcuda_comp.cu:236/341`.
- B2 RgTools / MaskTools / QTGMC stages 16-bit (`mvcuda_kobs.cu`, backblend in `mvcuda_resample.cu`, CudaKobs / CudaMt /
  CudaQtgmc gates): template kernels, half / max scaling; check RgTools 16-bit rounding (RemoveGrain 12 / 20); lutxy tables
  1024x1024 at 10 bit (from the CPU filter as now), expressions evaluated on the GPU above 10 bit or CPU there.
- B3 CudaNNEDI3 above 8 bit: new port of plugins_JPSDR's high-bit-depth prescreener / predictor (largest uncertainty).
- B4 QTGMC script: drop `BitsPerComponent(Input) == 8` (`make_qtgmc_gpu.py:66`), check the `!Is420` search-clip exclusion
  (`:49`, reason not written down), FastMA's 8-bit search-clip download.
- B5 Optional speed: packed 16-bit SAD (`__vsadu2`) in k_wave / k_search (10-bit analysis currently uses the unpacked form).
Tests: 10-bit 4:2:0 and 4:2:2 real material (e.g. XDCAM / ProRes 10-bit), framemd5 GPU = CPU per stage (run_cf_regression cases
+ negative controls), run_presetcheck -Fallback for every preset at 10 bit, FRC chain at 10 bit.
Estimates (code reading): M 1.5-2 weeks, B1 2-4 days, B2 1-2 weeks, B3 1-2 weeks, B4 1 day, B5 2-4 days.

**FRC baseline before Part M (2026-10-30, build_rel14, `profile\run_frc.ps1`, template `avs\m\frc.avs.tpl` = uc_FRC with
uc_frc_gpu, seg1_wires_pole.mp4 = UHD 4:2:2 10 bit, 112 output frames, Prefetch 3):** 8.71 s / 21.5 CPU-s (best of 3), CPU
fallback 20.2 s / 168 CPU-s, md5 BFE44775F811 everywhere. Decode alone 1.57 s. nsys (out\nsys\frc_base): GPU kernel time 6.1 s,
of it **k_search 5.84 s (95.5%, 3.66 ms x 1596, the meander=true slice kernel, 16-bit samples, unpacked SAD)**, k_recalc 0.19 s,
k_flow 0.02 s; copies 0.49 s H2D + 0.38 s D2H (pageable); host API time cudaMemcpyAsync 7.3 s and cudaStreamSynchronize 2.7 s
(summed over threads). Note: uc_FRC's two supers are different clips (RemoveGrain(22) full pyramid for MAnalyse, the source
levels=1 for MRecalculate / MFlowFps), so the "same picture uploaded twice" item mostly does not apply to it.
Consequence: Part M can at best close the gap between 6.1 s of kernels and 8.7 s wall (about -25%); the FRC lever is k_search
on 16-bit data (B5 packed `__vsadu2`, and the k_wave lessons of section 24 applied to the slice kernel).

**Order agreed 2026-10-30 (after a doom9 reply: trymany, large exhaustive radius, coarse searchparam > 3 and pel 4 all run on
the CPU today; same quality, no acceleration):** S1 faster k_search on 10/16 bit (packed SAD) -> S2 exhaustive radius > 3 and
coarse searchparam > 3 (MAnalyse, MRecalculate) -> S3 trymany -> S4 pel 4 with sub-pel samples computed on the fly -> Part M
-> rest of Part B. All bit-exact. neo-mv's new algorithms (predictors, chroma-wiener) are out of scope (other fork, no stable
CPU reference).

**S1 result (2026-10-30, committed in `1c7cf9b`, test set `profile\build_s1` = build_rel14 + new mvtools_cuda.dll):**
- Phase counters (`-DMVC_PHASE_PROF`) of the slice kernel on the FRC chain, per block at 768 threads: wait+pred 4,200 cycles,
  phase A 4,650, fold A 2,950, ring (24 candidates) 13,300, fold B 1,800. The ring cost the same with 12 warps as with 24.
- Cause: `__launch_bounds__(768, 1)` capped k_search at 80 registers; ptxas: 2.7 KB spill stores / 4.0 KB spill loads per
  thread (the compile-time row arrays of eval_candidate and the refine path). 384 threads still spill (0.8 / 3.2 KB), **256
  threads: no spills** (8 and 16 bit). Change: `SEARCH_THREADS = 256`.
- FRC chain: 8.71 -> **7.87 s** (-10%), md5 unchanged (BFE44775F811, = CPU fallback). meander=true QTGMC-type analyses
  (run_qtgmc_vec GPU run, wall incl. start-up): HD Placebo 13.8 -> 8.7 s, HD Slower 9.1 -> 5.5, HD Medium 6.0 -> 3.4, UMH ps16
  3.0 -> 2.4. Exactness: golden 85/85, run_qtgmc_vec 50/50 IDENTICAL, FRC = fallback.
- Tried and dropped (patch kept in `profile\out\frc\s1_window_experiment.patch`): a shared-memory window of the ring's
  reference area (cp.async 4 / 16 bytes, candidates of a warp evaluated together). Exact, but no gain: the window load cost
  6-7k cycles at level 0 (4 sub-pel planes) and the SADs from shared memory still ~10k cycles with 8 warps (2 warps per
  scheduler cannot hide the shared-load latency). Wall 8.14 s with it vs 7.90 without.
- What limits the slice kernel now: per block ~3 dependent round trips (predictors, phase A, ring), ~1,340 blocks per slice
  in series at UHD level 0, only 24 CTAs (= CPU thread count, fixed by exactness).
- **meander=false (wavefront kernel k_wave) on the same FRC chain: 5.51 s / 21 CPU-s (-37% vs 8.71)**, identical to the CPU
  fallback with meander=false (8DB9CD4E). Different vectors than uc_FRC's meander=true: a picture-quality decision for the
  user (uconvert.avsi), not done.

**B1 + B2 done (2026-10-30, committed in `1c7cf9b`, library API 15, test set `profile\build_b2`):**
- B1 `mvcuda_comp.cu`: k_blocks / k_subpel templated on the sample type; 16-bit overlap = 32-bit sum of sample * window,
  min((sum + 1024) >> 11, max) (Short2Bytes_Int32toWord16), degrain (sum + 128) >> 8 as at 8 bit, Wiener sub-pel clamped
  to max. Gates in MVCompensate.cpp / MVDegrain3.cpp accept pixelsize 2.
- B2 `mvcuda_kobs.cu` (RemoveGrain, mt ops, KeepOnlyBobShimmerFixes, SharpLimit) and the back blend (`mvcuda_resample.cu`)
  templated; RemoveGrain 20 on 16-bit containers is rg_mode20_sse_16: (sum + 4) * 3641 >> 15, saturated to 65535.
  Lookup tables up to 10 bit (`mvcuda_lut2_create_bits(luma, chroma, bits)`, API 15): 1024 x 1024 uint16 tables made by
  running the installed CPU filter on a 4:4:4 pattern clip. **Two tables, luma and chroma**: with use_expr (1 above 8 bit,
  3 always) masktools hands the expression to Expr, and Expr gives chroma other results than luma for expressions with
  scalef / range_half (found through QTGMC Fast at 10 bit: KeepOnlyBobShimmerFixes' second lut, use_expr=1). Table indices
  are clamped to max (AviSynth+ Sharpen leaves samples above 1023 in 10-bit clips; the CPU's Expr computes with them, a
  table cannot). masktools' i8 parameter scaling for mt_clamp / Sovs: 255 = max, else << (bits - 8) (`cf::scale_i8`).
  Not on the GPU: tables above 10 bit (and 9 bit: no Y9 / 4:4:4 9-bit pattern clip), SharpLimit at 16 bit (Merge rounds
  differently there).
- Tests: run_cf_regression 61 / 61 (MCompensate / MDegrain / KeepOnly / RemoveGrain / mt ops / SharpLimit / BackBlend /
  AddNoise now in 422p10, 420p10, 444p16 too; new cases 'mt_lutxy expr chroma', 'KeepOnly 4 chroma blur'); GPU use at
  10 bit confirmed per case (CUDAFILTERS_LOG / MVTOOLS_GPU_LOG / CUDAFILTERS_STATS).

**B4 done (2026-10-30):** `make_qtgmc_gpu.py` no longer requires 8-bit input for the GPU output side; the 4:2:0 search-clip
exclusion is narrowed to 4:2:0 **with ChromaMotion** (Slower / Very Slow / Placebo). Its reason, found by removing it: with
4:2:0 + ChromaMotion the GPU search clip differs from the CPU chain (Slower 8 and 10 bit; Fast and Medium at 4:2:0 are
identical) - **open: find the stage** (candidates: CudaBob / GaussResize chroma placement or a chroma lut on 4:2:0).
run_presetcheck -Fallback, 60 frames, all variants identical: 10-bit 4:2:2 Fast / Medium / Slower, 4:2:0 8 and 10 bit Fast
/ Slower. NNEDI3 stays on the CPU above 8 bit (B3 open, exact fallback).

**To do: QTGMC GPU default meander=true (user 2026-10-30: "match whatever stock QTGMC does").** Doom9's only measurement
(zorr, Zopti on MFlowFps, GMSD) found meander=false clearly worse. Measured 2026-10-30 with build_s1, PAL 1500 frames,
'gpu all', `QTGMC_gpu_meander = true`: output identical to stock CPU QTGMC (Fast EB74AA6E7731, Slower 61DA3321F664) but
**Fast 19.44 s / Slower 62.79 s vs CPU 6.83 / 13.96 and meander=false GPU 1.80 / 4.36**. The slice kernel is serial by
nature (a meander row starts where the row above ended; 24 slices = CPU threads) and still uses the host-synchronised
call path. Plan: M4 for the slice kernel (submitter, no host waits, many calls in flight), then the remaining per-block
latency of section 24 / S1; switch the default in `make_qtgmc_gpu.py` (and QTGMC_gpu.avsi) once 'gpu all' with meander=true
is clearly faster than the CPU. Until then the default stays false, QTGMC_gpu_meander=true gives the stock result.

**B3 done (2026-10-30, test set `profile\build_b3`, library API unchanged 15):** CudaNNEDI3 at 10 bit. Much smaller than the
estimate: at <= 10 bit plugins_JPSDR (opt 6) runs the same prescreener network (computeNetwork0new_AVX2), dotProd,
e0_m16_FMA3 and weightedAvgElliottMul5 as at 8 bit; its `_16` functions differ only in integer work (asm diffed):
uc2s64_C_16 copies the samples as int16, extract_m8_i16_AVX2_16 sums with pmaddwd (float tail identical), castScale_AVX_16
stores a word, processLine0_AVX2_ASM_16 computes the cubic in int32 + packusdw (same result as the 8-bit saturating form).
Changes: `mvcuda_nnedi3.cu` kernels templated on the sample type (network inputs read as int16 like pmaddwd, cubic as
uint16), one queue-counter sequence for both types; `CudaNnedi3.cpp` prescreener weights scaled with half = (2^bits - 1) / 2,
clamp 0 .. 2^bits - 1 (the plugin's default range 1), gate 8 or 10 bit. Above 10 bit the plugin takes extract_m8_i16_C_16_4
(other float order) -> CPU; 9 bit stays on the CPU (AviSynth+ ConvertBits cannot make 9-bit clips, untested).
Tests: run_cf_regression 65 / 65 (new: NNEDI3 Fast in 422p10 / 420p10 / 444p10 / y10 / 422p12 (fallback), big qual2 10 bit,
pscrn4 etype1 10 bit, samples above 1023 after Sharpen); negative control (nns 1 vs 0) differs on every frame; GPU use
confirmed (CUDAFILTERS_LOG). run_presetcheck -Fallback -Check Fast / Medium / Slower / Placebo at 10-bit 4:2:2, 10-bit 4:2:0
and 8-bit: ALL PASS, CudaNNEDI3 on the GPU in all, md5 unchanged from build_b2.
Speed (PAL 1500 frames, 'gpu all', Prefetch 12, best of 3): Fast 10 bit 3.34 s / 45.8 CPU-s -> **1.94 s / 12.0 CPU-s** (8 bit
1.76 / 8.5; CPU 15.07 s / 232 CPU-s); Slower 10 bit 5.58 s / 69.2 -> **5.61 s / 12.3** (8 bit 4.24; CPU 27.48 s): Slower at
10 bit is bound by the GPU analysis (unpacked 16-bit SAD in k_wave, B5), not by NNEDI3.
Left of Part B: lookup tables above 10 bit (B2), NNEDI3 above 10 bit, B5 packed 16-bit SAD, the 4:2:0 + ChromaMotion
search-clip difference (B4).

**B5 done (2026-10-30, committed in `9f08ddd`, test set `profile\build_b5` = build_b3 + new mvtools_cuda.dll, API unchanged 15):**
packed SADs for 16-bit samples, in both search kernels. A/B first: 8-bit QTGMC Slower with `MVCUDA_WAVE_PACK=0` took 5.40 s
(packed 4.35 s), about the 10-bit time (5.61 s): the missing packed form, not the doubled bytes, was the 10-bit gap.
- k_wave (`pk_load` / `pk_sum`): same lane layout as at 8 bit (4 columns of every 4th row); a row's 4 samples are read as the
  two aligned 8-byte words around them, picked out by funnel shifts (`pk_row`) and summed with two `__vsadu2`. Half the load
  instructions of the split form, the same registers. Gate: reference pitches % 4 samples, sub-plane bases 8-byte aligned.
- k_search (slice kernel, meander=true): new `eval_candidate_pk` (8 and 16 bit) for blksize 32 / 16 with 4:2:2 / 4:2:0
  chroma or luma only: a lane takes 4 columns of every 4th (bs 32) / 8th (bs 16) row, chroma U in lanes 0..15, V in 16..31;
  64 -> 32 load instructions per lane for the FRC chain's 32x32 4:2:2 block. Other geometries keep eval_candidate.
  `MVCUDA_SEARCH_PACK=0` switches it off (A/B).
- Speed: QTGMC PAL 1500 frames 10 bit Slower 5.61 -> **5.08 s** (8 bit 4.25, unchanged), Fast 1.96 (unchanged: small
  batches); uc_FRC chain (UHD 4:2:2 10 bit, run_frc.ps1) 7.76 -> **6.85 s**, md5 BFE44775F811 = CPU fallback.
- Exactness: golden 85/85 (tags 10, f8, f420, f420b8, f444), run_qtgmc_vec 50/50 with and without meander,
  run_cf_regression 65/65 + tokens 42/42, run_presetcheck -Fallback Fast / Slower / Placebo at 10-bit 4:2:2 and 8 bit ALL
  PASS, FRC = fallback.
- Left: 16-bit packed rows need twice the registers of 8-bit ones (the 8-byte window around 4 samples), which is the rest of
  the 10-bit Slower gap (5.08 vs 4.25 s); a scheme with shuffles from the neighbour lane instead of the second word was not tried.

**Open after B5 (2026-10-30), in the agreed order:**
1. S2: exhaustive radius > 3 and coarse searchparam > 3 on the GPU (MAnalyse, MRecalculate); today they fall back to the CPU.
2. S3: trymany.
3. S4: pel 4 with sub-pel samples computed on the fly.
   Doom9 (DTL, 2026-10-30): load the pel-1 reference window once into shared memory, each thread applies its own filter
   phase (MSuper's sharp filter, exact integer rounding as k_subpel) and computes the SAD, only the SAD leaves the core.
   Consequences: (a) retry the S1 shared-memory window with it: the window is the pel-1 area plus the filter margin,
   about 1/4 of the 4 stored planes that made the S1 window cost 6-7k cycles; the other S1 limit (shared-load latency
   with 8 warps) stays, so measure before keeping it. (b) Same path for pel 2 as a VRAM option (no 4x sub-pel planes
   per referenced frame; helps 4 concurrent jobs); expected speed gain at pel 2 small (k_search is latency-bound).
   (c) MSuper with `pelclip` has no filter to recompute: those planes stay stored.
   AreaMode (DTL's fork, shifted block grids, averaged vectors) noted, not planned: other algorithm, D5 / D7.
4. ~~Part M (M1-M7): MAnalyse / MRecalculate / MFlowFps on the cudafilters model (one upload per picture, pyramids in the
   registry budget, token in / out, submitter for every call).~~ Done (`4836ac9`, API 16, see "Part M done" below).
   Left from it: 4 concurrent uc_FRC jobs 17.4 -> 18.5 s (+6%), cause not found; next: nsys of the 4-job run
   (per-process timelines), the single upload stream per process.
5. QTGMC default meander=true once 'gpu all' with the slice kernel is clearly faster than the CPU (M4 is done, so this
   only waits for the measurement).
Smaller Part B leftovers: 10-bit k_wave registers (16-bit packed rows hold an 8-byte window per 4 samples; try neighbour-lane
shuffles instead of the second word; 10-bit Slower 5.08 s vs 8-bit 4.25 s); lookup tables and NNEDI3 above 10 bit; the 4:2:0
+ ChromaMotion search-clip difference (B4). After all of it: milestone F (FFT3DFilter on the GPU).

**Part M done (2026-10-30, committed in `4836ac9`, library API 16, test set `profile\build_m1`):** MAnalyse, MRecalculate and MFlowFps
on the cudafilters model; MCompensate / MDegrain take normal frames too.
- M1 `mvcuda_frame_upload_keyed(fmt, planes, key)`: a host picture is uploaded once under a content key
  (`GpuBackend::frame_key(clip, n)`); while that frame is in the registry, later requests get its token (stat
  `upload_hits`). No producer: a consumer that finds it evicted uploads again. The keys name clip addresses, so
  `mvcuda_cache_clear` (last GPU filter gone; MCompensate / MDegrain now count as users too) forgets them.
- M2 super pyramids (`DevSuper`) hang off the `DevFrame` (`frame_super`), counted in the registry budget
  (`frame_account_extra`); a stored one serves a request with the same pad / pel / sharp and at least its levels and
  chroma (rfilter only matters above 1 level). The separate 24-frame super cache, its host build path and
  `MVCUDA_CACHE_FRAMES` are gone; `mvcuda_*_host` are test wrappers (upload, device path, download).
  Lesson: an evicted frame stays reachable through the per-thread holds / pins / readers, and with it its pyramids (about
  5x the picture) outside the budget: peak VRAM went *up* with smaller budgets. Eviction now drops the pyramids and
  sub-pel planes of the evicted frame (made again by a later reader). And the start budget (128 frames of the first
  frame) is capped at 1 GB: 4.2 GB for UHD frames, while uc_FRC runs as fast at 1 GB.
- M3 `mvcuda_recalc_token`, `mvcuda_flow_token` (token out), `mvcuda_flow_token_host` (host out); MRecalculate's token
  refusal and MFlowFps's ResolveToken + re-upload are gone. `GpuBackend::Input` / `super_input` / `frame_input` /
  `with_inputs` replace `with_tokens` and MAnalyse's inline copy: tokens are regenerated, host pictures re-uploaded.
- M4 `leased_call(frames, issue)`: analysis (both kernels), recalc and flow are issued by the submitter on a leased stream,
  ordered after their inputs (an uploaded frame: its own copy event `DevFrame::up_done` on the upload stream; else a
  pipeline mark), one blocking-sync event wait; vectors up and results back through pinned staging (the slice kernel's
  payload header too). MFlowFps first ran on the pipeline stream; moved to a leased stream, because there it queued
  behind the wait-ops of every upload issued before it (output buffer from the upload pool, idle for any stream).
- M5 MFlowFps: token frames out when the source clip's frame is a token (producer + `gpu_regen`, `_gpu_mutex` over
  GetFrame because regeneration shares the vector clip state), else normal frames.
- M6 MCompensate / MDegrain with a lite super of a normal clip: pictures uploaded (keyed; reference numbers from
  `use_ref_frame`), result downloaded: normal frames in, normal frames out.
- M7 MAnalyse uses `with_inputs`; mvtools `write_token` writes the black rows like cudafilters; gpu.md updated (API 16,
  usage rule, readers, outputs, env vars). **Not done:** "`_gpu_mutex` not held over the CPU path" of MCompensate /
  MDegrain: the CPU path and a regeneration share the MVClip objects (Update), so releasing the lock would race; it is
  uncontended except during regenerations.
- Tests (all on build_m1): golden 85/85; `profile\run_partm.ps1` (new) 36/36: uc_FRC-type chain (thSCD1 raised so the
  GPU flow runs) as host / host with MVCUDA_CACHE_MB=1 / token (CudaUpload ... CudaDownload) / token with MVCUDA_CACHE_MB=1
  (187 regenerations) / fallback, token frames without CudaDownload differ (negative control), MDegrain2 + MCompensate
  host / tiny / fallback, in 8-bit 4:2:2, 10-bit 4:2:2, 4:2:0, 4:4:4, every one md5 = gpu=false; run_cf_regression 65/65
  (tokens 42/42); run_qtgmc_vec 50/50 with and without meander; run_presetcheck -Fallback Fast / Slower ALL PASS;
  uc_FRC md5 BFE44775F811 = CPU fallback.
- uc_FRC (UHD 4:2:2 10 bit, run_frc.ps1, Prefetch 3, best of 3): B5 6.78 s / 21.4 CPU-s -> **6.57 s / 21.1**; one upload per
  picture and super clip (296 uploads, 788 shared lookups; normal PAL test: exactly 2 x frames); copy time on the GPU
  0.77 -> 0.36 s (nsys), kernels unchanged (k_search 2.24 ms x 1603 = 95% of GPU time). Peak VRAM 5.20 -> **4.48 GB**.
  **4 concurrent jobs: 17.4 -> 18.5 s (+6%), open.** Ruled out by measurement: the submitter (direct issue from the worker
  threads: same), the budget (fixed 1 / 4 GB: same or worse), GPU work and copies (nsys: equal kernels, half the copy
  time), pinned-pool misses (12 per run), the pipeline-stream ordering of uploads (up_done events) and of MFlowFps (leased
  stream). Next idea: nsys of the 4-job run (per-process timelines), the single upload stream per process.
- QTGMC PAL 1500 frames 8 bit 'gpu all' (run_presettiming): Fast 1.80 -> 1.81 s, Slower 4.36 -> 4.36 s (unchanged, as
  expected: its search clip was on the device already). Final-build tests repeated: run_partm 36/36, cf_regression 65/65,
  qtgmc_vec 50/50 x2, run_presetcheck -Fallback Fast / Slower / Placebo ALL PASS, golden 85/85.
