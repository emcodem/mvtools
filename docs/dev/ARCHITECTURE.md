# Architecture of the GPU fork (developer notes)

How the GPU paths are built and why. User-facing docs: `../../README.md`, `../../Documentation/gpu.md`. Decisions:
[DECISIONS.md](DECISIONS.md). Open work: [TODO.md](TODO.md). The detailed design logs with all measurements were
GPU_PLAN.md, GPU_QTGMC_PLAN.md, GPU_PIPELINE_PLAN.md, VAPOURSYNTH_PLAN.md, REBASE_PLAN.md and NVOF_FILTER_PLAN.md;
their last versions are in git at commit af2e51e (`git show af2e51e:GPU_PIPELINE_PLAN.md`). Code comments are self-contained and do not
reference .md files (links to online resources are fine).

## Components

- `mvtools2.dll` / `libmvtools2.so`: the mvtools filters; the GPU versions are separate names (CudaMSuper, CudaMAnalyse,
  CudaMRecalculate, CudaMCompensate, CudaMDegrain1..6 / N, CudaMFlowFps; same parameters, D8); M* are pure CPU.
- `mvtools_cuda.dll` / `libmvtools_cuda.so`: every CUDA kernel, the device frame registry, NVOFA. C API in
  `Sources/cuda/mvcuda.h` (no CUDA types in the plugins), static CUDA runtime (only the driver is needed). Loaded at run
  time; missing library, no device (cc < 6) or an API mismatch = the Cuda* filters cannot run (creation error, or the
  CPU path with `cuda_cpu_fallback`).
- `cudafilters.dll` (GPL): every GPU filter that is not mvtools (CudaUpload / CudaDownload, Bob, TemporalSoften, Merge*,
  resizers, RemoveGrain, a mt_* subset, CudaNNEDI3, fused QTGMC stages, CudaNVOFAnalyse). Loads mvtools_cuda the same
  way, so mvtools2 stays a drop-in without it. All three must sit in one folder: one registry per library instance.
- `avstp.dll`: thread pool. With it the CPU search runs in `min(threads, 64, blkY / 4)` slices, which changes vectors at
  slice borders; the GPU uses the same count. Without a pool (VapourSynth, Linux) CPU and GPU use 1 slice.
- `mvtools_vsbridge`: AviSynth+ API shim for VapourSynth (below).
- **Library API version** (`MVCUDA_API_VERSION`, now 23) is checked at load; every bump needs all DLLs from one build.

## Exactness rule

- Integer stages are bit-exact against the CPU filter as it runs on SSE4.1 / AVX2 machines (not its C reference).
  CudaNNEDI3 is bit-exact too (the CPU's rcpps / rsqrtps results are measured at load into tables). FFT stages stay on
  the CPU; a GPU port would be non-exact and needs the user's decision.
- Every GPU filter keeps an exact CPU path. At creation (D8): a Cuda* filter that cannot run on the GPU at all (library,
  device, settings, format) stops with an error naming the reason (`GpuCheck` in GpuBackend.h, `cf::gpu_decide` in
  cudafilters), unless `cuda_cpu_fallback` (script global, else env `CUDA_CPU_FALLBACK`) allows the CPU filter, which
  then returns normal frames. Per frame (CUDA error, `MVCUDA_NEEDS_CPU`, field-based, unusable vectors, forced fallback):
  the CPU result is uploaded under the filter's producer id (`GpuBackend::cpu_result_token`, `cf::cpu_token`), and the
  regeneration repeats GPU-or-CPU, so the output stays a token clip. cudafilters build the original CPU filter in the
  constructor (`env->Invoke`) and feed its inputs through CudaDownload. Per-frame fallback is only safe because both
  agree bit for bit.
- MAnalyse reference: the CPU with the same slice count. meander=true = the multi-threaded CPU result (slice kernel);
  meander=false = CPU `meander=false, mt=false` (wavefront kernel; vectors differ from stock).
- Non-exact output only behind its own switch, default off (CudaNVOFAnalyse is a separate filter; MAnalyse stays exact).
- Validation: framemd5 GPU vs `MVTOOLS_GPU_FORCE_FALLBACK=1`; negative controls must differ; tiny budgets
  (`MVCUDA_CACHE_MB=1..16`) must give the same md5 (regeneration path). Vector clips are compared after the 256-byte
  MVAnalysisData header (uninitialised padding).

## Token frames and the device frame registry (`mvcuda_frames.cu`)

- A GPU filter with token output returns an ordinary frame whose luma row 0 holds `MvCudaTokenHeader` (magic `MVCUDATK`,
  version, token, producer, frame number, format) and every 8th luma row black (a CPU filter shows stripes). Consumers
  detect the magic; there is no `cuda_in` parameter.
- **Usage rule (D8):** every Cuda* filter takes token or normal frames (normal ones uploaded inside) and returns token
  frames; the last GPU stage before a CPU filter is `CudaDownload` (passes normal frames through). Vector clips stay
  normal frames. CudaDownload is a copy-engine D2H copy waiting for the producing op; a UVA-writing kernel instead was
  measured slower for QTGMC (2026-11-02).
- Registry: LRU with a byte budget; holders keep a `shared_ptr`, so eviction never frees a buffer in use; each thread
  holds its last 8 frames. Budget: `MVCUDA_CACHE_MB` fixes it, else auto: start max(256 MB, 128 frames) capped at 1 GB,
  grows on reuse near the LRU end and on misses of evicted frames, cap min(free / 3, 4 GB).
- Regeneration: an evicted token is made again by its producer (process-wide producer table); pins keep regenerated
  inputs alive; the registry redirects an evicted token to the newest live token of the same (producer, frame).
- Uploads keyed by content (`mvcuda_frame_upload_keyed`, `GpuBackend::frame_key(clip, n)`): one upload per picture.
- Super pyramids (`DevSuper`) and sub-pel planes hang off the `DevFrame`, count in the budget and are dropped on eviction.

## Submission model

- One submitter thread issues every pipeline-stream call in order (`pipe_submit`); a caller waits only for its own op.
  Reason: WDDM launch cost grows with the number of launching threads (9 us alone, 20-56 us with many).
- Uploads on their own stream from a pinned pool; own-input waits via `DevFrame::seq` and marks, not whole-pipeline waits.
- `leased_call(frames, issue)`: analysis, recalc, flow and NVOF run on leased streams, ordered after their inputs, with
  one blocking-sync event wait. Pooled events use `cudaEventBlockingSync`; context blocking-sync (`MVCUDA_SPIN=1` spins).
- Results go to pinned host memory written by kernels through UVA (`host_out`, `k_pay_out`): no D2H copies.
- Fused QTGMC kernels cut launches (`k_kobs`, `k_sharp_limit`, Binomial0, BackBlend).

## Search kernels (`Sources/cuda/mvcuda_analyse.cu`, `mvcuda_refine.cuh`)

- Bit-exact port of PseudoEPZSearch / Refine: a candidate wins only with strictly lower cost, first on ties. Batches
  around a fixed centre are evaluated in parallel and folded in CPU order. nLambda math IEEE-exact, no FMA contraction.
- **k_search** (meander=true): one CTA per CPU slice. A meander row starts where the row above ended, so each slice is a
  serial chain whose length the CPU slice count fixes; this is the main exact-mode limit. 256 threads (768 spilled).
  `k_search<TM>`: the plain instance has no trymany code (0 spills). Exhaustive radius <= 3, Hex2 range <= 3 and UMH
  range 2..3 (`wave_plan`) run as inline batches + the ring fold; other refines through mvc_refine (AnaChecker), whose
  out-of-line run_batch with local-memory candidate lists cost ~19k cycles per block for the default hex search vs ~6k
  inline. nLambda is computed per row in parallel above the smallest level (`s_nl`).
- **k_wave** (meander=false): one CTA of 4 warps per block row, rows by atomic ticket, per-row progress counters
  (release / acquire), all levels in one launch. Fixed plans for small Hex2 / UMH in the fast kernel, GEN kernel
  (mvc_refine) otherwise; register cap 96. Runs the bad-vector wide search itself (k_search sends such frames to the
  CPU). `MVCUDA_WAVE=0` forces k_search.
- **Packed SADs** (`pk_row`): a lane takes 4 columns of every 4th / 8th row; 8 bit `__vsadu4`, 16 bit `__vsadu2`.
  `MVCUDA_WAVE_PACK=0` / `MVCUDA_SEARCH_PACK=0` for A/B.
- **exhaustive_fast** (k_search, radius > 3): one block-wide minimum of (cost << 15 | ring index) when every cost part is
  >= 0, else sequential batches; reference window in shared memory, pruning by vector cost. `MVCUDA_EXH_FAST=0`.
- **trymany_par**: the 7 Refines in parallel warps (inputs via shared `TmIn`); dependent Refines afterwards in warp 0;
  exhaustive radius <= 3 shares one SAD table per distinct start vector; other types through `WarpChecker`. Also in
  k_wave GEN. `MVCUDA_TM_PAR=0` = old forms.
- search_coarse / trymanyall: refine type per level (`lvl_type[l]`), try_many at level 0 with 2+ levels.
- MRecalculate (`mvcuda_recalc.cu`): blocks independent (one warp each); slices only change the first rows' nLambda.
- Limits (`GpuBackend::search_supported`, `mvc_refine::max_param`): exhaustive radius 1..64, other types 24 (one
  MAX_BATCH = 160 candidate batch); above that the whole filter runs on the CPU.
- Profiling: `-DMVC_PHASE_PROF` (clock64 phase counters); nsys works, ncu needs admin rights.

## MSuper lite / CompleteLite

- `CudaMSuper` (pel 1 / 2 / 4, sharp 2, rfilter 2 / 4, no pelclip) stores only the picture (or the token header) and
  sets the lite bit; every CPU reader rebuilds the real frame through `MVSuper::CompleteLite` / `ResolveToken`. At pel 4
  the CPU MCompensate / MDegrain complete it (CudaMCompensate / CudaMDegrain have no pel-4 path yet).
- `SuperParams64Bits::param`: bit 0 upstream's YUY2 flag, bits 1-4 rfilter * 3 + sharp, bit 5 pelclip, bit 6 lite, bit 7
  our marker (valid only with bit 7 set); use the `super_*` helpers.
- CPU quirks the GPU reproduces: rfilter only for the first reduction (bilinear below); level 0 reduced before padding.

## CudaNVOFAnalyse (`cudafilters/CudaNvof.cpp`, `mvcuda_nvof_vec.cu`, `mvcuda_nvofa.*`)

- NVIDIA optical flow accelerator vectors as a MAnalyse-format vector clip (one level), accepted by MRecalculate,
  MFlowFps, MDegrain, MCompensate. Gray input, per-block median, clamp, own SAD from a levels=1 super. Not exact.
- nvofapi64 loaded at run time; without NVOFA the function returns MAnalyse itself; a failed frame gets MAnalyse's
  level 0. Experiment and visual judgement: `compare/nvofa_frc`.

## VapourSynth bridge (`vsbridge/`, DECISIONS.md D9 / D10)

- No VS rewrite: a shim implements the used part of the AviSynth+ C++ API on VS API 4; filter sources unchanged; output
  must equal AviSynth+ bit for bit. Namespaces `mvpf` (mvtools2) and `cuda` (cudafilters). Bridge decisions:
  1. Frame requests: base(n) = floor(n * frames_in / frames_out) plus offsets learned per input; a miss takes a
     synchronous `getFrame` (safe in current VS); only speed is affected.
  2. VideoInfo side channel: registry keyed by `VSNode*` with the full AviSynth VideoInfo (MSuper's audio fields,
     MVAnalysisData pointer).
  3. The shim provides the `AVS_Linkage` table (VideoInfo / AVSValue code from AviSynth+).
  4. Vector clips travel as Gray8 carriers; super clips need nothing.
  5. The bridge is its own library (`mvtools_vsbridge`, core mode; ODR); plugins get `VapourSynthEntry.cpp` ->
     `mvbridge_register`. Bridge missing: empty namespace and a reason on stderr, never a crash.
  6. VS functions keep AviSynth names and parameter names; unnamed positionals from a table, else `c1, c2`.
- MT modes: NICE -> fmParallel, MULTI_INSTANCE -> instance pool, SERIALIZED -> fmParallelRequests.
- cudafilters' CPU fallbacks under VS: stand-ins with the AviSynth VideoInfo; `MVBRIDGE_FALLBACK=strict` (default,
  error) or `nearest`. Exact masktools port `mt_port.cpp` (mt_lut / mt_lutxy / mt_adddiff, `-ffp-contract=off`).

## Builds

- CMake only (`build_windows.ps1`, `build_linux.sh`); upstream's `mvtools.sln` / vcxproj are not maintained here and
  have no GPU or VapourSynth parts. CUDA 12.8-12.x (12.9; CUDA 13 dropped Pascal / Volta); MSVC 14.4x toolset.
  Default archs: SASS 6.1-12.0 plus PTX 6.0 / 8.0.
- Linux: export only the entry points (`cmake/plugin_exports.map`; exporting everything makes `AVS_linkage` resolve to
  libavisynth's copy and crash). `dynlib::open` refuses truncated ELF files (glibc dlopen SIGBUS). Release builds in
  docker ubuntu:22.04 with dynamic libstdc++: static libstdc++ aborts when AviSynth errors cross .so boundaries.
- cudafilters uses only AviSynth+ interface V8 calls (users run 3.7.3).

## Lessons learned (measured)

- Pageable `cudaMemcpyAsync` is synchronous in the driver and stalled every thread's launches (163 us each): pinned
  staging only.
- H2D copies above ~100 KiB go to the copy engine, whose queue is in order across streams: one copy waiting on a kernel
  held every upload and serialised the HD / UHD slice search (E1). Use memset + small kernels, results through UVA.
- `cudaMalloc` synchronises the device (an oversized budget meant a malloc per frame); never destroy static pools at exit.
- Launches from many threads contend on the WDDM driver lock: hence one submitter thread. No MPS on Windows: processes
  time-slice the GPU.
- Register pressure decides speed: capping at 128 registers made k_wave 30-40% slower; 768-thread k_search spilled; any
  inlined trymany code spilled and slowed the plain search. Variants get their own template instance.
- Out-of-line functions with by-value arguments go through local memory (3-6k cycles per block); per-thread candidate
  arrays in local memory are effectively uncached (3.6k cycles per 8 reads): use shared memory or warp shuffles.
- k_wave's stack frame x resident threads is reserved as local memory (about 1 GB on a 5090): keep stacks small.
- Evicted frames stay reachable through holds and pins, with their pyramids: eviction must drop derived data.
- AviSynth's cache hands out evicted tokens again: redirect to the newest live token, or regeneration storms follow.
- An AviSynth error inside a Prefetch thread shows up in ffmpeg as an access violation; test scripts must fail on a
  non-zero exit (a crashed run once compared an old framemd5 as "identical").
- CPU SIMD differs from the C reference (TemporalSoften 16-bit division, Merge 16-bit rounding, RemoveGrain 11 runs mode
  12 on SSE4.1): port what the CPU actually runs. mt_lutxy tables come from the installed CPU filter, luma and chroma
  separately.
- Test material: LWLibav returns wrong frames on random access (MPEG-2 MXF): use ffms2. Yadifmod2 reads uninitialised
  memory at row ends. FFT3DFilter with FFTW_MEASURE is non-deterministic under load (`measure=false` fixes it).
- Benchmarks: runs under ~1 s are distorted by the GPU clock ramp; 300-frame runs by start-up allocations; use 1500
  frames. A test set without avstp.dll gives a different (valid) single-slice result.
- NVDEC cannot decode MPEG-2 4:2:2 (silent software decode). sm_120 has no hardware SAD (`__vsadu2` ~10 instructions).
- `core.autocrlf` turned merge resolutions into LF; check line endings after merges.
