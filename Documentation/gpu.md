# MVTools GPU reference

This page covers the CUDA part of this mvtools build:

- the `Cuda*` functions of mvtools2 (`CudaMSuper`, `CudaMAnalyse`, `CudaMRecalculate`, `CudaMCompensate`,
  `CudaMDegrain1..6`, `CudaMDegrainN`, `CudaMFlowFps`)
- the `cudafilters` plugin, with GPU versions of the AviSynth core, MaskTools2, RgTools and NNEDI3 filters that QTGMC uses,
  plus fused QTGMC stages
- the CPU fallback switch `cuda_cpu_fallback` and the environment variables

Until 2026-11-02 the mvtools2 GPU paths were switched on with a `gpu=true` parameter on MSuper, MAnalyse, ...; that
parameter is gone. MSuper, MAnalyse and the other M* functions are the CPU filters, as upstream.

This page describes what the functions do now (library API version 19). Design notes for developers:
`docs/dev/ARCHITECTURE.md`.

## Principles

- **Bit-exact.** Every GPU path gives exactly the result of the CPU filter it replaces, with the same arguments. The
  reference for mvtools2 is the CPU result with the same slice count (see `meander` under MAnalyse below). Since
  2026-11-02 that is the CPU code of upstream mvtools 2.7.48, whose symmetric rounding of luma vectors to the chroma
  grid (#59) changes the results of subsampled clips (4:2:0, 4:2:2, 4:1:1) with chroma; the GPU paths follow it, so
  their output there differs from builds of this branch before that date.
- **Explicit names.** A `Cuda*` function runs on the GPU; the function without the prefix is the CPU filter. A `Cuda*`
  function takes the same parameters and defaults as its CPU filter.
- **Token frames out.** Every `Cuda*` function (mvtools2 and cudafilters) accepts normal or token frames and returns
  token frames (normal input is uploaded inside the filter). End the GPU part of a script with `CudaDownload`. Vector
  clips are the exception: they stay normal frames.
- **Explicit fallback.** A `Cuda*` function that cannot run on the GPU at all stops with an error at creation, unless
  `cuda_cpu_fallback` is set (see "CPU fallback" below). Single frames that need the CPU during a run are computed on the
  CPU and uploaded; the result is the same either way.
  Set `MVTOOLS_GPU_LOG=1` (mvtools2) or `CUDAFILTERS_LOG=1` (cudafilters) to see on stderr which path every filter takes.
- **One exception: `CudaNVOFAnalyse`.** It is not a GPU version of a CPU filter but another motion search (NVIDIA's
  optical flow accelerator), so its vectors match no CPU result. Use it only where that is wanted (see below).

## Installation

| File | Built by | Notes |
|---|---|---|
| `mvtools2.dll` | root CMake | The plugin as before |
| `mvtools_cuda.dll` | root CMake (CUDA 12.9 toolkit; on Windows with the MSVC 14.4x toolset, v143. CUDA 13 is not used: it has no Pascal / Volta targets) | Must sit next to `mvtools2.dll`. Static CUDA runtime: only the NVIDIA driver is needed at run time |
| `cudafilters.dll` | root CMake | Optional. Must sit in the same folder as the other two |

Build with `build_windows.ps1` or `build_linux.sh` (README "Build Instructions"); on Linux the files are
`libmvtools2.so`, `libmvtools_cuda.so`, `libcudafilters.so`. Upstream's `mvtools.sln` / vcxproj files have no GPU or
VapourSynth parts.

All three files must come from one build: the plugins check the library API version; a library of another build counts
as "GPU not usable" (creation error of every `Cuda*` function, or the CPU with `cuda_cpu_fallback`).

Requirements:

- x64 AviSynth+ (Windows or Linux). cudafilters needs AviSynth+ 3.6 or newer (interface V8) and refuses to load on older
  versions; 3.7.5 is the tested version.
- VapourSynth (Windows and Linux): mvtools2 runs under VapourSynth through `mvtools_vsbridge` (namespace `mvpf`, README
  "VapourSynth"), the `Cuda*` functions included (`core.mvpf.CudaMSuper`, ...), with the same output as AviSynth+ (the
  GPU library next to the plugin). Every cudafilter is available in namespace `cuda` (`core.cuda.CudaDownload`, ...).
  VapourSynth has no script globals: the CPU fallback is switched with the environment variable `CUDA_CPU_FALLBACK`; for the limits of their CPU fallbacks under VapourSynth and
  `MVBRIDGE_FALLBACK` see README "VapourSynth".
- An NVIDIA GPU with compute capability 6.0 or newer (GTX 10xx / Pascal, Volta, RTX 20xx .. 50xx); older cards count as
  "no usable device". The first usable device is used; `MVCUDA_DEVICE=n` picks another. Pascal / Volta are verified
  only as PTX 6.1 on an RTX 50xx, not yet on real 6.x / 7.0 hardware.
- `CudaNNEDI3` needs the nnedi3 weights: from a loaded `plugins_JPSDR.dll` (nnedi3), else `nnedi3_weights.bin` next to
  cudafilters (under AviSynth+ its CPU fallback still needs nnedi3 itself). `CudaKeepOnlyBobShimmerFixes` needs
  `QTGMC.avsi` (`QTGMC_KeepOnlyBobShimmerFixes`) with masktools2 and RgTools loaded.

## Token frames

A GPU filter keeps its result in device memory. What it returns to AviSynth is a *token frame*: an ordinary frame of the
declared format whose first luma bytes hold a `MvCudaTokenHeader` (magic `MVCUDATK`, the device frame's registry key, the
filter and frame number that produced it, the format). Every 8th luma row of a token frame is black, so a filter that
does not understand tokens shows obvious stripes, not a plausible picture.

- **Readers.** Every `Cuda*` function reads token frames and uses the device frame directly. Of the CPU filters, `MSuper`,
  `MCompensate`, `MDegrain1..6`, `MAnalyse`, `MRecalculate` and `MFlowFps` accept token frames too and download the
  picture. Any other filter needs `CudaDownload` in front of it.
- **Usage rule.** One rule for mvtools2 and cudafilters: a `Cuda*` function accepts normal or token frames and always
  returns token frames; given normal frames, it uploads them itself (`CudaUpload` is not needed). End the GPU part with
  `CudaDownload` (it passes normal frames of any format through unchanged), and put it in front of every filter that
  does not read token frames. The mvtools2 functions upload each picture of a super clip once, shared by every GPU
  filter that reads that super clip (both CudaMAnalyse directions, CudaMRecalculate, CudaMFlowFps, CudaMCompensate,
  CudaMDegrain).
- **Vector clips** (CudaMAnalyse, CudaMRecalculate, CudaNVOFAnalyse) are normal frames: small host data that the CPU
  consumers (MMask, MFlowBlur, MBlockFps, MDepan, MStoreVect, ...) read directly.
- **Super clips.** CudaMSuper returns a "lite" super clip (the picture, or the token header; see CudaMSuper below), made
  for the `Cuda*` functions.
- **Eviction.** The device frames live in a registry with a memory budget (`MVCUDA_CACHE_MB`). When a frame is evicted
  before it is read, the filter that produced it makes it again (an uploaded picture is uploaded again), so evictions cost
  time but never change the result. The super pyramids that MAnalyse, MRecalculate and MFlowFps make on the device are
  kept with the device frame of their picture and counted in the same budget.
- **Supported formats.** Device frames are planar Y or YUV (no RGB, no alpha), 8 to 16 bit integer, chroma subsampling 1
  or 2 in each direction (4:2:0, 4:2:2, 4:4:4, Y only), and at least 72 bytes per luma row (the header must fit). Filters
  that work through a lookup table (CudaLutxy, CudaKeepOnlyBobShimmerFixes, CudaQtgmcSharpLimit with `smode` 1 / 2,
  CudaQtgmcAddNoise) and CudaNNEDI3 cover 8 and 10 bit; see each filter below. Any other format is a creation error
  (or the CPU filter with `cuda_cpu_fallback`).

A typical chain (mvtools2.dll, mvtools_cuda.dll and cudafilters.dll in one folder, from one build):

```
sup = CudaMSuper(c, pel=2)
bv = CudaMAnalyse(sup, isb=true, blksize=16, overlap=8)
fv = CudaMAnalyse(sup, isb=false, blksize=16, overlap=8)
CudaMDegrain1(c, sup, bv, fv, thSAD=400).CudaDownload()   # pixels again for the rest of the script
```

The same chain under VapourSynth: `core.mvpf.CudaMSuper`, `core.mvpf.CudaMAnalyse`, `core.mvpf.CudaMDegrain1`, then
`core.cuda.CudaDownload`.

Speed: the switch from `gpu=true` to the `Cuda*` names (token output plus CudaDownload) measured the same as before
(uc_FRC UHD 5.61 s vs 5.65 s, QTGMC Fast PAL 1500 frames 3.62 s vs 3.64 s, identical output).

## CPU fallback

| Setting | Where | Effect |
|---|---|---|
| `global cuda_cpu_fallback = true` / `false` | AviSynth+ script | Read when a `Cuda*` filter is created: set it before the first `Cuda*` call. Wins over the environment variable |
| `CUDA_CPU_FALLBACK=1` / `0` (also `true` / `false` / `yes` / `no`) | environment | Used when the script global is not set; the only switch under VapourSynth (no script globals) |

Default false. Then a `Cuda*` function that cannot run on the GPU at all (no usable device, `mvtools_cuda.dll` missing or
of another build, `MVTOOLS_GPU_DISABLE`, a format or setting without a GPU path) stops with an error at creation that
names the reason and the alternatives, for example:

```
CudaMSuper: cannot run on the GPU: only sharp=2.
Use MSuper for the CPU, or set global cuda_cpu_fallback = true (environment variable CUDA_CPU_FALLBACK=1) to run it on the CPU when the GPU cannot.
```

With fallback true, such a function runs its CPU filter and returns normal frames (CudaDownload passes them through),
so one script runs on machines with and without a GPU, with the same output.

Either way, single frames that need the CPU during a run (for example a field-based clip with pel > 1 in CudaMAnalyse,
`delta` 0 or less, frames with unusable vectors in CudaMFlowFps, a frame that cannot be made again after eviction, or
every frame with `MVTOOLS_GPU_FORCE_FALLBACK`) are computed on the CPU and uploaded: the output stays a token clip and
the result does not change. There is no per-call fallback parameter.

---

## mvtools2: the `Cuda*` functions

| Function | CPU filter | What runs on the GPU | Output |
|---|---|---|---|
| `CudaMSuper` | `MSuper` | Nothing: it makes a "lite" super clip that holds only the picture, so the GPU filters build the pyramid themselves | lite super clip |
| `CudaMAnalyse` | `MAnalyse` | Hierarchical search, all levels, with its own super pyramid | normal vector clip |
| `CudaMRecalculate` | `MRecalculate` | Refinement search | normal vector clip |
| `CudaMCompensate` | `MCompensate` | Block compensation | token frames |
| `CudaMDegrain1..6` | `MDegrain1..6` | Degraining | token frames |
| `CudaMDegrainN` | `MDegrainN` | as CudaMDegrain1..6, for `tr` 1 to 6 with `thSAD2` = `thSAD` and `thSADC2` = `thSADC` (the cases MDegrainN hands to MDegrain1..6) | token frames |
| `CudaMFlowFps` | `MFlowFps` | Flow interpolation (`mask=2`) | token frames |

Each takes exactly the parameters and defaults of its CPU filter. `MFlow`, `MFlowInter`, `MFlowBlur`, `MBlockFps`,
`MMask`, `MSCDetection`, `MShow`, `MDepan`, `MScaleVect`, `MStoreVect` and `MRestoreVect` have no GPU version.

The conditions below are the GPU paths' limits; outside them the function stops at creation with the reason (or runs
the CPU filter with `cuda_cpu_fallback`). Errors that the CPU filter raises at creation stay errors, e.g. since 2.7.48
`dct` 5 to 10 (SATD) with a block size not divisible by 4.

### CudaMSuper

Skips the CPU pyramid and the sub-pel planes: each frame holds only the picture inside its padding rows. Conditions:

- `pel` 1, 2 or 4, `sharp=2`, `rfilter` 2 or 4, no `pelclip`
- planar YUV (not Y only), 8 to 16 bit integer. YUY2 counts as 4:2:2: since upstream 2.7.48 every filter converts a YUY2
  clip to YV16 at its input (and its output back), so the GPU paths see YV16
- chroma subsampling 1 or 2 in each direction (4:2:0, 4:2:2, 4:4:0, 4:4:4)
- `hpad` divisible by the horizontal chroma ratio, `vpad` by the vertical one, height divisible by the vertical one

When the input frame is a token frame, the lite frame carries the token header at the picture position instead of the
pixels; the picture stays on the device.

Which filters accept a lite super:

- `CudaMAnalyse`, `CudaMRecalculate`, `CudaMFlowFps`, `CudaMCompensate`, `CudaMDegrain1..6` and the CPU filters
  `MAnalyse`, `MRecalculate`, `MFlowFps`, `MCompensate`, `MDegrain1..6`. Wherever they run on the CPU, they complete the
  frame themselves (pel 4 included), bit-identical to the normal super.
- `MFlow`, `MFlowInter`, `MFlowBlur`, `MBlockFps`, `MShow`, `MDegrainN` (the CPU class, i.e. `tr` above 6 or different
  `thSAD2`) and `MFinest` stop with an error. Give them a super made with `MSuper`.

### CudaMAnalyse

The GPU path is used when all of these hold:

| Setting | Supported on the GPU |
|---|---|
| super clip | from `CudaMSuper` or this build's `MSuper`, `sharp=2`, `rfilter` 2 or 4, no `pelclip`, `pel` 1, 2 or 4 |
| format | planar, 8 to 16 bit integer; with `chroma=true` chroma subsampling 1 or 2 |
| `blksize` | up to 32 |
| `levels` | up to 12 (counted levels) |
| `search` | 0 to 7 |
| `pelsearch` (finest level) | 1 to 24; 1 to 64 for `search=3` (exhaustive) |
| `searchparam` (coarse levels, when there is more than one level) | 1 to 64 when the coarse levels search exhaustively (`search_coarse=3`, the default); 1 to 24 for the other coarse types and for `search=6` / `7` (horizontal / vertical, every level) |
| `search_coarse` | 0 to 7 (upstream 2.7.48: search type of the coarse levels) |
| `trymany`, `trymanyall` | true or false |
| `divide`, `dct`, `temporal`, `multi` | 0 / false only |
| `outfile` | not set |

Settings outside the table are a creation error (or the CPU filter with `cuda_cpu_fallback`).

Single frames run on the CPU when the clip is field-based (a field shift is needed), when `delta` is 0 or less (static
reference), or, with `meander=true`, when the GPU reports that the frame needs the CPU's bad-vector wide search
(`badSAD` / `badrange`); with `meander=false` the GPU runs that search itself.

**Slices, `mt` and `meander`.** The CPU search with `mt=true` is split into horizontal slices, one per thread, and its
result depends on the number of slices. The GPU reproduces that:

- `meander=true` (default): the GPU searches in as many slices as the CPU uses. That is the avstp thread count with
  `mt=true`; 1 with `mt=false`, and 1 when there is no thread pool (avstp not loaded, VapourSynth, Linux), because the
  CPU then searches in one slice too. One slice is a long serial chain, so the GPU is slow there: under VapourSynth and
  on Linux use `meander=false` for speed. `MVTOOLS_GPU_SLICES=n` overrides the count, which gives a different vector
  field from the CPU default. (Before 2026-11-01c the GPU used 24 slices without a thread pool and so differed from
  the CPU and from its own CPU fallback.)
- `meander=false` with CudaMAnalyse: the GPU runs a wavefront search over the whole frame in one slice. This is the
  faster GPU mode. To keep the output independent of where a frame was searched, the CPU fallback of such a filter
  (per frame, or the whole filter with `cuda_cpu_fallback`) searches in one slice too, whatever `mt` says.

Before 2026-11-01b the slice search (`meander=true`) was serialised for frames larger than about 736x576: its result
buffer went through the GPU's copy engine, whose queue held every later upload. HD
QTGMC with `meander=true` was then slower than the CPU; it is now about 6x faster at the same, byte-identical result.

**Speed of the wider settings.** With `meander=true`, exhaustive radii above 3 (the finest level with `search=3`, every
coarse level with `searchparam` above 3) use a parallel exhaustive pass: the reference area of all candidates in
shared memory, candidates pruned by their vector cost like the CPU's, one parallel minimum. Measured (RTX 5090 against
24 CPU threads, MAnalyse only): PAL 1500 frames Prefetch 8, `pelsearch=24`: 8 bit 1.31 ms/frame (CPU 1.54), 10 bit 1.75
(CPU 2.12). UHD 10 bit 4:2:2, 32x32 blocks, 40 frames, Prefetch 3: `pelsearch=8` 0.52 s (CPU 0.85),
`pelsearch=24` 0.77 s (CPU 1.08); before 2026-11-01b the GPU searches of different frames did not run at the same time
there and the CPU was faster. With `meander=false` (wavefront kernel) large radii keep the sequential batches (exact,
slower).
`trymany` / `trymanyall` run their refines in parallel on the GPU, with `meander=true` and `meander=false` (wavefront
kernel) alike. PAL 500 frames, Prefetch 8, `search=4`, MAnalyse only, RTX 5090 against 24 CPU threads: default 0.71 s
(CPU 0.49), `trymany` 0.82 s (CPU 0.52), with `trymanyall` 0.93 s (CPU 0.68), `meander=false` + `trymany` 0.73 s (CPU
0.55); `search_coarse` 2 or 4 cost about the same as the default 3. At SD size the GPU search is still slower than 24
CPU threads; the GPU gains at HD / UHD and in whole GPU chains.
`pel=4` keeps 16 sub-pel planes per plane on the device (4 times the memory of pel 2).

The output is a normal vector clip; every consumer of vectors works unchanged.

### CudaMRecalculate

The same super, format, `blksize` and `divide` / `dct` / `outfile` conditions as CudaMAnalyse, and:

- `search` 0 to 7, with `searchparam` 1 to 24 (1 to 64 for `search=3`)
- `smooth` 0 or 1
- `tr` 0 (a single vector clip)

Single frames of a field-based clip run on the CPU. Slicing follows `mt` as for MAnalyse with `meander=true`. The output is
a normal vector clip.

### CudaMCompensate

The GPU path is used when all of these hold:

- `super` comes from `CudaMSuper`
- 8 to 16 bit, planar Y or YUV, at least 68 pixels wide (the token header must fit)
- `pel` 1 or 2
- `recursion=0`, `fields=false`, `tr=0`

The output is token frames. Frames without a usable vector field (scene change) come out as the source or reference
picture, as on the CPU. A frame that cannot be made on the GPU is made on the CPU and uploaded.

### CudaMDegrain1 .. CudaMDegrain6, CudaMDegrainN

The GPU path is used when all of these hold:

- `super` comes from `CudaMSuper`
- 8 to 16 bit, planar Y or YUV
- `pel` 1 or 2
- `limit` and `limitC` 255 (the default)
- `lsb` and `out16` false. Since upstream 2.7.48 `lsb=true` is `out16=true` plus a conversion to stacked 8-bit, so it has
  no GPU path either; `out32` no longer exists
- CudaMDegrainN: `tr` 1 to 6, `thSAD2` = `thSAD`, `thSADC2` = `thSADC`

The output is token frames; frames made on the CPU are uploaded.

### CudaMFlowFps

The GPU path is used when all of these hold:

- `mask=2` (the default)
- super from `CudaMSuper` or this build's `MSuper` with `sharp=2`, no `pelclip`, `pel` 1, 2 or 4
- planar YUV with three planes (not Y only, not RGB), 8 to 16 bit integer, chroma subsampling 1 or 2

The output is token frames (made again after eviction). Single frames go to the CPU whenever the GPU cannot make them,
for example frames whose vectors are not usable (scene change; see `thSCD1` / `thSCD2`) and frames at time 0 (the
source frame); those are uploaded.

---

## cudafilters

Every filter in `cudafilters.dll` builds the CPU filter or filter chain it replaces, with the same arguments, behind
`CudaDownload`. That CPU chain decides the output VideoInfo and parity and makes every frame the GPU cannot make (uploaded
again, so the output stays token frames). So each filter takes **the same parameters and defaults as its CPU
counterpart**, and the CPU plugin must be loaded. The tables below list the parameters and what part runs on the GPU
("On the GPU"); outside that the filter stops at creation with the reason, or runs the CPU chain with
`cuda_cpu_fallback` (see "CPU fallback"). For the meaning of each parameter, see the documentation of the CPU filter.

All filters return token frames, also when their input is normal frames, and are `MT_NICE_FILTER`. End a chain of them
with `CudaDownload`.

### Transfer

#### CudaUpload(clip)

Uploads every frame to the device and returns token frames. Token frames pass through unchanged. Planar Y / YUV, 8 to 16
bit, chroma subsampling 1 or 2 (no RGB); other formats, or no usable GPU, are a creation error (with
`cuda_cpu_fallback` the frames pass through as normal frames).

Upload is not needed: every `Cuda*` function uploads normal frames itself. Frames uploaded by `CudaUpload` can be made
again after eviction (it reads its input again); a filter that uploads normal frames itself uploads them again when
needed.

#### CudaDownload(clip)

Turns token frames into normal frames and lets normal frames of any format pass through (it never stops at creation).
An evicted frame is made again by its producer. It stops with an error when a token is of another format than the clip, or when the GPU library is not loaded.

### AviSynth+ core

| Function | Replaces | Parameters (defaults) | On the GPU |
|---|---|---|---|
| `CudaBob(clip, float b, float c, int height)` | `Bob` | `b` 1/3, `c` 1/3, `height` = clip height | frame-based input; any device frame format |
| `CudaTemporalSoften(clip, int radius, int luma_threshold, int chroma_threshold, int scenechange, int mode)` | `TemporalSoften` | `scenechange` 0, `mode` 1 (ignored, as on the CPU) | `radius` 1..7 and at least one threshold > 0 |
| `CudaMerge(clip, clip2, float weight)` | `Merge` | `weight` 0.5 | same format for both clips |
| `CudaMergeLuma(clip, clip2, float weight)` | `MergeLuma` | `weight` 1.0 | same format for both clips |
| `CudaMergeChroma(clip, clip2, float weight)` | `MergeChroma` | `weight` 1.0 | same format, YUV |
| `CudaGaussResize(clip, int width, int height, float src_left, float src_top, float src_width, float src_height, float p, float b, float s, int force, bool keep_center, string placement)` | `GaussResize` | `src_left`/`src_top` 0, `src_width`/`src_height` = source size, `p` 30, `b` 2, `s` 4, `force` 0, `keep_center` true, `placement` "auto" | any device frame format |
| `CudaBilinearResize(clip, int width, int height, float src_left, float src_top, float src_width, float src_height, int force, bool keep_center, string placement)` | `BilinearResize` | as CudaGaussResize, without `p` / `b` / `s` | any device frame format |

`radius`, `luma_threshold` and `chroma_threshold` of CudaTemporalSoften are required, as in TemporalSoften. With
`radius` 0 or both thresholds 0 TemporalSoften returns the clip unchanged; CudaTemporalSoften stops with an error
there: leave the filter out. The Merge
filters follow the AviSynth+ 3.7.5 arithmetic of its SSE2 / AVX2 code (including its scalar row tails), and return a
source frame unchanged where the CPU filter does (weight near 0, or near 1 for Merge). The resizers follow the
AviSynth+ 3.7.5 `FilteredResize` and read `_ChromaLocation` from frame 0 when `placement` is "auto".

### RgTools, MaskTools2 (8 to 16 bit on the GPU, CudaLutxy 8 and 10 bit)

| Function | Replaces | Parameters (defaults) | On the GPU |
|---|---|---|---|
| `CudaRemoveGrain(clip, int mode, int modeU, int modeV)` | `RemoveGrain` | `mode` 2, `modeU` = mode, `modeV` = modeU | every plane mode in -1, 0, 12, 20, and 11 on SSE4.1 CPUs (where RgTools computes mode 11 like 12) |
| `CudaLogic(clip a, clip b, string mode, int Y, int U, int V)` | `mt_logic` | `mode` "and", `Y` 3, `U` 1, `V` 1 | `mode` "min" / "max" |
| `CudaClamp(clip c, clip bright, clip dark, int overshoot, int undershoot, int Y, int U, int V)` | `mt_clamp` | `overshoot` 0, `undershoot` 0, `Y` 3, `U` 1, `V` 1 | `overshoot` / `undershoot` 0..255 |
| `CudaExpand(clip, string mode, int Y, int U, int V)` | `mt_expand` | `mode` "square", `Y` 3, `U` 1, `V` 1 | `mode` "vertical" |
| `CudaInpand(clip, string mode, int Y, int U, int V)` | `mt_inpand` | as CudaExpand | `mode` "vertical" |
| `CudaLutxy(clip x, clip y, string expr, int use_expr, int Y, int U, int V)` | `mt_lutxy` | `expr` "x", `use_expr` 0, `Y` 3, `U` 1, `V` 1 | any expression |

For the MaskTools2 filters, the GPU covers the plane modes 3 (process) and 2 (copy the first clip); any other plane mode
(such as the default `U=1, V=1`) has no GPU path (creation error, or the CPU filter with `cuda_cpu_fallback`). Pass
`U=3, V=3` or `U=2, V=2`.
CudaLutxy does not parse the expression: at creation it runs the installed `mt_lutxy` once over all (x, y) pairs (256 x
256 at 8 bit, 1024 x 1024 at 10 bit, in a 4:4:4 clip) and uses the result as two tables, one for luma and one for chroma
(with `use_expr` masktools hands the expression to `Expr`, whose chroma results can differ from luma), so it computes
exactly what the CPU filter computes. Samples above the bit depth's maximum (AviSynth+ `Sharpen` leaves such values in
10-bit clips) are read as the maximum. `CudaClamp`'s `overshoot` / `undershoot` are scaled as masktools scales them
(given in 8-bit units: 255 means the maximum, other values are shifted up).

### NNEDI3

#### CudaNNEDI3(clip, int field, bool dh, bool Y, bool U, bool V, int nsize, int nns, int qual, int etype, int pscrn, int threads, int opt, int fapprox)

Replaces `nnedi3` of plugins_JPSDR 4.1.0, which must be loaded (the network weights are read from it). The defaults are
nnedi3's: `field` -1, `dh` false, `Y`/`U`/`V` true, `nsize` 6, `nns` 1, `qual` 1, `etype` 0, `pscrn` 2, `opt` 0,
`fapprox` 15. `threads` is passed to the CPU fallback only.

The GPU reproduces the plugin's AVX2 + FMA code path (its `opt` 6 / 7). It runs on the GPU when all of these hold:

- 8 or 10-bit planar YUV or Y (above 10 bit the plugin takes a different code path; 9 bit is not tested)
- `dh=false`, `field` -2..3
- the plugin would take its AVX2 + FMA path on this CPU: `opt=0` on a CPU with AVX2 + FMA3 (or FMA4) and without
  AVX-512, or `opt` 6 / 7 set explicitly
- `pscrn` 2..4 (the new prescreener)
- `fapprox` with the int16 predictor and the fastest exp (bits 2 and 8 set; the default 15 has them)
- `qual` 1 or 2; any `nsize`, `nns`, `etype`
- this CPU's `rcpps` / `rsqrtps` lookup tables have the structure the GPU emulates (checked once per process)

Outside these conditions it stops at creation with the reason. Above 10 bit, and on CPUs where nnedi3 does not take
its AVX2 + FMA path (AVX-512 or older CPUs), set `cuda_cpu_fallback = true` to run nnedi3 itself.

### QTGMC

#### CudaKeepOnlyBobShimmerFixes(clip input, clip ref, int rep, bool chroma)

Replaces `QTGMC_KeepOnlyBobShimmerFixes` of `QTGMC.avsi` (which must be loaded). Defaults: `rep` 1, `chroma` true. On the
GPU: 8 or 10 bit, both clips of the same format, and `rep` with an edge mode other than 2 or 5 (the edge mode is `rep` itself
below 10, else its tens digit; for `rep` >= 10 the units digit must be 0..2).

#### CudaQtgmcBinomial0(clip b, int luma_threshold, int chroma_threshold, int scenechange, bool chroma)

QTGMC's search-clip binomial for `TR0=2`, in three kernels instead of eight:

```
b.TemporalSoften(1, lt, ct, sc, 2).MergeLuma(b.TemporalSoften(2, lt, ct, sc, 2), 0.357).MergeLuma(b, 0.125)
```

With `chroma=true` it uses `Merge` instead of `MergeLuma` (QTGMC's `ChromaMotion`). Defaults: `luma_threshold` 255,
`chroma_threshold` 0, `scenechange` 28, `chroma` false. Any device frame format.

#### CudaQtgmcSharpLimit(clip edi, clip degrain1, clip fComp1, clip bComp1, string resharpExpr, int rgBlur, int Sovs, clip degrain2, clip fComp3, clip bComp3, int tr1, int smode, bool precise, int nref)

QTGMC's output side from the binomial to the temporal sharpness limit, in one step. The first seven arguments are
required. Defaults: `degrain2`, `fComp3` and `bComp3` = `edi` (only read when used), `tr1` 1, `smode` 2, `precise` false,
`nref` 3.

```
binomial1 = tr1 0: edi
            tr1 1: degrain1.Merge(edi, 0.25)
            tr1 2: degrain1.Merge(degrain2, 0.2).Merge(edi, 0.0625)
smode 0:  resharp = binomial1
smode 1:  resharp = binomial1.mt_lutxy(binomial1.RemoveGrain(rgBlur), resharpExpr, use_expr=1, U=3, V=3)
smode 2:  vresharp = Merge(binomial1.mt_expand(mode="vertical", U=3, V=3), binomial1.mt_inpand(mode="vertical", U=3, V=3))
          precise: vresharp = vresharp.mt_lutxy(binomial1, "x y < x 1 scalef + x y > x 1 scalef - x ? ?", use_expr=2, U=3, V=3)
          resharp = binomial1.mt_lutxy(vresharp.RemoveGrain(rgBlur), resharpExpr, use_expr=1, U=3, V=3)
nref 3:   tMax / tMin = mt_logic "max" / "min" over edi, fComp1, bComp1 (nref 5: also fComp3, bComp3)
          result = resharp.mt_clamp(tMax, tMin, Sovs, Sovs, U=3, V=3)
nref 0:   result = resharp
```

With `smode=0, nref=0` it returns binomial1 itself (QTGMC's `lossed1`, the input of the back blend). It stops with an error
for `tr1` outside 0..2, `smode` outside 0..2, or `nref` other than 0 / 3 / 5. On the GPU: 8 or 10 bit (with `smode` 0: 8 to
15 bit), all clips of one format, `Sovs` 0..255 (in 8-bit units, scaled like `mt_clamp` does), and with `smode` 1 or 2
`rgBlur` 12 (or 11 on SSE4.1 CPUs).

#### CudaQtgmcBackBlend(clip thin, clip lossed, float src_width, float src_height, clip edi, clip fComp1, clip bComp1, clip fComp3, clip bComp3, int nref, int sovs)

QTGMC's back blend (`Sbb`), followed with `nref` > 0 by the same temporal limit as CudaQtgmcSharpLimit:

```
thin.mt_makediff(mt_makediff(thin, lossed, U=1, V=1).RemoveGrain(12, -1)
     .GaussResize(w, h, 0, 0, src_width, src_height, p=5), U=2, V=2)
```

QTGMC passes `src_width` / `src_height` = width / height plus a small epsilon. Defaults: `nref` 0, `sovs` 0. `nref` must
be 0, 3 or 5, and the reference clips it needs (`edi`, `fComp1`, `bComp1`, and for 5 `fComp3`, `bComp3`) must be given;
otherwise it stops with an error. On the GPU: 8 to 16 bit, all clips of one format, `sovs` 0..255 (8-bit units),
`src_width` != width and `src_height` != height, both > 0.

#### CudaQtgmcAddNoise(clip c, clip noise, string expr, int use_expr, bool clamp_float, int lutUV, int addUV)

QTGMC's grain / noise restore as one table lookup per sample:

```
c.mt_adddiff(noise.mt_lut(expr, use_expr=use_expr, clamp_float=clamp_float, U=lutUV, V=lutUV), U=addUV, V=addUV)
```

The tables are made by running that chain once on a 256 x 256 (10 bit: 1024 x 1024) pattern. Defaults: `use_expr` 0,
`clamp_float` false, `lutUV` 1, `addUV` 2. On the GPU: 8 or 10 bit, both clips of one format, and `addUV=2`, or
`lutUV=3` with `addUV=3`.

### Motion vectors from the optical flow accelerator

#### CudaNVOFAnalyse(clip c, bool isb, int delta, int blksize, int blksizeV, int overlap, int overlapV, int pel, int hpad, int vpad, bool chroma, int preset)

A vector clip in MAnalyse's format (accepted by MRecalculate, MFlowFps, MDegrain, MCompensate) from NVIDIA's optical flow
accelerator (NVOFA, `nvofapi64.dll` of the driver; Turing or newer). It takes the plain clip, not a super clip: no MSuper
pyramid is built for the search. Per block: the median of NVOFA's 4x4-grid flow under the block (x and y separately),
rounded to 1/pel pixel, clamped to the area MAnalyse could reach with that padding, and its SAD (luma + chroma as MAnalyse
with `scaleCSAD=0`). One level (`nLvCount` 1). **Not bit-exact with anything**: a different algorithm that follows true
motion better (fast pans) and minimises SAD less than MAnalyse. Made for uc_FRC's coarse pass, which MRecalculate refines.

Defaults: `isb` false, `delta` 1, `blksize` 16, `blksizeV` = blksize (MAnalyse's sizes), `overlap` 0, `overlapV` =
overlap, `pel` 2 (1 or 2), `hpad` / `vpad` 16 (must be those of the MSuper used downstream), `chroma` true, `preset` 2
(1 fast, 2 medium, 3 slow). Planar YUV or Y, 8 to 16 bit. Frames without a reference get invalid vectors, as MAnalyse.

Fallback: without NVOFA (or with `MVTOOLS_GPU_DISABLE`) the function stops at creation; with `cuda_cpu_fallback` it
returns `MAnalyse(MSuper(c, pel, hpad, vpad, chroma), isb, delta, blksize, blksizeV, overlap, overlapV, truemotion=true,
chroma)` (mvtools2.dll must be loaded). Use `CudaNVOFAvailable()` to choose a branch in the script instead. A single
frame NVOFA fails on gets that MAnalyse's level-0 vectors.

#### CudaNVOFAvailable()

True when CudaNVOFAnalyse will use NVOFA (library loaded, device present, a session can be made). For choosing a branch
in a script, as uconvert's `uc_frc_nvof` does.

### Using cudafilters with QTGMC

`qtgmc\make_qtgmc_gpu.py` makes a patched copy of `QTGMC.avsi` (`QTGMC_gpu.avsi`) that uses these filters and the
mvtools2 `Cuda*` functions. Run it with `python -I make_qtgmc_gpu.py [source.avsi] [dest.avsi]`. The source file is only
read. In the patched script, `global QTGMC_gpu = true` after the `Import` switches the GPU on (default false). The motion
analysis goes through the wrappers `QTGMC_MSuper` / `QTGMC_MAnalyse` / `QTGMC_MRecalculate` (the `Cuda*` function with
`QTGMC_gpu`, the CPU filter otherwise); the GPU output side calls CudaMSuper / CudaMCompensate / CudaMDegrain directly.
`QTGMC_gpu_meander` (default true) keeps CudaMAnalyse on the slice search, so the result is byte-identical to the stock
CPU QTGMC; `global QTGMC_gpu_meander = false` switches to the wavefront search (faster, slightly different vectors).
`global QTGMC_gpu_nvof = true` takes the MAnalyse vectors from the optical flow accelerator instead (CudaNVOFAnalyse on the
plain search clip, `QTGMC_gpu_nvof_preset` 1 fast / 2 medium / 3 slow; MRecalculate refines them as usual): not bit-exact. The search
itself costs about as much as the `meander=false` wavefront search (about 0.2 ms per 1080p frame, against 1.3 ms for
the exact search); the rest of QTGMC is unchanged. SubPel 4 and GPUs without NVOFA use the exact search.
A stage without a GPU path (above 10 bit: lookup tables and NNEDI3; NNEDI3 also on CPUs without its AVX2 + FMA path)
stops the script at creation; set `global cuda_cpu_fallback = true` before QTGMC to run such stages on the CPU.

The base is QTGMC 3.393s (A.SONY / Dogway mod, realfinder AVS-Stuff, Community branch, `avs 2.6 and up/QTGMC.avsi`);
the generator stops if a pattern it patches is not found as often as in that version. Its 3.39x options on the GPU:
`RefineMotion2`, `RefineMotionAll` / `RefineMotionSearch` / `RefineMotionRadius`, `trymany`, `MEus` (the upsizing
LanczosResize runs on the CPU, the analysis on the GPU), `pscrn` 2..4. `xflateMaTo` and `PhosphorSoft` keep the stages
they change on the CPU. No GPU path (creation error unless `cuda_cpu_fallback`): `temporal=true` (MAnalyse) and
`pscrn` 0 / 1 (NNEDI3). `temporal=true` is not reproducible run to run with stock CPU QTGMC either.

The Windows release bundle wraps this: `QTGMC_gpu_load.avsi` (source: `qtgmc\bundle\`) loads every DLL and script from
its own folder and defines `QTGMC_gpu_on()` / `QTGMC_gpu_off()`, which set all the QTGMC GPU switches (`QTGMC_gpu`,
`QTGMC_gpu_search`, `QTGMC_gpu_tokens`, `QTGMC_gpu_edi`, `QTGMC_gpu_out`) at once. `qtgmc\bundle\examples\` has example
scripts for QTGMC, a GPU / CPU comparison and frame-rate conversion (exact search and CudaNVOFAnalyse).

**Reported on Ampere (open).** With the `gpu-2026-11-01c` bundle (commit 0288391) on an RTX 3090 Ti (compute 8.6,
driver 616.56) the whole SD QTGMC chain is slower on the GPU than the stock CPU QTGMC, not faster: 720x480, 1500
interlaced frames, `Prefetch(12)`, against a Ryzen 9 9950X3D (16C/32T), Fast 26.96 s on the GPU against 4.44 s on the
CPU, Slower 74.67 s against 7.23 s. `global QTGMC_gpu_meander = false` recovers about 3x (Fast 8.68 s) but stays about
2x slower than the CPU. The same script with `MVTOOLS_GPU_FORCE_FALLBACK=1` takes 3.66 s, so the GPU compute is the
cause and not the script path; the output is bit-exact either way. Not seen on the RTX 5090 the bundle was tested on,
where SD Fast is 1.9 s on the GPU against 3.4 s on the CPU.

---

## Environment variables

Read once per process.

### mvtools2 (`mvtools2.dll`)

| Variable | Effect |
|---|---|
| `CUDA_CPU_FALLBACK` | `1` / `0`: the CPU fallback switch when the script global `cuda_cpu_fallback` is not set (see "CPU fallback"); read by cudafilters too |
| `MVTOOLS_GPU_DISABLE` | Do not load `mvtools_cuda.dll`: every `Cuda*` function counts as "GPU not usable" (creation error, or the CPU with the fallback switch). cudafilters honours it too |
| `MVTOOLS_GPU_LOG` | Print every GPU decision and per-frame fallback to stderr, and the device frame registry statistics when the last GPU filter is destroyed |
| `MVTOOLS_GPU_FORCE_FALLBACK` | Every GPU call reports "use the CPU": every frame is computed on the CPU and uploaded (for testing the fallback paths) |
| `MVTOOLS_GPU_SLICES=n` | Slice count of MAnalyse (`meander=true`) and MRecalculate on the GPU, only with `mt=true`; changes the vector field compared with the CPU |
| `MVTOOLS_GPU_PROFILE` | Wall time per named phase, printed to stderr when the process ends |

### cudafilters (`cudafilters.dll`)

| Variable | Effect |
|---|---|
| `CUDAFILTERS_LOG` | Print for every filter whether it runs on the GPU or the CPU and why, and per-frame fallbacks |
| `CUDAFILTERS_STATS` | When CudaUpload / CudaDownload are destroyed: device frame registry statistics (frames, MB, uploads, downloads, evictions, regenerations) |

### GPU library (`mvtools_cuda.dll`)

| Variable | Effect |
|---|---|
| `MVCUDA_DEVICE=n` | Use CUDA device n (it must have compute capability 6.0 or newer) |
| `MVCUDA_CACHE_MB=n` | Fixed byte budget of the device frame registry (token frames, uploaded pictures and the super pyramids kept with them), in MB. Default: automatic; starts at 256 MB (or 128 frames, whichever is larger, at most 1024 MB) and grows while frames have to be made again, up to a third of the free device memory at first use, at most 4096 |
| `MVCUDA_WAVE=0` | MAnalyse with `meander=false` uses the slice kernel instead of the wavefront kernel (for comparisons) |
| `MVCUDA_SPIN` | Waits for the GPU spin a CPU core instead of sleeping |
| `MVCUDA_TIMING` | Kernel timings to stderr |

For comparisons only (same output, older or slower code forms): `MVCUDA_TM_PAR=0` (trymany refines one after the
other), `MVCUDA_EXH_FAST=0` (no parallel exhaustive pass), `MVCUDA_SEARCH_PACK=0` / `MVCUDA_WAVE_PACK=0` (no packed
SADs), `MVCUDA_NOPIN` (no pinned host staging).
