# MVTools GPU reference

This page covers the CUDA part of this mvtools build:

- the `gpu` parameter of the mvtools2 functions
- the `cudafilters` plugin, with GPU versions of the AviSynth core, MaskTools2, RgTools and NNEDI3 filters that QTGMC uses,
  plus fused QTGMC stages
- the environment variables

The design notes and measurements are in `GPU_PLAN.md`, `GPU_QTGMC_PLAN.md` and `GPU_PIPELINE_PLAN.md`. They describe how
the work got here; this page describes what the functions do now (library API version 13).

## Principles

- **Bit-exact.** Every GPU path gives exactly the result of the CPU filter it replaces, with the same arguments. The
  reference for mvtools2 is the CPU result with the same slice count (see `meander` under MAnalyse below).
- **Silent fallback.** `gpu=true` is a request. When the GPU library or device is missing, or the format or settings are
  not covered, the filter runs its CPU code and returns the same output. Single frames can fall back too, for example
  when a device frame was evicted and cannot be made again. Nothing in the script has to change.
  Set `MVTOOLS_GPU_LOG=1` (mvtools2) or `CUDAFILTERS_LOG=1` (cudafilters) to see on stderr which path every filter takes.
- **Default off.** All `gpu` parameters default to `false`.

## Installation

| File | Built by | Notes |
|---|---|---|
| `mvtools2.dll` | `mvtools.sln` | The plugin as before |
| `mvtools_cuda.dll` | `Sources\cuda\build.bat` (CUDA 13.x toolkit + MSVC build tools) | Must sit next to `mvtools2.dll`. Static CUDA runtime: only the NVIDIA driver is needed at run time |
| `cudafilters.dll` | `cudafilters\cudafilters.vcxproj` (see `cudafilters\README.md`) | Optional. Must sit in the same folder as the other two |

All three files must come from one build: the plugins check the library API version and run on the CPU when it does not
match.

Requirements:

- x64 AviSynth+. cudafilters needs AviSynth+ 3.6 or newer (interface V8) and refuses to load on older versions; 3.7.5 is the
  tested version.
- An NVIDIA GPU with compute capability 8.0 or newer (RTX 30xx, 40xx, 50xx, A100, H100). Older cards count as "no usable
  device". The first device with cc >= 8.0 is used; `MVCUDA_DEVICE=n` picks another.
- `CudaNNEDI3` and `CudaKeepOnlyBobShimmerFixes` call into other plugins: they need `plugins_JPSDR.dll` (nnedi3), and
  `QTGMC.avsi` (`QTGMC_KeepOnlyBobShimmerFixes`) with masktools2 and RgTools, loaded.

## Token frames

A GPU filter keeps its result in device memory. What it returns to AviSynth is a *token frame*: an ordinary frame of the
declared format whose first luma bytes hold a `MvCudaTokenHeader` (magic `MVCUDATK`, the device frame's registry key, the
filter and frame number that produced it, the format). Every 8th luma row of a token frame is black, so a filter that
does not understand tokens shows obvious stripes, not a plausible picture.

- **Readers.** Every `Cuda*` filter, `MSuper`, `MCompensate`, `MDegrain1..6`, `MAnalyse`, `MRecalculate` and `MFlowFps` accept
  token frames as input. They use the device frame directly, or download it when they run on the CPU. Any other filter
  needs `CudaDownload` in front of it.
- **Mixed input.** Every `Cuda*` filter also accepts normal frames and uploads them.
- **Eviction.** The device frames live in a registry with a memory budget (`MVCUDA_CACHE_MB`). When a frame is evicted
  before it is read, the filter that produced it makes it again, so evictions cost time but never change the result.
- **Supported formats.** Device frames are planar Y or YUV (no RGB, no alpha), 8 to 16 bit integer, chroma subsampling 1
  or 2 in each direction (4:2:0, 4:2:2, 4:4:4, Y only), and at least 72 bytes per luma row (the header must fit). Many
  filters cover only 8 bit on the GPU; see each filter below. In any other format a filter runs on the CPU.

A typical chain:

```
CudaUpload()                         # or start with a Cuda* filter
...Cuda* filters...
sup = MSuper(pel=2, gpu=true)        # "lite" super of the token clip
...MAnalyse / MDegrain / MCompensate with gpu=true...
CudaDownload()                       # pixels again for the rest of the script
```

---

## mvtools2: the `gpu` parameter

| Function | What runs on the GPU | Output |
|---|---|---|
| `MSuper(gpu=true)` | Nothing: it makes a "lite" super clip that holds only the picture, so the GPU filters build the pyramid themselves | lite super clip |
| `MAnalyse(gpu=true)` | Hierarchical search, all levels, with its own super pyramid | normal vector clip |
| `MRecalculate(gpu=true)` | Refinement search | normal vector clip |
| `MCompensate(gpu=true)` | Block compensation | token frames |
| `MDegrain1..6(gpu=true)` | Degraining | token frames |
| `MFlowFps(gpu=true)` | Flow interpolation (`mask=2`) | normal frames |

`MDegrainN`, `MFlow`, `MFlowInter`, `MFlowBlur`, `MBlockFps`, `MMask`, `MSCDetection`, `MShow`, `MDepan`, `MScaleVect`,
`MStoreVect` and `MRestoreVect` have no GPU path.

Unless noted otherwise, the GPU paths need the super clip to come from `MSuper` of this build with `sharp=2` and without
`pelclip`. The super clip can be a lite super or a normal one.

### MSuper

`gpu` (bool, default false)

With `gpu=true`, MSuper skips the CPU pyramid and the sub-pel planes. When all of the following hold, each frame holds
only the picture inside its padding rows:

- the GPU library is usable
- `pel` 1 or 2, `sharp=2`, `rfilter` 2 or 4, no `pelclip`
- planar YUV (not Y only, not YUY2), 8-bit or 16-bit container
- chroma subsampling 1 or 2 in each direction
- `hpad` divisible by the horizontal chroma ratio, `vpad` by the vertical one, height divisible by the vertical one

Otherwise it makes the normal super clip. When the input frame is a token frame, the lite frame carries the token header
at the picture position instead of the pixels; the picture stays on the device.

Which filters accept a lite super:

- `MAnalyse`, `MRecalculate`, `MFlowFps`, `MCompensate` and `MDegrain1..6` use it. Wherever they run on the CPU, they
  complete the frame themselves, bit-identical to the normal super.
- `MFlow`, `MFlowInter`, `MFlowBlur`, `MBlockFps`, `MShow` and `MDegrainN` stop with an error. Give them a super made
  without `gpu=true`.

Lite super frames cost almost nothing on the CPU. Use one wherever all consumers are on the list above.

### MAnalyse

`gpu` (bool, default false)

The GPU path is used when all of these hold:

| Setting | Supported on the GPU |
|---|---|
| super clip | from this build's `MSuper`, `sharp=2`, `rfilter` 2 or 4, no `pelclip`, `pel` 1 or 2 |
| format | planar, 8-bit or 16-bit container; with `chroma=true` chroma subsampling 1 or 2 |
| `blksize` | up to 32 |
| `levels` | up to 12 (counted levels) |
| `search` | 0 to 7 |
| `pelsearch` (finest level) | 1 to 24; 1 to 3 for `search=3` (exhaustive) |
| `searchparam` (coarse levels, when there is more than one level) | 1 to 3; 1 to 24 for `search=6` / `7` (horizontal / vertical) |
| `divide`, `dct`, `temporal`, `trymany`, `multi` | 0 / false only |
| `outfile` | not set |

Single frames run on the CPU when the clip is field-based (a field shift is needed), when `delta` is 0 or less (static
reference), or when the GPU reports that the frame needs the CPU's bad-vector wide search (`badSAD` / `badrange`).

**Slices, `mt` and `meander`.** The CPU search with `mt=true` is split into horizontal slices, one per thread, and its
result depends on the number of slices. The GPU reproduces that:

- `meander=true` (default): the GPU searches in as many slices as the CPU uses. That is the avstp thread count with
  `mt=true`; 24 when `mt=true` but no thread pool is available; 1 with `mt=false`. `MVTOOLS_GPU_SLICES=n` overrides the
  count, which gives a different vector field from the CPU default.
- `meander=false` together with `gpu=true`: the GPU runs a wavefront search over the whole frame in one slice. This is the
  faster GPU mode. To keep the output independent of where a frame was searched, the CPU fallback of such a filter
  searches in one slice too, whatever `mt` says.

The output is a normal vector clip; every consumer of vectors works unchanged.

### MRecalculate

`gpu` (bool, default false)

The same super, format, `blksize` and `divide` / `dct` / `outfile` conditions as MAnalyse, and:

- `search` 0 to 7, with `searchparam` 1 to 24 (1 to 3 for `search=3`)
- `smooth` 0 or 1
- `tr` 0 (a single vector clip)

Single frames of a field-based clip run on the CPU. Slicing follows `mt` as for MAnalyse with `meander=true`. The output is
a normal vector clip.

### MCompensate

`gpu` (bool, default false)

The GPU path is used when all of these hold:

- `super` is a lite super (`MSuper(gpu=true)`) **of a token clip**, so the pictures are on the device
- 8 bit, planar Y or YUV
- `pel` 1 or 2
- `recursion=0`, `fields=false`, `tr=0`

The output is token frames. Frames without a usable vector field (scene change) come out as the source or reference
picture, as on the CPU. When a frame cannot be made on the GPU, it is made on the CPU and returned as a normal frame.

### MDegrain1 .. MDegrain6

`gpu` (bool, default false)

The GPU path is used when all of these hold:

- the source clip is a token clip, and `super` is a lite super of a token clip
- 8 bit, planar Y or YUV
- `pel` 1 or 2
- `limit` and `limitC` 255 (the default)
- `lsb`, `out16`, `out32` false

The output is token frames, or normal frames for frames made on the CPU. MDegrainN has no `gpu` parameter.

### MFlowFps

`gpu` (bool, default false)

The GPU path is used when all of these hold:

- `mask=2` (the default)
- super from this build's `MSuper` with `sharp=2`, no `pelclip`, `pel` 1 or 2
- planar YUV with three planes (not Y only, not RGB), 8-bit or 16-bit container, chroma subsampling 1 or 2

The output is normal frames. Single frames go to the CPU whenever the GPU cannot make them.

---

## cudafilters

Every filter in `cudafilters.dll` builds the CPU filter or filter chain it replaces, with the same arguments, behind
`CudaDownload`. That CPU chain decides the output VideoInfo and parity and makes every frame the GPU cannot make. So each
filter takes **the same parameters and defaults as its CPU counterpart**, and the CPU plugin must be loaded. The tables
below list the parameters and what part runs on the GPU. For the meaning of each parameter, see the documentation of the
CPU filter.

All filters return token frames when they run on the GPU, and are `MT_NICE_FILTER`.

### Transfer

#### CudaUpload(clip)

Uploads every frame to the device and returns token frames. Token frames pass through unchanged. If the GPU or the format
is not usable, the frames pass through as normal frames; every consumer accepts those.

Upload is not needed in front of a `Cuda*` filter, because they all upload normal frames themselves. It is useful in
front of `MSuper(gpu=true)` when `MCompensate` / `MDegrain` should run on the GPU. Frames uploaded by `CudaUpload` can be
made again after eviction (it reads its input again); frames uploaded inside other filters cannot.

#### CudaDownload(clip)

Turns token frames into normal frames and lets normal frames pass through. An evicted frame is made again by its
producer. It stops with an error when a token is of another format than the clip, or when the GPU library is not loaded.

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

`radius`, `luma_threshold` and `chroma_threshold` of CudaTemporalSoften are required, as in TemporalSoften. The Merge
filters follow the AviSynth+ 3.7.5 arithmetic of its SSE2 / AVX2 code (including its scalar row tails), and return a
source frame unchanged where the CPU filter does (weight near 0, or near 1 for Merge). The resizers follow the
AviSynth+ 3.7.5 `FilteredResize` and read `_ChromaLocation` from frame 0 when `placement` is "auto".

### RgTools, MaskTools2 (8 bit on the GPU)

| Function | Replaces | Parameters (defaults) | On the GPU |
|---|---|---|---|
| `CudaRemoveGrain(clip, int mode, int modeU, int modeV)` | `RemoveGrain` | `mode` 2, `modeU` = mode, `modeV` = modeU | every plane mode in -1, 0, 12, 20, and 11 on SSE4.1 CPUs (where RgTools computes mode 11 like 12) |
| `CudaLogic(clip a, clip b, string mode, int Y, int U, int V)` | `mt_logic` | `mode` "and", `Y` 3, `U` 1, `V` 1 | `mode` "min" / "max" |
| `CudaClamp(clip c, clip bright, clip dark, int overshoot, int undershoot, int Y, int U, int V)` | `mt_clamp` | `overshoot` 0, `undershoot` 0, `Y` 3, `U` 1, `V` 1 | `overshoot` / `undershoot` 0..255 |
| `CudaExpand(clip, string mode, int Y, int U, int V)` | `mt_expand` | `mode` "square", `Y` 3, `U` 1, `V` 1 | `mode` "vertical" |
| `CudaInpand(clip, string mode, int Y, int U, int V)` | `mt_inpand` | as CudaExpand | `mode` "vertical" |
| `CudaLutxy(clip x, clip y, string expr, int use_expr, int Y, int U, int V)` | `mt_lutxy` | `expr` "x", `use_expr` 0, `Y` 3, `U` 1, `V` 1 | any expression |

For the MaskTools2 filters, the GPU covers the plane modes 3 (process) and 2 (copy the first clip); any other plane mode
(such as the default `U=1, V=1`) means the CPU filter. In practice, pass `U=3, V=3` or `U=2, V=2` to stay on the GPU.
CudaLutxy does not parse the expression: at creation it runs the installed `mt_lutxy` once over all 256 x 256 (x, y)
pairs and uses the result as the table, so it computes exactly what the CPU filter computes.

### NNEDI3

#### CudaNNEDI3(clip, int field, bool dh, bool Y, bool U, bool V, int nsize, int nns, int qual, int etype, int pscrn, int threads, int opt, int fapprox)

Replaces `nnedi3` of plugins_JPSDR 4.1.0, which must be loaded (the network weights are read from it). The defaults are
nnedi3's: `field` -1, `dh` false, `Y`/`U`/`V` true, `nsize` 6, `nns` 1, `qual` 1, `etype` 0, `pscrn` 2, `opt` 0,
`fapprox` 15. `threads` is passed to the CPU fallback only.

The GPU reproduces the plugin's AVX2 + FMA code path (its `opt` 6 / 7). It runs on the GPU when all of these hold:

- 8-bit planar YUV or Y
- `dh=false`, `field` -2..3
- the plugin would take its AVX2 + FMA path on this CPU: `opt=0` on a CPU with AVX2 + FMA3 (or FMA4) and without
  AVX-512, or `opt` 6 / 7 set explicitly
- `pscrn` 2..4 (the new prescreener)
- `fapprox` with the int16 predictor and the fastest exp (bits 2 and 8 set; the default 15 has them)
- `qual` 1 or 2; any `nsize`, `nns`, `etype`
- this CPU's `rcpps` / `rsqrtps` lookup tables have the structure the GPU emulates (checked once per process)

`CUDAFILTERS_LOG=1` prints the reason when it stays on the CPU.

### QTGMC

#### CudaKeepOnlyBobShimmerFixes(clip input, clip ref, int rep, bool chroma)

Replaces `QTGMC_KeepOnlyBobShimmerFixes` of `QTGMC.avsi` (which must be loaded). Defaults: `rep` 1, `chroma` true. On the
GPU: 8 bit, both clips of the same format, and `rep` with an edge mode other than 2 or 5 (the edge mode is `rep` itself
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
for `tr1` outside 0..2, `smode` outside 0..2, or `nref` other than 0 / 3 / 5. On the GPU: 8 bit, all clips of one format,
`Sovs` 0..255, and with `smode` 1 or 2 `rgBlur` 12 (or 11 on SSE4.1 CPUs).

#### CudaQtgmcBackBlend(clip thin, clip lossed, float src_width, float src_height, clip edi, clip fComp1, clip bComp1, clip fComp3, clip bComp3, int nref, int sovs)

QTGMC's back blend (`Sbb`), followed with `nref` > 0 by the same temporal limit as CudaQtgmcSharpLimit:

```
thin.mt_makediff(mt_makediff(thin, lossed, U=1, V=1).RemoveGrain(12, -1)
     .GaussResize(w, h, 0, 0, src_width, src_height, p=5), U=2, V=2)
```

QTGMC passes `src_width` / `src_height` = width / height plus a small epsilon. Defaults: `nref` 0, `sovs` 0. `nref` must
be 0, 3 or 5, and the reference clips it needs (`edi`, `fComp1`, `bComp1`, and for 5 `fComp3`, `bComp3`) must be given;
otherwise it stops with an error. On the GPU: 8 bit, all clips of one format, `sovs` 0..255, `src_width` != width and
`src_height` != height, both > 0.

#### CudaQtgmcAddNoise(clip c, clip noise, string expr, int use_expr, bool clamp_float, int lutUV, int addUV)

QTGMC's grain / noise restore as one table lookup per sample:

```
c.mt_adddiff(noise.mt_lut(expr, use_expr=use_expr, clamp_float=clamp_float, U=lutUV, V=lutUV), U=addUV, V=addUV)
```

The table is made by running that chain once on a 256 x 256 pattern. Defaults: `use_expr` 0, `clamp_float` false, `lutUV` 1,
`addUV` 2. On the GPU: 8 bit, both clips of one format, and `addUV=2`, or `lutUV=3` with `addUV=3`.

### Using cudafilters with QTGMC

`qtgmc\make_qtgmc_gpu.py` makes a patched copy of `QTGMC.avsi` (`QTGMC_gpu.avsi`) that uses these filters and the
mvtools2 `gpu` parameters. Run it with `python -I make_qtgmc_gpu.py [source.avsi] [dest.avsi]`. The source file is only
read. In the patched script, `global QTGMC_gpu = true` after the `Import` switches the GPU on (default false), and
`global QTGMC_gpu_meander = true` keeps MAnalyse on the slice search instead of the wavefront search.

---

## Environment variables

Read once per process.

### mvtools2 (`mvtools2.dll`)

| Variable | Effect |
|---|---|
| `MVTOOLS_GPU_DISABLE` | Do not load `mvtools_cuda.dll`; everything runs on the CPU. cudafilters honours it too |
| `MVTOOLS_GPU_LOG` | Print every GPU decision and per-frame fallback to stderr |
| `MVTOOLS_GPU_FORCE_FALLBACK` | Every GPU call reports "use the CPU" (for testing the fallback paths) |
| `MVTOOLS_GPU_SLICES=n` | Slice count of MAnalyse (`meander=true`) and MRecalculate on the GPU; changes the vector field compared with the CPU |
| `MVTOOLS_GPU_PROFILE` | Wall time per named phase, printed to stderr when the process ends |

### cudafilters (`cudafilters.dll`)

| Variable | Effect |
|---|---|
| `CUDAFILTERS_LOG` | Print for every filter whether it runs on the GPU or the CPU and why, and per-frame fallbacks |
| `CUDAFILTERS_STATS` | When CudaUpload / CudaDownload are destroyed: device frame registry statistics (frames, MB, uploads, downloads, evictions, regenerations) |

### GPU library (`mvtools_cuda.dll`)

| Variable | Effect |
|---|---|
| `MVCUDA_DEVICE=n` | Use CUDA device n (it must have compute capability 8.0 or newer) |
| `MVCUDA_CACHE_MB=n` | Byte budget of the device frame registry (token frames), in MB. Default: a quarter of the free device memory, at most 2048 |
| `MVCUDA_CACHE_FRAMES=n` | Capacity of the device super frame cache used by MAnalyse / MRecalculate / MFlowFps, in frames (default 24, minimum 2) |
| `MVCUDA_WAVE=0` | MAnalyse with `meander=false` uses the slice kernel instead of the wavefront kernel (for comparisons) |
| `MVCUDA_SPIN` | Waits for the GPU spin a CPU core instead of sleeping |
| `MVCUDA_TIMING` | Kernel timings to stderr |
