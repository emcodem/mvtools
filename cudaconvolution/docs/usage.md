# CudaConvolution - AviSynth+ plugin: progressive -> interlaced, with optional integer downscale

One self-contained 64-bit DLL. It converts progressive material of any size and any frame rate R into interlaced
material at R/2 (50p -> 25i, 59.94p -> 29.97i, ...): field assembly plus a vertical interlace (anti-twitter) filter.
Optionally it downscales by an integer factor 1..8 per axis in the same step. The built-in preset `balanced` holds the
filters per factor:

| factor (input / output) | horizontal | vertical | origin |
|---|---|---|---|
| 1 | none | interlace filter, 9 taps | designed to match the factor-2 vertical response |
| 2 | measured filter | measured filter | matches the measured reference (UHD -> 1080i: luma bit-exact) |
| 3..8 | designed | designed | same look as factor 2, transferred to the factor (`tools/design_taps.py`) |

Runs on the GPU (NVIDIA CUDA) by default, or on the CPU with `cpu=true`; both give identical output.

## Requirements

- Windows x64, AviSynth+ 3.7 or newer.
- GPU engine: NVIDIA GPU, compute capability 6.0+ (Pascal e.g. GTX 10xx / Quadro P, Volta, Turing, Ampere e.g. RTX A4000, Ada,
  Hopper, Blackwell), NVIDIA driver
  R525 or newer. Nothing else: the CUDA runtime is built into the DLL, no VC++ redistributable needed.
- CPU engine: any x64 CPU (AVX2/FMA3 used when available).

## Install

Copy `CudaConvolution.dll` into the AviSynth+ 64-bit plugin folder (e.g. `C:\Program Files (x86)\AviSynth+\plugins64+`)
or load it explicitly with `LoadPlugin("...\CudaConvolution.dll")`.

Check the installation once per machine (and after driver updates) with `examples\selftest.avs`. It shows one line
per engine (`CPU scalar`, `CPU AVX2`, `GPU <name>, driver ...`) and `RESULT: PASS`. If the GPU cannot be used, the
line says why (no driver, driver too old, unsupported GPU, ...).

## Usage

```
FFVideoSource("uhd50p.mov", threads=0)  # or LWLibavVideoSource, ...; e.g. YUV422P10 3840x2160 progressive
Prefetch(1)                             # directly after the source: decode ahead, in order (see below)
CudaConvolution(width=1920, height=1080)       # -> 1920x1080i25 TFF (factor 2: measured filter)

FFVideoSource("hd1080p50.mov", threads=0)
Prefetch(1)
CudaConvolution()                              # same size: 1080p50 -> 1080i25
```

Put `Prefetch(1)` directly after the source filter, not a `Prefetch(N)` at the end of the script: with several
threads requesting frames out of order, ffms2 keeps seeking and the whole chain gets up to 5x slower. CudaConvolution does
its own parallel work (GPU, or CPU worker threads), so it does not need `Prefetch` after it.

Behind a slow filter that wants its frames in order (e.g. a motion-compensated frame-rate converter), put
`Prefetch(1)` directly in front of CudaConvolution (and after it): that filter and CudaConvolution then run in their own threads
instead of one after the other. Example: `QTGMC(...).Prefetch(8).CudaSplatFps(60000, 1001).Prefetch(1).CudaConvolution().Prefetch(1)`
(RTX 5050, 1080i25 -> 1080i29.97: 0.91x -> 1.16x real time).

`CudaConvolution(clip c, int "width", int "height", string "preset", string "field_order", int "clip_min", int "clip_max",
int "device", bool "cpu", int "threads", bool "convert", bool "token_out")`

| parameter | default | meaning |
|---|---|---|
| `width`, `height` | input size | output size; input / output must be an integer 1..8 per axis. If only one is given, the other axis uses the same factor |
| `preset` | built-in `balanced` | path to your own preset file (JSON) to use instead |
| `field_order` | `"tff"` | `"tff"` or `"bff"` |
| `clip_min`, `clip_max` | none | output limits for all planes, in the input's bit depth (e.g. 10 bit: `clip_min=64, clip_max=940`). Without them only the storage range applies (0..1023 at 10 bit) |
| `device` | 0 | CUDA device index |
| `cpu` | false | true: CPU engine instead of the GPU |
| `threads` | 0 = all | CPU engine worker threads per frame (results identical for any value); use 1 only with a `Prefetch(N)` after CudaConvolution |
| `convert` | true | unsupported input formats are converted automatically (see below); false: stop with an error instead |
| `token_out` | false | true: return GPU token frames (see "GPU token frames" below); needs the GPU engine and the GPU plugins' `mvtools_cuda.dll` |

### GPU token frames (optional)

The companion GPU filter plugins pass *token frames* between their filters: ordinary frames whose first row holds a
reference to a picture in GPU memory, kept by their library `mvtools_cuda.dll`. CudaConvolution understands them when that
library is usable: it takes the instance the GPU plugins have already loaded, or the copy in CudaConvolution's own folder.
Without it nothing changes, except that a token frame CudaConvolution cannot read is an error, never picture data.

- **Input:** token frames are read back from the GPU automatically (frames the library evicted are made again by
  their producer), so no `CudaDownload()` is needed in front of CudaConvolution; the result is identical to the
  `CudaDownload()` chain. Every frame is checked (clips may mix token and ordinary frames), also before the automatic
  format conversion.
- **Output:** `token_out=true` returns token frames; put `CudaDownload()` after the last GPU stage that follows it.
- **Zero-copy:** with a library that offers it (token interface 1, `mvcuda_frame_run`), token input frames of 10, 12
  or 16 bit on CudaConvolution's GPU are read in place, and with `token_out=true` the result stays on the GPU: no copies
  between GPU and CPU for CudaConvolution at all. Otherwise (older library, 8 bit, another GPU, an input evicted meanwhile)
  the frames are copied as above; the output is identical either way.
- **One library per process:** two different copies of `mvtools_cuda.dll` loaded at the same time are two separate
  registries; CudaConvolution stops with an error then. Load CudaConvolution.dll and all GPU plugins from one folder.
- Only library versions with a known token interface are used (otherwise: "update CudaConvolution" in the error).
- A CPU filter that changes a token frame without overwriting its first row (e.g. `Subtitle`, `MergeChroma`) has no
  effect: CudaConvolution, like `CudaDownload()`, reads the picture from the GPU. Put `CudaDownload()` before such filters.
- With `token_out=true`, do not destroy a CudaConvolution while frames of it are still being made again by another thread
  (e.g. CudaConvolution inside a per-frame `ScriptClip`) with a library older than API 31: those do not wait for running
  regenerations.
- `CudaConvolution_Version()` reports whether token frames are available and why not. Set `CUDACONVOLUTION_NO_TOKENS=1` to
  switch the interop off, `CUDACONVOLUTION_LOG=1` for a message when it is loaded.

Non-integer ratios and upscaling are rejected with a message naming the valid alternatives, e.g.
`width 4096 -> 1920 is not an integer ratio (2.133); use width=2048 (factor 2), or crop the input width to 3840 (factor 2)`.
Crop, pad or resize before CudaConvolution with the usual AviSynth filters; the plugin never changes the aspect on its own.

No colour or range conversion happens inside: YUV stays YUV, RGB stays RGB, frame properties (`_Matrix`, `_ColorRange`,
...) pass through, and nothing is clipped unless `clip_min` / `clip_max` are given. Per-plane legal ranges can be
applied afterwards with `Limiter()`.

To reproduce the measured reference exactly (UHD -> 1080i, YUV422P10):

```
CudaConvolution(width=1920, height=1080, clip_min=1, clip_max=1020)
```

- Input formats: YUV422P8/10/12/16 (bit-exact path: YUV422P10), YUV444P8/10/12/16, RGBP8/10/12/16 (planar, no
  alpha). Anything else is converted to the nearest of these with the AviSynth+ built-ins (with `convert=false` it
  is an error naming the format):

  | input | conversion |
  |---|---|
  | YUV 4:2:0, 4:1:1, Y (greyscale), YUY2 | `ConvertToYUV422()` |
  | YUVA, planar RGBA | `RemoveAlphaPlane()` |
  | packed RGB24 / RGB32 / RGB48 / RGB64 | `ConvertToPlanarRGB()` (alpha dropped) |
  | 14 bit, 32-bit float | `ConvertBits(16)` |

  RGB stays RGB and YUV stays YUV: no colour matrix is guessed. For other targets (e.g. 4:2:0 -> 4:4:4, or a
  specific chroma resampler) convert explicitly before CudaConvolution.
- 8-bit input gives 8-bit output computed with the 10-bit arithmetic: identical to
  `ConvertBits(10).CudaConvolution(...).ConvertBits(8)` (rounded, no dither), same speed. `clip_min` / `clip_max` are then
  8-bit codes. If the source is 10-bit, feed it to CudaConvolution before any reduction to 8 bit (e.g. a `ConvertBits(8)`
  added by a workflow tool): reducing UHD with error-diffusion dither is slow, and the dither noise then goes
  through the filter.
- Input sizes: any. 4:2:2 needs an even output width whose chroma width divides the input chroma width; interlaced
  output needs an even height.
- Output: same format, size from `width` / `height`, frame rate halved, frame count halved (a trailing odd frame is
  dropped), audio unchanged. Output frame k = top field from input frame 2k, bottom field from input frame 2k+1 (TFF; BFF swapped).
- Frame properties: `_FieldBased` = 2 (TFF) / 1 (BFF); others passed through.
- Errors are never silent: if the GPU engine cannot run, `CudaConvolution()` stops with the reason and suggests `cpu=true`.
- `CudaConvolution_Version()` returns the version and the detected GPU / driver; `CudaConvolution_SelfTest()` runs the 30 built-in
  test vectors (factors 1..8, all formats) on all engines.

## Own preset files

`presets\balanced.json` in this package is the built-in preset; copy it and change what you need, then
`CudaConvolution(preset="my_preset.json")`. A preset file contains:

- `target`: `scan` (`interlaced` / `progressive`) and `field_order`;
- `profiles`: arithmetic sets (`arith` = `f32_fma_trunc` or `round_half_up`, `offset_code`, `intermediate_clip`,
  `output_clip`, `border`), e.g. `measured` and `designed`;
- `h`: per factor (`"1"`..`"8"`) a `profile` and the kernels `luma` and `chroma`;
- `v`: per factor a `profile` and the `kernel`.

A kernel has `first` (offset of the first tap), `taps` and `divisor` (integers / divisor, or real values with divisor 1,
stored as float32); H kernels may have `edges` (`left` / `right` lists of `{src_first, taps}`) replacing the kernel at
the outermost output columns. The H and V entries of the requested factors are combined; their profile is used if both
name the same one, otherwise `designed`. A file only needs the factors you use; errors name the offending key.

## Speed

Dev PC (24-thread Xeon, RTX 5050 on PCIe Gen3 x8), UHD 50p -> 1080i25, real time = 25 output fps:

| | output fps |
|---|---|
| CudaConvolution GPU, filter only | 32-69 (limited by the PCIe upload of the UHD frames) |
| CudaConvolution CPU, filter only (all threads) | 26.6 |
| ffms2 + `Prefetch(1)` + CudaConvolution GPU | 34.6 |
| ffms2 + `Prefetch(1)` + CudaConvolution CPU | 21.1 |
| full encode 60 s UHD ProRes -> ProRes HQ 1080i25 with audio | 68 s (0.88x real time; previous fmtconv solution: 369 s) |

Details and the comparison with the reference on real material: `validation.md` (repo `docs/`).

## Encoding with ffmpeg

Set the colour and field tags with `setparams` (in some ffmpeg builds the output `-color_*` options insert a colour
conversion instead of only tagging). Audio from the original file via a second input.

ProRes 422 HQ (MOV; the remux sets the `fiel` atom to `02 01`, top field first):

```
ffmpeg -i in.avs -i source.mov -map 0:v -map 1:a -vf "setparams=field_mode=tff:range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709" ^
  -c:v prores_ks -profile:v 3 -vendor apl0 -flags +ildct -c:a pcm_s24le tmp.mov
ffmpeg -i tmp.mov -c copy -field_order tt out_prores_hq.mov
```

XDCAM HD422 (MPEG-2 4:2:2 50 Mbit/s, MXF OP1a; check strict compliance against your target system):

```
ffmpeg -i in.avs -i source.mov -map 0:v -map 1:a -vf "setparams=field_mode=tff:range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709" ^
  -c:v mpeg2video -pix_fmt yuv422p -b:v 50M -minrate 50M -maxrate 50M -bufsize 17825792 -rc_init_occupancy 17825792 ^
  -g 12 -bf 2 -flags +ilme+ildct -intra_vlc 1 -non_linear_quant 1 -dc 10 -qmin 1 -qmax 12 -lmin "1*QP2LAMBDA" ^
  -c:a pcm_s24le out_xdcam.mxf
```

Without `-flags +ildct` prores_ks writes progressive pictures even though the container says interlaced.

## Accuracy

- Factor 2, YUV422P10 (UHD -> 1080i) with `clip_min=1, clip_max=1020`, against the reference output on two
  measurement probes (6 s, 311 million samples): luma identical (0 differences incl. borders); chroma identical except
  +-1 on 0.002 % of samples (float rounding cases near .5), edge columns included. Without the clip parameters only
  values beyond the reference's limits differ. Only 50p was measured; other rates use the same filter.
- ProRes sources: Apple and ffmpeg ProRes decoders differ slightly, so on ProRes input the match is limited by the
  decoder, not the filter.
- Factor 2 is the measured filter for any input size; factors 1 and 3..8 and the other formats use the same contract
  with designed filters (no reference measurement).
