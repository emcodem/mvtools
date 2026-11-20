# UConvert design

UConvert is one AviSynth+ entry point that takes virtually any input file and a target specification (frame size,
scan, frame rate), decides the filter chain itself, and delivers the converted clip (video + audio) to the encoder.
The work is done by existing plugins; `uconvert.avsi` holds the decision logic, so it stays editable without a
compiler. Encoding, container and output channel mapping stay outside (e.g. ffmpeg, see `encode_xdcamhd.py`).

The usage, parameters and global switches are documented in the header of `uconvert.avsi`. This file describes the
rules behind the chain; `tests/run_matrix.py` checks every generated case against an independent model of the
decision tables below.

## Principles

- **Any material**: any constant frame rate to any frame rate, any scan to any scan, any size to any size, including
  odd sizes, rates, SARs and pixel formats. Pulldown/telecine and VFR are not handled: the file is converted as
  flagged.
- **Colour is never touched**: no matrix, primaries, transfer, range or chroma-subsampling conversion. The output has
  the input's pixel format (colour family, subsampling, bit depth) and colour properties, e.g. YUV422P10 BT.2020 HLG
  in -> YUV422P10 BT.2020 HLG out.
- **Lossless detours only**: formats a stage plugin cannot take go through a lossless detour and are restored at the
  output (see [Format detours](#format-detours)); never a detour that changes values (no RGB <-> YUV).
- **Metadata is trusted**: scan type and SAR come from the source's metadata or the caller's overrides; there is no
  content-based scan detection and no active-area handling (720 vs 704: the full stored width is the picture).
- **Exact output**: the output always has exactly the requested size, scan and frame rate. The decision tables only
  choose how the output pictures are made.

## Stages

```
1 Source   -> clip with video + audio, frame properties, _uc_source
2 Probe    -> scan, SAR, rate, pixel format (caller overrides win)
3 Plan     -> temporal + spatial decision tables
4 Execute  -> deinterlace, scale, frame-rate conversion, interlace
5 Output   -> input pixel format, colour properties, scan / SAR tags, audio aligned to the video
```

The output carries the frame properties `_uc_source` (which source filters were used, and why a fallback was taken)
and `_uc_plan` (the chosen chain), e.g.
`in=1920x1080 tff 30000/1001 YUV422P10 sar 1:1 | deint=qtgmc-gpu tff | frc=splat 50/1 | interlace=tff | out=1920x1080 tff 25/1 YUV422P10`.

### 1. Source

| input | video | audio | fallback |
|---|---|---|---|
| `.mxf` | `MxfSource` | `MxfSource(use_audio="always")`: all tracks merged in track order | audio fails: video from MxfSource, audio from ffms2; open fails: ffms2 for both |
| `.mov`, `.mp4`, `.m4v` | `LSMASHVideoSource` (container sample tables, no index) | `LSMASHAudioSource`, one call per track, merged | open fails: ffms2 for both; no audio track L-SMASH can open (e.g. big-endian PCM in some cameras' MP4): audio from ffms2 |
| everything else | `FFVideoSource` (index file); `.ts` `.m2ts` `.mts` `.mpg` `.mpeg` `.vob` `.m2v`: no `Prefetch` on the source | `FFAudioSource`, one call per stream, merged in stream order | - |

- `source=` forces one of `"mxf"`, `"lsmash"`, `"ffms2"`. LWLibav is not used.
- Index files (ffms2 `.ffindex`) go to `cachedir=` of the call, else the global `uc_index_dir`,
  else the OS temp folder (`MTempDir()` of mvtools2; AviSynth scripts cannot read environment variables). They are
  named after the full source path, never written next to the source, and reused while the file is unchanged.
  ffms2 indexes video and all audio tracks in one pass (`FFIndex(indexmask=-1)`); every ffms2 call reads that file.
- Why ffms2 and not LWLibav (2026-11-15, per-frame md5 against a linear decode, read linear / under `Prefetch(3)` /
  in shuffled 13-frame segments): MxfSource and L-SMASH were exact everywhere; ffms2 was exact on MXF, MP4/MOV and
  MKV; LWLibav served wrong frames on H.264 whose container timestamps disagree with the picture order (MXF written
  by ffmpeg, and MP4/TS/MOV copies of it: 875 of 1500 wrong even read linearly) and after any seek in an all-intra
  interlaced MPEG-2 stream in MXF and MKV (exact in TS/MOV/M2V). ffms2 serves wrong frames on MPEG-2 in TS/PS
  after a seek (read in order it is exact; `seekmode=0` is exact but decodes from the start on every backward
  request: 43-166 s instead of 2.5-4 s for 1500 frames under `Prefetch(3)`), so those files get no `Prefetch` on the
  source and are read in order.
- Audio search: every stream index from 0 is tried until 4 in a row are not audio (with the index file a probe only
  reads that file).
- Audio is handled as plain PCM streams: never mixed or gain-changed. Streams with different sample types are merged
  losslessly (wider integer, or float); streams with different sample rates are all resampled to 48 kHz.
- Open errors are caught in the script and trigger the fallbacks. Decode errors while frames are fetched cannot be
  caught: they reach the encoder as end of file (ffmpeg exits with code 0 and a short output). The caller must
  compare the output duration with the script's, as `encode_xdcamhd.py` does. A repair such as `FreezeFrame` over
  undecodable frames must sit before the source's `Prefetch`, which reads ahead.

### 2. Probe

- Scan from `_FieldBased` (2 = TFF, 1 = BFF, 0 = progressive) at up to five frames spread over the clip; the first
  interlaced one wins. No property = progressive. Override: `scan_in`.
- SAR from `_SARNum` / `_SARDen`; missing or invalid = square pixels. Override: `sar_num` / `sar_den` (metadata is
  often wrong, e.g. SD files flagged 1:1). MxfSource derives it from the MXF picture descriptor's AspectRatio and
  display rectangle (D-10 PAL 16:9: 64:45, 4:3: 16:15; at 720 lines and more only 16:9 is taken).
- Display window: MxfSource's `MxfDisplayLeft` / `Top` / `Width` / `Height` (set only when the descriptor's display
  rectangle is smaller than the frame, e.g. the 32 VBI lines of D-10 720x608) are cropped first, so the SAR applies to
  the picture alone (plan: `crop=720x576+0+32`). Not when the output has the full frame's size (D-10 -> D-10).
- Output SAR: the input SAR if the frame size is unchanged, else 1:1. Override: `out_sar_num` / `out_sar_den`.

### 3. Plan

#### Temporal

Definitions (all rates exact as num/den):

- input motion rate `r` = frame rate (progressive input) or 2 x frame rate (interlaced input). Interlaced input is
  always deinterlaced to full field-rate progressive first, so all rate decisions see progressive input at `r`.
- output motion rate `m` = frame rate (progressive output) or 2 x frame rate (interlaced output).
- picture rate `p`: the rate at which distinct pictures are made. `p = m / 2` ("pictures x2": each picture shown
  twice) when `r < m / 2` and `m / r` is not an integer, else `p = m`. Only x1 or x2, never more: 12p -> 50p
  makes 25 Hz pictures, each shown twice.

Frame-rate conversion from `r` to `p`:

| condition | conversion | example |
|---|---|---|
| `r = p` | none | 50i -> 50p, 50p -> 25i |
| `r / p` integer | drop: every n-th frame, no motion compensation | 50p -> 25p, 50i -> 25p |
| `p / r` integer | repeat: each frame n times, no motion compensation | 25p -> 50p, 25p -> 75p (x3) |
| otherwise | motion-compensated (MC) | 59.94i -> 50i, 24p -> 25p, 50p -> 23.976p |

Low picture rates: 24p -> 50p gives MC 24 -> 25 then pictures x2; 24p -> 25i the same, then interlace (each
output frame holds one picture in both fields). The output never has fewer distinct pictures than the source: a
source faster than `m / 2` is converted to the full motion rate, so 29.97p -> 25i gives MC 29.97 -> 50 and fields
from 50 different pictures, and 29.97p -> 50p gives MC 29.97 -> 50. A source at or below `m / 2` is not given
motion it does not have: 24p keeps its film look in 25i / 50p (MC to 25, each picture twice).

Timing and cuts:
- Output picture k is taken at input time `k x r / p` exactly, without latency.
- Clean cut: at scene changes pictures are never blended; the picture spanning a cut is a plain copy of the nearest
  input frame.

Passthrough: same frame size, same frame rate and same scan -> the clip is passed through untouched (no
deinterlace / re-interlace), only the output properties are set.

#### Spatial

One resize in the work pixel format with an AviSynth+ core resizer (default Spline36Resize); no matrix, range or
subsampling change. The kernel is chosen per axis: `uc_scale_down_h` / `uc_scale_down_v` where the source window is
larger than the scaled picture (`uc_scale_down_v_interlaced` for interlaced output, where the interlace step's
vertical low-pass follows), `uc_scale_up` otherwise; each is a resizer name with optional arguments
(`"SinPowerResize, p=3.0"`). Two different kernels run one after the other (horizontal first). The
display aspect comes from input SAR and output SAR; `aspect_mode`:

- `pillar` (default): fit the whole picture, pad with black (pillar- or letterbox);
- `crop`: fill the frame, cut the excess (sub-pixel source window);
- `stretch`: ignore the aspect.

Output sizes must be multiples of the chroma subsampling (x2 vertically for interlaced output).

#### Order of the stages

```
deinterlace (interlaced input) -> scale, if it makes the picture smaller -> MC or drop -> scale, otherwise
-> repeat -> pictures x2 -> interlace (interlaced output)
```

Expensive temporal stages run at the smaller of the two sizes (UHD -> HD is scaled before the frame-rate conversion,
SD -> HD after it). Repetition runs after the scaling, so identical frames are scaled once.

### 4. Execute

| stage | GPU chain (`gpu=true`, default) | CPU chain (`gpu=false`) |
|---|---|---|
| deinterlace | QTGMC (`QTGMC_gpu.avsi`) with its GPU switches; motion vectors optionally from the NVIDIA optical flow accelerator (`uc_qtgmc_nvof`) or NVENC's motion estimation (`uc_qtgmc_nvenc`, nvencmv) | QTGMC; preset `uc_qtgmc_preset` (default "Fast") in both chains |
| frame-rate conversion (MC) | `CudaSplatFps`; candidates optionally from the NVIDIA optical flow accelerator (`uc_frc_nvof`: `CudaNVOFAnalyse` via `MSplatVectors`) or NVENC's motion estimation (`uc_frc_nvenc`: `CudaNVENCAnalyse`) | MVTools `MSuper` / `MAnalyse` (truemotion) / `MBlockFps(mode=3, blend=false)`; up to 2.5 Mpixel pel 4 / 16x16, above pel 2 / 32x32 |
| scale | `Cuda*Resize` (cudafilters, bit-identical with the CPU resizer); borders on the CPU | the AviSynth+ core resizer (`uc_scale_*`) |
| interlace | `CudaConvolution`, built-in preset `balanced` | `mt_convolution` with the same taps |

One-step scale + interlace (GPU chain, `uc_cc_onestep`): progressive input at the output's field rate (no frame-rate
conversion), interlaced output, no crop or borders, an integer ratio of 1..8 per axis (e.g. UHD 50p -> 1080i25), 4:2:2 /
4:4:4 / planar RGB without alpha, up to 16 bit. The scale step and the interlace step are replaced by one
`CudaConvolution(width, height, preset=balanced_halfup.json)`; `_uc_plan` shows `scale+interlace=cc`. The preset is
CudaConvolution's `balanced` with `round_half_up` in the measured profile: the stock `f32_fma_trunc` rounding adds +1 code
to every sample below mid-grey (flat areas, 10 bit). Against the original (UHD 50p scaled to HD, Carmen clip, PSNR Y/U/V):
halfup 42.7 / 49.0 / 46.5 dB, stock 42.6 / 48.6 / 46.5 dB, alch 37.6 / 40.8 / 43.8 dB.

- GPU chain frames: QTGMC (`QTGMC_gpu_out_tokens`, not with alpha) and the Cuda resizers keep the pictures on the GPU
  as token frames up to the next stage that reads normal frames (borders, bit-depth change, CudaSplatFps, 4:2:0 /
  greyscale interlacing, the output); there a `CudaDownload()` is inserted. CudaConvolution reads token frames itself.
- The GPU chain stops with an error when no usable NVIDIA GPU is found. Single stages without a GPU path for a format
  (e.g. greyscale or alpha in QTGMC) run on the CPU with the same result (`cuda_cpu_fallback`).
- Deinterlacing and MC frame-rate conversion run in 8 bit by default (`frc_8bit=true`) and are converted back to the
  input bit depth afterwards; `frc_8bit=false` keeps the source bit depth (32-bit float: 16 bit).
- Interlace step: CudaConvolution's built-in preset `balanced`. The preset holds one filter set per scale factor
  (input size / output size): factor 1 (same size) is a 9-tap vertical low-pass filter and no horizontal filter;
  factors 2..8 downscale and filter in one step (e.g. factor 2: 31 taps horizontal, 17 vertical). UConvert scales
  first (scale step) and runs the interlace step at the output size, so it always uses factor 1, also for
  UHD -> HD: taps `43 -195 -765 4211 9796 4211 -765 -195 43` / 16384 (`uc_interlace_kernel` /
  `uc_interlace_divisor`; `""` takes the fields without filtering) on the progressive picture, then each output
  frame's fields from two consecutive progressive frames. 4:2:0 is filtered at 4:2:2 and subsampled per field, so
  each field's chroma comes from that field's lines. Other taps: a CudaConvolution preset file with the same taps in
  `uc_interlace_preset` and the taps in `uc_interlace_preset_kernel`.

### 5. Output

Back to the input pixel format; `_Matrix`, `_Primaries`, `_Transfer`, `_ColorRange` and `_ChromaLocation` copied
from the input unchanged (so the encoder writes the original tags); `_FieldBased` and `_SARNum` / `_SARDen` set for
the new scan and geometry. Audio is aligned to the video length (trimmed or padded with silence); the frame-rate
conversion keeps the duration, so audio is otherwise untouched.

## Format detours

All verified bit-exact as round trips (`tests/run_tests.py exact_`):

- YUY2 -> YV16, packed RGB24/32/48/64 -> planar RGB(A), back at the output.
- Planar RGB(A) is relabelled as YUV(A) 4:4:4 (G in Y, B in U, R in V; no value change) so QTGMC and MVTools take it.
- 4:1:1 -> 4:4:4 by sample duplication, back by taking every 4th sample (MVTools crashes on 4:1:1).
- Alpha: deinterlaced as a separate luma clip; the frame-rate conversion compensates it with the picture's motion
  vectors; scaling and the interlace filter process it directly.
- Float (32 bit) cannot go through QTGMC / MVTools: with `frc_8bit=false` deinterlacing and frame-rate conversion
  run in 16 bit. ffmpeg cannot read float output from AviSynth.

## Threading

`UConvert` puts the source in `Prefetch(3)` (not MPEG-2 TS/PS files, see 1.) and passes `threads` (default -1: automatic) to `UConvertClip`, which
places the thread pool itself. Automatic means `uc_gpu_threads` (10) on the GPU chain and 3 on the CPU chain. The GPU
needs many frames in flight: one picture's QTGMC motion search (`CudaMAnalyse`, 95 % of the GPU time of deinterlacing)
occupies only about 24 of an RTX 5090's 170 SMs at 1080p, so with 3 threads most of the chip idles although
`nvidia-smi` shows it 99 % busy. Measured on the RTX 5090: deinterlacing 1080i25 -> 1080p50 147 -> 353 fps (3 -> 10
threads), whole GPU chain 75 -> 131 fps; the deinterlaced output is identical for any thread count.

- CPU chain: `Prefetch(threads)` on the progressive chain before the interlace step, `Prefetch(3)` at the end. Each
  interlaced frame needs two adjacent progressive frames; with the pool after the interlace step, threads compute
  the same deinterlaced / converted frames twice (measured: half speed).
- GPU MC frame-rate conversion: `CudaSplatFps` takes one frame at a time and wants its frames in order, so the pool
  sits in front of it, with `Prefetch(1)` after it and at the end.

Callers must not add a large `Prefetch` after UConvert. Index files belong on a local disk (`cachedir` /
`uc_index_dir`), not on network shares. Two jobs that open the same never-indexed file at the same moment both
write its index file.

## Not handled

- VFR, pulldown / telecine, content-based scan detection, active-area (704) handling.
- MXF with MxfSource video but ffms2 audio (audio fallback): MxfSource trims the pre-roll (Origin), ffms2 may not;
  whether the audio then lines up is untested.
