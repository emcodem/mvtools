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
`in=1920x1080 tff 30000/1001 YUV422P10 sar 1:1 | deint=qtgmc-gpu tff | frc=splat 25/1 | pictures x2 | interlace=tff | out=1920x1080 tff 25/1 YUV422P10`.

### 1. Source

| input | video | audio | fallback |
|---|---|---|---|
| `.mxf` | `MxfSource` | `MxfSource(use_audio="always")`: all tracks merged in track order | audio fails: video from MxfSource, audio from LWLibav; open fails: LWLibav for both |
| `.mov`, `.mp4`, `.m4v` | `LSMASHVideoSource` (container sample tables, no index) | `LSMASHAudioSource`, one call per track, merged | open fails: LWLibav |
| everything else | `LWLibavVideoSource` (index file in `cachedir`, never next to the source; no `cachedir`: not written) | `LWLibavAudioSource`, one call per stream, merged in stream order | - |

- `source=` forces one of `"mxf"`, `"lsmash"`, `"lwlibav"`.
- LWLibav re-indexes the whole file for every stream index it probes that is not audio, so the audio search assumes
  contiguous audio streams: from stream 1 up to the first gap, stream 0 only if streams 1 and 2 are not audio.
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
  often wrong, e.g. SD files flagged 1:1).
- Output SAR: the input SAR if the frame size is unchanged, else 1:1. Override: `out_sar_num` / `out_sar_den`.

### 3. Plan

#### Temporal

Definitions (all rates exact as num/den):

- input motion rate `r` = frame rate (progressive input) or 2 x frame rate (interlaced input). Interlaced input is
  always deinterlaced to full field-rate progressive first, so all rate decisions see progressive input at `r`.
- output motion rate `m` = frame rate (progressive output) or 2 x frame rate (interlaced output).
- picture rate `p`: the rate at which distinct pictures are made. `p = m / 2` ("pictures x2": each picture shown
  twice) when `r < 0.75 x m` and `m / r` is not an integer, else `p = m`. Only x1 or x2, never more: 12p -> 50p
  makes 25 Hz pictures, each shown twice.

Frame-rate conversion from `r` to `p`:

| condition | conversion | example |
|---|---|---|
| `r = p` | none | 50i -> 50p, 50p -> 25i |
| `r / p` integer | drop: every n-th frame, no motion compensation | 50p -> 25p, 50i -> 25p |
| `p / r` integer | repeat: each frame n times, no motion compensation | 25p -> 50p, 25p -> 75p (x3) |
| otherwise | motion-compensated (MC) | 59.94i -> 50i, 24p -> 25p, 50p -> 23.976p |

Low picture rates: 24p -> 50p gives MC 24 -> 25 then pictures x2; 29.97p -> 25i gives MC 29.97 -> 25 then
pictures x2, then interlace. The 0.75 boundary sits between the cases that make full-rate pictures (`r` close to
`m`) and those that make half-rate ones (`r` near or below `m / 2`).

Timing and cuts:
- Output picture k is taken at input time `k x r / p` exactly, without latency.
- Clean cut: at scene changes pictures are never blended; the picture spanning a cut is a plain copy of the nearest
  input frame.

Passthrough: same frame size, same frame rate and same scan -> the clip is passed through untouched (no
deinterlace / re-interlace), only the output properties are set.

#### Spatial

One resize (`z_ConvertFormat`, Spline36) in the work pixel format; no matrix, range or subsampling change. The
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
| deinterlace | QTGMC (`QTGMC_gpu.avsi`) with its GPU switches; motion vectors optionally from the NVIDIA optical flow accelerator (`uc_qtgmc_nvof`) | QTGMC; preset `uc_qtgmc_preset` (default "Fast") in both chains |
| frame-rate conversion (MC) | `CudaSplatFps` | MVTools `MSuper` / `MAnalyse` (truemotion) / `MBlockFps(mode=3, blend=false)`; up to 2.5 Mpixel pel 4 / 16x16, above pel 2 / 32x32 |
| scale | `z_ConvertFormat` (avsresize), Spline36 | same |
| interlace | `CudaConvolution` with `uc_interlace.json` | `mt_convolution` |

- The GPU chain stops with an error when no usable NVIDIA GPU is found. Single stages without a GPU path for a format
  (e.g. greyscale or alpha in QTGMC) run on the CPU with the same result (`cuda_cpu_fallback`).
- Deinterlacing and MC frame-rate conversion run in 8 bit by default (`frc_8bit=true`) and are converted back to the
  input bit depth afterwards; `frc_8bit=false` keeps the source bit depth (32-bit float: 16 bit).
- Interlace step: a 9-tap vertical low-pass filter on the progressive picture (taps
  `31 -66 -235 1075 2486 1075 -235 -66 31` / 4096, `uc_interlace_kernel`; `""` takes the fields without
  filtering), then each output frame's fields from two consecutive progressive frames. 4:2:0 is filtered at 4:2:2
  and subsampled per field, so each field's chroma comes from that field's lines. The GPU path reads the same taps
  from `uc_interlace.json`; a changed kernel needs a matching preset file.

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

`UConvert` puts the source in `Prefetch(3)` and passes `threads` (default 3) to `UConvertClip`, which places the
thread pool itself:

- CPU chain: `Prefetch(threads)` on the progressive chain before the interlace step, `Prefetch(3)` at the end. Each
  interlaced frame needs two adjacent progressive frames; with the pool after the interlace step, threads compute
  the same deinterlaced / converted frames twice (measured: half speed).
- GPU MC frame-rate conversion: `CudaSplatFps` takes one frame at a time and wants its frames in order, so the pool
  sits in front of it, with `Prefetch(1)` after it and at the end.

Callers must not add a large `Prefetch` after UConvert. LWLibav index files belong on a local disk (`cachedir`),
not on network shares.

## Not handled

- VFR, pulldown / telecine, content-based scan detection, active-area (704) handling.
- MXF with MxfSource video but LWLibav audio (audio fallback): MxfSource trims the pre-roll (Origin), LWLibav does
  not; whether the audio then lines up is untested.
