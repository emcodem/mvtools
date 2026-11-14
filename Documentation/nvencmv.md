# nvencmv: motion vectors from NVIDIA's video encoder

`nvencmv.dll` (Linux: `libnvencmv.so`) makes MVTools vector clips with the motion estimation of NVIDIA's video encoder
(NVENC, "motion-estimation-only" mode). The vectors are an encoder's block matching, not MAnalyse's search, plus an
optional candidate selection of our own. It is meant for experiments: every engine option is a parameter.

- **Output**: a one-level vector clip in MAnalyse's format, the same layout as `CudaNVOFAnalyse`. MRecalculate,
  MFlowFps, MDegrain*, MCompensate etc. of any MVTools2 build accept it.
- **Input**: the plain clip, not a super clip (no MSuper pyramid is searched). Ordinary frames are uploaded once per
  picture; token frames of the `Cuda*` filters are used where they are, without a copy.
- **Needs**: `mvtools_cuda.dll` (Linux: `libmvtools_cuda.so`) from the same build next to the plugin, and an NVIDIA GPU
  with NVENC ME-only mode: Pascal (GTX 10xx) or newer. Driver with NVENC API 13.0 or newer (R570+). Tested only on an
  RTX 5090 so far.
- **Not bit-exact with anything.** With `session="reset"` or `"fresh"` (default `"reset"`) the output does not depend
  on thread order (stable md5 under Prefetch).

## Known problems

- **`codec="hevc"` gives zero vectors.** HEVC ME-only output is unusable here (RTX 5090, driver 616.92, NVENC API
  13.0): zero vectors on the synthetic shift test, also in the earlier prototype (most CUs come back as `cuType 0` with
  zero vectors, the rest off). Use `codec="h264"` (the default). The cause (driver, our reading of
  `NV_ENC_HEVC_MV_DATA`, or the configuration) is not known yet.
- **`lambda` works against `channels="yuv"` / `"rgb"`**: the smoothness term pulls towards the median of the first
  (luma / red) run, so motion that is only in the chroma is pulled back to the luma's vectors.
- **`session="kept"` is not deterministic** under Prefetch (the encoder uses the previous call's field as predictors,
  and the calls come in thread order). Use `"reset"` (default) or `"fresh"`.
- **NVENC ignores chroma** in its own search (H.264 SDK and D3D12 alike): colour reaches it only through `channels`.
- Tested on an RTX 5090 only; Pascal / Turing, Linux and VapourSynth (through the bridge) are untested.

Measured on a UHD 4:2:2 clip (frame-rate conversion chain with MRecalculate):
[compare/nvencme_colour](../compare/nvencme_colour/README.md). Summary: NVENC + MRecalculate land at about the quality
of MAnalyse's exact search (better in the skip-frame test with the defaults), far behind NVOFA. The point of NVENC is
cards without NVOFA (Pascal, Turing GTX 16xx) and moving work from the CUDA cores to the encoder chip, which runs next
to them.

## CudaNVENCAnalyse

`CudaNVENCAnalyse(clip c, bool "isb", int "delta", int "blksize", int "blksizeV", int "overlap", int "overlapV",
int "pel", int "hpad", int "vpad", bool "chroma", string "codec", int "preset", int "qp", bool "intra", bool "p16",
string "session", bool "half", string "channels", bool "swap", bool "neighbours", int "lambda", int "fbpenalty",
bool "temporal", clip "hintclip", string "stats")`

How a frame is made:
1. Grey 8-bit pictures of the frame and of its reference (n ± delta): the luma, or three channels (`channels`),
   optionally downscaled 2x2 (`half`).
2. One NVENC run per channel, plus (`swap`) one run of the first channel with the two pictures swapped. Each gives
   a vector per 8x8 cell of the ME picture, quarter pel.
3. Per cell, the candidate with the lowest cost wins. The candidates are each run's vector (with `neighbours`, also its
   8 neighbours' vectors) and the swapped run's negated vector. The cost is the luma SAD plus the scaled chroma SAD at
   integer pel, + `lambda` term + `fbpenalty` term.
4. Per MVTools block, the median (x and y separately) of the chosen cells under it, rounded to 1/pel, clamped to the
   area MAnalyse could reach with that padding, with its SAD (luma + chroma as MAnalyse with `scaleCSAD=0`).

### Block layout (as MAnalyse / CudaNVOFAnalyse)

| parameter | default | |
|---|---|---|
| `isb` | false | backward vectors (reference n + delta) |
| `delta` | 1 | frame distance |
| `blksize`, `blksizeV` | 16, = blksize | MAnalyse's block sizes up to 64 |
| `overlap`, `overlapV` | 0, = overlap | at most half the block size |
| `pel` | 2 | 1 or 2 |
| `hpad`, `vpad` | 8 | MSuper's defaults; **must be those of the MSuper used downstream**: the MVTools filters do not check it, and a mismatch makes MCompensate / MDegrain read outside the frame and crash |
| `chroma` | true | chroma in the output SADs and in the selection (with `channels="yuv"` / `"rgb"` the selection always uses chroma) |

`blksize=8, overlap=0` gives the engine's cells one to one (one cell per block).

### Engine

| parameter | default | |
|---|---|---|
| `codec` | `"h264"` | `"h264"` or `"hevc"`. **`"hevc"` gives zero vectors** (see Known problems); use H.264 |
| `preset` | 4 | NVENC P1..P7 |
| `qp` | 10 | constant QP 0..51, or -1 for the preset's rate control. The encoder's cost is SAD + λ(QP) · vector bits: high QP gives smoother but less accurate fields, low QP the most accurate. `qp=0` behaved like -1 in our tests |
| `intra` | true | false: no intra search (H.264) |
| `p16` | false | true: 16x16 partitions only (H.264) |
| `session` | `"reset"` | the encoder keeps the previous field as predictors. `"reset"`: one run on a flat picture before each real run (same output as `"fresh"`, about 4x faster); `"fresh"`: a new encoder per run; `"kept"`: neither (fastest, but the output depends on the thread order) |
| `half` | false | ME on the picture downscaled 2x2: twice the search range, cells of 16 pixels |

Sessions: at most 4 open at once by default (`MVTOOLS_NVENC_SESSIONS`); GeForce drivers limit concurrent NVENC
sessions, and other programs (ffmpeg, OBS) count too.

### Candidates

| parameter | default | |
|---|---|---|
| `channels` | `"y"` | `"y"`: the luma. `"yuv"`: Y, U, V (chroma at luma size) as three grey runs. `"rgb"`: R, G, B (BT.709 limited range) as three runs. NVENC's motion estimation ignores chroma, so this is the only way colour reaches it; measured: motion that is only in the chroma is found |
| `swap` | true | one more run of the first channel with the pictures swapped; its negated field adds candidates (and is the backward field of `fbpenalty`) |
| `neighbours` | false | the 8 neighbouring cells of every run are candidates too |
| `lambda` | 0 | smoothness: cost += lambda · d² / 256, d = distance in half pel to the median of the 3x3 cells of the first run. It pulls towards the luma field, so it works against `channels` on colour-only motion |
| `fbpenalty` | 0 | with `swap`: cost += fbpenalty when the vector fails the forward-backward check (more than 1 pixel off) |

### Hints (H.264 only)

| parameter | default | |
|---|---|---|
| `temporal` | false | the field of the previous pair of the same direction (n-1, computed again without hints) gives 2 hint candidates per 16x16 macroblock: the cell under its centre and the 3x3 median |
| `hintclip` | | any MVTools vector clip of the same picture size, direction and delta (MAnalyse, CudaNVOFAnalyse, another CudaNVENCAnalyse ...): its finest level gives 2 hint candidates per macroblock the same way |

### Diagnostics

| parameter | default | |
|---|---|---|
| `stats` | | file name: at the end, one line per filter with how often each candidate won (index: run × (neighbours ? 9 : 1) + cell; runs = channel runs, then the swapped run; cell 0 = own, 1..8 = neighbours) |

## CudaNVENCAvailable(string "codec")

True when NVENC ME-only can be used for that codec (`"h264"` default). Without it, `CudaNVENCAnalyse` stops with the
reason, unless `global cuda_cpu_fallback = true` (or the environment variable `CUDA_CPU_FALLBACK`): then it returns
`MAnalyse(MSuper(c, pel, hpad, vpad, chroma), isb, delta, blksize, blksizeV, overlap, overlapV, truemotion=true, chroma)`
(mvtools2 must be loaded). A single frame NVENC fails on gets that MAnalyse's level-0 vectors.

## Examples

```avisynth
# frame-rate conversion, NVENC for the coarse vectors, MRecalculate refines (as uc_FRC's NVOFA branch)
pre = src.RemoveGrain(22)
sup = MSuper(src, pel=2, levels=1)
bv  = CudaNVENCAnalyse(pre, isb=true,  blksize=32, overlap=16, channels="yuv")
fv  = CudaNVENCAnalyse(pre, isb=false, blksize=32, overlap=16, channels="yuv")
bv  = MRecalculate(sup, bv, blksize=16, overlap=8, thSAD=100)
fv  = MRecalculate(sup, fv, blksize=16, overlap=8, thSAD=100)
MFlowFps(src, sup, bv, fv, num=60000, den=1001)

# the engine field as it comes: one 8x8 cell per block, no selection, the encoder's own rate control
raw = CudaNVENCAnalyse(src, blksize=8, overlap=0, qp=-1, swap=false)

# NVOFA vectors as hints for NVENC
nv  = CudaNVOFAnalyse(src, isb=true, blksize=16, overlap=8)
bv  = CudaNVENCAnalyse(src, isb=true, blksize=16, overlap=8, hintclip=nv)
```

VapourSynth: through `mvtools_vsbridge` (namespace `nvencmv`), as mvtools2 and splatfps (not tested yet).

Defaults are a first choice from the frame-rate conversion tests (QP 10 + swapped run: best accuracy there); they are
to be tuned for QTGMC (block sizes 8-16, pel 1-2, several deltas).
