# uc_FRC with NVIDIA's optical flow accelerator in place of MAnalyse's search

> **Now a filter.** The `MVCUDA_NVOFA` hook described here was removed from MAnalyse (it is exact-only again); the same
> computation is `CudaNVOFAnalyse` of cudafilters.dll (library API 17, [NVOF_FILTER_PLAN.md](../../NVOF_FILTER_PLAN.md)),
> switched on in uconvert with `uc_frc_nvof`. `CudaNVOFAnalyse(pre, blksize=32, overlap=16, pel=2, preset=2)` ->
> MRecalculate -> MFlowFps gives the medium md5 FA834251D140 below, and its vectors and SADs equal level 0 of the hook's
> MAnalyse output in all 300 vector frames. The scripts here that set `MVCUDA_NVOFA` need the experiment's build
> (`profile\build_nvofa`, source: commit 574cd22).

Experiment of [opticalflow.md](../../opticalflow.md): MAnalyse's GPU block search (k_search) replaced by NVOFA vectors
(Optical Flow SDK, `nvofapi64.dll` of the driver). MRecalculate refines them and MFlowFps interpolates, both unchanged.
Switched by the environment of the process: `MVCUDA_NVOFA=fast|medium|slow`, off when unset. **Not bit-exact** with the
CPU search (another algorithm); with the variable unset the output is unchanged (md5 BFE44775F811 = CPU fallback).

Clip: UHD 10-bit 4:2:2, pole and wires against sky (`seg1_wires_pole.mp4`). Chain as in our frame-rate converter:
MAnalyse 32/16 → MRecalculate 16/8, `thSAD=100, search=3, searchparam=2, truemotion=true, chroma=true`, pel 2 →
MFlowFps 24000/1001 `mask=2, ml=200`. Script: [scripts/avs/frc_flow.avs](scripts/avs/frc_flow.avs).
RTX 5090, driver 616.92, CUDA 13.4, test set `profile\build_nvofa` (= build_m1 + the new mvtools_cuda.dll).

## Speed

`profile\run_frc.ps1 -Dir profile\build_nvofa [-SetEnv MVCUDA_NVOFA=<level>]`, 112 frames, Prefetch 3, best of 3:

| search | wall | CPU-s | md5 |
|---|---:|---:|---|
| exact GPU search (k_search) | 6.80 s | 22.5 | BFE44775F811 (= CPU fallback) |
| NVOFA fast | **4.88 s** | 22.0 | 6C85DE040D85 |
| NVOFA medium | **4.93 s** | 22.7 | FA834251D140 |
| NVOFA slow | 5.25 s | 22.3 | B8B830F31B09 |

Each NVOFA level gives the same md5 in every run (temporal hints off: deterministic).

GPU kernel time (nsys `cuda_gpu_kern_sum`, one run):

| | exact search | NVOFA medium |
|---|---:|---:|
| k_search | 3.610 s (1603 launches, 92.9%) | none |
| k_of_blocks (median of the flow per block + SAD) | | 0.039 s (228) |
| k_recalc | 0.188 s | 0.170 s (53.9%) |
| all kernels | 3.88 s | **0.32 s** |

NVOFA runs on its own engine, not in the kernel list. From the probe (scratchpad `nvof_probe`, UHD, grid 4): one pair takes
3.0 ms at medium (2.2 ms fast, 5.4 ms slow) in one thread; with 2 or more threads the single engine delivers 2.64 ms per
pair (medium) / 1.90 ms (fast): about 0.6 s of engine time for the 229 analyses of this run, in parallel to the SMs.
The rest of the 4.9 s is decode (1.57 s alone), the CPU side and k_recalc / k_flow.

## Result

Output frames 10–17 of the script = frames 18–25 of the plain MFlowFps output (same frames as
[compare/dtl_smooth2](../dtl_smooth2/)). Frames 3 and 7 fall on source frames: identical in every variant. The interpolated
ones, PSNR of the whole frame against the exact search (a pointer only, judge by eye):

| frame | 1 | 2 | 4 | 5 | 6 | 8 |
|---|---:|---:|---:|---:|---:|---:|
| medium | 47.2 | 48.5 | 48.9 | 47.8 | 48.3 | 48.1 |
| fast | 46.7 | 48.2 | 46.3 | 45.8 | 46.2 | 45.3 |

Crops (960×540 at x 2880, y 240, the pole and wires; overview: [img/frame_overview.jpg](img/frame_overview.jpg)):
`img/crop_<exact|medium|fast>_<1..8>.jpg`, differences to the exact search ×8: `img/diff_<medium|fast>_<n>.jpg`.
The differences sit at the pole / wire crossings (occlusions), e.g. [img/diff_medium_1.jpg](img/diff_medium_1.jpg).

Motion: [img/side_by_side.mp4](img/side_by_side.mp4), the crop region over the whole run, exact | medium | fast.

Everything above: [scripts/render.ps1](scripts/render.ps1).

### Slow motion, judged by eye (1.7–2.6 s, trees rushing through)

[img/slowmo_1.7-2.6.mp4](img/slowmo_1.7-2.6.mp4): exact | medium | fast, every frame held 4×, 120 px grid with cells
A1..H5 (column letter, row number), header with the clip time and the output frame number
([scripts/slowmo.py](scripts/slowmo.py)). Judgement (user, 2026-10-31):

| frame | exact search | NVOFA medium | NVOFA fast |
|---|---|---|---|
| 41 | B1, B2, C2: pink/red missing, blue sky instead; D1, D2: black smearing; D3, E3: pink/red missing, sky instead; H1: dark block, a quarter of the cell | no visible issues | B2: pink/red missing, blocky |
| 42 | D2 blocky, the cells around it smear in odd directions | no visible issues | B4: oddly formed smearing, about a quarter of a cell |

### Why: the vectors

[scripts/render_vec.ps1](scripts/render_vec.ps1) dumps the vectors MFlowFps uses for these frames (source 29.97 fps →
23.976: output frame 41 = source 61.25, bv 61 + fv 62; frame 42 = source 62.5, bv 62 + fv 63), after MRecalculate
(`avs/frc_vec_recalc.avs`, 16/8 blocks) and straight from MAnalyse (`avs/frc_vec_analyse.avs`, 32/16, level 0);
[scripts/vecviz.py](scripts/vecviz.py) draws them over the same grid. Row 1: vectors (hue = direction, saturation =
length, white = no motion); row 2: SAD (grey, log scale). Outlined: blocks whose vector differs from the median of its 8
neighbours by more than 3 px.

- [img/vec_recalc_41_bv.png](img/vec_recalc_41_bv.png), [img/vec_recalc_41_fv.png](img/vec_recalc_41_fv.png),
  [img/vec_recalc_42_bv.png](img/vec_recalc_42_bv.png), [img/vec_recalc_42_fv.png](img/vec_recalc_42_fv.png)
- the same before MRecalculate: `img/vec_analyse_<41|42>_<bv|fv>.png`

The camera pans fast here: NVOFA finds about (64, −150) px per source frame in the whole crop (80% of the blocks 135–175
px vertically). **The exact search does not find that motion**: its backward vectors are (0, 0) for most blocks (80%
within ±8 px), the forward ones catch part of it (median (−24, 48), spread 0..124 px), with islands of other vectors in
between. MRecalculate (thSAD 100, radius 2 around the given vectors) cannot repair a vector that is 160 px off. The
exact path is bit-identical to the CPU MAnalyse, so this is MVTools' own result with these settings, not a GPU defect.

| crop region, MRecalculate vectors | exact | NVOFA medium | NVOFA fast |
|---|---:|---:|---:|
| frame 41 bv: mean SAD / outliers (of 7973) | 10661 / 138 | 2106 / 68 | 2388 / 726 |
| frame 41 fv | 5979 / 1204 | 1958 / 22 | 2101 / 285 |
| frame 42 bv | 9236 / 128 | 2617 / 82 | 2976 / 585 |
| frame 42 fv | 5850 / 1582 | 2184 / 10 | 2409 / 294 |

(The outlier count understates the exact search's backward vectors: a field that is uniformly (0, 0) and wrong has few
outliers; its SAD shows it.) NVOFA fast is right in most of the crop but has a wrong region in A1–B4 (bv) and A1–B3 (fv),
outliers along its edge: the cells where fast was judged blocky (B2, B4). Outliers in the judged cells, exact / medium /
fast, fv of frame 41: B1 34/0/34, B2 63/0/60, C2 39/0/0, D3 64/0/0, E3 16/2/1, H1 15/1/18; fv of frame 42: B4 71/0/0,
D2 44/0/0 (all cells: `stats.json` of vecviz.py).

### Cause: truemotion's lambda

Per pyramid level (MAnalyse dump of the exact search, crop region; level 6 = 2×1 blocks for the whole picture):
neither search finds anything on level 6, and from there on the exact search keeps (0, 0) (bv) or stops at (−32, 64) px
(fv: level 5 at the edge of its ±2 radius, then no growth on the finer levels), while NVOFA has (64, −150) on every level.

[scripts/why_pan.py](scripts/why_pan.py): MAnalyse (exact search, same GPU = CPU path) with one parameter changed at a
time, level-0 vectors of bv 61 / fv 62; 'match' = blocks within 16 px of NVOFA medium's vector of the same block.

| variant | crop bv: match / SAD | crop fv: match / SAD | frame bv: match / SAD | frame fv: match / SAD |
|---|---|---|---|---|
| uc_FRC (truemotion, lambda 16000, plevel 1) | 0% / 41870 | 7% / 23358 | 61.5% / 11127 | 62.5% / 9131 |
| **lambda=0** (same: lsad=0) | **73% / 7622** | **88% / 6730** | 78.6% / 4320 | 80.9% / 3818 |
| truemotion=false | 72% / 5996 | 83% / 6043 | 79.5% / 3535 | 78.8% / 3497 |
| plevel=0 | 24% / 14647 | 70% / 7947 | 71.6% / 5511 | 78.2% / 4277 |
| plevel=0, lambda=4000 | 56% / 9552 | 82% / 7280 | 74.6% / 4964 | 80.0% / 4043 |
| plevel=0, lambda=8000 | 32% / 12890 | 76% / 7494 | 72.8% / 5274 | 78.6% / 4134 |
| lambda=1000 / 2000 / 4000 / 8000 | 39 / 8 / 0 / 0% | 83 / 66 / 50 / 13% | 73.7 / 70.2 / 61.8 / 61.6% | |
| searchparam=4 / 8 | 1 / 4% | 55 / 57% | | |
| pnew=0, pzero=0, global=false, levels=−1 / −2 | no improvement | | | |

Each candidate vector costs `lambda × (distance to the predictor)²` on top of its SAD, and with plevel 1 lambda is
multiplied by 2^level (64× on level 6). The predictor starts at (0, 0) on the coarse levels; in blurred, low-texture
foliage the SAD gain of the true 160 px vector is smaller than that penalty, so the search stays near zero, and every
finer level inherits the wrong predictor and is held near it the same way. A larger radius alone does not help (bv).
Lambda has to drop to about 1/16 (1000) to free the backward vectors on this pair, or plevel 0 with lambda 4000.

Whole run with these settings: [img/slowmo_lambda.mp4](img/slowmo_lambda.mp4), 2×2: uc_FRC (lambda 16000) | MAnalyse
`plevel=0, lambda=4000` / MAnalyse `lambda=1000` | NVOFA medium, every frame held 4×, same grid
([scripts/render_lambda.ps1](scripts/render_lambda.ps1), scripts `avs/frc_flow_pl0_l4000.avs`, `avs/frc_flow_l1000.avs`;
MRecalculate unchanged).

### Pre-screen and visual check of the whole run

[scripts/prescreen.py](scripts/prescreen.py) (numpy) scores every interpolated output frame per 120 px cell of the full
UHD frame (columns A..AF, rows 1..18) from the MRecalculate vectors of the four variants (outliers + temporal jumps of
single blocks, SAD), lists the worst spots of each variant and the spots where a lambda variant is worse than uc_FRC,
and cuts 2×2 crops (`img/spot_<nn>.jpg`). The 25 spots were judged by eye (user, 2026-10-31):
[spots_check.csv](spots_check.csv), 18 answered, 7 skipped (5 of the 11 tree-section spots).

Verdict:
- **NVOFA medium: overall the best candidate**, uc_FRC clearly the worst (spots 3, 17). NVOFA keeps detail the others
  lose (wire detail in AB3 / AC1, spot 21; Z14, AB11, spots 17 / 19). Its own defects: a black dot in AB15 that does not
  belong there (spot 20), unnatural smearing in AF13–AF16 (spot 8), a dark spot in Z13 that may be an artefact or kept
  detail (spot 17).
- **uc_FRC**: smearing / defect blocks in AA12 (spot 4), X14, X15, Z15, AA15 (spot 9), X14 (spot 10), Y11 / Y12 and Z12
  (spot 5); AA4 (spot 12) and AC1 (spot 21) are bad or blocky in every variant except NVOFA medium.
- **Lambda variants** (MAnalyse `plevel=0, lambda=4000` / `lambda=1000`): better than uc_FRC and nearly as good as NVOFA
  medium where uc_FRC smears (spot 9), but blocky in Y11 (spot 5), and AE14 looks worst with plevel 0 (spot 8). The two
  flagged regressions against uc_FRC (spots 2, 22) are not visible.
- **The score**: none of the flat-sky spots (1, 12 Y2, 23, 24, 25) shows anything (scattered vectors in featureless sky
  are harmless), and in the tree section the flagged cell itself rarely shows the defect: the visible defects were mostly
  in neighbouring, unflagged cells (spots 4, 5, 8, 9, 10, 12, 21). Confirmed at the flagged cell: spot 5 (uc_FRC Z12),
  spot 20 (NVOFA AB15), spot 8 (plevel 0 AE14, mild). So the per-cell score finds the problem areas, not the exact cells;
  a next version should weight by texture (no flat sky) and point at regions, not single cells.

Open: whether a smaller lambda costs quality elsewhere (lambda is what keeps truemotion vectors coherent, e.g. in flat
sky); only this frame pair is measured.

## How it works

`mvcuda_analyse.cu` `analyse_dev`, with `MVCUDA_NVOFA` set: the super pyramids are built as usual; the level-0 luma of
source and reference goes into 8-bit NVOFA buffers (10-bit rounded to 8, `k_of_gray`), NVOFA computes the flow source →
reference on a 4×4 grid on the leased stream, then per level-0 block `k_of_blocks` takes the median of the flow cells under
the block (x and y separately), rounds it to 1/pel, clamps it to MAnalyse's search area and computes its SAD (luma +
scaled chroma, the analysis's own `sad_block`). Coarser levels get the scaled vector of the cell under the block centre
and SAD 0 (`k_of_coarse`; nobody downstream reads them). Sessions: `mvcuda_nvofa.cu`, a pool per picture size,
nvofapi64.dll loaded at run time (no new link dependency). Headers: `Sources/cuda/third_party/nvof` (NVIDIA, BSD-3,
API 2.0; the driver reports 5.0).
