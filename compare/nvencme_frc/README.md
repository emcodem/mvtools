# uc_FRC with NVENC ME-only vectors (prototype, ahead of NVENCMEAnalyse)

> **Status 2026-10-31: NVENC ME is not good enough for frame interpolation; recommendation: do not build
> NVENCMEAnalyse.** Judged by eye (user): plain P4 "extremely bad", the hinted pyramid (P4, 3 levels, intra off) in
> [img/slowmo_pyramid_1.7-2.6.mp4](img/slowmo_pyramid_1.7-2.6.mp4) "still extremely bad". Every NVENC lever was
> tried: presets P1/P4/P7, half resolution, chroma (ignored by the ME), intra off, 16x16 only, a hinted pyramid
> (fixes the search range, SADs near NVOFA's, fields even less coherent), colour mixed into the luma (worse); AV1 has
> no ME-only mode, the D3D12 motion estimator (DTL's DX12_ME path) is the best of them but still far behind, HEVC ME-only output
> is broken. The cause is the cost function: the encoder's ME picks the cheapest match per block, not the true motion,
> and the API has no smoothness control. What remains would be our own processing (smoothing the field, stronger
> MRecalculate), i.e. not NVENC's quality. NVOFA covers Ampere+; NVENC would only have added Pascal / Turing, which
> cannot be tested here. The spot crops and `spots_check.csv` below were not judged (not needed after the clips).
>
> Prototype switches (environment of the AviSynth process): `NVME_FRESH=0` keep sessions (faster, not deterministic),
> `NVME_NOINTRA=1`, `NVME_CHROMA=1`, `NVME_P16=1` (16x16 partitions only), `NVME_PSEUDO=<k>` (colour into the luma), `NVME_SESSIONS=<n>` (pool size, default 2),
> `NVME_TUNING=<n>` (tuning used for the preset config).

Early visual check for [NVENCMEAnalyse.md](../../docs/dev/NVENCMEAnalyse.md) step 5, done before the plugin exists: a throwaway
AviSynth filter `NVENCMEProto` ([proto/](proto/), standalone mode only, Windows) takes the place of MAnalyse in the
uc_FRC chain of [compare/nvofa_frc](../nvofa_frc/) — same clip, same chain (MRecalculate 16/8 → MFlowFps 24000/1001),
same crops, slow motion and spot workflow. Output layout as CudaNVOFAnalyse (one level, 32/16 blocks, pel 2); SADs are
written as 0 and recomputed by MRecalculate (`PlaneOfBlocks.cpp:3358-3362`).

Per pair: 8-bit luma of both frames (CPU) → NVENC input buffers (NV12, neutral chroma) → `nvEncRunMotionEstimationOnly`
→ 8x8 quarter-pel vectors (H.264: four per macroblock by partition type) → per block the lower median of the cells
under it, to 1/pel, clamped to MAnalyse's search area. `half=true`: ME on the luma downscaled 2x2.

RTX 5090, driver 616.92, NVENC API 13.0 (header: FFmpeg/nv-codec-headers n13.0.19.1).

## Probe (`proto/probe.exe`, synthetic smooth-noise texture, uniform shift)

| | H.264 P1 | H.264 P4 | H.264 P7 |
|---|---:|---:|---:|
| shift (5, -3), (17, 9): cells exact | 100 / 99.8 % | 100 / 99.5 % | 100 / 100 % |
| shift (40, -24) | 35 % | 63 % | 86 % |
| shift (64, 32), (120, -60), (-150, 0), (0, 100) | 0 % | 0 % | 0 % |
| ms per pair 1080p, 1 / 2 / 4 threads (kept session) | 1.64 / 0.84 / 0.51 | 2.65 / 1.32 / 0.68 | 2.61 / 1.29 / 0.67 |

- **Search range** ends between 40 and 64 px per frame (P7 reaches furthest). `half=true` doubles it.
- **Session cap**: 12 concurrent sessions, the 13th fails with NVENCSTATUS 21.
- **ME-only rejects every `tuningInfo`** in `NV_ENC_INITIALIZE_PARAMS` (it must be UNDEFINED; the preset config is
  still fetched with one). B frames off (`frameIntervalP = 1`).
- **A session has memory**: the same pair gives different vectors depending on the previous call (the last field is
  used as predictors; even a direct repeat differs in 56 % of the cells). `nvEncReconfigureEncoder` is not implemented
  in ME-only mode (NVENCSTATUS 22). With kept sessions and Prefetch 3 the output md5 changes from run to run;
  single-threaded it is stable. The prototype therefore opens a **fresh session per call** (`NVME_FRESH=0` turns it
  off): deterministic, but 42 ms instead of 10 ms per UHD pair. For the plugin: a session per clip and direction, called
  in frame order (temporal predictors as intended), or fresh sessions — open question.
- **AV1: no ME-only mode.** The driver lists the AV1 encoder, but `NV_ENC_CAPS_SUPPORT_MEONLY_MODE` is 0 for AV1
  (1 for H.264 and HEVC), and SDK 13.0 defines no AV1 MV output layout (only `NV_ENC_H264_MV_DATA` /
  `NV_ENC_HEVC_MV_DATA`).
- **HEVC: not usable yet.** Same setup, but most CUs come back as `cuType 0` with zero vectors and the rest are off
  (shift (5, -3) → (5.25, -0.5)), also at 64-aligned sizes. Left out of the comparison; to be investigated.

## Speed (uc_FRC chain, 112 output frames, Prefetch 3, `scripts/render.py`, best of 3, indicative)

| variant | wall | md5 (ffmpeg -f md5) |
|---|---:|---|
| exact search | 6.61 s | B5D0E69C4823 |
| NVOFA medium | 4.64 s | 3F01579928CD |
| NVENC H.264 P1 | 9.41 s | AF7192744378 |
| NVENC H.264 P4 | 9.92 s | 6CD3B0843418 |
| NVENC H.264 P7 | 10.08 s | A0EE87211759 |
| NVENC H.264 P7 half-res | 7.98 s | DD68F94A7594 |

With kept sessions (not deterministic) P1 / P4 / P7 / P7 half took 5.46 / 5.77 / 5.82 / 4.71 s. The prototype also
converts the luma on the CPU in one thread per frame; the plugin's numbers will differ.

## Vectors (`scripts/prescreen.py`, MRecalculate output, score as in nvofa_frc)

| variant | score all | score tree section | mean SAD | cells > 0.5 |
|---|---:|---:|---:|---:|
| exact search | 0.038 | 0.031 | 1287 | 158 |
| NVOFA medium | 0.009 | 0.011 | 820 | 2 |
| NVENC H.264 P1 | 0.096 | 0.114 | 1309 | 3062 |
| NVENC H.264 P4 | 0.098 | 0.124 | 1219 | 3240 |
| NVENC H.264 P7 | 0.108 | 0.141 | 1176 | 3493 |
| NVENC H.264 P7 half-res | 0.132 | 0.182 | 1123 | 3497 |

The NVENC outliers sit in the trees (columns Z..AF, rows 3..18, where the exact search has its highest SADs too), not
in the sky: 3–4× the exact search's score and 10× NVOFA's, with SADs between the two. A pointer only — judge by eye.

### DTL's points (doom9), tested

| change (P4, fresh sessions) | score all | score tree | mean SAD | cells > 0.5 |
|---|---:|---:|---:|---:|
| as above | 0.098 | 0.124 | 1219 | 3240 |
| + real chroma in the NV12 input (`NVME_CHROMA=1`) | 0.098 | 0.124 | 1219 | 3240 |
| intra search off (`NVME_NOINTRA=1`) | 0.086 | 0.152 | 959 | 2718 |
| chroma + intra off + 16x16 partitions only (`NVME_P16=1`) | 0.087 | 0.152 | 987 | 2698 |

- **Chroma is not used by the H.264 ME** (driver 616.92, API 13.0): identical vectors with and without it, and a
  probe picture with flat luma and texture only in U / V gives zero vectors in 32636 of 32640 cells. Luma-only ME is
  what NVENC does here, not a prototype limitation.
- **Intra search off** (`NV_ENC_CONFIG_H264_MEONLY::disableIntraSearch`): fewer outliers and 21 % lower SAD overall,
  but worse in the tree section. 16x16 only changes little.
- The codec union: filling it from the encoder preset (as before) or clearing it and setting the ME-only struct gives
  identical H.264 vectors; HEVC stays broken either way.
- **Search range with fresh sessions is shorter**: shift (40, -24) went from 63 % exact (kept session, hints from the
  previous call) to 0 %. The missing predictors, not the block size or chroma, look like the main weakness on fast
  motion. Next lever: `meExternalHints` (H.264 only) from a half-resolution pass or from the previous frame of the same
  direction — deterministic predictors instead of whatever the session saw last.

### D3D12 video motion estimator (DTL's DX12_ME path), same board

DTL's request (doom9): test the same NVENC engine through `ID3D12VideoMotionEstimator`, the API of his MAnalyse
`optSearchOption=5 / 6`. [proto/dx12me.h](proto/dx12me.h) makes the same calls as his `Init_DX12_ME` /
`EstimateMotion` / `ResolveMotionVectorHeap` (NV12, quarter pel, defaults, no hints); `NVENCMEProto(..., dx12=8|16)`
uses it in place of the NVENC SDK, [proto/dx12probe.cpp](proto/dx12probe.cpp) runs the probe tests.

Probe (RTX 5090, driver 616.92; supports 8x8 and 16x16, quarter pel only, 32x32..4096x4096):

| | 8x8, 1080p | 8x8, UHD | 16x16, UHD |
|---|---:|---:|---:|
| shift (5, -3), (17, 9): blocks exact | 100 / 99.3 % | 100 / 99.9 % | 100 / 100 % |
| shift (40, -24) | 72 % | 62 % | 79 % |
| shift (64, 32) and larger | 0 % | 0 % | 0 % |
| flat luma, texture only in U / V | 0 exact, all zero | 0 exact, all zero | 0 exact, all zero |
| same pair after another pair / repeated | identical | identical | identical |
| ms per pair, 1 thread, incl. upload + readback | 1.81 | 5.82 | 4.16 |

- **Chroma is ignored through D3D12 as well**: not an issue of the NVENC SDK, the engine's ME is luma-only here.
- **No session memory**: the D3D12 estimator gives the same vectors regardless of call order, so it needs no fresh
  sessions (deterministic: md5 5495CC13974D twice).
- Search range like the NVENC SDK without hints.

On the clip:

| variant | score all | score tree | mean SAD | cells > 0.5 | wall (112 frames) |
|---|---:|---:|---:|---:|---:|
| NVENC SDK P4 (fresh sessions) | 0.098 | 0.124 | 1219 | 3240 | 9.92 s |
| **D3D12 ME 8x8** | **0.059** | 0.116 | 1007 | 2080 | 7.16 s |
| D3D12 ME 16x16 | 0.087 | 0.152 | 987 | 2698 | 6.68 s |
| D3D12 ME 8x8 half-res | 0.072 | 0.140 | 970 | 2452 | |
| *exact search* | *0.038* | *0.031* | *1287* | *158* | *6.61 s* |
| *NVOFA medium* | *0.009* | *0.011* | *820* | *2* | *4.64 s* |

D3D12 8x8 is the best NVENC-engine variant so far (its defaults beat the SDK's P1–P7 presets on coherence), but
in the tree section it is no better than the SDK's P4 and still about 10x NVOFA. Clip for the eye:
[img/slowmo_dx12_1.7-2.6.mp4](img/slowmo_dx12_1.7-2.6.mp4) (exact | NVOFA medium / NVENC SDK P4 | D3D12 8x8).

Prototype note: the first D3D12 integration hung at process exit — estimators in `thread_local` storage released
their D3D12 objects under the loader lock. They are now created once per thread and never destroyed (as the NVENC
session pool).

### Colour into the luma (`NVME_PSEUDO=k`), DTL: "luma only ME can greatly degrade ME quality on colour footage"

NVENC ignores chroma, so the colour goes into the luma that NVENC sees: Y + k (U − 128) − k (V − 128) (8 bit, chroma at
the luma position).

| variant | score all | score tree | mean SAD | cells > 0.5 |
|---|---:|---:|---:|---:|
| P4 | 0.098 | 0.124 | 1219 | 3240 |
| P4 pseudo-luma k = 1 | 0.104 | 0.129 | 1228 | 3438 |
| P4 pseudo-luma k = 2 | 0.119 | 0.143 | 1255 | 3815 |
| P4 pyramid 3, intra off | 0.121 | 0.192 | 875 | 2592 |
| P4 pyramid 3, intra off, pseudo-luma k = 2 | 0.277 | 0.356 | 949 | 8491 |
| *NVOFA medium (luma only, 8-bit grey)* | *0.009* | *0.011* | *820* | *2* |

Worse with every step of k, much worse in the pyramid (the downscaled levels mix colour noise into the hints). On this
clip luma-only input is not what holds NVENC back: NVOFA gets the same 8-bit luma (`k_of_gray`) and gives the most
coherent fields; MRecalculate (`chroma=true`) adds chroma to every variant's final vectors. DTL's point stays valid for
scenes where objects differ in colour only, for NVOFA as well.

### Hinted pyramid (`levels=2..4`)

NVENC ME on the picture downscaled 2x per level, coarsest first; each finer level gets two hint candidates per 16x16
macroblock (`meExternalHints`, integer pel): the coarse vector under the MB centre and the median of the 3x3 coarse
cells around it. Fresh sessions, so deterministic (md5 E11CE147046A twice for pyramid 3 no intra). Probe, UHD,
P4, cells exact:

| shift | 1 level | 2 levels | 3 levels |
|---|---:|---:|---:|
| (40, -24) | 84 % | 100 % | 100 % |
| (64, 32) | 0 % | 71 % | 98 % |
| (0, 100) | 0 % | 0 % | 88 % |
| (120, -60) / (-150, 0) | 0 % | 0 % | 26 % / 8 % |
| ms per pair (fresh sessions) | ~48 | ~87 | ~118 |

On the clip:

| variant | score all | score tree | mean SAD | cells > 0.5 |
|---|---:|---:|---:|---:|
| P4 (1 level) | 0.098 | 0.124 | 1219 | 3240 |
| P4 pyramid 2 | 0.116 | 0.156 | 1152 | 3861 |
| P4 pyramid 3 | 0.125 | 0.174 | 1105 | 4201 |
| P4 pyramid 4 | 0.126 | 0.177 | 1078 | 4253 |
| P4 pyramid 3, intra off | 0.121 | 0.192 | 875 | 2592 |
| P7 pyramid 3 | 0.131 | 0.185 | 1087 | 4391 |
| *NVOFA medium* | *0.009* | *0.011* | *820* | *2* |

The range is fixed (SAD falls towards NVOFA's: 875 vs 820 with intra off), but the fields get *less* coherent, not
more. The search range was not what separates NVENC from NVOFA: the encoder's ME picks the cheapest match per block,
which in the trees jumps from block to block. Whether that is visible: [img/slowmo_pyramid_1.7-2.6.mp4](img/slowmo_pyramid_1.7-2.6.mp4)
(exact | NVOFA medium / P4 | P4 pyramid 3 intra off). Speed of that variant: 21.2 s for 112 frames (three session
re-opens per pair; kept sessions with hints would be far faster but need a fix for the session state).

## For the visual check

- [img/slowmo_1.7-2.6.mp4](img/slowmo_1.7-2.6.mp4): exact | NVOFA medium (top), NVENC H.264 P4 | P7 half-res (bottom),
  the crop at x 2880, y 240, every frame held 4×, cells A1..H5 — the section judged for nvofa_frc (frames 41, 42).
- [spots_check.csv](spots_check.csv) with [img/spot_01.jpg](img/spot_01.jpg) .. `spot_10.jpg`: the worst cells of
  P4 and P7 half-res (all in the tree section, frames 34–54), each also a regression against NVOFA medium; 2×2 crops
  exact | NVOFA medium / P4 | P7 half-res, full-frame cell names, flagged cell in orange. One question per spot,
  y/n columns per variant and notes.

Reproduce: build `proto/` (`build.bat <nv-codec-headers include\ffnvcodec>`), copy `nvencme_proto.dll` to
`profile\build_nvme_proto`, then `scripts/prescreen.py <work> --panels p4,p7h`, `scripts/render.py <work>`,
`scripts/slowmo.py <work> 1.7 2.6 4 --variants ... --cols 2` (python with numpy for prescreen).
