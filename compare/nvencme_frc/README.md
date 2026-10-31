# uc_FRC with NVENC ME-only vectors (prototype, ahead of NVENCMEAnalyse)

> **Status 2026-10-31: not judged by eye yet, on purpose.** First look (user): "extremely bad". On the vector scores
> every NVENC variant is far behind NVOFA medium (outlier score ~0.09 vs 0.009). DTL's points are tested below:
> chroma is ignored by NVENC's H.264 ME, intra off helps only partly. The likely main cause is the missing predictors
> (fresh session per call, see the probe). **Next:** a hinted variant (`meExternalHints` from a half-resolution pass or
> from the previous frame of the same direction), re-scored; the spots and the slow-motion clip are judged only if it
> comes close to NVOFA. Gate for NVENCMEAnalyse.md steps 2–4: build the plugin only then; otherwise NVOFA covers
> Ampere+, and NVENC would only add Pascal / Turing, which cannot be tested here.
>
> Prototype switches (environment of the AviSynth process): `NVME_FRESH=0` keep sessions (faster, not deterministic),
> `NVME_NOINTRA=1`, `NVME_CHROMA=1`, `NVME_P16=1` (16x16 partitions only), `NVME_SESSIONS=<n>` (pool size, default 2),
> `NVME_TUNING=<n>` (tuning used for the preset config).

Early visual check for [NVENCMEAnalyse.md](../../NVENCMEAnalyse.md) step 5, done before the plugin exists: a throwaway
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
