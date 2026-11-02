# RefineMotionSearch=5, RefineMotionRadius=50, search=4, searchparam=2, trymany=true, PNew=5

Recalculate needs both trymany and a very large radius to recover from poor predictions.

## State today (gpu_work, checked 2026-11-01 from code gates + Documentation\gpu.md, not run)

- `profile\qtgmc\QTGMC_gpu.avsi` has no `RefineMotionSearch`, `RefineMotionRadius` or `TryMany` arguments (only
  `RefineMotion` bool; MRecalculate reuses `Search` / `SearchParam` / `PNew`; MAnalyse never gets `trymany`). No QTGMC
  .avsi on this machine has them. Passing them now = script error "does not have a named argument".
- MAnalyse `search=4`, `searchparam=2`, `pnew=5`, `trymany=true`: GPU-capable (trymany since S3, API 18).
- MRecalculate `search=5` (UMH) radius 50: **CPU fallback** (exact, not faster). `mvc_refine::max_param()`
  (`Sources\cuda\mvcuda_refine.cuh:42`) = 64 for exhaustive, 24 for every other type; checked at
  `Sources\cuda\mvcuda_recalc.cu:276`.
- MRecalculate has **no `trymany` parameter** at all (CPU or GPU) - needs adding in MVRecalculate.cpp first.
- ~~`trymany` + `meander=false` drops to the slow one-slice slice kernel (k_wave lacks it).~~ Fixed 2026-11-02
  (08e5ff9): trymany runs in k_wave, trymany Refines in parallel warps (GPU_PIPELINE_PLAN.md section 33).

## TODO

1. QTGMC_gpu.avsi / make_qtgmc_gpu.py: add `RefineMotionSearch`, `RefineMotionRadius`, `TryMany`; pass `trymany` to
   every MAnalyse, `search=RefineMotionSearch, searchparam=RefineMotionRadius` (+ trymany) to every MRecalculate.
2. MRecalculate: add `trymany` (CPU path first, then GPU via `LevelArgs::try_many` as in MAnalyse); bit-exact vs CPU.
3. Raise the non-exhaustive radius cap (UMH at least 50) in `max_param`, and make large UMH radii fast (today sequential
   batches: PAL trymany+UMH6 4.9 s GPU vs 0.1 s CPU, GPU_PIPELINE_PLAN.md section 27; section 29 did this for exhaustive).
4. **DONE 2026-11-02 (08e5ff9):** trymany in k_wave (meander=false), or document the slowdown.
5. Tests: run_s234 / run_qtgmc_vec with these settings, exact vs gpu=false; benchmark PAL + HD against the CPU.
   **Partly done 2026-11-02:** run_s234 has the MAnalyse trymany / trymanyall / search_coarse cases (exact); DTL's
   settings (MRecalculate UMH radius 50 + trymany) not tested yet, needs 1-3.

## Hex (search=4) / UMH (search=5) above radius 24 on the GPU

### Why it stops at 24 today

- `mvc_refine::max_param()` (`Sources\cuda\mvcuda_refine.cuh:42`) returns 64 for exhaustive and 24 for every other type.
  Checked per level in MAnalyse (`mvcuda_analyse.cu:3284`) and in MRecalculate (`mvcuda_recalc.cu:276`). In MAnalyse
  the coarse levels always search exhaustively (limit 64), so the hex/UMH cap of 24 only hits `pelsearch` (finest
  level); in MRecalculate it hits `searchparam` directly (= QTGMC's `RefineMotionRadius`). Above it the whole filter falls back to the CPU (exact, not faster).
- The cap comes from the batch size, not from the search itself. `umh()` (`mvcuda_refine.cuh:269`) builds the whole
  UMH candidate list in **one** batch in per-thread arrays `xs[MAX_BATCH], ys[MAX_BATCH]` with `MAX_BATCH = 160`
  (line 31, comment: "UMH with search range 24: cross 48 + 6 grid rings of 16" = 144). Radius 50 needs
  cross 2 x 2 x 25 = 100 + hex grid 16 x floor(50/4) = 192 = **292** candidates: the array would overflow.
  Simply raising MAX_BATCH is bad: those local arrays grow every kernel's stack frame and so the device-wide
  local-memory reservation (`mvcuda_analyse.cu:2508` notes 1.3 KB per thread for a MAX_BATCH array in k_wave).
- Hex2 itself (`hex2()`, line 229) has no array limit, only the shared cap: it walks up to `range / 2 - 1` dependent
  steps of 3 candidates, then one radius-1 ring. Radius 50 = up to 24 serial steps per block.
- Horizontal / vertical (search 6 / 7) have fixed `xs[2 * 24]` arrays (lines 349 / 361): same cap, not needed here.

### Why it is exact to lift

- UMH's cross + hex grid are all relative to one fixed centre (`omx, omy`), so the CPU's CheckMV sequence over them
  equals folding consecutive batches in CPU order with the running minimum carried over - the same argument as the
  exhaustive batches (S2, GPU_PIPELINE_PLAN.md section 27). `wave_bad_search` (`mvcuda_analyse.cu:2486-2530`) already
  does exactly this for the bad-vector UMH at pel 2 (list longer than one batch, written straight into the shared
  list `g_ws.xs / g_ws.ys` instead of local arrays).
- Hex2 steps are already separate dependent batches; a larger range only means more iterations.

### Why it matters

- DTL's settings (`RefineMotionSearch=5, RefineMotionRadius=50`) are a UMH refine at radius 50 in MRecalculate;
  DTL's `RC_smooth=2` (his mvtools fork only) goes up to radius 100.
- UMH radius 50 = about 292 + hex walk candidates per block, exhaustive radius 50 = 10200: UMH is the cheap way to get
  a wide range, so it is the type people will use for large radii.

### Work

1. `umh()`: emit cross + grid in CPU order and flush every MAX_BATCH (`run_batch`, running minimum carried), like the
   exhaustive case; keep the single-batch path for radius <= 24 (unchanged code and speed).
2. Avoid the local arrays: write the list into the checker's shared list as `wave_bad_search` does (k_search,
   k_wave GEN, recalc kernel), or generate candidate k on the fly (closed form like `expanding_point`).
3. Raise `max_param` for S_UMH / S_HEX2 (proposal: 128, covers RC_smooth=2's 100); keep `GpuBackend.h:190` comment and
   `Documentation\gpu.md` (MAnalyse / MRecalculate tables) in sync.
4. Speed: the long Hex2 walk and several UMH batches are serial per block, and with meander=true each slice is serial
   block after block (latency-bound, 16-24 CTAs). Options: evaluate a UMH batch with all warps of the CTA (it is one
   parallel minimum like the exhaustive fast pass, section 29: prune by vector cost, then (cost << 15 | index) min),
   reference window in shared memory when the batch area fits.
5. Tests: run_s234-style vector/SAD compare vs gpu=false for UMH / hex radius 25, 32, 50, 64, 100 at pel 1 / 2 / 4,
   8 and 10 bit, MAnalyse (pelsearch and coarse searchparam) and MRecalculate; negative control; then QTGMC with DTL's
   settings (after TODO 1 + 2 above) and timing vs the CPU.

## Predictor architecture (DTL post #2, #7) - checked against our code

### What the code does today (CPU PlaneOfBlocks.cpp; the CUDA port is bit-exact with it)

- MAnalyse candidates per block, in this order (`PseudoEPZSearch`, line 875; `FetchPredictors`, line 694):
  1. zero, cost with `pzero` (line 908)
  2. global, cost with `pglobal` (line 931)
  3. main predictor = `InterpolatePrediction` (line 502): 9/3/3/1 bilinear blend of the 4 nearest coarse-level
     vectors (median on the coarsest level); no penalty, and the anchor of the lambda cost
  4. `predictors[0]` median of 1..3, `[1]` left (right on reversed meander rows), `[2]` up, `[3]` bottom-right
     (still holds the coarse interpolated value; else up-right; else zero), `[4]` temporal (only `temporal=true`)
- Acceptance is strict `<`, so order and penalties decide ties.
- **No deduplication**: the check in `CheckMV0` (line 2605) is behind `ONLY_CHECK_NONDEFAULT_MV`, commented out in
  `MVInterface.h:48`. trymany=false: a duplicate costs one SAD, never changes the result. trymany=true: every
  duplicate gets a full Refine (up to 7 refines per block, 8 with temporal) - real waste at large radii.
- MRecalculate (line 3260-3440): one predictor (`smooth=0` nearest, `smooth=1` bilinear), SAD measured, refined only
  if SAD > `thSAD` (line 3365); no other predictors, no trymany. DTL's QTGMC: thSAD = thSAD2/2, 0 with
  RefineMotionAll. GPU kernel (`mvcuda_recalc.cu`) identical.
- **Parent predictors: not present** anywhere (CPU, CUDA, wt_upstream). The 4 coarse vectors only enter as the
  blend; at a motion boundary the blend can point between both objects.

### TODO

6. Parent predictors in MAnalyse: the 4 coarse vectors of the blend as extra `CheckMV0` candidates (after the
   existing ones, so with the switch off the output stays stock). New opt-in parameter (changes vectors vs stock
   mvtools). CPU first, then CUDA (k_search + k_wave phase A, trymany path too); +4 SADs per block without trymany.
   Compare with neo-mv branch feature/analyse-parent-predictors for order and semantics before choosing.
7. Optional dedup of identical candidates (opt-in; changes results on ties, mostly a trymany speed-up). Do it
   together with 6, since parents often repeat existing predictors.
8. Later / design work, not a port: adaptive predictor set (DTL post 2050759, tile the area the predictors span with
   fixed-radius searches) vs smooth=2 adaptive radius (post 2050471); refinement folded into MAnalyse (input vector
   clip) with MRecalculate as legacy. Decide after 6 is measured.

## smooth=2 and DTL's two-parameter-set idea (doom9 t=187172, posts 5-7)

- smooth=2 (DTL's fork only): DTL confirmed the nSearchParam bug (post 5) and that a larger search area fails on
  repeating patterns (post 7); our compare/dtl_smooth2 (gpu_work, 74d0486) shows it on the wires clip. No action.
- **Two parameter sets blended through a motion mask can be done as of today**, script only, no mvtools change:
  two MAnalyse (+ MRecalculate) chains (e.g. small radius + high `pzero` vs large radius), a motion mask
  (MMask or similar), then a masked merge of the two processed clips (mt_merge / Overlay). Costs two analyses.
  Note: a zero penalty only exists in MAnalyse (`pzero`); MRecalculate has no `pzero` argument (Interface.cpp:704).

## Sub-pel shifts at runtime, chroma precision, area mode (DTL post #2, #3)

### State today

- Our GPU stores sub-pel planes in VRAM (pel 2: 4 per plane, pel 4: 16 via `k_quarter_v`; MCompensate / MDegrain
  cache pel-2 Wiener planes per frame). Computing them at runtime was already an option (GPU_PIPELINE_PLAN.md line
  761) and S4's original plan; S4 stored them instead. Rejected earlier only: texture-unit interpolation (8-bit
  fixed-point weights, not MSuper's filter, section 28).
- Chroma: stock mvtools halves and rounds a 4:2:0 vector for chroma (`MvToChromaX/Y`, `PlaneOfBlocks.h:384`), losing
  one sub-pel bit. neo-mv feature/chroma-subpixel-precision keeps it (changes vectors vs stock).
- Area mode: not in pinterf or our code. In DTL's fork (`profile\dtl_dl_r2746a29`, r.2.7.46-a.29): MAnalyse and
  MRecalculate `AreaMode` (steps), `AMstep`, `AMoffset`, `AMpel`, `AMflags` (1 diagonals, 2 sides), `AMavg`,
  `AMdiffSAD`; 1 + 4..8 x AreaMode grid positions per block, each a full search, combined by median (`toMedian`,
  PlaneOfBlocks.cpp:1248-1360) or `AMavg`. Cost 5-17x the analysis for AreaMode 1-2.

### TODO

9. Runtime sub-pel on the GPU (bit-exact, MSuper's integer filter): copy one pel-1 window to shared memory and
   interpolate there instead of copying every stored sub-pel plane. Helps the section 29 exhaustive pass (window
   copy ~23k cycles per block) and VRAM (pel 4 = 4x pel 2). Exact mode is mostly latency-bound (meander slices,
   section 24 / 30), so expect gains at large radii / pel 4, not QTGMC Fast. Measure before building.
10. Full chroma sub-pel precision (opt-in, port from neo-mv): CPU first, then GPU; cheap once 9 exists.
11. Area mode (opt-in, port from DTL's fork): CPU first; GPU-friendly (positions are independent searches). Lower
    priority than parent predictors (item 6).

## Hardware motion estimation: NVENC, NVOFA, DX12 (DTL post #5, #7, #8, #13)

### State today (NVENCMEAnalyse.md, NVOF_FILTER_PLAN.md, compare/nvencme_frc, compare/dtl_dx12_frc)

- Outlier score in the uc_FRC chain (all / tree section, lower = better): NVOFA medium (`CudaNVOFAnalyse`, built,
  API 17) 0.009 / 0.011; exact search 0.047 / 0.035; our D3D12 prototype (median of engine cells per block)
  0.058 / 0.115; DTL's own DX12_ME build r.2.7.46-a.29 8x8 0.106 / 0.186, 16x16 0.145 / 0.224; NVENC SDK ME-only
  0.086-0.12. User's eye: NVENC "extremely bad"; DX12 frames not judged yet.
- Measured NVENC problems: (1) chroma ignored through the NVENC SDK and D3D12 (flat luma + textured U/V gives zero
  vectors) - DTL's Gemini research confirms luma-only is general; (2) incoherent fields: the encoder takes the
  cheapest match per block, no smoothness control. (2) was the main cause of the bad result.
- DTL's other points: pel scaling and SAD are solved in `CudaNVOFAnalyse` (1/32 px -> 1/pel, own luma + chroma SAD);
  NVOFA's dense 4x4 grid gives any block size / overlap; his 16x16 ran for us, but `optSearchOption=5` crashes under
  Prefetch(3) and `Compute.cso` is read from the current directory.
- NVOFA input is luma only today (`NV_OF_BUFFER_FORMAT_GRAYSCALE8`, `Sources\cuda\mvcuda_nvofa.cu:118`); the SDK
  also takes NV12 and ABGR8 (`nvOpticalFlowCommon.h:259-260`) - whether it then uses colour is untested.
- Why NVENC still matters: D11 supports compute capability 6.0+; Pascal has NVENC but no OFA engine.
- DTL's chroma ideas: 3 single-channel runs (Y/U/V or R/G/B) + per-block selection by full-colour SAD
  (`MAverage mode=8` in his fork) or 3 extra predictors; interleaving Y/U/V (or nibble-split U/V) into one wide
  8-bit luma frame; his bit-count argument (4:2:0 = 12 bits per sample position, 8-bit input loses 4).

### TODO

12. Extra predictor clip(s) for MAnalyse (DTL: `additional_predictor_mvclip`): one or more hardware vector clips as
    additional candidates next to the own predictors (lambda / pnew / refine apply), opt-in. CPU first, then GPU
    (one more candidate per block per clip). Fits next to parent predictors (item 6); should help NVOFA's known
    defects (AB15 / AF13-16 / Z13, blocks straddling two motions).
13. NVOFA chroma: run `CudaNVOFAnalyse` with NV12 / ABGR8 input on the flat-luma synthetic clip. If NVOFA ignores
    colour: 3 runs + per-block colour-SAD selection in `k_of_blocks` (SAD kernel exists), opt-in, judged on real
    material. Cost 3 OFA runs (~2.6 ms each UHD medium, 1 OFA engine on the 5090).
14. NVENC revisited along DTL's lines, target Pascal-Turing cards without OFA, in this order:
    (a) Y/U/V (or R/G/B) runs on separate NVENC engines (3 on the 5090: about one run's wall time), selection by
        colour SAD + median of engine cells per block; score with `compare/nvencme_frc/scripts/prescreen.py`
        against the old prototype.
    (b) the same vectors fed to MAnalyse through item 12 (software search supplies the coherence NVENC lacks);
        compare with the exact search and NVOFA, user judges by eye.
    (c) DTL's interleaving ideas only if (a) / (b) fall short. Sceptical: the engine's sub-pel interpolation mixes
        channels, nothing aligns x-vectors to the channel layout, nibble-packing makes SAD meaningless.

## Quality test against a reference (DTL's proposal) - needed once features change the output

Bit-exact comparison proves nothing for opt-in features (items 6, 7, 10-14). DTL's pipeline: clean UHD source ->
downscale to 1080p progressive reference -> interlace with vertical filtering -> add grain (low / medium / high) ->
QTGMC or SMDegrain -> compare with the reference (PSNR / SSIM / VIF) -> tune switches and parameters; start with the
default presets and each new feature on / off. Publish results per footage type as user documentation.

### Material (checked 2026-11-01: ffprobe + ffmpeg mpdecimate over the first 100 frames, rough check)

- `TESTFILES\1_CrowdRun`, `2_ParkJoy`, `5_OldTownCross` (`_1080p50_CgrLevels_SINC_FILTER_SVTdec05_.mxf`) and
  `Dancers_HD_50p.mkv`: 1920x1080, 50 fps, progressive, yuv422p10le, no duplicate frames in the first 100. SVT
  clips are sinc-downscaled from the UHD 50p masters: steps 1-2 done. Not checked: `3_DucksTakeOff`, `4_InToTree`.
- Step 3 exists for one clip: `1_CrowdRun_1080i25_tinterlace_cvlpf_prores.mov` (tinterlace + vertical lowpass).
- A note in another session claimed "TESTFILES UHD clips are 25p doubled": not in this memory or the docs; which
  clips it means is unknown - check them the same way before use.

### Our changes to the pipeline (user's rules: no PSNR-only claims, single bad blocks matter, user's eye decides)

- Score per block and per frame too (worst block, outlier count, as `compare/nvencme_frc/scripts/prescreen.py`),
  not only global PSNR / SSIM / VIF; show the worst frames to the user.
- Automatic tuning must not maximise an average alone (PSNR rewards blur): combined score incl. worst-block
  terms; the final decision stays with the user's eye.
- Grain vs a clean reference rewards denoising: right for SMDegrain; for QTGMC also run without grain to judge
  deinterlacing alone.
- QTGMC ground truth: bobbed output frame n = 50p reference frame n (field n came from it).

### TODO

15. Build the test: interlace the four 50p clips (vertical lowpass as the CrowdRun 1080i25 file), grain variants,
    QTGMC presets (default, then each opt-in feature on / off), scoring script with global + per-block metrics,
    worst-frame crops for the user. Then the same for SMDegrain.
