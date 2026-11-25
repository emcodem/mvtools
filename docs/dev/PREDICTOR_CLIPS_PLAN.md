# Temporal predictor clips (DTL's zero / global predictor clips) - CPU port + CUDA

Branch `predictor-clips` (worktree `C:\dev\mvtools\wt_predclip`, from gpu_work 3360915). Goal (user, 2026-11-25):
DTL's changes on the GPU so that he can try them on his own. Source of the idea: DTL2020/mvtools r.2.7.46-a.32
(17dfdb5), local clone `C:\dev\mvtools_dtl`; background and decisions in DTL_COMMENTS_TODO.md items 17, 18, 22-24.
Eye test 2026-11-25 with his DLL (profile\dtl_a32\eye): no visible gain on our clips nor on his DV clip; the user
wants it built anyway so DTL can test it.

## Not bit-exact with his DLL

His logic sits on his own search (candidate dedup `IsVectorChecked`, `GetDM`, reworked predictors); ours is
pinterf's. We port the logic; our CPU path is the reference, the GPU must match it bit for bit.

## Script interface (his names, appended, all off by default = output unchanged)

- MAnalyse / CudaMAnalyse: `super_przero` (c), `Nframes_przero` (i), `SADadj_przero` (f, 0 = auto),
  `super_prglobal` (c), `Nframes_prglobal` (i), `SADadj_prglobal` (f, 0 = auto). Checks as his: same super format and
  frame count, Nframes not in -1..1; auto SADadj = sqrt(2N / (N + 1)) for N > 1, sqrt(2|N| / (|N| - 1)) for N < -1.
  dct != 0 with predictor clips: error (the DCT of the source block would be stale).
- MAnalyse / CudaMAnalyse: `pglobal_coarse` (i, default = pglobal): global penalty on the coarse levels. Replaces his
  `pglobal < -256` mode (user decision): his `pglobal=-1000` becomes `pglobal=-1000, pglobal_coarse=0`.
- MDegrainN / CudaMDegrainN: `SkipFrame` (i, default 0): the reference frame at that signed offset gets weight 0.
  Sign as his: -k = the isb (backward) vector of delta k = reference frame n + k (MAnalyse: isb -> nref = nsrc + delta).

## CPU (reference)

- PlaneOfBlocks::WorkingArea: second aligned block buffer per predictor kind (`pPr_temp[2][3]`), filled in
  search_mv_slice like the source copy; same pitch (nSrcPitch), so the SAD / Refine code is used unchanged by swapping
  `workarea.pSrc` (no ALIGN_SOURCEBLOCK change, unlike his build).
- PseudoEPZSearch, his logic on our search:
  - zero clip after the zero candidate (cost before the trymany refine = nMinCost_zero); with trymany: refine around
    zero on the clip, SADadj scaling of the result, into bestMVMany[0] if better; without: adjusted SAD / cost become
    the current best, raw values kept for the final refine;
  - global clip after the global candidate (+ its trymany refine), target cost trymany ? cost(global) :
    min(cost(global), current best);
  - final refine (no trymany): on the clip whose candidate is still the best (flag + same vector + same cost; his
    build compares cost and vector only and can take the raw values of a clip that did not win), SADadj after it.
  - SADadj scaling: sad *= adj (float, truncated); cost recomputed with the candidate's penalty if the vector stayed,
    else cost += new_sad - old_sad.
- GroupOfPlanes: zero clip on every level; global clip not on the coarsest level (global vector still zero) and not
  where the global vector equals the zero vector while a zero clip exists (his 98397ea). pglobal on level 0,
  pglobal_coarse on the others.
- MVAnalyse: predictor supers loaded like the source (CompleteLite for CudaMSuper lite supers).
- MDegrainN: SkipFrame in every per-reference loop (usable flag && k != skip).

## CUDA

- CudaMAnalyse: per level the predictor clips' source planes (integer positions only, no sub-pel) - the same pyramid
  build as the source; slice kernel + wavefront kernel get 0..2 extra source plane sets and the extra candidates.
- CudaMDegrainN: SkipFrame; note: his zero clip uses MDegrainN with tr 10 (global 6), CudaMDegrainN covers tr 1..6
  today (MDegrain1-6 path) - tr up to 10 needed for his script to stay on the GPU.
- pglobal_coarse in the GPU search arguments.

## CUDA steps (survey 2026-11-25, line numbers of d8ade6a)

User decisions 2026-11-25: steps 1-5 including tr up to 10. Predictor clips in the wavefront kernel (meander=false: the
faster GPU path, meander=true was never faster on the GPU and DTL does not need it), as a separate kernel variant so the
normal meander=false path keeps its static shared memory; meander=true + predictor clips refuse (D8).

1. pglobal_coarse: per level in LevelArgs.pglobal (filled mvcuda_analyse.cu ~4999-5005, today one value for all).
2. CudaMDegrain SkipFrame: MVDegrainX::gpu_make sets usableB/F (MVDegrain3.cpp ~614-627); unusable = src token
   + weight 0 + NORMWEIGHTS, no kernel change. MVDegrainX CPU path needs SkipFrame too (CPU == GPU), then
   CudaMDegrainN tr <= 6 with SkipFrame can use it.
3. CudaMAnalyse multi=true: refusal at MVAnalyse.cpp ~512 without a technical reason found (gpu_search takes nsrc /
   nref, geometry the same for all deltas); lift and test interleaved output vs CPU.
4. CudaMDegrainN tr > 6: MVDegrainX fixed to 6 (ctor args, NORMWEIGHTS<1..6>, MAXR arrays in mvcuda_comp.cu,
   with_inputs <= 16 inputs); kernel loops over L at runtime. Needs a GPU path in MDegrainN. Optional.
5. CudaMAnalyse predictor clips: pr supers via super_input / with_inputs (in[2..3]), device super per clip
   (frame_super); slice kernel k_search<TM>: second / third source block in dynamic shared memory next to s_y/s_u/s_v
   (issue_src), zero / global candidates and their refines (trymany_par start states st_x/st_c[0..1], dedup key must
   include the source; sequential mx/mc; final refine via AnaChecker with swapped s_y/s_u/s_v). Refuse (D8) for now:
   meander=false (k_wave: g_wsrc hard-wired) and k_search_fix.

## Tests (new cases in all suites)

- CPU: arguments off == gpu_work md5 (whole run_m249_cpu-style set); on: deterministic, mt=true/false.
- GPU == CPU: run_s234 cases with predictor clips (trymany on / off, meander on / off, chroma, pel 1/2/4, 8 / 16 bit),
  run_partm MDegrainN SkipFrame, his full script CPU vs GPU.
- VS compare.py case.

## Progress

- [x] CPU port (MAnalyse predictor clips + pglobal_coarse, MDegrainN SkipFrame). Tests: arguments off == gpu_work in
  run_m249_cpu (82 / 84, the 2 = upstream MBlockFps edgepad crash); profile\run_predclip_cpu.ps1 24 / 24
  (deterministic, differs from the chain without clips; 8 / 10 bit, trymany, mt, luma, pel 4, meander off).
  CudaMAnalyse with predictor clips / pglobal_coarse != pglobal and CudaMDegrainN with SkipFrame refuse (D8) for now.
  Note: CudaMAnalyse also refuses multi=true, which his chain uses for the zero / global clips.
- [x] CUDA steps 1-3 (f7c5b31): pglobal_coarse per level (API 34), MDegrain1-6 class with skip_frame (CPU + GPU,
  MDegrainN tr <= 6 with SkipFrame uses it again), CudaMAnalyse multi=true. run_s234 S8 15 / 15, run_partm 66 / 66
  (degrainN_skip / _skip2).
- [x] CUDA step 4: CudaMDegrainN tr > 6 - own GPU path in MDegrainN (same kernel as MDegrain1-6, MVCUDA_MAX_DEGRAIN 10,
  with_inputs up to 24), weights from MDegrainN's own use_block / norm_weights (generic MDegrainN == MDegrain1-6 class
  on luma, tr 3 / 6, checked), lite supers in the CPU path (CompleteLite). run_partm 75 / 75 incl. tr 10 + SkipFrame
  and tr 8 + thSAD2.
- [x] CUDA step 5: CudaMAnalyse predictor clips in the wavefront kernel (meander=false; meander=true refuses, D8). The
  clip's block is loaded into g_wsrc while its candidate is measured / refined, then the source block again (no extra
  shared memory; the clip planes per level in WaveCall, not in LevelArgs); predictor clips force the general (GEN)
  kernel; trymany: the clip refines after trymany_par replace slot 0 / 1, then the selection again. Tokens + SADadj in
  MvCudaAnalyseParams (API 34). The predictor supers need the super clip's rfilter on the GPU.
  Tests: run_s234 S9 24 / 24 (GPU == CPU: zero / zero+global, trymany, trymanyall, hex, luma, auto SADadj with
  pglobal_coarse; meander=true refused), full run_s234 243 / 243, run_partm 75 / 75, run_frc_pel4 6 / 6,
  run_cf_regression 86 / 86.
- [x] End to end (profile\run_predclip_dtl.ps1): DTL's a.32 release script (MDegrain3, zero clips tr 10, global clips tr
  6, pglobal_coarse=0 instead of his pglobal < -256) on his DV clip, progressive (SeparateFields: odd deltas get a field
  shift, which the GPU search has always left to the CPU), 50 frames: CPU == GPU md5. Times: CPU 14.0 s / GPU 19.2 s
  without Prefetch, CPU 4.1 s / GPU 6.8 s with Prefetch(4) - the GPU chain is slower on SD here; not profiled yet.
- [x] Profile 2026-11-26 (run_predclip_dtl.ps1 -Variant, DV clip progressive, Prefetch(4), 300 frames, RTX 5090; all
  variants CPU == GPU):

  | variant | CPU s | GPU s |
  |---|---|---|
  | full (his script: trymany, predictor clips) | 23.0 | 40.9 |
  | nopr (same MDegrain3 chain without clips: no generation) | 10.1 | 17.2 |
  | cheappr (clips = the super itself: search with clips, no generation), 50 frames vs nopr | 1.7 vs 1.8 | 3.4 vs 3.3 |
  | nopr, trymany=false | 5.9 | 3.3 (~91 fps) |
  | nopr, meander=true mt=true (slice kernel) | 5.9 | 15.4 |
  | nopr, trymany=false, meander=true | 5.0 | 5.6 |
  | full, trymany=false | 18.3 | 28.2 |

  nsys (full, 100 frames): k_wave GEN (6 main MAnalyse, trymany + clips) 54 % of the kernel time, 36 ms per call;
  k_wave fast (the generation's multi MAnalyse with forced zero / global vectors) 44 %, 5.3 ms per call; k_blocks
  (MDegrain / MDegrainN tr 10) 1.4 %; super builds ~0.3 %.
  Conclusions: the predictor clips in the search cost ~nothing; the GPU loses on (1) trymany (36 ms per SD call, the
  trymany wavefront / slice paths), (2) the generation's searches for forced vectors (~25 s of 28 s with trymany off).
  Built-in generation (item 23) removes (2); (1) is a separate GPU trymany optimisation (QTGMC trymany presets too).

## Next (user decisions 2026-11-26)

1. [x] DONE - field shift on the GPU (LevelArgs.fshift, level 0 only; MvCudaAnalyseParams.field_shift, API 34): zero
   vector (0, fshift) in the zero candidate, the default predictors, the trymany start states (trymany_par and the
   sequential form) and the predictor clip anchors; global predictor y + fshift; the zero candidate's chroma SAD from an
   extra (0, 0) candidate (8th phase A candidate) as the CPU's ChromaSAD(0, 0); k_search_fix is not used on a level with
   a field shift (the general slice kernel takes it). MVAnalyse calls the GPU for fieldShift != 0. Tests: run_s234 S10
   36 / 36 (separated fields, odd deltas, every kernel, trymany, hex, pel 1 / 4, global=false, multi, predictor clips;
   CPU frames 0), full run_s234 279 / 279, run_partm 75 / 75, run_frc_pel4 6 / 6, run_cf_regression 86 / 86.
   Was: MAnalyse on SeparateFields clips (odd delta: zero vector y = fieldShift at the finest level,
   global predictor y + fieldShift, the default predictors and the "not the zero vector" checks of the searches) in the
   slice and the wavefront kernel, so such frames no longer fall back to the CPU (gpu_search requires fieldShift == 0).
2. trymany on the GPU: 36 ms per SD call (wavefront and slice kernel) - find the cause, speed it up.
   Diagnosis 2026-11-26 (profile\run_tm_speed.ps1: one CudaMAnalyse, DV clip, 100 frames; dev build of mvtools_cuda.dll
   with -DMVC_PHASE_PROF via Sources\cuda\build_dev.ps1 -Flags:-DMVC_PHASE_PROF, its unit list fixed for nvenc):
   8x8 / ov 4: no trymany 7.2 ms per frame, trymany (coarse levels) 20.3, trymanyall 38.8; 16x16: 4.7 / 8.1.
   Wavefront phases (cycles per block): refine phase 53k-114k with trymany vs ~5k without (growing with the level
   size: co-resident CTAs), all of it in trymany_par's eval (37k-104k; starts / syncs / fold / select ~10k): the
   exhaustive batch of every distinct start vector's 24 ring candidates (radius 2) in eval_multi groups of 4 per warp.
   Tried: groups of 8 (eval_multi<8>, half the rounds): no gain (19.4 vs 20.3 ms), reverted - the rounds are not the
   limit. Next ideas: the reference window of each distinct start in shared memory (as exhaustive_fast of k_search),
   or all 7 Refines' candidates as one CTA-wide batch; needs a closer look at eval_multi_pk's memory pattern first.
3. Built-in predictor clip generation, the simpler variant (not bit-identical to his MDegrainN chain): MAnalyse computes
   the predictor block itself. Proposal: zero clip = plain average of the current frame and its neighbours n-r..n+r at
   the same position without the reference frame of the delta (sum once per frame, minus the reference); global clip =
   the same along the global vector to each neighbour (source of those vectors still to decide). The separate-filter
   form (super_przero / super_prglobal, SkipFrame) stays.
