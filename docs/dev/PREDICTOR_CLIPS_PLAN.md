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
- [ ] CUDA MDegrainN SkipFrame (+ tr > 6?)
- [ ] CUDA MAnalyse predictor clips
- [ ] tests, build for DTL
