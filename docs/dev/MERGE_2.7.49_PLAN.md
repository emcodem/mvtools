# Merge of upstream pinterf/mvtools 2.7.49 into gpu_work

Started 2026-11-25. Worktree `C:\dev\mvtools\wt_merge249`, branch `merge-2.7.49` (from gpu_work 5924e89), merging
`upstream/mvtools-pfmod` 2a699ae (21 commits, 2026-11-12..14, all by pinterf). Base of the previous merge: 2.7.48
(a74dcb6, merged 2026-11-02, see REBASE_PLAN.md).

## What 2.7.49 brings

| Change | Output |
|---|---|
| Fixes: MScaleVect divide>0 crash (#70), MShow out-of-frame writes, MDepan with mask not deterministic, MFlowBlur halo offset, MSuper rejects hpad / vpad not a multiple of the chroma subsampling | only in those cases |
| New: MSuper `edgepad` / `edgepadV` (#28): one extra block column / row over the right / bottom edge, processed by all vector filters | unchanged without edgepad |
| Internal: super params in a process-wide registry (`SuperParams.h`), index in `vi.sample_type`; the legacy 8 bytes are still written | none |
| MAnalyse `temporal=true` fixed (finest level had a zero temporal predictor) + vectors of the last 8 frames cached | changes with temporal=true |
| MAnalyse `mt=true`: badcount per slice, same output in every run | can change with mt=true |
| MSuper: faster padding | none |

## Decisions

- edgepad on the GPU (user, 2026-11-25): the D8 rule, like every other unsupported option. A GpuCheck need in
  CudaMSuper, CudaMAnalyse, CudaMRecalculate, CudaMDegrain1-6, CudaMCompensate, CudaMFlowFps: error 'edgepad is not
  supported' at creation, the CPU path with `global cuda_cpu_fallback = true` / CUDA_CPU_FALLBACK=1. Vector filters
  see edge-block vectors through `MVFilter::_edge_blocks`. GPU support for edgepad = separate later item.
- MScaleVect: take upstream's version, drop our own divide>0 fix (73af318), same bug.

## Conflicts (trial merge: 18 files)

1. Super params, 13 files (1-2 hunks each): our `SuperParams64Bits` reads -> upstream `GetSuperParams(vi)`. Our bits
   1-7 of `param` (rfilter / sharp / pelclip / lite / ours, SuperParams64Bits.h) survive: `SuperParams::param` is an
   int carried through the registry and written to the legacy 8 bytes. Helpers (`super_param_valid`,
   `MV_REJECT_LITE_SUPER`, ...) get overloads for `SuperParams`. GpuBackend reads through `GetSuperParams` too.
   Check: the registry index in `vi.sample_type` survives token frames, lite supers and vsbridge.
2. `MVSuper.cpp/.h`: constructor gets both `edgepad, edgepadV` and our `gpu_flag`; `param = super_param_make(...)`
   plus the edge pad fields; CudaMSuper lite needs `edgepad == 0`.
3. `Interface.cpp`: `edgepad` / `edgepadV` for MSuper and CudaMSuper.
4. `MScaleVect.cpp`: upstream.
5. `PlaneOfBlocks.cpp`, `MVAnalyse.cpp`, MFlow*, `mvtools2.html`: includes / comments / golden log, keep both.

## GPU follow-up

- edgepad: CPU fallback in all Cuda* filters (CudaMSuper, CudaMAnalyse, CudaMRecalculate, CudaMDegrain1-6 / N,
  CudaMCompensate, CudaMFlowFps, other GpuBackend users that take a super clip).
- mt=true badcount per slice: no GPU change needed. The slice kernel never applies badcount: any block above badSAD
  sets `bad_flag` and the level goes to the CPU (MVCUDA_NEEDS_CPU), whose result is now the same in every run; the
  one-slice wavefront kernel counts over the whole plane (= one CPU slice).
- temporal=true: GPU rejects it already (CPU fallback), nothing to do.
- MSuper padding: CudaMSuper parity suite must stay bit-identical.

## Tests (dev: Win AVS + 5090; new cases go into all suites)

1. Build MSVC + CUDA (nightly arch set).
2. CPU vs upstream 2.7.49 built as is: same md5, with and without edgepad.
3. Before vs after the merge, default params: same md5 (allowed: mt=true with many bad blocks, temporal=true).
4. GPU vs CPU parity suite + new edgepad cases (expect CPU fallback, same md5 as CPU).
5. QTGMC + UConvert end to end: md5 and speed.

Then a merge commit (emcodem). Push only on request.

## Progress

- [x] conflicts resolved, builds (build_merge249, sm_120, 96 s); MBlockFps: upstream's edge-block bounds (nBlkXE / nBlkYE)
  ported into our threaded compensate_overlap_slice; GpuBackend / CompleteLite read through GetSuperParams
- [x] edgepad GpuCheck needs in the Cuda* filters
- [x] mt=true badcount on the GPU: nothing to do (see above)
- [x] test 1: CMake builds - merged (build_merge249, sm_120), upstream 2.7.49 as is (wt_up249 with our CMake files
  copied in, GPU / VS / plugins off: build_up249), pre-merge gpu_work 5924e89 (-NoCuda: build_pre249)
- [x] tests 2 + 3: `profile\run_m249_cpu.ps1`, 28 cases x 3 formats (422, 420, 422p10), 82 / 84 PASS:
  - merged == upstream in every case that runs, edgepad cases included
  - merged == pre except vec_bad_mt (per-slice badcount, expected)
  - FAIL edge_blockfps 422 and 420: MBlockFps with an edgepad super crashes (0xC0000374 heap corruption) in
    **upstream 2.7.49 too**, 8 bit only (422p10 runs). Upstream bug, not from the merge; report to pinterf (ask user)
  - test gap: temporal=true gives the same output as temporal=false on this MXF (also with lambda 4000 / trymany),
    so the upstream temporal fix is not exercised; needs material where the temporal predictor wins
- [x] GPU vs CPU `run_s234` (old cases): 195 / 195
- [x] `run_s234 -Only s7` (edgepad) 9 / 9, `run_partm` (+ frc_edge / degrain_edge) 57 / 57, `run_frc_pel4` 6 / 6,
  `run_cf_regression` all PASS
- [x] MBlockFps edgepad crash narrowed (bottom block row + real motion; upstream as is): issue draft
  `profile\out\m249_bfps\ISSUE_DRAFT.md`, the user posts it
- [x] test 5a QTGMC (`run_qtgmc_full.ps1`, one call per preset; seg1_wires_pole, 40 frames, Prefetch 4): Medium and
  Slower CPU == GPU in both sets, merged == pre (build_pre249g, sm_120) for CPU and GPU md5
- [x] test 5b UConvert: matrix generated (gen_matrix.py, 120 cases, UCONVERT_TEST_DATA = profile\uconvert_data;
  run_matrix.py needs numpy: Python 3.14), `profile\run_m249_uconvert.ps1` stages each set and restores the staged
  plugins: GPU chain 120 / 120 with pre and with merged
- [x] merge commit on merge-2.7.49 (not pushed)
- MBlockFps edgepad crash: root cause found later - the scene-change blend branch of MVBlockFps::GetFrame blends
  chroma with the local nHeightUV / nWidthUV of the extended grid (shadowing the members), writing past dst. Fix and
  repro in the issue draft; left as upstream in this merge.
