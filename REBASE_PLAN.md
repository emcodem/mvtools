# Bringing gpu_work up to upstream mvtools-pfmod

Started 2026-11-01 on branch `gpu_work_rebase` (from `gpu_work` f9553c5). `gpu_work` itself stays untouched until
the result is tested.

## Situation

- Upstream: `https://github.com/pinterf/mvtools` branch `mvtools-pfmod`, head a74dcb6 (2.7.48, 20261029), fetched as
  `upstream/mvtools-pfmod`. Our fork's `origin/mvtools-pfmod` is still at the merge base cdf04a5 (2026-04-24).
- Upstream has 97 commits that we don't, `gpu_work` has 55 that upstream doesn't. 149 files changed upstream, 304 on
  our side, 36 on both.
- Main upstream changes that matter to us: YUY2 removed from the filters (converted to YV16 at the filter boundary,
  `Yuy2Planar.h`, `yuy2planes.cpp/.h` deleted); lsb (stacked) kernels removed (`lsb=true` → out16 + conversion);
  MDegrain `out32` parameter removed; MAnalyse `search_coarse` / `trymanyall`; MCompensate `cclip` fix; AVX2
  interpolation and SATD; RB2F 16 bit SSE2, 8 bit SAD (block width 6 / 12 / 24) and SSE2 vector upsizing fixes;
  aarch64 NEON; ARM64 llvm-mingw build; DePan motion header reserved field zeroed; SSE2 base for GCC / Clang.

## Trial merge (2026-11-01, aborted again)

`git merge --no-commit upstream/mvtools-pfmod` on `gpu_work_rebase`: 23 files with conflicts, the rest merges
cleanly (including all 23 files that only we added, and `vsbridge/`, `cudafilters/`, `Sources/cuda/`).

| File | Hunks | Conflict lines | Kind |
|---|---:|---:|---|
| `CMakeLists.txt`, `Sources/CMakeLists.txt` | 3, 3 | 109, 94 | build: we rewrote, upstream added |
| `DePan/CMakeLists.txt`, `DePanEstimate/CMakeLists.txt` | 2, 2 | 92, 92 | build |
| `README.md` | 7 | 298 | docs |
| `Documentation/mvtools2.html` | 2 | 19 | docs |
| `Sources/SimpleResize.cpp` | 5 | 139 | code: our GPU / slicing vs upstream SSE2 fix |
| `Sources/MVBlockFps.cpp` | 1 | 117 | code: we moved the block loop into slices, upstream removed YUY2 in it |
| `Sources/MVDegrain3.cpp`, `.h` | 4, 3 | 51, 22 | code: GPU path vs lsb / out32 / YUY2 removal |
| `Sources/MVSuper.cpp`, `.h` | 3, 2 | 35, 8 | code: GPU lite super vs YUY2 flag |
| `Sources/Interface.cpp` | 6 | 33 | parameter lists (see below) |
| `Sources/MVCompensate.cpp`, `.h` | 2, 1 | 15, 5 | code |
| `Sources/MVFlowFps.cpp`, `.h` | 2, 2 | 12, 6 | code |
| `Sources/MDegrainN.cpp`, `MVMask.cpp`, `GroupOfPlanes.cpp` | 1 each | 5-6 | code |
| `Sources/MVAnalyse.cpp`, `.h`, `SuperParams64Bits.h` | 1 each | 3 | code |

Leftovers of removed upstream code after the merge (`yuy2planes`, `DstPlanes`, `SrcPlanes`, `lsb_flag`, `out32`) are
all in the conflicting files, so resolving the conflicts takes care of them.

## Issues git does not report

1. **`SuperParams64Bits::param` collision.** Upstream: bit 0 = `SUPER_PARAM_YUY2_SOURCE` (read in `Interface.cpp`
   to convert MShow's output back to YUY2). Ours: `0x80 | rfilter (bits 0-2) | sharp << 3 | pelclip << 5 | lite << 6`.
   After a plain merge, every super clip with an odd `rfilter` would count as "YUY2 source". All 8 bytes of the
   struct are used, so the bits have to be redistributed: see D4. Checked 2026-11-01: neither `vsbridge/` nor
   `cudafilters/` decode the bits (the bridge copies the 8 bytes unchanged), so only `Sources/` changes.
2. **Positional parameter order.** Upstream added `search_coarse`, `trymanyall` to MAnalyse at the index where our
   `gpu` is (args[33]). `gpu` moves to the end of each list (after upstream's parameters) so that positional calls
   written for upstream mvtools2 still mean the same. MDegrain1-6: upstream dropped `out32`, so our `gpu` index
   shifts. The VapourSynth bridge takes its signatures from `AddFunction`; check that nothing in `vsbridge/` or the
   test scripts hardcodes indices or the removed `out32`.
3. **New CPU features on the GPU path.** `search_coarse != 3` and `trymanyall=true` are not in the CUDA kernels:
   until ported, `gpu=true` with them must fall back to the CPU (same rule as other unsupported settings, see
   `Documentation/gpu.md`), never give different vectors silently. Same check for the SATD block-size rule (dct 5-10 needs a
   block size divisible by 4), which upstream now enforces at creation.
   **Done 2026-11-02 (API 19):** both options are on the GPU, bit-exact (GPU_PIPELINE_PLAN.md section 32).
4. **lsb=true with gpu=true.** Upstream turns lsb into out16 + stacked conversion. Our GPU MDegrain either supports
   out16 and gets the same conversion, or falls back; decide when merging `MVDegrain3.cpp` / `MDegrainN.cpp`.
5. **MCompensate `cclip`.** Upstream fixed `cclip` being read from the `thSAD2` argument. Check what our GPU
   MCompensate does with `cclip` (it may have copied the bug or ignore cclip).
6. **Reference changes in CPU output.** Upstream's kernel fixes (RB2F 16 bit SSE2, SAD width 6 / 12 / 24, SSE2
   vector upsizing with negative vectors in the last columns) change CPU output for some settings. Our GPU kernels
   were made to match the old CPU output; where the old CPU SIMD was the buggy one, GPU and new CPU should now agree;
   where the GPU copied a SIMD bug, it has to be fixed too. AVX2 interpolation / SATD are meant to equal the C code.
7. **DePan reserved field.** Upstream zeroes the reserved field of the motion data header; relevant for MDepan under
   VapourSynth (still open in `VAPOURSYNTH_PLAN.md`).
8. **Upstream changed CPU output on purpose; our CUDA kernels copy the old behaviour.** Found in step 0 (below). Two
   2.7.48 fixes change results with default settings: **#59 chroma shift** (luma vector → chroma grid rounds
   symmetrically now, in MAnalyse / MRecalculate chroma SAD and in every consumer: MDegrain1-6, MDegrainN,
   MCompensate, MBlockFps, MFlow, MFlowInter, MFlowFps, MFlowBlur; 4:2:0 / 4:2:2 / 4:1:1 only, `chroma=true` is
   MAnalyse's default) and **overlap windows summing to exactly 2048** (MCompensate, MDegrain, MBlockFps with
   overlap). After the merge, GPU = CPU fails for subsampled clips and for overlapped compensation until both are
   ported to the CUDA kernels (`Sources/cuda/`: the chroma vector conversion in the search's chroma SAD and in the
   compensation / degrain / flow kernels; the window table). Also check the rest of CHANGELOG 2.7.48 for kernels we
   mirror: float overlap (#69), MDegrain limitC for float, MFlowBlur chroma vpad, the vector upsizer (negative
   vectors in the last columns), dct=1..4 rounding, MSuper with odd chroma size.
9. **avstp changes MAnalyse output.** With `avstp.dll` next to mvtools2, MAnalyse / MRecalculate search in slices
   (`min(threads, 64, blkY / 4)`), and the vectors at slice borders differ from the one-slice search. Builds may only
   be compared with the same libraries in the plugin folder; `baseline.py` writes `plugins.txt` and `diff` warns.
   (Also a GPU = CPU question: which slicing does the GPU path match? Our baseline shows GPU = CPU with avstp.)

## Decisions

- **D1: merge, not rebase.** One merge commit of `upstream/mvtools-pfmod` into `gpu_work_rebase`. Replaying our 55
  commits would hit the same files again and again (`MVFlowFps.cpp`, `MVDegrain3.cpp`), each time with a conflict.
  `git rerere` is enabled for the repo. (Open: if a linear history is wanted later, squash into groups first.)
- **D2: upstream wins for CPU code, ours is re-applied on top.** In every conflicting filter file, take upstream's
  version and add our GPU hooks to it again; don't keep our copy of code upstream deleted (YUY2, lsb, out32).
- **D3: CMake is our only build; the Visual Studio solution stays exactly upstream's.** (Decided 2026-11-01.) We
  build and test with CMake only (`build_windows.ps1` / `build_linux.sh`), so nothing has to be tested twice.
  `mvtools.sln`, `mvtools.slnx`, every `*.vcxproj(.filters)` and upstream's `.bat` build helpers are taken from
  upstream unchanged in every merge (`git checkout upstream/mvtools-pfmod -- <file>`), our earlier vcxproj edits are
  dropped; they then build only upstream's CPU mvtools2 (our new sources are not in them), which is fine. Not
  deleted: deleting them would make every upstream change to them a conflict. CMake files: ours win, plus
  upstream's additions: llvm-mingw ARM64 toolchain (`cmake/toolchains/`), `ENABLE_NEON_SIMD` and the ISA filters of
  upstream's `Sources/Files.cmake` (taken as upstream has it), version resource for MinGW builds,
  `cmake_minimum_required(3.8.2...3.31)` style range. Upstream's "use the sln for MSVC" guard is not taken. README /
  CMake messages that point to the solution for clang-cl / 32 bit are reworded (not supported by our build).
  Step 0 showed our CMake files build upstream's sources unchanged (only `-DMVTOOLS_GPU=OFF` needed).
- **D4: `param` layout (issue 1), keeping the 0x80 marker.** The marker says "written by our MSuper, the other bits
  are real". It was added because the original MSuper never wrote `param` (uninitialised stack bytes); upstream now
  always writes it, but a super clip from another / older mvtools2 DLL in the same script can still carry garbage,
  and garbage that looks like "rfilter 2 / 4, sharp 2, no pelclip" would let the GPU rebuild the subpixel planes with
  the wrong filter: different vectors without an error. Nine bits are needed (upstream's YUY2 flag + ours), so rfilter
  (0-4) and sharp (0-2) share one 4-bit code (`rfilter * 3 + sharp`, 15 values):

  | Bit | Meaning |
  |---|---|
  | 0 | `SUPER_PARAM_YUY2_SOURCE` (upstream, unchanged) |
  | 1-4 | `rfilter * 3 + sharp` |
  | 5 | pelclip |
  | 6 | lite (`MSuper(gpu=true)`, picture only) |
  | 7 | marker: written by our MSuper |

  Valid only when bit 7 is set and the code is < 15. All reads and writes go through helpers in
  `SuperParams64Bits.h` (`super_param_make`, `super_valid`, `super_rfilter`, `super_sharp`, `super_pelclip`,
  `super_lite`); the hand-written decoding in `MVAnalyse.cpp`, `MVRecalculate.cpp`, `MVFlowFps.cpp`,
  `MVSuper.cpp` (`CompleteLite`) and `MV_SUPER_IS_LITE` goes away. Upstream's own reader (`param &
  SUPER_PARAM_YUY2_SOURCE`) keeps working unchanged. Later: ask upstream to reserve bits 1-7 (or bit 7 as "extended
  layout") so a future upstream flag does not collide again.

## Steps

0. Before the merge: run all tests on `gpu_work` and keep the results (`vsbridge/test/*.md5`, compare scripts, the
   17 CPU / 7 GPU / 17 cudafilters chains, the GPU = CPU chains). Also build plain upstream a74dcb6 and run the CPU
   chains with it: after the merge, our CPU output must equal upstream's, and our GPU output must equal our CPU.
   Add `upstream` as a remote permanently (`git remote add upstream https://github.com/pinterf/mvtools`).
1. Merge, resolve build files first (D3) so it compiles at all: `CMakeLists.txt` ×4, `Files.cmake`; solution,
   vcxproj and `.bat` files reset to upstream's.
2. `SuperParams64Bits.h` and `MVSuper.cpp/.h` (issue 1, D4).
3. `Interface.cpp` (issue 2), then the filters: `MVDegrain3` / `MDegrainN` (issue 4), `MVCompensate` (issue 5),
   `MVAnalyse` / `GroupOfPlanes` (issue 3), `MVFlowFps`, `MVBlockFps`, `SimpleResize`, `MVMask`.
4. Build Windows (`build_windows.ps1`, MSVC + Ninja, CUDA 12.9) and Linux (`build_linux.sh`); fix what the clean
   merges broke (auto-merged files that both sides changed: `MVFinest`, `MVFlow`, `MVFlowBlur`, `MVFlowInter`,
   `MVRecalculate`, `MVShow`, `PlaneOfBlocks`, `MVBlockFps.h`, `conc/Interlocked.hpp`).
5. Port upstream's output changes to the CUDA kernels (issue 8), then tests: CPU = upstream a74dcb6 (step 0); GPU = CPU; VapourSynth = AviSynth+ (`vsbridge/test/compare.py`,
   `cf_chains.py`, `stand_ins.py`, `mt_tables.py`); missing-library fallbacks (no `mvtools_cuda`, no NVOFA) still load
   and run on the CPU. Investigate every differing chain against issue 6 before accepting new references.
6. Docs: README (7 hunks) and `mvtools2.html` merged section by section; `Documentation/gpu.md` for the new fallback rules
   (issue 3); `VAPOURSYNTH_PLAN.md` todo list updated (its Windows entry is stale already).
7. When everything passes: fast-forward `gpu_work` to `gpu_work_rebase`; update `origin/mvtools-pfmod` to upstream.
   `gpu_portable` (Vulkan, development stopped): leave as it is unless it is still wanted.

## Status

- 2026-11-01: branch created, trial merge done and aborted, plan written; D4 decided (marker kept, combined
  rfilter / sharp code); repo git identity set to emcodem. Next: step 0.
- 2026-11-01, step 0 (ours): `vsbridge/test/baseline.py` (AviSynth+ framemd5 of compare.py's chains into a folder;
  `diff` between two folders; GPU chains checked against their twins without `gpu=true`). Baseline of `gpu_work`
  f9553c5 (build `C:\dev\mvtools\profile\build_rel20\bin`, Windows, RTX 5090) in
  `C:\dev\mvtools\profile\baseline\gpu_work_f9553c5`: 50 chains (20 CPU, 6 GPU CPU twins, 7 GPU, 17 cudafilters),
  all run; GPU = CPU for all 6 GPU chains and lin_gpu = lin_cpu; a second run (`..._run2`) is identical in every
  chain.
- 2026-11-01, step 0 (references): upstream's CMake refuses MSVC ("use the sln"), and the solution needs NASM
  integration per toolset, so upstream a74dcb6 and the merge base cdf04a5 were built with **our** CMake files copied
  into worktrees (`C:\dev\mvtools\wt_upstream`, `wt_base`; `-DMVTOOLS_GPU=OFF`): same flags, only the sources differ.
  Results (all with `avstp.dll` in the plugin folder, issue 9):
  - base cdf04a5 (MSVC 19.44) = gpu_work in all 25 CPU / gpucpu chains: our 55 commits do not change CPU output.
  - upstream a74dcb6 (MSVC 19.51, toolset not pinned by `-NoCuda`) differs in every 4:2:0 chain and in
    `gpu_flowfps@cpu` (4:2:2), equal in the 4:4:4 `flowfps`, `gpu_vectors_normalsuper@cpu`, `scdetection`: matches
    CHANGELOG 2.7.48 #59 + overlap window sum (issue 8). Rebuilt with `-VcvarsVer 14.44` (MSVC 19.44, as ours):
    identical to the 19.51 build in every chain, so the compiler plays no part. Reference for the merged CPU code:
    `C:\dev\mvtools\profile\baseline\upstream_a74dcb6_msvc1944`.
  - `lin_cpu` does not run on base / upstream (`mt=` in MFlowFps is ours); expected.
- 2026-11-01, steps 1-4: merge committed (17c5bb5). Issues 1-5 resolved in it: D4 layout with helpers (issue 1),
  `gpu` last in every signature (2), MAnalyse GPU off for `search_coarse != 3` / `trymanyall` (3), lsb with gpu=true
  falls back because the GPU MDegrain excludes out16 (4), GPU MCompensate never sees cclip (needs tr = 0) (5).
  Beyond the conflicts: MVBlockFps's sliced compensation got upstream's #59 `pos_to_chroma` (git took our side
  silently); SimpleResize combines upstream's per-pixel C limiter with our row ranges; the bridge needed the
  AVS_Linkage V13 / V14 entries for upstream's avisynth.h. A script compared every line upstream added in files both
  sides changed with the merged tree: only our deliberate signature / param changes differ.
  Tests (Windows, build `C:\dev\mvtools\profile\build_merge\bin` + avstp): merged CPU = upstream a74dcb6 in all 25
  chains (`baseline\merge_wip1`); AviSynth+ = VapourSynth in all 44 compare.py chains; GPU = CPU only for the 4:4:4
  chains (`gpu_vectors_normalsuper`, `gpu_compensate`), the 4:2:0 / 4:2:2 ones differ as predicted (issue 8).
  Not done yet: Linux build and tests. Next: step 5, port #59 to the CUDA kernels.
- 2026-11-02, step 5 (issue 8): upstream's #59 rounding ported to the CUDA kernels: `mvc::mv_to_chroma` /
  `pos_to_chroma` (`mvcuda_internal.h`, same formula as `commonfunctions.h`) in the search's chroma SAD
  (`mvcuda_analyse.cu`, every candidate path and the exhaustive window bounds), `mvcuda_recalc.cu`, the NVOF vector
  refinement (`mvcuda_nvof_vec.cu`), MCompensate / MDegrain1-6 (`mvcuda_comp.cu`) and MFlowFps's chroma vectors
  (`mvcuda_flow.cu`, as `VectorSmallMaskYToHalfUV`). One trap: `exh_runs` (exhaustive search, radius > 3) evaluates
  8 candidates `pel << log_x_uv` apart and took candidate k's chroma k samples after candidate 0's, which the new
  rounding breaks for a run of odd vectors crossing x = 0 (-1 -> -1, +1 -> +1); such runs now take the sums of
  their cx > 0 candidates from a second pass at cxa + 1. Without it every exhaustive 4:2:0 / 4:2:2 chain differed,
  so compare.py got `gpu_vectors_exh` (hd420p8, pel 2, searchparam 8; all old chains use radius 2).
  The other 2.7.48 fixes need nothing on the GPU: the overlap windows are the CPU's `OverlapWindows` uploaded as they
  are (the 2048 sum came with the merge); the GPU MSuper builds level 0 with its padding before reducing (= the odd
  chroma fix); the GPU vector upsizer uses signed weights (= the C path, which was right); the GPU has no float,
  dct != 0, MFlowBlur or MBlockFps paths (#69, limitC float, dct rounding, MFlowBlur vpad).
  Tests (build `build_merge\bin` + avstp, `baseline\merge_step5`): CPU = upstream a74dcb6 in all 25 chains; GPU = CPU
  in all 7 GPU chains and lin_gpu = lin_cpu; AviSynth+ = VapourSynth in all 45 compare.py chains. Ad hoc: exhaustive
  search radius 4 / 8 / 16, pel 1 / 2, hd420p8 and uhd422p10: GPU = CPU in all 12 (all 12 differ with the second
  pass disabled). The cf_* chains with vectors changed against `merge_wip1`, as expected.
  Not done yet: missing-library fallback test, Linux build and tests, step 6 docs (`Documentation/gpu.md`
  search_coarse / trymanyall fallback, `VAPOURSYNTH_PLAN.md` todo list).
- 2026-11-02, step 5 (missing-library fallbacks): plugin folder `build_merge\bin_nocuda` = `bin` without
  `mvtools_cuda.dll` (avstp kept). `baseline.py run` (`baseline\merge_nocuda`): all 52 chains run, GPU = CPU in all 7
  GPU chains and lin_gpu = lin_cpu, and every chain equals `merge_step5` except `cf_nvof` / `cf_nvof_degrain`
  (CudaNVOFAnalyse falls back to MSuper / MAnalyse, as documented). `MVTOOLS_GPU_LOG=1` shows "mvtools_cuda.dll not
  loaded from the plugin folder (error 126), using the CPU". compare.py cpu gpu cf without the library: AviSynth+ =
  VapourSynth in all 28 MVTools chains and in cf_roundtrip / degrain2 / nvof / nvof_degrain / rgb_*; the 11 cf chains
  whose CPU fallback is an AviSynth+ core filter (Merge*, Bob, resizers, TemporalSoften) stop under VapourSynth with
  the bridge's error (`MVBRIDGE_FALLBACK=strict`, the default, README), no hang or crash. NVOFA (System32, so tested
  with a stand-in `nvofapi64.dll` in the host's application folder, which Windows searches first): a damaged file
  (error 193) and a DLL without `NvOFAPICreateInstanceCuda` both log once and give exactly the no-`mvtools_cuda`
  output in cf_nvof / cf_nvof_degrain, AviSynth+ = VapourSynth with the damaged file.
- 2026-11-02, before step 7: `gpu_work` had one commit more than this branch (0288391: without a thread pool the GPU
  search uses 1 slice like the CPU, was 24), so it was merged in (3383de1; a merge, so that step 7 stays a
  fast-forward). The merge 17c5bb5 had stored `README.md`, `Documentation/mvtools2.html` (also without its BOM) and
  `Sources/MVDegrain3.h` with LF while both parents have CRLF (core.autocrlf normalises conflict resolutions): restored
  in 3383de1 / 3b1411e, so their diffs against upstream are only our changes again. `build_linux.sh` is executable in
  git now (was 100644 since f9553c5: `./build_linux.sh` failed on a fresh clone).
- 2026-11-02, step 6 (docs, 99a9228): README and `mvtools2.html` were already merged section by section in 17c5bb5
  (checked against every upstream change since cdf04a5); `gpu.md`: trymanyall / search_coarse != 3 / lsb fall back,
  the SATD block-size error stays an error, YUY2 reaches the GPU as YV16, the CPU reference is 2.7.48 (#59);
  `VAPOURSYNTH_PLAN.md` todo list. Checked ad hoc on Windows: YUY2 MSuper / MAnalyse / MDegrain1 / MFlowFps with
  gpu=true run on the GPU and equal the CPU; search_coarse=2, trymanyall=true, MDegrain lsb=true log "not supported
  here" and equal the CPU.
- 2026-11-02, final tests at 3383de1 (code unchanged since). **Windows** (`build_merge\bin`, avstp, RTX 5090):
  `baseline\merge_final` = `merge_step5` in all 52 chains, GPU = CPU in all 7 GPU chains, lin_gpu = lin_cpu; without
  avstp (`bin_noavstp`, `merge_final_noavstp`): GPU = CPU in all 7 (0288391), the slicing differences against the avstp
  run are issue 9; without `mvtools_cuda.dll` (`merge_final_nocuda`): as in the fallback test above. compare.py cpu gpu
  cf: AviSynth+ = VapourSynth in all 45 chains with and without avstp. VapourSynth against the stored references:
  cf_chains 19 / 19 (needs `nnedi3_weights.bin` in the plugin folder, `bin_vs`), stand_ins 22 / 22, mt_tables
  100 / 100; without `mvtools_cuda.dll` stand_ins and mt_tables pass, the cf_chains cases whose fallback is an
  AviSynth-only plugin stop with the bridge's strict-mode error (by design).
  **Linux** (WSL Ubuntu, GCC 15.2, CUDA 12.9 with g++-14 as host compiler, `build_linux.sh`, `~/build/merge_lin`):
  builds without errors (SSE2 base, `-mavx2` per file; no libcuda / fftw NEEDED); warnings only the known nvcc ones,
  DePan's unused fgets result (upstream) and a GCC `-Wstringop-overflow` false positive in `CudaNvof.cpp`
  (memset of the header into `GetWritePtr()`). Upstream a74dcb6 built with upstream's own CMake crashes in
  `AvisynthPluginInit3` under our AviSynth+ (`~/.local/lib/libavisynth.so.11`): its library exports all 3636 symbols,
  including `AVS_linkage`, which libavisynth exports as read-only data too, so the plugin's store goes to libavisynth's
  copy (ELF interposition) and faults; our build exports only the entry points (`cmake/plugin_exports.map`). An upstream
  issue, worth reporting (needs `-fvisibility=hidden` or a version script). Reference therefore built from a74dcb6 with
  our CMake files (`~/build/up2_lin`, as on Windows in step 0): merged CPU = upstream in all 26 chains (lin_cpu does not
  run upstream). Ours (`~/build/test/baseline/m_full`): all 52 chains run, GPU = CPU in all 7, lin_gpu = lin_cpu (1
  slice now); without `libmvtools_cuda.so` and with a truncated one (4 MB of 95 MB): identical to each other and to the
  full run except cf_nvof / cf_nvof_degrain (MAnalyse fallback), no crash. compare.py cpu gpu cf: AviSynth+ = VapourSynth
  in all 45 chains; without the library the 11 AviSynth-core-fallback cf chains give the strict-mode error, as on
  Windows. cf_chains 19 / 19, stand_ins 22 / 22, mt_tables 100 / 100 (without the library: as on Windows).
  Not tested: Clang (not installed in WSL), aarch64. Next: step 7 (fast-forward `gpu_work`, update
  `origin/mvtools-pfmod`) waits for the user.
