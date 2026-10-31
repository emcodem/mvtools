# NVOFA as a standalone filter: CudaNVOFAnalyse (cudafilters.dll) + uc_FRC option

Implementation plan, written 2026-10-31 for a new session. Background and measurements: [opticalflow.md](opticalflow.md)
(experiment plan + results) and [compare/nvofa_frc/README.md](compare/nvofa_frc/README.md) (speed, visual judgement,
vector analysis, cause of the exact search's failure).

## Handoff: state at the time of writing

Branch `gpu_work`, last commit `6bfdc14` (pushed). **Uncommitted** (all from the experiment):
- `Sources/cuda/mvcuda_nvofa.{h,cu}`: NVOFA session pool, nvofapi64.dll loaded at run time, preset from env var
  `MVCUDA_NVOFA=fast|medium|slow` (read once). Keep this module, it is reused.
- `Sources/cuda/third_party/nvof/nvOpticalFlow{Common,Cuda}.h`: NVIDIA Optical Flow SDK headers, API 2.0, BSD-3,
  from github.com/NVIDIA/NVIDIAOpticalFlowSDK (driver 616.92 reports API 5.0; 2.0 works).
- `Sources/cuda/mvcuda_analyse.cu`: the experiment hook (env var) in `analyse_dev` + kernels `k_of_gray`,
  `of_to_vec`, `k_of_blocks`, `k_of_coarse` (search for "MVCUDA_NVOFA experiment").
- build lists: `Sources/cuda/build.bat`, `build_dev.bat`, `build_prof.bat`, `build_dev.ps1` (unit `mvcuda_nvofa.cu`).
- `compare/nvofa_frc/` (README, scripts, img, `spots_check.csv` = the user's visual judgement), `opticalflow.md`.
- Outside the repo: `C:\dev\mvtools\profile\run_frc.ps1` got `-SetEnv NAME=VALUE,...`; test set
  `C:\dev\mvtools\profile\build_nvofa` (= build_m1 + the experiment's mvtools_cuda.dll).
- The previous session's scratchpad (vector dumps `vec_analyse_*.raw`, crop clips, the numpy venv) is gone:
  regenerate with `compare/nvofa_frc/scripts/render.ps1` / `render_vec.ps1` into a work folder; `prescreen.py` needs
  numpy (make a venv; the base conda env has none, and no Pillow).

Reference numbers (uc_FRC, `profile\avs\m\frc.avs.tpl`, UHD 4:2:2 10 bit, 112 frames, Prefetch 3, RTX 5090):
exact GPU search 6.80 s, md5 **BFE44775F811** (= CPU fallback); NVOFA medium via the hook 4.93 s, md5
**FA834251D140** (fast 6C85DE040D85, slow B8B830F31B09). NVOFA medium is the variant the user judged best.

**Step 0 of the implementation: commit the experiment as it stands** (one commit: hook + session module + headers +
build lists + compare/nvofa_frc + opticalflow.md + this file), so the standalone work starts from a recorded state.
Commit identity: this shell has no git user; previous commits used `git -c user.name=emcodem -c
user.email=emcodem@ffastrans.com` (the repo's author). End commit messages with the Co-Authored-By line.

## Context
The experiment replaced MAnalyse's search via an env var: uc_FRC 6.80 -> 4.93 s, and in the user's visual check
NVOFA medium is the best variant overall (uc_FRC clearly worst; the exact search misses fast pans because of
truemotion's lambda). Decisions (user, 2026-10-31):
- NVOFA becomes a **standalone filter in cudafilters.dll**, not a MAnalyse parameter: MAnalyse stays bit-exact and
  unchanged (GPU_PIPELINE_PLAN.md rule 3), and NVOFA takes the plain clip, so no MSuper pyramid (`supf`) is built for
  the search (DTL's point on doom9). MRecalculate / MFlowFps still need their `levels=1` super.
- Preset as an int: 1 fast, 2 medium, 3 slow.
- uc_FRC: new global `uc_frc_nvof = 0` (off); set 2 after the Import to use it. Existing jobs stay bit-identical.

## Design
**Script API** (cudafilters.dll):
- `CudaNVOFAnalyse(clip c, bool "isb"=false, int "delta"=1, int "blksize"=16, int "blksizeV"=blksize, int "overlap"=0,
  int "overlapV"=overlap, int "pel"=2, int "hpad"=16, int "vpad"=16, bool "chroma"=true, int "preset"=2)` -> a vector
  clip in MAnalyse's format (accepted by MRecalculate, MFlowFps, MDegrain, MCompensate). pel 1 or 2. hpad / vpad must be
  those of the MSuper used downstream (header check `nWidth + 2*hpad == super width`).
- `CudaNVOFAvailable()` -> bool (library loaded, device has NVOFA, session can be made); lets scripts choose a branch.

**Vector clip** (cudafilters side, new `cudafilters/CudaNvof.cpp`):
- Include `Sources/MVAnalysisData.h` (shared struct layout, `#pragma pack(16)`, 92 bytes; version 5, magic 0x564D)
  and `Sources/AnaFlags.h`. Fill: magic, version, blk sizes, pel, **nLvCount 1** (MRecalculate's own output uses 1
  level, `MVRecalculate.cpp:257`; MDegrain / MFlowFps accept it), delta, isBackward, nFlags (`MOTION_IS_BACKWARD` 0x40
  if isb, `MOTION_USE_CHROMA_MOTION` 0x800 if chroma), width / height, overlaps, nBlkX / nBlkY with MAnalyse's formula
  incl. its overlap-0 fallback (`MVAnalyse.cpp:238-252`), pixelType, x/yRatioUV, chromaSADScale 0, pixelsize,
  bits_per_pixel, h/vpad. Block sizes: MAnalyse's allowed list (`MVAnalyse.cpp:165-178`).
- VideoInfo as `ClipFnc::format_vector_clip(vi, true, ..., "rgb32", 256 + 4*words)` does (`ClipFnc.cpp:77-138`): BGR32,
  width = ceil(bytes/4), height 1, audio_samples_per_second 0; plus the pointer hack in `nchannels` / `sample_type`
  (`MVAnalyse.cpp:520-542`, x64 form: `nchannels = 0x80000000 | (p >> 32)`, `sample_type = p & 0xffffffff`) pointing at
  the filter's own MVAnalysisData member (valid for the filter's lifetime). MVClip (`MVClip.cpp:25-76`) decodes it and
  throws if nchannels is 0..8 or the magic is wrong; on the first GetFrame it re-reads the struct from the frame
  header (`MVClip.cpp:150-173`, checks magic and version 5).
- Frame: int 256, the struct, zero padding to byte 256, then the payload `[3+3N, 1, 1+3N, (vx, vy, sad) * N]`
  (vectors in 1/pel pixel). Frames without a reference (n - delta < 0 / n + delta >= frames): validity 0, vectors 0,
  sad = 3*blkX*blkY << bits (as MAnalyse, `PlaneOfBlocks.cpp:96`).
- Downstream checks to satisfy: `MVFilter::CheckSimilarity` (`MVFilter.cpp:60`: width, height, block size, pel, overlap,
  ratios, pixelsize, bits); MFlowFps `nWidth + 2*nHPadding == super.width` (`MVFlowFps.cpp:117`, `:136`); MDegrain
  `nWidth == superWidth - 2*superHPad` (`MDegrainN.cpp:691`, `MVDegrain3.cpp:403`); MRecalculate reads nWidth, nHeight,
  pixelType, nDeltaFrame, isBackward and checks them against its super (`MVRecalculate.cpp:124-208`). MRecalculate
  recomputes the SAD of every incoming vector (`PlaneOfBlocks.cpp:3358-3362`), so for uc_FRC the SAD only matters for
  direct MFlowFps / MDegrain use.
- Inputs: src = frame n, ref = n + delta (isb) / n - delta, uploaded keyed by (clip, n) so each picture is uploaded once
  (add `mvcuda_frame_upload_keyed` to `CudaLib`, as mvtools2's `GpuBackend::frame_key`, `GpuBackend.cpp:191-229`);
  token inputs used directly (`input_token` / `with_inputs`, `cudafilters/CudaCommon.h:61-177`).
- Fallback (library / NVOFA unavailable, or a call fails): the exact `MAnalyse(MSuper(c, pel, hpad, vpad), isb, delta,
  blksize, overlap, truemotion=true, chroma)` built in the constructor with `env->Invoke` (pattern: CudaMerge's `_cpu`,
  `cudafilters/CudaMerge.cpp`), per frame; documented as such. `SetCacheHints`: MT_NICE_FILTER.

**Library** (mvtools_cuda.dll, API 16 -> 17):
- New exports in `mvcuda.h`: `int mvcuda_nvof_token(const MvCudaNvofParams *p, unsigned long long src_token,
  unsigned long long ref_token, int *payload, int payload_words)` (writes the 1-level payload above) and
  `int mvcuda_nvof_available(void)`. `MvCudaNvofParams`: width, height, x/y ratio, bits, chroma, chroma_sad_scale,
  blk_x/y, overlap_x/y, n_blk_x/y, pel, hpad, vpad, preset. Model: `mvcuda_analyse_token` (`mvcuda_analyse.cu`, token
  entry point near the end: argument check, `use_device()`, `frame_get` both tokens -> `MVCUDA_TOKEN_GONE`, worker).
- New unit `mvcuda_nvof_vec.cu` + `mvcuda_nvof_vec_8.cu` / `_16.cu` wrappers (pix-dependent, like
  `mvcuda_analyse_8/16.cu`): `k_of_gray` (from the **unpadded** DevFrame luma, `DevFrame::plane/pitch`,
  `mvcuda_internal.h:172-235`; 9..16 bit rounded to 8), NVOFA execute (`mvcuda_nvofa.cu` session pool; preset passed per
  call instead of the env var; sessions keyed by size and preset), `k_of_blocks` (median of the 4x4-grid flow cells
  under each block, x and y separately; round 1/32 px to 1/pel; clamp to `[-pel*x0, pel*(ext - x0 - blk))` with
  x0 = hpad + block origin and ext = width + 2*hpad; SAD luma + scaled chroma). SAD at half-pel from
  `mvc::frame_subpel(fr, hpad, vpad)` (`mvcuda_comp.cu:373-429`, the MSuper pel-2 Wiener planes, padded); the full-pel
  plane: DevSubpel's if it has one, else the unpadded DevFrame with clamped coordinates (= MSuper's edge padding).
  Issued with `mvc::leased_call` like `analyse_dev`.
- Remove the MVCUDA_NVOFA hook from `mvcuda_analyse.cu` (MAnalyse exact-only again); `k_of_gray` / `k_of_blocks` /
  `of_to_vec` move to the new unit (adapted from LevelArgs / super planes to DevFrame + DevSubpel); `k_of_coarse`
  goes (1 level only). `mvc_nvofa::perf_level()` (env var) is replaced by the preset argument; an env override for A/B
  tests is optional.
- Wiring: exports in `cuda/mvcuda.cu` (8/16 dispatch like `mvcuda_analyse_token`, `mvcuda.cu:187-197`); build lists
  (`build.bat`, `build_dev.bat`, `build_prof.bat`, `build_dev.ps1` unit map incl. header deps); `CudaLib` members +
  `CF_LOAD` + `ok` chain (`CudaCommon.h:19-53`, `CudaFilters.cpp:16-69`); registration (`CudaFilters.cpp:314-343`);
  `cudafilters.vcxproj` (new .cpp, include path to `..\Sources`). mvtools2's `GpuBackend` only needs the version bump
  (`GpuBackend.cpp:326-328` checks `MVCUDA_API_VERSION`; it does not call the new exports). All three DLLs must be
  rebuilt together (`profile\build_all_dev.ps1`).

**uc_FRC** (`C:\dev\uconvert\uconvert.avsi`, NOT a git repo: copy it to a backup first):
- `global uc_frc_nvof = 0` next to `uc_frc_gpu` (~line 66), with a comment: 1 fast / 2 medium / 3 slow, not
  bit-identical to the CPU, needs uc_frc_gpu and an NVIDIA GPU with NVOFA; falls back to the normal branch when
  CudaNVOFAvailable() is false.
- In `uc_FRC` (~line 393): `nv = uc_frc_gpu && uc_frc_nvof > 0 && CudaNVOFAvailable()`; when nv:
  `bv = CudaNVOFAnalyse(pre, isb=true, blksize=bs, overlap=bs/2, pel=2, hpad=16, vpad=16, chroma=true, preset=uc_frc_nvof)`
  (and fv with isb=false), no `supf`, `uc_frc_coarse_half` ignored; MRecalculate / MFlowFps unchanged. Input `pre`
  (RemoveGrain(22)), as in the judged experiment (the hook fed NVOFA level 0 of supf = pre). HD (pel 4): vectors at
  pel 2, MRecalculate rescales them to its pel-4 super (`PlaneOfBlocks.cpp:3328`) — verify; if a check refuses it, HD
  keeps the normal branch.
- cudafilters.dll must be loaded by uconvert: check how uconvert loads its dlls (`C:\dev\uconvert\dlls`); add a
  LoadPlugin if needed. CudaNVOFAvailable must only be called when cudafilters is present (guard like the existing
  uc_frc_gpu handling).

## Critical files
- new: `cudafilters/CudaNvof.cpp`, `Sources/cuda/mvcuda_nvof_vec.cu` (+ `_8.cu`, `_16.cu`)
- `Sources/cuda/mvcuda.h`, `mvcuda.cu`, `mvcuda_nvofa.{h,cu}`, `mvcuda_analyse.cu` (hook removed), build scripts
- `cudafilters/CudaCommon.h`, `cudafilters/CudaFilters.cpp`, `cudafilters/cudafilters.vcxproj`, `cudafilters/README.md`
- `Documentation/gpu.md` (API 17, the new filter), `GPU_PIPELINE_PLAN.md` (new section), `compare/nvofa_frc/README.md`
- `C:\dev\uconvert\uconvert.avsi`

## Verification
1. Build all three DLLs: `profile\build_all_dev.ps1 -Out C:\dev\mvtools\profile\build_nvof2`.
2. Exact paths unchanged: golden 85/85 (`cd profile\golden; $env:MVCUDA_TEST_DLL=<build>\mvtools_cuda.dll;
   python -I run_all.py 10 f8 f420 f420b8 f444`), `run_partm.ps1 -Dir <build>` 27/27 (run with pwsh; Windows
   PowerShell blocks scripts), `run_cf_regression.ps1`, `run_frc.ps1 -Dir <build> -Fallback` md5 BFE44775F811.
3. Same result as the judged experiment: chain `CudaNVOFAnalyse(pre, blksize=32, overlap=16, pel=2, preset=2)` ->
   MRecalculate -> MFlowFps (as `profile\avs\m\frc.avs.tpl`) must give md5 **FA834251D140** (MRecalculate reads only
   level 0, and level 0 is the same computation on the same picture); also its level-0 vectors = level 0 of the
   hook's MAnalyse output (dump it from the step-0 commit with `compare/nvofa_frc/scripts/render_vec.ps1` before the
   hook is removed).
4. Speed: timing of that chain vs 4.93 s (expected lower: no supf pyramid); determinism over 3 runs; 4 concurrent
   jobs (one OFA engine; the exact path's 4-job figure was 17.4 -> 18.5 s).
5. Format coverage: CudaNVOFAnalyse -> MFlowFps directly and -> MDegrain2 / MCompensate; 8 / 10 bit; 4:2:0 / 4:2:2 /
   4:4:4; pel 1 / 2; delta 1 / 2; isb both; first / last frames (validity 0) — scripts load, no errors, sane output.
6. Synthetic shift (clip vs `Crop/AddBorders` copy shifted by (+5, -3)): interior vectors exactly (10, -6) at pel 2
   (NVOFA flow points from the input frame into the reference: content at x in src is at x + flow in ref).
7. Fallbacks: `MVTOOLS_GPU_DISABLE=1` -> CudaNVOFAvailable() false, uc_FRC takes the normal branch (md5 = exact);
   CudaNVOFAnalyse alone falls back to its internal MAnalyse.
8. uconvert: UConvertClip on the UHD clip with uc_frc_gpu=true, uc_frc_nvof=2 (and 0 = unchanged md5); an HD clip
   for the pel-4 path.
9. The user judges the visual result by eye (no PSNR-only claims); offer crops / slow-motion with the grid
   (`compare/nvofa_frc/scripts/slowmo.py`) if anything changed.

Commits on gpu_work: step 0 = the experiment as it stands; then the standalone filter (API 17, hook removed);
uconvert.avsi edited in place (backup kept). Push only when the user asks.

## Known NVOFA defects to look at later (from the user's judgement, not part of this plan)
Black dot in AB15 at output frame 53, unnatural smearing in AF13–AF16 at frame 42, a dark spot in Z13 at frame 50
(full-frame 120 px grid, `compare/nvofa_frc/spots_check.csv`). Suspect: the per-block median when a 32x32 block
straddles two motions.
