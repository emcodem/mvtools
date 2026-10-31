# NVOFA test: Optical Flow SDK vectors in place of k_search (uc_FRC)

## Context
On uc_FRC, k_search takes about 95% of GPU kernel time (2.24 ms x 1603, ~3.6 s of 6.57 s wall). NVIDIA's Optical Flow Accelerator
(NVOFA) is a fixed-function unit, so a search run there would leave the SMs. The user wants two things: the speed gain, and how
the FRC output looks. This is an experiment, not a feature. It runs behind an env var, is off by default, and breaks
exactness on purpose. The CPU fallback and the golden / md5 tests stay unchanged when it is off.

What already exists on this machine:
- `C:\Windows\System32\nvofapi64.dll` is the runtime and comes with the driver.
- The SDK headers are missing. `nvOpticalFlowCommon.h` and `nvOpticalFlowCuda.h` are MIT and available from github.com/NVIDIA/NVIDIAOpticalFlowSDK.
  They go into `Sources\cuda\third_party\nvof\`.
- Load the DLL with `LoadLibrary` / `NvOFAPICreateInstanceCuda`, so no import lib and no change to the link line.

## Approach: swap the search inside `analyse_dev`, everything else unchanged
Insert at `Sources\cuda\mvcuda_analyse.cu` `analyse_dev`, after `frame_super` (~2588-2593), where the level-0 planes, `d_pay`
and `offs[]` are available. If `MVCUDA_NVOFA` is set to `fast|medium|slow` (the NVOFA perf preset), skip the
k_global / k_interp / k_search launches and fill the payload from NVOFA instead. There are no API, export or
`mvtools2.dll` changes. MAnalyse still gets the same `(vx, vy, sad)` payload (`mvcuda.h:129-131`). In uc_FRC,
MRecalculate (blksize 16, thSAD 100) refines the vectors and recomputes their SAD, then MFlowFps interpolates. Both are
unchanged and read only level 0.

New file `Sources\cuda\mvcuda_nvofa.cu`, ~400-500 lines, called from `analyse_dev`:
1. **Session cache.** `NvOFAPICreateInstanceCuda` runs once. One NVOFA handle per (width, height, preset) is kept in a
   small mutex-protected pool, because Prefetch 3 means concurrent calls. Settings: grid size 4 (fall back to the
   smallest supported size reported by `nvOFGetCaps`), output S10.5 (1/32 px), cost buffer off. The CUDA context comes
   from the runtime's primary context via `cudaGetDriverEntryPoint` (no `cuda.lib`). `nvOFSetIOCudaStreams` uses the
   leased stream of the call.
2. **Input.** A small kernel copies the level-0 luma `sub[0]` (unpadded window, from `hpad`/`vpad`) into an NVOFA
   GRAYSCALE8 buffer. 16-bit storage is shifted down by `bits_per_pixel - 8`. NVOFA takes 8-bit input only, which
   costs some precision on 10-bit sources. src = NVOFA input, ref = NVOFA reference, so the flow points from the src
   block into ref, as an MVTools vector does.
3. **Flow to blocks (level 0).** For each block (32x32, overlap 16 in uc_FRC), take the median of the grid vectors under
   the block, for x and y separately. Convert 1/32 px to 1/pel with rounding (`v * pel / 32`). Clamp to
   `dx_min/dx_max`, `dy_min/dy_max`, the same formulas as `mvcuda_recalc.cu:95-98`.
4. **SAD.** A warp per block computes luma SAD plus scaled chroma SAD for the chosen vector. This copies `block_sad` and
   `scale_chroma` from `mvcuda_recalc.cu:53-68` and the `Checker::luma/chroma` pattern (166-176). It is not shared,
   because those helpers are file-local.
5. **Coarse levels.** Downscale the vectors (halve per level, pel 1 above level 0) with sad = 0. The levels only have to
   exist with the right sizes (`FakeGroupOfPlanes::Update`); no consumer in uc_FRC reads them.
6. **Timing.** With `MVCUDA_TIMING` set, print the convert / NVOFA execute / SAD times, matching the existing phase printout.

Build: add `mvcuda_nvofa.cu` to the unit map in `Sources\cuda\build_dev.ps1` (18-24) and to the unit lists in
`build.bat` / `build_dev.bat`. Test set: `profile\build_nvofa` from `profile\build_all_dev.ps1 -Out ...` (mvtools2 and
cudafilters do not change, but building all three keeps them on the same API version).

Step 0 is a go/no-go check, about 1 hour. A tiny test (`MVCUDA_NVOFA=probe`) prints `nvOFGetCaps` (grid sizes, max
size, formats) on the RTX 5090 before anything else is built. If NVOFA is missing or rejects UHD there, stop and report.

## What gets measured / shown
- **Speed:** `run_frc.ps1 -Dir build_nvofa`. Add an `-Env` parameter (e.g. `-Env MVCUDA_NVOFA=medium`) that sets the
  variable for the ffmpeg process. Compare best-of-3 wall and CPU-s against the current 6.57 s / 21.1 CPU-s, for the
  fast / medium / slow presets. Run one nsys of the medium run to confirm k_search is gone and to show where the time goes
  now: NVOFA engine time, k_recalc, k_flow.
- **Result:** the md5 will differ, as expected. For visual judgement, render the same frames from the bit-exact path and
  the NVOFA path (the medium preset, plus slow if it differs): PNGs of fixed frames, with emphasis on the problem areas of
  seg1_wires_pole (wires, pole edges, occlusions). Also render a side-by-side / interleaved clip. Put both on a compare
  page in the same style as `compare/dtl_smooth2`, with the exact scripts and commands listed. The user judges by eye; no
  PSNR-only claims.
- **Optional second variant**, only if the first looks promising: NVOFA vectors fed in at the finest level of the normal
  search as extra predictors (k_search still runs, with fewer iterations). This would show whether the hardware helps the
  exact pipeline converge, not only as a replacement.

## Critical files
- `Sources\cuda\mvcuda_analyse.cu` (hook in `analyse_dev`); new `Sources\cuda\mvcuda_nvofa.cu`; new
  `Sources\cuda\third_party\nvof\*.h`
- `Sources\cuda\build_dev.ps1`, `build.bat`, `build_dev.bat` (unit lists)
- `profile\run_frc.ps1` (`-Env`); new compare page / scripts under `compare\nvofa_frc\`
- `GPU_PIPELINE_PLAN.md`: a short "experiment" note with the numbers

## Verification
1. Off by default: with `MVCUDA_NVOFA` unset, run golden 85/85, `run_frc.ps1` md5 BFE44775F811 PASS with `-Fallback`, and
   `run_partm.ps1` 36/36, all on `build_nvofa`.
2. On: uc_FRC renders all 112 frames without `MVCUDA_NEEDS_CPU` fallbacks (`MVTOOLS_GPU_LOG=1`). As a sanity check on
   static areas, the vectors must be (0,0) there, and an artificially shifted test clip (`Crop`/`AddBorders` by +4 px) must
   give vx = 4*pel.
3. Timing table and the compare page are handed to the user.

Effort estimate: 1-2 days including the compare page. The main unknown is Step 0 (NVOFA caps and UHD on Blackwell).

## Results (2026-10-31)
- Step 0: NVOFA on the RTX 5090 (driver 616.92, API 5.0; headers 2.0 from github.com/NVIDIA/NVIDIAOpticalFlowSDK, BSD-3):
  grid 1/2/4, up to 8192x8192. UHD grid 4: 3.0 ms per pair medium / 2.2 fast / 5.4 slow in one thread; one engine,
  2+ threads give 2.64 ms (medium) / 1.90 ms (fast) per pair. Synthetic shift (5, -3): median exact, 95.7% of cells exact.
- k_search was 1603 launches = 229 analyses x 7 levels, about 15.7 ms of SM time per analysis.
- uc_FRC (run_frc.ps1, best of 3): exact 6.80 s -> NVOFA fast 4.88 / medium 4.93 / slow 5.25 s (-27%); CPU-s unchanged;
  each level deterministic. Kernel time 3.88 -> 0.32 s (k_search gone; k_recalc now the largest kernel).
- Result: frames on source frames identical; interpolated frames 45-49 dB PSNR vs exact, differences at the pole / wire
  crossings. compare/nvofa_frc (README, crops, diffs, side_by_side.mp4), for judging by eye.
- Off (variable unset): run_frc md5 BFE44775F811 = CPU fallback PASS, golden 85/85, run_partm 27/27.
- Code: Sources/cuda/mvcuda_nvofa.{h,cu}, hook + k_of_gray / k_of_blocks / k_of_coarse in mvcuda_analyse.cu, unit added to
  build.bat / build_dev.bat / build_prof.bat / build_dev.ps1; profile/run_frc.ps1 -SetEnv. Nothing committed.
