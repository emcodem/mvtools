# NVENCMEAnalyse: motion vectors from the NVENC motion estimator (plan)

**Status 2026-10-31: plan only, nothing built.** Decisions in this file are the user's unless marked "proposal".
Related: [NVOF_FILTER_PLAN.md](NVOF_FILTER_PLAN.md) (the NVOFA sibling, implemented), GPU_PIPELINE_PLAN.md section 26,
[opticalflow.md](opticalflow.md), [compare/nvofa_frc/README.md](compare/nvofa_frc/README.md).

## Context
CudaNVOFAnalyse (cudafilters.dll, library API 17) gives MAnalyse-format vectors from NVIDIA's optical flow engine
(NVOFA). This plan adds a sibling that uses the **video encoder's motion estimation (NVENC "ME-only" mode)** and that
**works without optical-flow hardware, to cover more cards**, on **Windows and Linux**, for **AviSynth+ and
VapourSynth**.

Findings that shape the design:
- `mvtools_cuda.dll` runs only on compute capability **8.0+**: `mvcuda_core.cu:36` skips `p.major < 8` because
  MAnalyse's kernels use `cp.async` (`mvcuda_analyse.cu:445`); `build.bat` builds sm_80..sm_120 with CUDA 13.4, which
  cannot target anything older than sm_75. Every GeForce card that passes this gate already has an OFA engine, so an
  NVENC path that *requires* mvtools_cuda would add almost no cards.
- NVENC needs no CUDA kernels. An ME-only session runs on a CUDA context made with the **driver API** alone
  (`nvcuda.dll` / `libcuda.so.1`, shipped with the driver, no compute-capability gate): luma copied into a device buffer
  registered with NVENC, vectors from `nvEncCreateMVBuffer` read back. D3D11 would also work but is Windows-only.
- Target: **Pascal and newer** (GTX 10xx): H.264 and HEVC ME-only, 8-bit NV12 input; run-time capability check
  (`NV_ENC_CAPS_SUPPORT_MEONLY_MODE`) and an MAnalyse fallback.
- Test setup: **WSL2 on the RTX 5090 machine** only. No native Linux box, no Pascal card: Pascal stays untested and is
  documented as such.

## What NVENC ME-only is, compared with NVOFA
| | NVOFA (CudaNVOFAnalyse) | NVENC ME-only |
|---|---|---|
| API / library | Optical Flow SDK, `nvofapi64.dll` | Video Codec SDK `nvEncodeAPI.h`; `nvEncodeAPI64.dll` / `libnvidia-encode.so.1`; `enableMEOnlyMode`, `nvEncRunMotionEstimationOnly` |
| Output | dense flow, 4x4 grid, 1/32 px | H.264: four 8x8 vectors per 16x16 MB (`NV_ENC_H264_MV_DATA`); HEVC: per 8x8 CU (`NV_ENC_HEVC_MV_DATA`); **1/4 px**; cost / partition type |
| Optimises | smooth "true" motion | encoder rate/distortion cost (SATD + MV bits): block matching, close to MAnalyse |
| Hardware (RTX 5090) | 1 OFA engine (~2.6 ms/pair UHD medium, saturated with 2 threads) | 3 NVENC engines, separate from OFA |
| Limits | none in practice | GeForce concurrent-session cap (driver-dependent; ME-only sessions count); limited search range, especially vertical |
| Cards | Turing+ (in practice Ampere+, our library's gate) | Pascal+ for H.264 + HEVC ME-only |

## Usage (planned script API)
```avisynth
# frame-rate conversion, as the uc_FRC NVOFA branch
pre = src.RemoveGrain(22)
bv  = NVENCMEAnalyse(pre, isb=true,  blksize=16, overlap=8, pel=2, hpad=16, vpad=16, preset=2)
fv  = NVENCMEAnalyse(pre, isb=false, blksize=16, overlap=8, pel=2, hpad=16, vpad=16, preset=2)
sup = MSuper(src, pel=2, hpad=16, vpad=16, levels=1)
bv  = MRecalculate(sup, bv, blksize=8, overlap=4)
fv  = MRecalculate(sup, fv, blksize=8, overlap=4)
MFlowFps(src, sup, bv, fv, num=60000, den=1001)

# engine choice per machine
nvof  = FunctionExists("CudaNVOFAvailable") ? CudaNVOFAvailable() : false
nvenc = FunctionExists("NVENCMEAvailable")  ? NVENCMEAvailable()  : false
bv = nvof  ? CudaNVOFAnalyse(pre, isb=true, blksize=16, overlap=8, pel=2) \
   : nvenc ? NVENCMEAnalyse (pre, isb=true, blksize=16, overlap=8, pel=2) \
   :         MAnalyse(MSuper(pre, pel=2), isb=true, blksize=16, overlap=8, truemotion=true)

# denoising: MRecalculate in between so thSAD sees real SADs
b1 = MRecalculate(sup, NVENCMEAnalyse(src, isb=true, delta=1, blksize=16, overlap=8), blksize=16, overlap=8)
```
```python
# VapourSynth (standalone mode)
bv = core.nvencme.Analyse(pre, isb=True, blksize=16, overlap=8, pel=2)
bv = core.mv.Recalculate(sup, bv, blksize=8, overlap=4)
```
Input frames: CPU frames and GPU token frames are both accepted; no `CudaUpload()` / `CudaDownload()` needed.

## Expected benefits and drawbacks (to be confirmed by the probe and the visual check)
Benefits:
- Runs on Pascal / Turing, where the whole MVTools chain is CPU-only today: MAnalyse's search, usually the biggest cost,
  moves to fixed-function hardware.
- No dependency on the CUDA toolkit, mvtools_cuda or NVOFA in standalone mode; one plugin for Windows and Linux,
  AviSynth+ and VapourSynth.
- No MSuper search pyramid (only the `levels=1` super for MRecalculate / MFlowFps).
- On new cards: 3 NVENC engines next to the single OFA engine (concurrent jobs; NVOFA and NVENC in parallel).
- Same parameters as CudaNVOFAnalyse: drop-in swap.

Drawbacks / risks:
- **Quality unknown.** Encoder ME minimises cost, not true motion; likely closer to MAnalyse than to NVOFA, which won
  the user's visual judgement because it follows motion. The limited search range may repeat the fast-pan misses.
- 8x8 granularity, 1/4 px, 8-bit luma only (chroma only in the SAD).
- Not bit-identical to the CPU; may differ across GPU generations and driver versions.
- Session cap shared with other NVENC users (ffmpeg / OBS exports): frames then fall back to the CPU MAnalyse.
- Standalone mode: PCIe upload and host-side block processing.
- NVIDIA only; Pascal untested.

MRecalculate downstream (recommended): it accepts the clip like any MAnalyse output (`MVRecalculate.cpp:124-208`),
rescales the vectors to its own pel (`PlaneOfBlocks.cpp:3328`), searches a small radius around each one and recomputes
the SAD (`:3358`). It fixes the 1/4-px rounding, the 8x8 coarseness and the SADs; it cannot recover vectors the NVENC
search missed beyond its radius. On Pascal it runs on the CPU, on Ampere+ with `gpu=true`.

## Phase 0: CMake as the only build system, Linux for the whole repo (before the plugin)
Starting point: mvtools2, DePan, DePanEstimate already have a cross-platform CMake build from upstream (root
`CMakeLists.txt:200-204`, GCC / Clang / MinGW / macOS branches `:180-198`, `Sources/Files.cmake` drops `AvstpFinder` off
Windows). Windows / MSVC-only is what this branch added:
- `mvtools_cuda`: `Sources/cuda/build.bat`, `build_dev.bat`, `build_prof.bat`, `build_dev.ps1` (nvcc + `/MT`).
- `cudafilters`: vcxproj only.
- Win32 loader calls: `Sources/GpuBackend.cpp:1-4` (`<windows.h>` unguarded) and `:302-316`, `cudafilters/CudaCommon.h`,
  `Sources/cuda/mvcuda_nvofa.cu:40-53`. Since `Files.cmake` globs every `*.cpp`, the Linux build of mvtools2 on this
  branch most likely fails today (to be confirmed).
- Test and profile scripts: PowerShell with Windows paths.

Work:
1. `Sources/cuda/CMakeLists.txt`: `LANGUAGES CXX CUDA`, `CMAKE_CUDA_ARCHITECTURES 80;86;89;90;120` + PTX 80, the
   `_8` / `_16` unit list from `build.bat`, static runtime on Windows, `-fPIC` on Linux; output `mvtools_cuda.dll` /
   `libmvtools_cuda.so`. `cudafilters/CMakeLists.txt` likewise. Option `MVTOOLS_GPU=ON/OFF` (OFF: mvtools2 without
   `GpuBackend`, as upstream).
2. `Sources/dyn_lib.h`, shared by GpuBackend, cudafilters' `CudaLib`, `mvcuda_nvofa.cu` and later nvencme: open /
   symbol / directory of this module (`LoadLibraryW` / `GetProcAddress` / `GetModuleFileNameW` vs `dlopen` / `dlsym` /
   `dladdr`), library names per OS (`nvofapi64.dll` / `libnvidia-opticalflow.so.1`, etc.).
3. Windows first: the CMake (Visual Studio generator) DLLs must reproduce today's results **bit for bit**: golden 85/85,
   run_partm 27/27, run_cf_regression, uc_FRC BFE44775F811, NVOF FA834251D140. Compiler flags (`/O2`, `/MT`, fp model)
   taken over from the vcxproj / bat files.
4. Then **CMake becomes the only build system**: `mvtools.vcxproj`, `cudafilters.vcxproj`, `DePan*.vcxproj`,
   `mvtools.sln` / `.slnx`, `Sources/cuda/build*.bat`, `build_dev.ps1` are removed. `profile\build_all_dev.ps1` (outside
   the repo) and the test scripts switch to `cmake --build`; CMake's dependency tracking replaces build_dev.ps1's unit
   map. Build instructions updated in `README.md`, `Documentation/gpu.md`, `cudafilters/README.md`.
5. Linux (WSL2): all targets build with GCC + nvcc; AviSynth+ for Linux loads them, MVTOOLS_GPU off and on.
6. Linux reference results: CPU-path md5s are not comparable 1:1 with Windows (no avstp.dll: the CPU fallback runs
   single-sliced, section 26: 09FDEBF10F3F vs BFE44775F811; GCC vs MSVC float code may differ too). So a Linux reference
   set of its own, plus a check that the GPU paths (same nvcc kernels) match Windows. Test runner: the PowerShell
   scripts under `pwsh` with parameterised paths, else a small Python runner.

## Plugin structure: one engine, two front ends, one shared library
The library exports `AvisynthPluginInit3` and `VapourSynthPluginInit2`; each host ignores the other's entry point.
Built by `nvencme/CMakeLists.txt`; output `nvencme.dll` / `libnvencme.so`; links nothing NVIDIA; plain C++ (no nvcc).
- `nvencme/core/`: host-independent C++: library loading (`dyn_lib.h`), CUDA driver context, NVENC session pool,
  8-bit gray conversion, ME run, per-block median + clamp, SAD. In: plane pointers, pitches, format; out:
  `(vx, vy, sad)` per block at the requested pel. No AviSynth / VapourSynth headers.
- `nvencme/avs/`: AviSynth+ front end: `NVENCMEAnalyse`, `NVENCMEAvailable`, `NVENCMEMode`; this repo's vector clip;
  MAnalyse fallback via `env->Invoke`; token frames and the integrated mode.
- `nvencme/vs/`: VapourSynth front end: `nvencme.Analyse`, `nvencme.Available`; vectors in the format of the
  VapourSynth MVTools port (dubhater/vapoursynth-mvtools) so `mv.Recalculate`, `mv.FlowFPS`, `mv.Degrain*` accept them;
  fallback `core.mv.Analyse(core.mv.Super(...))` through the VS API. API 4 (R55+), C API (`VapourSynth4.h`),
  `fmParallel`, frames n and n±delta requested in `arInitial`. Standalone mode only (token frames are AviSynth-only).
- `nvencme/third_party/nvEncodeAPI.h` (Video Codec SDK, MIT licence). NVIDIA's driver branch 580 is the last with
  Maxwell / Pascal / Volta support: use an NVENC API version that branch accepts, and check
  `NvEncodeAPIGetMaxSupportedVersion` at run time.

**Research item before the VapourSynth front end** (read the VS port's source): how its vector clips carry the
analysis header and vectors (frame properties vs frame data; field list and order of its MVAnalysisData; versioning),
its downstream checks (block sizes, overlap, pel, padding vs its Super), how its Recalculate uses incoming vectors.
`mvsf` (float port): not covered by default.

## Two modes (AviSynth+ front end)
| | Standalone mode | Integrated mode |
|---|---|---|
| When | no usable mvtools_cuda (absent, GPU < CC 8.0, MVTOOLS_GPU_DISABLE=1); always for VapourSynth | mvtools_cuda loads and accepts the device (Ampere+) |
| NVENC device | own CUDA context via the driver API (`NV_ENC_DEVICE_TYPE_CUDA`) | the library's CUDA context |
| CPU frame in | luma copied into a device buffer registered with NVENC | uploaded with `mvcuda_frame_upload_keyed` (`mvcuda.h:251`): the same device copy is reused by MRecalculate / MFlowFps with `gpu=true` |
| Token frame in | only with `mode="standalone"` forced: `mvcuda_frame_download` (`mvcuda.h:255`) + host path | used in place, no copy; MVCUDA_TOKEN_GONE → regenerate + pin as `with_inputs` (`cudafilters/CudaCommon.h:99-177`) |
| 8-bit gray | CPU | library kernel (`k_of_gray`, `mvcuda_nvof_vec.cu:37`), pointer registered with `nvEncRegisterResource` (CUDADEVICEPTR) |
| Block vectors + SAD | CPU port of `k_of_blocks` (`mvcuda_nvof_vec.cu:87`) | `k_of_blocks` on the GPU, SAD from `frame_super` (as CudaNVOFAnalyse) |
| Code | `nvencme/core` | session pool + `mvcuda_nvencme_token` / `mvcuda_nvencme_available` in mvtools_cuda (siblings of `mvcuda_nvofa.cu` / `mvcuda_nvof_vec.cu`; API bump), called by the plugin |

- `mode="auto"` (default): integrated when mvtools_cuda loads, `mvcuda_device_count() > 0` and its API version has the
  NVENC exports; else standalone. `"standalone"` / `"cuda"` force a mode (`"cuda"` errors if unavailable). The library
  is loaded at run time (the `CudaLib` pattern, `cudafilters/CudaCommon.h:19-53`), never linked.
- `NVENCMEAvailable()`: true in either mode (NVENC + CUDA driver libraries load, `cuInit` finds an NVIDIA device, a
  session opens, ME-only cap set for the codec). `NVENCMEMode()`: "cuda" / "standalone" / "".
- Whether both modes must give bit-identical results: decided in the probe (NVENC vectors are the same for the same
  input; only the median / SAD code differs, so matching is feasible).
- Rejected alternative: the integrated mode as a second filter in cudafilters.dll. One function for all cards is what
  makes it a drop-in.

## Filter details
**Script API** (same parameters as CudaNVOFAnalyse, `cudafilters/CudaNvof.cpp:265-301`):
`NVENCMEAnalyse(clip c, bool "isb"=false, int "delta"=1, int "blksize"=16, int "blksizeV"=blksize, int "overlap"=0,
int "overlapV"=overlap, int "pel"=2, int "hpad"=16, int "vpad"=16, bool "chroma"=true, int "preset"=2,
string "codec"="h264", string "mode"="auto")`. `preset` 1 / 2 / 3 → NVENC P1 / P4 / P7. pel 1 or 2. hpad / vpad must
match the MSuper used downstream.

**Per frame**
1. Luma of frame n and its reference (n ± delta) → 8-bit (9..16 bit rounded, as `k_of_gray`), NV12 with neutral
   chroma (128). Each picture uploaded once per (clip, n) through a small cache (each frame is input of two pairs).
2. `nvEncRunMotionEstimationOnly` → MV buffer → 8x8 vectors in 1/4 px.
3. Per MVTools block: median of the covered 8x8 vectors (x and y separately), 1/4 px → 1/pel, clamp to the padded
   frame; SAD luma + scaled chroma. SAD method (own code with edge clamping = MSuper padding, or mvtools' SAD functions)
   decided in the probe; it matters only for direct MFlowFps / MDegrain use.
4. Payload as CudaNVOFAnalyse: one level, `[3+3N, 1, 1+3N, (vx, vy, sad) * N]`; frames without a reference: validity 0,
   verybigSAD.

**Shared vector-clip code** (new header, e.g. `Sources/VectorClipOut.h`, used by cudafilters and nvencme/avs):
MVAnalysisData fill (block count incl. MAnalyse's overlap-0 fallback), BGR32 VideoInfo + x64 pointer hack, frame
layout, `make_manalyse` fallback and `from_fallback` repack (`cudafilters/CudaNvof.cpp:20-125`). CudaNvof.cpp moves to
it in the same change; its md5 must stay FA834251D140.

**Sessions**: pool keyed by (device, size, codec, preset), acquire / run / release like `mvc_nvofa`. Pool capped
(default 2, env override) because of the GeForce session cap; when a session cannot be opened, the frame uses the
MAnalyse fallback (stderr message once). `SetCacheHints`: MT_NICE_FILTER.

**uc_FRC** (`C:\dev\uconvert\uconvert.avsi`, not in git: back up first): `global uc_frc_nvencme = 0`; branch order:
NVOFA if `uc_frc_nvof > 0` and available, else NVENC ME if `uc_frc_nvencme > 0` and
`FunctionExists("NVENCMEAvailable") ? NVENCMEAvailable() : false`, else the normal branch. Same `pre` input, no `supf`;
independent of `uc_frc_gpu` (must work where mvtools_cuda refuses the device).

## Steps
0. Phase 0 (above): CMake for everything, bit-identical on Windows, old build files removed, Linux build in WSL2.
1. **Probe** (scratchpad console program, Windows and WSL2): CUDA driver API + NVENC ME-only, H.264 and HEVC, UHD
   8-bit pair: ms per pair with 1..4 threads, session cap, synthetic shifts (+5,-3 and +120,-60 px) for the search range;
   own driver-API context vs the library's context (vectors expected identical). In WSL2 first check that NVENC is
   exposed (`libnvidia-encode.so.1` under `/usr/lib/wsl/lib`, a session opens); if not, Linux stays build-tested only.
2. Core + AviSynth+ front end, standalone mode; shared vector-clip header, CudaNvof.cpp moved to it.
3. Integrated mode: session pool + exports in mvtools_cuda (API bump, `GpuBackend` version check), mode selection,
   token-frame input.
4. VapourSynth front end (after the research item).
5. uc_FRC option; render the judgement spots (`compare/nvofa_frc/scripts/render.ps1`); the user judges by eye (crops +
   CSV, one question per spot).
6. Docs: `Documentation/gpu.md` or a new `Documentation/nvencme.md`; GPU_PIPELINE_PLAN.md new section.

## Verification
- Synthetic shift: interior blocks exact at pel 2.
- md5 stable over 3 runs; CudaNVOFAnalyse unchanged (FA834251D140); exact paths unchanged (golden 85/85, run_partm
  27/27, run_cf_regression, uc_FRC BFE44775F811).
- Formats: 4:2:0 / 4:2:2 / 4:4:4 × 8 / 10 bit × pel 1 / 2, both directions, delta 1 + 2, into MRecalculate, MFlowFps,
  MDegrain2, MCompensate. MCompensate PSNR vs none / NVOFA / MAnalyse on the 960x540 test.
- Independence: works with mvtools_cuda and the NVOFA library absent, and with MVTOOLS_GPU_DISABLE=1;
  `NVENCMEAvailable()` false → the filter returns MAnalyse (md5 = MAnalyse's).
- Token-frame input (CudaUpload / a cudafilters chain in front) gives the same md5 as plain-frame input, also with the
  token cache forced small (eviction → regenerate); a script without token frames in standalone mode never loads
  mvtools_cuda.
- VapourSynth (Windows + WSL2): `nvencme.Analyse` → `mv.Recalculate` → `mv.FlowFPS` / `mv.Degrain2` run; vectors per
  block equal the AviSynth+ front end's for the same input; fallback = `mv.Analyse` output.
- Linux (WSL2): builds with CMake, loads in AviSynth+ and VapourSynth; runtime tests if NVENC is exposed.
- Speed on the uc_FRC chain (reference: NVOFA 4.74 s, exact 6.75 s), plus 2–4 concurrent jobs (session cap).
