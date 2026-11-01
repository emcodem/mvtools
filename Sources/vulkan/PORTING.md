# mvtools_vulkan: porting notes

`mvtools_vulkan` is the Vulkan build of the GPU backend. It exports exactly the C API of `mvtools_cuda`
(`Sources/cuda/mvcuda.h`, API 18), and its results must be **bit-identical** to the CUDA library's and the CPU's.
mvtools2 and cudafilters load one of the two libraries at run time (`Sources/cuda/dyn_lib.h` `open_gpu_backend`):
`MVTOOLS_GPU_BACKEND=cuda|vulkan|auto` (auto: CUDA when it has a usable device, else Vulkan).

## Layout

| File | Port of | Content |
|---|---|---|
| `vkrt.h/.cpp` | (CUDA runtime) | Vulkan loader, device, pooled memory (`dev_alloc` -> 64-bit device address), `Cmd` (record, `submit_wait`), embedded kernels |
| `vk_internal.h` | `mvcuda_internal.h` | `DevFrame`, `DevSuper`, `DevSubpel`, registry functions, `frame_super` |
| `vk_frames.cpp` | `mvcuda_frames.cu` | token registry (budget, LRU, ghosts, pins, producers), upload / download |
| `vk_core.cpp` | `mvcuda.cu`, `mvcuda_core.cu` | version, devices, selftest, `frame_super`, cache clear |
| `vk_merge.cpp` + `shaders/merge.pix.comp` | `mvcuda_merge.cu` | **reference port**: read it first |
| `vk_<x>.cpp` + `shaders/<x>*.comp` | `mvcuda_<x>.cu` | one file per CUDA file; stubs return -4 (callers then use the CPU) until ported |

## Execution model

* One API call = one or more `vkrt::Cmd`; `submit_wait()` returns when the GPU has finished. So: no streams, events,
  marks, `pipe_submit`, `pipe_keep`, `leased_call`; buffers can be freed right after `submit_wait`. Use
  `vkrt::DevBuf` for scratch buffers (RAII).
* Inside one `Cmd` the commands run **in recording order** (an automatic full barrier is inserted before every command),
  i.e. like a CUDA stream. Kernel sequences port 1:1.
* Several threads call concurrently; the runtime spreads them over up to 8 queues. Shared mutable host state needs a mutex
  as in the CUDA code.
* `frame_alloc` / `frame_put` / `frame_get` as in CUDA; `frame_put` only after `submit_wait` succeeded.
* Host-memory test entry points (`*_host`): `mvk::upload_picture`, then the token code path.

## Shaders

* GLSL 4.60 compute, compiled by glslangValidator at build time (`--target-env vulkan1.2`), embedded. File `x.comp` gives
  kernel `x`; `x.pix.comp` gives `x_8` (`MVC_PIX8` defined, 8-bit samples) and `x_16` (16-bit samples). Look up with
  `vkrt::kernel("x_8")`; specialization constants (`layout(constant_id = i) const int NAME = ...;`) via
  `vkrt::kernel("x_8", {v0, v1, ...})` replace CUDA template parameters. Kernel names must be unique across all files:
  prefix them with your area (`ana_`, `flow_`, `comp_`, ...).
* Include `common.glsl` (see its header). Pointers are `uint64_t` device addresses: `PU8(addr).v[i]`, `PU16`, `PI32`, ...
  `ldpix` / `stpix` for the compile-time sample size. Arguments: a C struct (only `int`, `unsigned`, `float`, `double`,
  `uint64_t`/`unsigned long long` members and arrays of them; put 64-bit members first or pad so offsets match) passed to
  `Cmd::dispatch`, declared identically in GLSL and read as `A.member`. No size limit.
* Device requirements (checked at device selection): Vulkan 1.2, bufferDeviceAddress, 8/16-bit storage, scalar block
  layout, shaderInt64. 8/16-bit values are **storage only**: convert to `int`/`uint` right after loading. Optional:
  `shaderFloat64` (`vkrt::has_float64()`; if a kernel needs doubles and the device has none, return -4 so the caller
  uses the CPU), `shaderInt16`, `shaderInt8`.
* **No subgroup (warp) operations** unless you also handle every subgroup size (NVIDIA 32, AMD 32/64, Intel 8/16/32,
  lavapipe 4-16): replace `__shfl*`, warp votes and warp-synchronous code with shared memory + `barrier()`. `barrier()`
  must be reached by all invocations of the workgroup (uniform control flow).
* `__syncthreads()` -> `barrier()` (+ `memoryBarrierShared()` where shared data is published; `barrier()` alone orders
  shared memory in Vulkan's memory model for compute shaders in practice, but write `memoryBarrierShared(); barrier();`).
* Inter-workgroup waiting (spin loops on another workgroup's progress, tickets) has **no forward-progress guarantee** in
  Vulkan. It works on NVIDIA / AMD when all workgroups of the dispatch are resident; it can deadlock elsewhere (lavapipe,
  some Intel). Prefer one dispatch per dependent step; when spinning is unavoidable, launch no more workgroups than can be
  resident, use `coherent`/`volatile` buffers (`PI32V`) and atomics, and keep a bounded spin with a failure flag that
  makes the host return -4 (CPU fallback) instead of hanging. The runtime treats a GPU wait > 60 s as device lost.
* Floating point: GLSL may contract `a * b + c` into fma. Where the CUDA code relies on unfused arithmetic (`__dmul_rn`,
  `__fmul_rn`) mark the variables `precise`; where it uses `fmaf`/`__fmaf_rn` use `fma()`. Integer code is exact anyway.
* `atomicAdd/Min/Max/Or/Exchange/CompSwap` work on `PI32(addr).v[i]` and shared variables.
* `cp.async`, `__nanosleep`, `prefetch`, `%globaltimer`, `clock64`, profiling counters: drop them (plain loads into shared).

## Build and test (Windows)

* Build only the library: `cmd /c C:\dev\mvtools\profile\vkbuild.bat <build dir>` (Ninja, MSVC; about 30 s).
  Use your **own** build dir (several people build at the same time).
* `python -I <scratchpad>\vk_smoke.py <dll>` style ctypes checks; golden tests take another DLL through
  `MVCUDA_TEST_DLL=<path to mvtools_vulkan.dll>`: `python -I C:\dev\mvtools\profile\golden\run_all.py 10 f8 f420 f420b8 f444`
  (super / recalc / analyse / flow host entry points vs CPU dumps).
* Plugin-level tests: copy your `mvtools_vulkan.dll` into a test folder of your own that also has `mvtools2.dll`,
  `cudafilters.dll`, `avstp.dll` from `C:\dev\mvtools\profile\build_vkt` (built with the new loader), set
  `MVTOOLS_GPU_BACKEND=vulkan`, and run `C:\dev\mvtools\profile\run_cf_regression.ps1 -Dir <folder>` (cudafilters stages
  vs CPU) or AviSynth scripts through ffmpeg. **Check that the GPU path really ran**: `MVTOOLS_GPU_LOG=1`,
  `CUDAFILTERS_LOG=1`, `MVVK_LOG=1` print every call / fallback; a stub or an error silently falls back to the CPU and
  then "passes".
* `MVVK_DEVICE=1` selects the second Vulkan device (on the development PC: the Intel iGPU, a non-NVIDIA driver with
  other subgroup sizes) - run your tests there too. `MVVK_VALIDATION=1` enables the validation layer if installed.
* Linux: the same sources build with the root CMake (`MVTOOLS_VULKAN=ON`); Mesa lavapipe (`MVVK_ALLOW_CPU=1`) runs the
  kernels on the CPU for functional tests.

## Status (2026-11-01)

Every entry point of mvcuda.h is ported except NVOFA (`mvcuda_nvof_token` returns MVCUDA_NEEDS_CPU; `CudaNVOFAnalyse`
then uses its MAnalyse fallback). Verified bit-identical on an RTX 5090 and an Intel iGPU (no float64):

| Check | Result |
|---|---|
| golden `run_all.py 10 f8 f420 f420b8 f444` (super, analyse, recalc, flow) | 85/85 |
| C API cross-checks against mvtools_cuda (super/flow 120, analyse/recalc 124, comp 120, nnedi3 35 random configurations) | all identical |
| `run_cf_regression.ps1` (pwsh 7) with every stage on Vulkan | 65/65 PASS, tokens 42/42, 0 failed GPU calls |
| QTGMC Fast / Slower, 1500 PAL frames, Prefetch 12, all GPU stages | md5 = CUDA = CPU fallback |
| VapourSynth vs AviSynth+ (`vsbridge/test/compare.py`), Windows and Linux | 44/44 |
| Linux build (GCC 15), lavapipe selftest | ok |

Speed on the RTX 5090 (QTGMC, 1500 frames): Fast 3.40 s Vulkan vs 1.82 s CUDA vs 6.9 s CPU; Slower 8.53 / 4.33 / 14.3 s.
Throughput per stage with several threads equals CUDA; the gap is the synchronous call model (every call waits for its
fence) and MAnalyse's packed SAD / next-block prefetch, which are not ported (1.1-1.9x slower per call).

Open:
* Asynchronous submission: per-frame timeline semaphore values (consumers wait instead of producers), deferred
  `dev_free` until the GPU passed the last use; would remove most of the gap to CUDA.
* MAnalyse packed SADs (4 samples per load) and next-block prefetch.
* Not run on AMD hardware or real Pascal hardware yet (Pascal: Vulkan path tested on Intel and NVIDIA RTX only; the CUDA
  Pascal path as PTX 6.1 JIT on the RTX 5090).
* `super_covers` in vk_flow.cpp duplicates `covers` in vk_core.cpp (keep them equal).
