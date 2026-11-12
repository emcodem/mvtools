# CudaConvolution

AviSynth+ plugin: progressive -> interlaced (R p -> R/2 i) for any size, with optional integer downscale 1..8 per
axis. The built-in preset `balanced` holds filters per factor: factor 2 is the measured reference filter (UHD -> 1080i:
luma bit-exact), factor 1 the interlace filter, 3..8 designed with the same look. One self-contained x64 DLL (static
CUDA runtime, static MSVC runtime, presets embedded). User documentation: `docs/usage.md`; validation on
real material: `docs/validation.md`.

Status: CPU and GPU engines done and bit-identical; factor 2 luma bit-exact to the reference on the probes, chroma
+-1 on 0.002 % of samples.

## Build

Part of the mvtools build: the root `CMakeLists.txt` / `build_windows.ps1` build `CudaConvolution.dll` with the
other plugins (option `MVTOOLS_CUDACONVOLUTION`, MSVC only; with the CUDA engine when the root builds `mvtools_cuda`,
tests off: `-DCUDACONVOLUTION_TESTS=ON`). The standalone build of this folder below is the one with the tests and the
release package. GPU architectures: `CUDACONVOLUTION_CUDA_ARCH` (default sm_60 .. sm_120 + PTX; `tools/build.ps1
-CudaArch 61-virtual` builds only the Pascal PTX, which the driver JIT-compiles for the local GPU: runs the tests on the
Pascal code).

Requirements: Visual Studio 2022 (C++ workload, its bundled CMake and Ninja work), CUDA Toolkit 12.8 / 12.9
(release builds use 12.9 so the DLL runs on R525+ drivers, e.g. the A4000 machines with 537.70; optional: without it
the CPU-only DLL is built). Toolkit as a conda env:
`mamba create -p <env folder> -c nvidia -c conda-forge cuda-nvcc=12.9.* cuda-cudart-dev=12.9.* cuda-cudart-static=12.9.* cuda-version=12.9`

```
pwsh -File tools\build.ps1 -Cuda <env folder>\Library -BuildDir build-cuda12   # GPU build -> build-cuda12\CudaConvolution.dll
pwsh -File tools\build.ps1 -Cuda "" -BuildDir build-cpu                                 # CPU-only build
pwsh -File tools\package.ps1                                                            # release build + tests + dist\*.zip
```

The build script runs CMake + Ninja inside the VS x64 environment and then the tests:

| test | what |
|---|---|
| `unit` | engines without AviSynth: CPU scalar = AVX2 = GPU, self-test vectors, presets, formats |
| `probe` | bit-exact against the reference output (needs `CUDACONVOLUTION_DATA`, skipped otherwise) |
| `avisynth` | the built DLL inside AviSynth+: automatic format conversion, 8-bit route, `clip_min` / `clip_max`, output clip and frame properties, parameter errors, `CudaConvolution_SelfTest()`; needs AviSynth+ installed (fails otherwise) |
| `dll_dependencies` | the DLL imports nothing besides system DLLs |

After an intended change of the arithmetic or the presets, regenerate
the self-test vectors with `build-cuda12\cudaconvolution_tests.exe "[.gen-selftest]"` and rebuild.

## Usage

See `docs/usage.md`. In short:

```
FFVideoSource("uhd50p.mov", threads=0)  # e.g. YUV422P10 3840x2160, any rate R
Prefetch(1)                             # decode ahead; not a Prefetch(N) at the end (ffms2 then seeks, 5x slower)
CudaConvolution(width=1920, height=1080)       # -> 1920x1080 interlaced R/2, TFF, GPU (default output size = input size)
# bit-exact to the measured reference: CudaConvolution(width=1920, height=1080, clip_min=1, clip_max=1020)
```

`CudaConvolution(clip c, int "width", int "height", string "preset", string "field_order", int "clip_min", int "clip_max",
int "device", bool "cpu", int "threads", bool "convert", bool "token_out")`,
`CudaConvolution_SelfTest(int "device")`, `CudaConvolution_Version()`. `preset` = path to an own preset file (format:
`presets/balanced.json`, regenerate the designed factors with `python -I tools/design_taps.py`).

## Layout

| path | content |
|---|---|
| `src/core` | presets, formats, CPU engine (scalar reference + AVX2), self-test vectors (no CUDA) |
| `src/cuda` | CUDA engine, runtime dependency checks |
| `src/avs` | AviSynth+ wrapper |
| `presets/*.json` | built-in presets, embedded into the DLL at build time |
| `selftest/vectors.json` | expected MD5 of the self-test vectors, embedded |
| `tests/` | Catch2 tests (internal), `tests/data` MD5 lists of the probe |
| `tools/` | python reference model (`ref_model.py`), probe generators, fits, verification, build and package scripts |
| `examples/` | example scripts (shipped) |
| `third_party/` | AviSynth+ 3.7.3 headers, nlohmann/json 3.11.3 (Catch2 is fetched by CMake) |

Python verification (needs the decoded probe and reference output, see `tools/pio.py`):
`CUDACONVOLUTION_DATA=<folder> python -I tools/p_verify.py replicate 0 100`; golden C++ test: `cudaconvolution_tests "[probe]"`.
