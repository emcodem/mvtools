# Changelog

## Unreleased

- Moved into the mvtools repository (`cudaconvolution/`): built with the other plugins by the root CMake build
  (`MVTOOLS_CUDACONVOLUTION`, MSVC only; CUDA engine when the root builds `mvtools_cuda`); the standalone build from
  this folder (`tools/build.ps1`, `tools/package.ps1`) works as before.
- GPU engine on Pascal (compute capability 6.0 / 6.1, e.g. GTX 10xx, Quadro P) and Volta (7.0): the fatbin now has
  SASS sm_60/61/70/75/80/86/89/90/120 + PTX 12.0. Same kernels, same output (bit-identical to the CPU engine).
  The minimum compute capability in the device check follows the built architecture list
  (`CUDACONVOLUTION_CUDA_ARCH`, `tools/build.ps1 -CudaArch`).

## 0.10.0 - 2026-11-12

- Renamed from ConvCuda to **CudaConvolution**, without aliases: `CudaConvolution()`, `CudaConvolution_Version()`,
  `CudaConvolution_SelfTest()`, `CudaConvolution.dll`; environment variables `CUDACONVOLUTION_NO_TOKENS`,
  `CUDACONVOLUTION_LOG` (dev: `CUDACONVOLUTION_DATA`). Scripts calling `ConvCuda()` must be updated. Output unchanged.

## 0.9.3 - 2026-11-12

- GPU token frames of the companion GPU filter plugins (`mvtools_cuda.dll`, optional, loaded at run time; nothing
  changes without it): token input is read automatically (no `CudaDownload()` needed in front), every frame checked,
  also before the automatic format conversion. A token frame that cannot be read is an error, never picture data.
- Zero-copy with `mvtools_cuda.dll` API 31+ (token interface 1): token inputs of 10/12/16 bit are read in place on the
  GPU; new `token_out=true` keeps the result on the GPU for following GPU filters. Same output as the copy path.
- Uses the library instance the GPU plugins loaded; two different copies in one process, an unknown token interface
  or header layout are refused with an error.
- `ConvCuda_Version()` reports whether token frames are available.

## 0.9.2 - 2026-10-16

- Unsupported input formats are converted automatically with the AviSynth+ built-ins: 4:2:0 / 4:1:1 / Y / YUY2 ->
  `ConvertToYUV422()`, alpha -> `RemoveAlphaPlane()`, packed RGB -> `ConvertToPlanarRGB()`, 14 bit / float ->
  `ConvertBits(16)`. YUV stays YUV, RGB stays RGB. `convert=false` restores the error.

## 0.9.1 - 2026-10-16

- 8-bit input: YUV422P8, YUV444P8, RGBP8 -> 8-bit output, computed with the 10-bit arithmetic (samples x 4, result
  rounded to 8 bit): identical to `ConvertBits(10).ConvCuda(...).ConvertBits(8)`, no extra cost. Native 8-bit
  arithmetic is not used: the measured trunc rule would lift every value below the offset by up to a full 8-bit code.
  `clip_min` / `clip_max` are 8-bit codes for 8-bit input.
- `ConvCuda_SelfTest()`: 30 vectors (6 new 8-bit cases).

## 0.9.0 - 2026-10-14

First release candidate.

- Progressive -> interlaced for any input size and frame rate (R p -> R/2 i), optional integer downscale 1..8 per
  axis via `width` / `height` (default: same size). Non-integer ratios and upscaling are rejected with the valid
  alternatives in the message.
- Built-in preset `balanced`: factor 2 = measured reference filter (UHD -> 1080i luma bit-exact), factor 1 = interlace
  filter matched to its vertical response, factors 3..8 designed with the same look. `preset=` takes an own preset file.
- GPU engine (CUDA, built into the DLL, NVIDIA driver R525+) and CPU engine (scalar + AVX2/FMA3), bit-identical.
- Input YUV422P10/12/16, YUV444P10/12/16, RGBP10/12/16.
- Runtime checks of AviSynth version, NVIDIA driver, GPU and memory with explicit error messages; no silent fallback.
- `ConvCuda_SelfTest()` (24 built-in vectors on all engines), `ConvCuda_Version()`.
- Factor 2 chroma from a dedicated chroma probe: asymmetric float32 H filter and own kernels for the 4 outermost
  output columns per side; chroma matches the reference except +-1 on 0.002 % of samples.
- Preset format: float taps (`"divisor": 1`) and edge kernels (`edges`) for H filters.
- CPU engine: internal worker threads (persistent pool, `threads`, default all), 3.1 -> 26.6 fps UHD -> 1080i;
  results identical for any thread count.
- Recommended script: `Prefetch(1)` directly after the source (a `Prefetch(N)` at the end makes ffms2 seek: 5x
  slower). 60 s UHD ProRes -> ProRes HQ 1080i25: 68 s (previous fmtconv solution 369 s).
- No output clipping by default; `clip_min` / `clip_max` (input bit depth, all planes). The reference match needs
  `clip_min=1, clip_max=1020`; its intermediate clip is part of the measured factor-2 arithmetic.
