# cudafilters

AviSynth+ plugin (x64 Windows and Linux, also usable from VapourSynth) with GPU versions of the AviSynth core, MaskTools2, RgTools and plugins_JPSDR (NNEDI3) stages that
QTGMC uses, plus fused QTGMC stages. Every filter is bit-exact with the CPU filter it replaces, and every filter builds that
CPU filter chain too: it makes a frame the GPU cannot make (uploaded again, the output stays token frames). A filter that
cannot run on the GPU at all (no device or library, format or settings not covered) stops with an error at creation,
unless `global cuda_cpu_fallback = true` or the environment variable `CUDA_CPU_FALLBACK=1` is set: then it runs that CPU
chain and returns normal frames. The kernels live in `mvtools_cuda.dll`
(`..\Sources\cuda`); this plugin loads it from its own folder at run time, so `cudafilters.dll`, `mvtools2.dll` and
`mvtools_cuda.dll` must sit in one folder and come from one build (library API version check).

GPU filters return *token frames* (also for normal input, which they upload themselves): ordinary AviSynth frames whose
first luma row holds a `MvCudaTokenHeader` while the picture stays in device memory. `CudaDownload` turns them back into
pixels (normal frames pass through); the `Cuda*` functions of mvtools2 (`CudaMSuper`, `CudaMAnalyse`,
`CudaMRecalculate`, `CudaMFlowFps`, `CudaMCompensate`, `CudaMDegrain1..6` / `CudaMDegrainN`) read them directly and
return token frames too, apart from the vector clips (see `..\Documentation\gpu.md`). Design: `..\docs\dev\ARCHITECTURE.md`.

Filters: `CudaUpload`, `CudaDownload`, `CudaBob`, `CudaTemporalSoften`, `CudaMerge`, `CudaMergeLuma`, `CudaMergeChroma`,
`CudaKeepOnlyBobShimmerFixes`, `CudaRemoveGrain`, `CudaGaussResize`, `CudaBilinearResize`, `CudaNNEDI3`, `CudaLogic`,
`CudaClamp`, `CudaExpand`, `CudaInpand`, `CudaLutxy`, `CudaQtgmcSharpLimit`, `CudaQtgmcBinomial0`, `CudaQtgmcBackBlend`,
`CudaQtgmcAddNoise`, and `CudaNVOFAnalyse` / `CudaNVOFAvailable` (motion vectors from NVIDIA's optical flow
accelerator, not bit-exact with anything). QTGMC uses them through the patched script made by `..\qtgmc\make_qtgmc_gpu.py`. Parameters, defaults
and what each filter covers on the GPU: `..\Documentation\gpu.md`.

VapourSynth (Windows and Linux): every filter in namespace `cuda` through `mvtools_vsbridge` (`..\docs\dev\DECISIONS.md`
D9), bit-identical with AviSynth+; CPU fallbacks that are AviSynth+ core filters need `MVBRIDGE_FALLBACK`
(`..\README.md` "VapourSynth").

Build: with the root CMake (`..\build_windows.ps1` / `../build_linux.sh`, `..\README.md` "Build Instructions").
`cudafilters.vcxproj` is not maintained (no VapourSynth entry point).

Licence: GPL v2 or later, like mvtools. `Resample.cpp` is a port of the AviSynth+ 3.7.5 resampling program code (GPL v2 or
later); `CudaNnedi3.cpp` prepares the plugins_JPSDR 4.1.0 nnedi3 weights with that plugin's own arithmetic (GPL v2 or later).
