# cudafilters

AviSynth+ plugin (x64) with GPU versions of the AviSynth core, MaskTools2, RgTools and plugins_JPSDR (NNEDI3) stages that
QTGMC uses, plus fused QTGMC stages. Every filter is bit-exact with the CPU filter it replaces, and every filter builds that
CPU filter chain too: it is the exact fallback when the GPU cannot make a frame. The kernels live in `mvtools_cuda.dll`
(`..\Sources\cuda`); this plugin loads it from its own folder at run time, so `cudafilters.dll`, `mvtools2.dll` and
`mvtools_cuda.dll` must sit in one folder and come from one build (library API version check).

GPU filters return *token frames*: ordinary AviSynth frames whose first luma row holds a `MvCudaTokenHeader` while the picture
stays in device memory. `CudaDownload` turns them back into pixels; `MSuper(gpu=true)`, `MAnalyse`, `MCompensate` and
`MDegrain` of mvtools2 read them directly. See `GPU_PIPELINE_PLAN.md` (sections 16 onwards) for the design and the test
results.

Filters: `CudaUpload`, `CudaDownload`, `CudaBob`, `CudaTemporalSoften`, `CudaMerge`, `CudaMergeLuma`, `CudaMergeChroma`,
`CudaKeepOnlyBobShimmerFixes`, `CudaRemoveGrain`, `CudaGaussResize`, `CudaBilinearResize`, `CudaNNEDI3`, `CudaLogic`,
`CudaClamp`, `CudaExpand`, `CudaInpand`, `CudaLutxy`, `CudaQtgmcSharpLimit`, `CudaQtgmcBinomial0`, `CudaQtgmcBackBlend`,
`CudaQtgmcAddNoise`. QTGMC uses them through the patched script made by `..\qtgmc\make_qtgmc_gpu.py`.

Build (Visual Studio 2026 Build Tools, toolset v145):

    MSBuild cudafilters.vcxproj /p:Configuration=Release /p:Platform=x64 /p:OutDir=<dir>\ /p:IntDir=<dir>\obj_cf\

Licence: GPL v2 or later, like mvtools. `Resample.cpp` is a port of the AviSynth+ 3.7.5 resampling program code (GPL v2 or
later); `CudaNnedi3.cpp` prepares the plugins_JPSDR 4.1.0 nnedi3 weights with that plugin's own arithmetic (GPL v2 or later).
