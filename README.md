# mvtools
Motion estimation and compensation plugin for Avisynth+ and Avisynth v2.6 family.\
Supporting YUY2, 4:2:0, 4:2:2, 4:4:4 at native 8, 10, 12, 14 and 16 bit depths, 32bit float in selected filters.\
Still supporting Windows XP.
x86 and x64 versions
From December 20, 2020: Linux port
This branch (gpu_work): NVIDIA GPU paths for AviSynth+ on Windows x64 (`gpu=true`, the `cudafilters` plugin and QTGMC
on the GPU, see "GPU" below); VapourSynth support for mvtools2 and cudafilters through `mvtools_vsbridge` (namespaces
`mvpf` and `cuda`, output bit-identical with AviSynth+; Linux done, Windows to follow; see "VapourSynth" below)

File: mvtools2.dll

## Credits: 

- Manao, Fizick, Tsp, TSchniede, SEt, Vit, Firesledge, cretindesalpes 

## Links
- Doom9 forum: https://forum.doom9.org/showthread.php?t=173356
- Avisynth wiki: http://avisynth.nl/index.php/MVTools
- Project host: https://github.com/pinterf/mvtools under mvtools-pfmod branch
- http://avisynth.nl/index.php/External_filters#64-bit_filters 
- http://avisynth.org.ru/mvtools/mvtools2.html

For more information see also documents folder.

Note:
For a more experimental feature set (new parameters, use hw devices for motion estimation, but Windows only)
have a look at DTL's fork: https://github.com/DTL2020/mvtools

## External dependencies: 

- libfftw3f-3.dll (or renamed to FFT3W.DLL)
from http://www.fftw.org/ or look at ICL builds at http://forum.doom9.org/showthread.php?t=173229
  It is used only at specific dct parameter values
  
- May require Microsoft Visual C++ Redistributables https://www.visualstudio.com/downloads/
  

Others
Modification base:

- mvtools2, 2.6.0.5 64 bit version from
http://avisynth.nl/index.php/AviSynth%2B#AviSynth.2B_x64_plugins

Source code:

## Build Instructions

Note:

Windows MSVC builds are using external assembler source - if there exists.

Other builds are using internal SIMD code, governed by defines in def.h

### Windows MSVC

- Prerequisite: for asm compilation use nasm https://www.nasm.us/ 
  Visual Studio integration: https://github.com/ShiftMediaProject/VSNASM/releases

  Compiler nasm.exe can appear in e.g. `C:\Program Files\Microsoft Visual Studio\18\Community\VC\`

  You can check the files nasm.targets, nasm.props and nasm.xml in e.g. `C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations\`

  For XP compatible (v141_xp) build, copy them to the v150 BuildCustomizations folder as well:
  `C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Microsoft\VC\v150\BuildCustomizations\`

  **Important for VS2026 + v141_xp:** The VSNASM files released for VS2026 use task parameters
  (`TrackFileAccess` on `GetOutOfDateItems`, and `AcceptableNonZeroExitCodes`, `StdOutEncoding`,
  `StdErrEncoding`, `UseMsbuildResourceManager` on `ParallelCustomBuild`/`CustomBuild`) that are
  not supported by the older `Microsoft.Build.CppTasks.Common.dll` (v15.0) shipped with the v150
  toolset folder. The v170 copy is fine as-is. In the **v150 copy only**, remove those attributes
  from `nasm.targets`:
  - In both `<GetOutOfDateItems>` calls: remove `TrackFileAccess="$(TrackFileAccess)"`
  - In `<ParallelCustomBuild>`: remove `AcceptableNonZeroExitCodes`, `StdOutEncoding`, `StdErrEncoding`, `UseMsbuildResourceManager`
  - In `<CustomBuild>`: remove `AcceptableNonZeroExitCodes`, `StdOutEncoding`, `StdErrEncoding`

  Some nasm versions (e.g. the one valid around 2022.Dec.) will throw a lot of warnings on x265 assembler files. Temporarily they are silenced with -w-macro-params-legacy parameter.

* build from IDE

### Windows Visual Studio + Intel C++ Compiler ICX and ICL

- Prerequisite: Intel C++ compilers + Visual Studio integration

* build from IDE

## Windows GCC
(mingw installed by msys2)
From the 'build' folder under project root:

    del ..\CMakeCache.txt
    cmake .. -G "MinGW Makefiles" -DENABLE_INTEL_SIMD:bool=on
    @rem test: cmake .. -G "MinGW Makefiles" -DENABLE_INTEL_SIMD:bool=off
    cmake --build . --config Release  

## Linux build instructions

* Clone repo

        git clone https://github.com/pinterf/mvtools
        cd mvtools
        cmake -B build -S .
        cmake --build build

  Useful hints:        
  build after clean:

      cmake --build build --clean-first

  delete CMake cache

      rm build/CMakeCache.txt

* GPU part (this branch): `libmvtools_cuda.so` and the `libcudafilters.so` plugin are built when a CUDA compiler is
  found (CUDA 12.8+ / 13.x, compute capability 8.0+). Point CMake at nvcc if it is not on the PATH, or switch it off:

        cmake -B build -S . -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
        cmake -B build -S . -DMVTOOLS_GPU=OFF                      # CPU plugins only
        cmake -B build -S . -DMVTOOLS_CUDA_ARCH=120-real           # development: local RTX 50xx only, faster

  `libmvtools_cuda.so` must sit in the same folder as `libmvtools2.so` / `libcudafilters.so`; it needs only the NVIDIA
  driver at run time (static CUDA runtime). Without the library, the driver or a usable GPU, the filters use the CPU.
  CudaNNEDI3 reads `nnedi3_weights.bin` from the folder of `libcudafilters.so`. Without avstp (Windows only) the CPU
  search runs single-sliced, while `gpu=true` reproduces the 24-thread CPU result (set `MVTOOLS_GPU_SLICES=1` to match
  the single-sliced CPU result).

* Find binaries at

        build/Sources/libmvtools2.so
        build/DePan/libdepan.so
        build/DePanEstimate/libdepanestimate.so
        build/Sources/cuda/libmvtools_cuda.so
        build/cudafilters/libcudafilters.so
        build/vsbridge/libmvtools_vsbridge.so

* Install binaries

        cd build
        sudo make install

## GPU (this branch)

CUDA paths for MSuper / MAnalyse / MRecalculate / MCompensate / MDegrain1..6 / MFlowFps (`gpu=true`, default false),
bit-exact with the CPU result; settings, formats or machines the GPU does not cover run the CPU code with the same
output. Needs an NVIDIA GPU with compute capability 8.0+ (RTX 30xx and newer) and `mvtools_cuda.dll` next to
`mvtools2.dll`.

* Reference of every GPU parameter, the `cudafilters` plugin and the environment variables: `Documentation/gpu.md`.
* `cudafilters/`: GPU versions of the AviSynth core, MaskTools2, RgTools and NNEDI3 stages QTGMC uses, plus
  `CudaNVOFAnalyse` (vectors from NVIDIA's optical flow accelerator).
* `qtgmc/`: `make_qtgmc_gpu.py` makes `QTGMC_gpu.avsi` from an unmodified QTGMC.avsi; `qtgmc/bundle/` holds the loader
  and the example scripts of the Windows release bundle.
* Windows release bundles (AviSynth+ x64, DLLs, patched QTGMC, examples, docs): GitHub releases `gpu-qtgmc-*` of
  github.com/emcodem/mvtools.
* Design notes and measurements: `GPU_PLAN.md`, `GPU_QTGMC_PLAN.md`, `GPU_PIPELINE_PLAN.md`, `DECISIONS.md`.

## VapourSynth (this branch; Linux, Windows to follow)

Not part of the Windows release bundles yet.

mvtools2 runs under VapourSynth (API 4.0, so R55 or newer; tested with R81) unchanged: `mvtools_vsbridge` is a small AviSynth+ host on top of
the VapourSynth API, and `libmvtools2.so` / `mvtools2.dll` only gained a `VapourSynthPluginInit2` entry point that loads
it. Under AviSynth+ nothing changes. Design, decisions and test results: `VAPOURSYNTH_PLAN.md`.

### Quick start

Put these files into one folder (VapourSynth's plugin folder for autoloading, or any folder you load from):

| File | Needed for |
|---|---|
| `libmvtools2.so` | mvtools2 (`core.mvpf`) |
| `libmvtools_vsbridge.so` | always: the bridge both plugins load from their own folder |
| `libmvtools_cuda.so` | optional: `gpu=true` and the cudafilters' GPU paths (without it: CPU, see below) |
| `libcudafilters.so` | optional: cudafilters (`core.cuda`) |
| `nnedi3_weights.bin` | optional: CudaNNEDI3 on Linux |

Denoising on the CPU (the AviSynth function and parameter names, `Documentation/mvtools2.html` applies):

    import vapoursynth as vs
    core = vs.core
    core.std.LoadPlugin('/path/to/plugins/libmvtools2.so')   # not needed when autoloaded

    clip = core.lsmas.LWLibavSource('input.mkv')             # any source filter
    sup = core.mvpf.MSuper(clip, pel=2)
    bv1 = core.mvpf.MAnalyse(sup, isb=True, delta=1, blksize=16, overlap=8)
    fv1 = core.mvpf.MAnalyse(sup, isb=False, delta=1, blksize=16, overlap=8)
    bv2 = core.mvpf.MAnalyse(sup, isb=True, delta=2, blksize=16, overlap=8)
    fv2 = core.mvpf.MAnalyse(sup, isb=False, delta=2, blksize=16, overlap=8)
    core.mvpf.MDegrain2(clip, sup, bv1, fv1, bv2, fv2, thSAD=400).set_output()

The same on the GPU (`Documentation/gpu.md`): add `gpu=True` to MSuper, MAnalyse and MDegrain2. To keep the frames on
the GPU between filters, wrap the chain in cudafilters' upload / download:

    core.std.LoadPlugin('/path/to/plugins/libcudafilters.so')
    u = core.cuda.CudaUpload(clip)
    sup = core.mvpf.MSuper(u, pel=2, gpu=True)
    bv = core.mvpf.MAnalyse(sup, isb=True, blksize=16, overlap=8, gpu=True)
    fv = core.mvpf.MAnalyse(sup, isb=False, blksize=16, overlap=8, gpu=True)
    out = core.mvpf.MDegrain1(u, sup, bv, fv, thSAD=400, gpu=True)
    core.cuda.CudaDownload(out).set_output()

Frame-rate conversion works the same way: `core.mvpf.MFlowFps(clip, sup, bv, fv, num=60000, den=1001, gpu=True)`.
Check stderr on the first run: a missing bridge or GPU library is reported there, never as a crash. Without
`libmvtools_cuda.so`, mvtools2 computes `gpu=True` on the CPU with the same output; a cudafilter whose CPU fallback is an
AviSynth-only filter stops with an error instead (set `MVBRIDGE_FALLBACK=nearest` to accept a close, not bit-identical,
VapourSynth replacement; details below).

* Build: on by default (`-DMVTOOLS_VAPOURSYNTH=OFF` to skip); no VapourSynth installation needed (`VapourSynth4.h` is
  in `Sources/include`). Status: Linux done; Windows waits for the CMake build of this branch (the MSVC solution does
  not build the bridge yet).
* Install: `libmvtools_vsbridge.so` must sit in the same folder as `libmvtools2.so` (VapourSynth's plugin folder, or
  any folder used with `core.std.LoadPlugin`). Without it the namespace `mvpf` exists but is empty, and the reason is
  printed to stderr.
* Use: namespace `mvpf` (dubhater's VapourSynth port keeps `mv`, both can be loaded), the AviSynth function and
  parameter names, so `Documentation/mvtools2.html` applies; booleans are 0 / 1 or True / False:

        sup = core.mvpf.MSuper(clip, pel=2)
        bv = core.mvpf.MAnalyse(sup, isb=True, blksize=16, overlap=8)
        fv = core.mvpf.MAnalyse(sup, isb=False, blksize=16, overlap=8)
        out = core.mvpf.MDegrain1(clip, sup, bv, fv, thSAD=400)

  Positional clips have the documentation's names (`clip`, `super`, `vectors`, `mvbw`, `mvfw`, `mvbw2`, ...;
  `MStoreVect` takes a list `clips`). Parameter names are case-sensitive in VapourSynth; where mvtools2 spells a name
  differently between filters (`thsad` / `thSAD`), the most common spelling is used everywhere.
* Formats: Gray, YUV 4:2:0 / 4:2:2 / 4:4:4, 8-16 bit, 32 bit float where mvtools2 supports it. Vector
  clips appear in VapourSynth as Gray8 clips; pass them only to `mvpf` filters: a vector or super clip that went through
  another VapourSynth filter (e.g. `std.Trim`) is not recognised.
### Details

* Output is bit-identical with AviSynth+ on the same system (17 CPU, 7 GPU and 17 cudafilters test chains,
  `vsbridge/test/compare.py`; further cudafilters tests against AviSynth+ on Windows in `vsbridge/test/`).
  `gpu=true` works as under AviSynth+ (`Documentation/gpu.md`): `libmvtools_cuda.so` goes into the same folder; without
  it, or with a damaged one, the GPU filters run on the CPU. Not verified yet: MDepan.
* cudafilters: namespace `cuda` (`core.cuda.CudaUpload`, ...), `libcudafilters.so` next to the bridge. All filters work
  and give AviSynth+'s output bit for bit (tested against AviSynth+ with the original masktools2 / RgTools / nnedi3 /
  QTGMC.avsi, with mvtools2 `gpu=true` chains on token frames, and CudaNVOFAnalyse with NVOFA or its MAnalyse
  fallback, which needs `libmvtools2.so` loaded too). CudaNNEDI3 needs `nnedi3_weights.bin` (the standard
  13574928-byte file) next to `libcudafilters.so` on Linux. Planar RGB passes through, but has no GPU path.
  A cudafilter's CPU fallback (AviSynth filters such as `Merge`, `Bob`, `RemoveGrain`, `nnedi3`, masktools) does not exist
  under VapourSynth; it runs only when the GPU is unusable or fails for a frame. `MVBRIDGE_FALLBACK=strict` (default) makes
  such a frame an error, so the output is AviSynth+'s or an error; `MVBRIDGE_FALLBACK=nearest` uses the closest
  VapourSynth filter where one exists (`std.Merge`, `std.Expr`, `std.Maximum` / `Minimum`, `resize.Bicubic` /
  `Bilinear`), which is not bit-identical; TemporalSoften, GaussResize, RemoveGrain, nnedi3 and
  QTGMC_KeepOnlyBobShimmerFixes have none. Exception: `mt_lutxy` / `mt_lut` / `mt_adddiff` (CudaLutxy,
  CudaQtgmcAddNoise and the tables of CudaKeepOnlyBobShimmerFixes / CudaQtgmcSharpLimit) are computed by the bridge
  itself, bit-identical with masktools2 2.2.30 under AviSynth+ 3.7.5, also as CPU fallback (expressions with pow, %,
  exp, log or trigonometric functions are refused).
* Debugging: `MVBRIDGE_DEBUG=1` prints frame-request misses, instance counts and a backtrace on a crash;
  `MVBRIDGE_SERIAL=MAnalyse,MFlowFps` (or `all`) runs the named filters single-threaded.
