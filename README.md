# mvtools
Motion estimation and compensation plugin for Avisynth+ and Avisynth v2.6 family.\
Supporting YUY2, 4:2:0, 4:2:2, 4:4:4 at native 8, 10, 12, 14 and 16 bit depths, 32bit float in selected filters.\
Still supporting Windows XP.
x86 and x64 versions
From December 20, 2020: Linux port

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

* Install binaries

        cd build
        sudo make install
