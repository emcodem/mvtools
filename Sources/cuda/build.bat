@echo off
rem Builds mvtools_cuda.dll with nvcc. Needs the CUDA toolkit (CUDA_PATH) and the MSVC build tools.
rem Output: x64\Release\mvtools_cuda.dll (static CUDA runtime, only the NVIDIA driver is needed at run time).
setlocal
if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "PATH=%CUDA_PATH%\bin;%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
if not exist x64\Release mkdir x64\Release
nvcc -shared -o x64\Release\mvtools_cuda.dll mvcuda.cu mvcuda_super.cu mvcuda_recalc.cu mvcuda_analyse.cu -DMVCUDA_EXPORTS -O3 --std=c++17 ^
  -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120 ^
  -Xcompiler "/MT /O2" || exit /b 1
echo built x64\Release\mvtools_cuda.dll
