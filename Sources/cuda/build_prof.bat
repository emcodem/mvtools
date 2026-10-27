@echo off
rem Profiling build of mvtools_cuda.dll (clock64 counters per phase of the analysis kernel), output x64\Prof\
setlocal
if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "PATH=%CUDA_PATH%\bin;%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
if not exist x64\Prof mkdir x64\Prof
nvcc -shared -o x64\Prof\mvtools_cuda.dll mvcuda.cu mvcuda_super.cu mvcuda_recalc.cu mvcuda_analyse.cu -DMVCUDA_EXPORTS -DMVCUDA_PROFILE -O3 --std=c++17 ^
  -gencode arch=compute_120,code=sm_120 -Xcompiler "/MT /O2" || exit /b 1
echo built x64\Prof\mvtools_cuda.dll
