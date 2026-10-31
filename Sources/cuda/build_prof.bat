@echo off
rem Builds mvtools_cuda.dll with nvcc. Needs the CUDA toolkit (CUDA_PATH) and the MSVC build tools.
rem Output: x64\Prof\mvtools_cuda.dll (static CUDA runtime, only the NVIDIA driver is needed at run time).
rem Targets compute capability 8.0 and newer (RTX 30xx / A100 = 8.0 / 8.6, RTX 40xx = 8.9, H100 = 9.0, RTX 50xx = 12.0) as machine
rem code, plus PTX for 8.0 that the driver can JIT for anything newer. The code needs cp.async (8.0+); older cards report "no usable device".
setlocal
if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "PATH=%CUDA_PATH%\bin;%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
if not exist x64\Prof mkdir x64\Prof
nvcc -shared -o x64\Prof\mvtools_cuda.dll mvcuda.cu mvcuda_core.cu mvcuda_frames.cu mvcuda_resample.cu mvcuda_temporal.cu mvcuda_merge.cu mvcuda_kobs.cu mvcuda_nnedi3.cu mvcuda_comp.cu mvcuda_nvofa.cu mvcuda_super_8.cu mvcuda_super_16.cu mvcuda_recalc_8.cu mvcuda_recalc_16.cu mvcuda_analyse_8.cu mvcuda_analyse_16.cu mvcuda_flow_8.cu mvcuda_flow_16.cu mvcuda_nvof_vec_8.cu mvcuda_nvof_vec_16.cu -DMVCUDA_EXPORTS -DMVC_PHASE_PROF -O3 --std=c++17 ^
  -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 ^
  -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_80,code=compute_80 ^
  -Xcompiler "/MT /O2" || exit /b 1
echo built x64\Prof\mvtools_cuda.dll
