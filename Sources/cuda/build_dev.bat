@echo off
rem Development build of mvtools_cuda.dll: machine code for the local RTX 50xx (sm_120) only, so it compiles several times
rem faster than build.bat. Not for distribution (build.bat makes the multi-architecture release DLL).
rem   build_dev.bat <output dir> [extra nvcc flags, e.g. -DMVC_PHASE_PROF]
setlocal
if "%~1"=="" (echo usage: build_dev.bat outdir [nvcc flags] & exit /b 1)
set "OUT=%~1"
shift
set "EXTRA="
:args
if "%~1"=="" goto run
set "EXTRA=%EXTRA% %1"
shift
goto args
:run
if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "PATH=%CUDA_PATH%\bin;%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
if not exist "%OUT%" mkdir "%OUT%"
nvcc -shared -o "%OUT%\mvtools_cuda.dll" mvcuda.cu mvcuda_core.cu mvcuda_frames.cu mvcuda_resample.cu mvcuda_temporal.cu mvcuda_merge.cu mvcuda_kobs.cu mvcuda_nnedi3.cu mvcuda_comp.cu mvcuda_nvofa.cu mvcuda_super_8.cu mvcuda_super_16.cu mvcuda_recalc_8.cu mvcuda_recalc_16.cu mvcuda_analyse_8.cu mvcuda_analyse_16.cu mvcuda_flow_8.cu mvcuda_flow_16.cu mvcuda_nvof_vec_8.cu mvcuda_nvof_vec_16.cu -DMVCUDA_EXPORTS -O3 --std=c++17 %EXTRA% ^
  -gencode arch=compute_120,code=sm_120 -Xcompiler "/MT /O2" || exit /b 1
echo built %OUT%\mvtools_cuda.dll
