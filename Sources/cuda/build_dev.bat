@echo off
rem Development build of mvtools_cuda.dll: machine code for the local RTX 50xx (sm_120) only, so it compiles several times
rem faster than build.bat. Not for distribution (build.bat makes the multi-architecture release DLL).
rem   build_dev.bat <output dir> [extra nvcc flags, e.g. -DMVC_PHASE_PROF]
rem Toolkit and host compiler as build.bat: CUDA 12 (MVCUDA_CUDA_PATH, else CUDA_PATH_V12_9, else the default 12.9
rem folder) with MSVC 14.44 (MVCUDA_VCVARS_VER, "default" = the installation's default toolset).
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
set "CUDA_DIR=%MVCUDA_CUDA_PATH%"
if not defined CUDA_DIR set "CUDA_DIR=%CUDA_PATH_V12_9%"
if not defined CUDA_DIR set "CUDA_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
if not exist "%CUDA_DIR%\bin\nvcc.exe" (echo nvcc not found in "%CUDA_DIR%\bin": install CUDA 12.9 or set MVCUDA_CUDA_PATH & exit /b 1)
if not defined MVCUDA_VCVARS_VER set "MVCUDA_VCVARS_VER=14.44"
set "VCARG="
if /i not "%MVCUDA_VCVARS_VER%"=="default" set "VCARG=-vcvars_ver=%MVCUDA_VCVARS_VER%"
set "PATH=%CUDA_DIR%\bin;%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" %VCARG% >nul || exit /b 1
if not exist "%OUT%" mkdir "%OUT%"
nvcc -shared -o "%OUT%\mvtools_cuda.dll" mvcuda.cu mvcuda_core.cu mvcuda_frames.cu mvcuda_resample.cu mvcuda_temporal.cu mvcuda_merge.cu mvcuda_kobs.cu mvcuda_nnedi3.cu mvcuda_comp.cu mvcuda_nvofa.cu mvcuda_super_8.cu mvcuda_super_16.cu mvcuda_recalc_8.cu mvcuda_recalc_16.cu mvcuda_analyse_8.cu mvcuda_analyse_16.cu mvcuda_flow_8.cu mvcuda_flow_16.cu mvcuda_nvof_vec_8.cu mvcuda_nvof_vec_16.cu -DMVCUDA_EXPORTS -O3 --std=c++17 %EXTRA% ^
  -gencode arch=compute_120,code=sm_120 -Xcompiler "/MT /O2" || exit /b 1
echo built %OUT%\mvtools_cuda.dll
