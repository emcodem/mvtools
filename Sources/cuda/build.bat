@echo off
rem Builds the release mvtools_cuda.dll with nvcc: machine code for every supported architecture. Needs a CUDA toolkit and
rem the MSVC build tools. Output: x64\Release\mvtools_cuda.dll (static CUDA runtime: only the NVIDIA driver is needed at
rem run time).
rem
rem Architectures. The device code runs on compute capability 6.0 and newer (mvcuda_internal.h, MVC_ARCH_BELOW).
rem   CUDA 12.x (12.8 or newer, 12.9 recommended): machine code for 6.1 (GTX 10xx), 7.0 (V100 / Titan V), 7.5 (RTX 20xx),
rem     8.0 (A100), 8.6 (RTX 30xx), 8.9 (RTX 40xx), 9.0 (H100), 12.0 (RTX 50xx), PTX 6.0 (Tesla P100: JIT-compiled by the
rem     driver on first use) and PTX 8.0 (newer architectures, e.g. 10.0, JIT-compiled from the fast 8.0 code).
rem   CUDA 13.x cannot compile for Pascal (6.x) or Volta (7.0) any more: with CUDA 13 the DLL gets 7.5 .. 12.0 + PTX 8.0
rem     only, and those older cards report "no usable device" (no code for them in the DLL).
rem
rem Toolkit: MVCUDA_CUDA_PATH if set, else CUDA_PATH_V12_9 (set by the CUDA 12.9 installer), else CUDA_PATH, else the
rem default CUDA 12.9 install folder.
rem Host compiler: CUDA 12.x accepts MSVC up to Visual Studio 2022 (toolset v143, 14.3x / 14.4x); the STL of VS 2026
rem (14.5x) does not compile with it. vcvars: MVCUDA_VCVARS (default: the VS 18 Build Tools vcvars64.bat) with
rem -vcvars_ver=%MVCUDA_VCVARS_VER% (default 14.44 with CUDA 12, i.e. the "MSVC v143" component of the VS 2026
rem installer; "default" = the installation's default toolset). With a VS 2022 installation, point MVCUDA_VCVARS at its
rem vcvars64.bat and set MVCUDA_VCVARS_VER=default.
rem Optional: MVCUDA_OUT (output folder, default x64\Release), MVCUDA_NVCC_FLAGS (extra nvcc flags).
setlocal
cd /d "%~dp0"
set "CUDA_DIR=%MVCUDA_CUDA_PATH%"
if not defined CUDA_DIR set "CUDA_DIR=%CUDA_PATH_V12_9%"
if not defined CUDA_DIR set "CUDA_DIR=%CUDA_PATH%"
if not defined CUDA_DIR set "CUDA_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
if not exist "%CUDA_DIR%\bin\nvcc.exe" (echo nvcc not found in "%CUDA_DIR%\bin" & exit /b 1)
set "NVCC=%CUDA_DIR%\bin\nvcc.exe"
set "NVCC_REL="
for /f "tokens=5 delims=, " %%a in ('call "%NVCC%" --version ^| findstr release') do set "NVCC_REL=%%a"
set "NVCC_MAJOR="
for /f "tokens=1 delims=." %%m in ("%NVCC_REL%") do set "NVCC_MAJOR=%%m"
if not defined NVCC_MAJOR (echo cannot read the nvcc version & exit /b 1)
set "ARCH=-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_80,code=compute_80"
if %NVCC_MAJOR% LSS 13 (
  set "ARCH=-gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 %ARCH% -gencode arch=compute_60,code=compute_60 -Wno-deprecated-gpu-targets"
  if not defined MVCUDA_VCVARS_VER set "MVCUDA_VCVARS_VER=14.44"
) else (
  echo note: CUDA %NVCC_REL% has no Pascal / Volta targets; this DLL needs compute capability 7.5 or newer. Use CUDA 12.9 for 6.x / 7.0.
)
if not defined MVCUDA_VCVARS set "MVCUDA_VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "VCARG="
if defined MVCUDA_VCVARS_VER if /i not "%MVCUDA_VCVARS_VER%"=="default" set "VCARG=-vcvars_ver=%MVCUDA_VCVARS_VER%"
set "PATH=%CUDA_DIR%\bin;%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "%MVCUDA_VCVARS%" %VCARG% >nul || (echo vcvars failed: "%MVCUDA_VCVARS%" %VCARG% & exit /b 1)
if not defined MVCUDA_OUT set "MVCUDA_OUT=x64\Release"
if not exist "%MVCUDA_OUT%" mkdir "%MVCUDA_OUT%"
echo CUDA %NVCC_REL% from "%CUDA_DIR%", %VCARG%
"%NVCC%" -shared -o "%MVCUDA_OUT%\mvtools_cuda.dll" mvcuda.cu mvcuda_core.cu mvcuda_frames.cu mvcuda_resample.cu mvcuda_temporal.cu mvcuda_merge.cu mvcuda_kobs.cu mvcuda_nnedi3.cu mvcuda_comp.cu mvcuda_nvofa.cu mvcuda_super_8.cu mvcuda_super_16.cu mvcuda_recalc_8.cu mvcuda_recalc_16.cu mvcuda_analyse_8.cu mvcuda_analyse_16.cu mvcuda_flow_8.cu mvcuda_flow_16.cu mvcuda_nvof_vec_8.cu mvcuda_nvof_vec_16.cu -DMVCUDA_EXPORTS -O3 --std=c++17 --threads 0 %MVCUDA_NVCC_FLAGS% ^
  %ARCH% -Xcompiler "/MT /O2" || exit /b 1
echo built %MVCUDA_OUT%\mvtools_cuda.dll
