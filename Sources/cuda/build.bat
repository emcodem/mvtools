@echo off
rem Builds the release mvtools_cuda.dll with nvcc: machine code for every supported architecture. Needs the CUDA 12
rem toolkit and the MSVC build tools. Output: x64\Release\mvtools_cuda.dll (static CUDA runtime: only the NVIDIA driver is
rem needed at run time).
rem
rem Toolkit: CUDA 12 only, 12.8 or newer (the first with RTX 50xx code), 12.9 recommended. CUDA 13 cannot compile for
rem Pascal (6.x) or Volta (7.0) any more and is refused. Taken from MVCUDA_CUDA_PATH if set, else CUDA_PATH_V12_9 /
rem CUDA_PATH_V12_8 (set by the CUDA installers), else the default CUDA 12.9 install folder, else CUDA_PATH.
rem Architectures: the device code runs on compute capability 6.0 and newer (mvcuda_internal.h, MVC_ARCH_BELOW); machine
rem code for 6.1 (GTX 10xx), 7.0 (V100 / Titan V), 7.5 (RTX 20xx), 8.0 (A100), 8.6 (RTX 30xx), 8.9 (RTX 40xx), 9.0
rem (H100), 12.0 (RTX 50xx), PTX 6.0 (Tesla P100: JIT-compiled by the driver on first use) and PTX 8.0 (newer
rem architectures, e.g. 10.0, JIT-compiled from the fast 8.0 code).
rem Host compiler: CUDA 12 accepts MSVC up to Visual Studio 2022 (toolset v143, 14.3x / 14.4x); the STL of VS 2026
rem (14.5x) does not compile with it. vcvars: MVCUDA_VCVARS (default: the VS 18 Build Tools vcvars64.bat) with
rem -vcvars_ver=%MVCUDA_VCVARS_VER% (default 14.44, i.e. the "MSVC v143" component of the VS 2026 installer; "default" =
rem the installation's default toolset). With a VS 2022 installation, point MVCUDA_VCVARS at its vcvars64.bat and set
rem MVCUDA_VCVARS_VER=default.
rem Optional: MVCUDA_OUT (output folder, default x64\Release), MVCUDA_NVCC_FLAGS (extra nvcc flags).
setlocal
cd /d "%~dp0"
set "CUDA_DIR=%MVCUDA_CUDA_PATH%"
if not defined CUDA_DIR set "CUDA_DIR=%CUDA_PATH_V12_9%"
if not defined CUDA_DIR set "CUDA_DIR=%CUDA_PATH_V12_8%"
if not defined CUDA_DIR if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" set "CUDA_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
if not defined CUDA_DIR set "CUDA_DIR=%CUDA_PATH%"
if not exist "%CUDA_DIR%\bin\nvcc.exe" (echo nvcc not found in "%CUDA_DIR%\bin": install CUDA 12.9 or set MVCUDA_CUDA_PATH & exit /b 1)
set "NVCC=%CUDA_DIR%\bin\nvcc.exe"
set "NVCC_REL="
for /f "tokens=5 delims=, " %%a in ('call "%NVCC%" --version ^| findstr release') do set "NVCC_REL=%%a"
set "NVCC_MAJOR="
set "NVCC_MINOR="
for /f "tokens=1,2 delims=." %%m in ("%NVCC_REL%") do (set "NVCC_MAJOR=%%m" & set "NVCC_MINOR=%%n")
if not defined NVCC_MINOR (echo cannot read the nvcc version & exit /b 1)
if not "%NVCC_MAJOR%"=="12" (echo CUDA %NVCC_REL% in "%CUDA_DIR%": mvtools_cuda is built with CUDA 12.8 .. 12.x only, 12.9 recommended; set MVCUDA_CUDA_PATH & exit /b 1)
if %NVCC_MINOR% LSS 8 (echo CUDA %NVCC_REL% in "%CUDA_DIR%" has no RTX 50xx target: CUDA 12.8 or newer 12.x needed, 12.9 recommended & exit /b 1)
set "ARCH=-gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_60,code=compute_60 -gencode arch=compute_80,code=compute_80 -Wno-deprecated-gpu-targets"
if not defined MVCUDA_VCVARS_VER set "MVCUDA_VCVARS_VER=14.44"
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
