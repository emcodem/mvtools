# Configure + build + test with Ninja inside the VS 2022 x64 environment.
# usage: pwsh -File tools\build.ps1 [-Cuda <toolkit root containing bin\nvcc.exe>] [-BuildDir build-cuda] [-Config Release]
#        [-CudaArch 61-virtual]  (CUDA architectures; default: CMakeLists.txt, Pascal .. Blackwell)
# Without -Cuda: uses $env:CUDA_PATH if set, otherwise builds the CPU-only DLL.
param([string]$Cuda = $env:CUDA_PATH, [string]$BuildDir = "build-cuda", [string]$Config = "Release", [string]$CudaArch = "")
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$cmakeDir = "$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
$cudaArgs = if ($Cuda) { $c = $Cuda -replace '\\', '/'; "-DCUDACONVOLUTION_CUDA=ON -DCMAKE_CUDA_COMPILER=$c/bin/nvcc.exe -DCUDAToolkit_ROOT=$c" }
            else { "-DCUDACONVOLUTION_CUDA=OFF" }
if ($Cuda -and $CudaArch) { $cudaArgs += " `"-DCUDACONVOLUTION_CUDA_ARCH=$CudaArch`"" }

# vcvars64.bat calls vswhere by name: put the installer folder on PATH
$env:PATH = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer;$env:PATH"
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && cd /d `"$root`" && `"$cmakeDir\cmake.exe`" -S . -B $BuildDir -G Ninja -DCMAKE_BUILD_TYPE=$Config $cudaArgs && `"$cmakeDir\cmake.exe`" --build $BuildDir && `"$cmakeDir\ctest.exe`" --test-dir $BuildDir --output-on-failure"
exit $LASTEXITCODE
