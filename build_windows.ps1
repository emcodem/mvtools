<#
.SYNOPSIS
  Builds every plugin of this repo with CMake + Ninja + MSVC (x64, Release, static CRT) and collects the DLLs in one folder.

.DESCRIPTION
  Output (-Out, default <repo>\build\win-x64\bin): mvtools2.dll, DePan.dll, DePanEstimate.dll, mvtools_vsbridge.dll,
  cudafilters.dll, mvtools_cuda.dll (when a CUDA toolkit is found), mvtools_vulkan.dll (when glslangValidator is found).

  Tools are taken from the parameters, else from the environment, else from the usual install locations:
    MSVC     newest Visual Studio / Build Tools with the C++ workload (vswhere); vcvars64.bat is run here
    CMake / Ninja  the copies inside Visual Studio, else the PATH
    NASM     -Nasm, %NASMPATH%, the PATH, C:\dev\tools\nasm, Visual Studio's VC folder
    nvcc     -Nvcc, %CUDA_PATH%\bin, newest toolkit under "NVIDIA GPU Computing Toolkit"; -NoCuda skips mvtools_cuda
    glslang  -Glslang, %VULKAN_SDK%\Bin, the PATH; -NoVulkan skips mvtools_vulkan
    -VcvarsVer 14.44 selects an older MSVC toolset (vcvars64.bat -vcvars_ver); CUDA 12.x (needed for Pascal / GTX 10xx
    in mvtools_cuda) does not accept the VS 2026 STL (14.5x), so it needs the VS 2022 toolset (v143) installed

.EXAMPLE
  .\build_windows.ps1
  .\build_windows.ps1 -BuildDir C:\tmp\b -Nvcc 'C:\cuda129\bin\nvcc.exe' -CudaArch '61-real;75-real;86-real;120-real'
  .\build_windows.ps1 -NoCuda -NoVulkan          # CPU plugins + bridge + cudafilters only, about 1 minute
#>
[CmdletBinding()]
param(
  [string]$BuildDir = (Join-Path $PSScriptRoot 'build\win-x64'),
  [string]$Out = '',
  [ValidateSet('Release', 'RelWithDebInfo', 'Debug')][string]$Config = 'Release',
  [string]$Nasm = '',
  [string]$Nvcc = '',
  [string]$CudaArch = '',
  [string]$Glslang = '',
  [string]$VcvarsVer = '',
  [switch]$NoCuda,
  [switch]$NoVulkan,
  [switch]$NoVapourSynth,
  [switch]$Clean,
  [string[]]$CMakeArgs = @()
)
$ErrorActionPreference = 'Stop'

function Find-First([string[]]$candidates) {
  foreach ($c in $candidates) { if ($c -and (Test-Path -LiteralPath $c)) { return (Resolve-Path -LiteralPath $c).Path } }
  return $null
}
function Find-OnPath([string]$name) {
  $cmd = Get-Command $name -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($cmd) { return $cmd.Source } else { return $null }
}

# --- MSVC environment -----------------------------------------------------------------------------------------------
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsPath = $null
if (Test-Path $vswhere) {
  $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
}
if (-not $vsPath) { throw 'No Visual Studio / Build Tools with the C++ x64 tools found (vswhere).' }
$vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
if ($VcvarsVer -or -not $env:VCToolsInstallDir -or $env:VSCMD_ARG_TGT_ARCH -ne 'x64') {
  Write-Host "MSVC: $vcvars"
  $vcArgs = if ($VcvarsVer) { " -vcvars_ver=$VcvarsVer" } else { '' }
  $envDump = & cmd.exe /s /c "`"$vcvars`"$vcArgs >nul 2>&1 && set"
  if ($LASTEXITCODE -ne 0) { throw "vcvars64.bat failed" }
  foreach ($line in $envDump) {
    $i = $line.IndexOf('=')
    if ($i -gt 0) { Set-Item -Path ("env:" + $line.Substring(0, $i)) -Value $line.Substring($i + 1) }
  }
}

$vsCMake = Join-Path $vsPath 'Common7\IDE\CommonExtensions\Microsoft\CMake'
$cmake = Find-First @((Join-Path $vsCMake 'CMake\bin\cmake.exe'), (Find-OnPath 'cmake'))
$ninja = Find-First @((Join-Path $vsCMake 'Ninja\ninja.exe'), (Find-OnPath 'ninja'))
if (-not $cmake) { throw 'cmake.exe not found' }
if (-not $ninja) { throw 'ninja.exe not found' }

# --- NASM (mvtools2's assembler sources) ----------------------------------------------------------------------------
if (-not $Nasm) {
  $Nasm = Find-First @($(if ($env:NASMPATH) { Join-Path $env:NASMPATH 'nasm.exe' }), (Find-OnPath 'nasm'),
    'C:\dev\tools\nasm\nasm.exe', (Join-Path $vsPath 'VC\nasm.exe'), (Join-Path $env:ProgramFiles 'NASM\nasm.exe'))
}
if (-not $Nasm) { throw 'nasm.exe not found (-Nasm or NASMPATH); mvtools2 needs it with MSVC' }

# --- CUDA -------------------------------------------------------------------------------------------------------------
$cudaArgs = @()
if ($NoCuda) {
  $cudaArgs += '-DMVTOOLS_CUDA=OFF'
} else {
  if (-not $Nvcc) {
    $tk = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA'
    $newest = if (Test-Path $tk) { Get-ChildItem $tk -Directory | Sort-Object { [version]($_.Name.TrimStart('v')) } -Descending | Select-Object -First 1 }
    $Nvcc = Find-First @($(if ($env:CUDA_PATH) { Join-Path $env:CUDA_PATH 'bin\nvcc.exe' }),
      $(if ($newest) { Join-Path $newest.FullName 'bin\nvcc.exe' }), (Find-OnPath 'nvcc'))
  }
  if ($Nvcc) {
    $cudaArgs += '-DMVTOOLS_CUDA=ON'
    $cudaArgs += "-DCMAKE_CUDA_COMPILER=$($Nvcc -replace '\\','/')"
    if ($CudaArch) { $cudaArgs += "-DMVTOOLS_CUDA_ARCH=$CudaArch" }
  } else {
    Write-Warning 'nvcc not found: mvtools_cuda.dll is not built (-Nvcc or CUDA_PATH)'
    $cudaArgs += '-DMVTOOLS_CUDA=OFF'
  }
}

# --- Vulkan shaders -------------------------------------------------------------------------------------------------
$vkArgs = @()
if ($NoVulkan) {
  $vkArgs += '-DMVTOOLS_VULKAN=OFF'
} else {
  if (-not $Glslang) {
    $Glslang = Find-First @($(if ($env:VULKAN_SDK) { Join-Path $env:VULKAN_SDK 'Bin\glslangValidator.exe' }),
      (Find-OnPath 'glslangValidator'), (Find-OnPath 'glslang'))
  }
  $vkArgs += '-DMVTOOLS_VULKAN=ON'
  if ($Glslang) { $vkArgs += "-DMVTOOLS_GLSLANG=$($Glslang -replace '\\','/')" }
  else { Write-Warning 'glslangValidator not found: mvtools_vulkan.dll is not built (-Glslang or VULKAN_SDK)' }
}

$vsArgs = @()
$vsArgs += $(if ($NoVapourSynth) { '-DMVTOOLS_VAPOURSYNTH=OFF' } else { '-DMVTOOLS_VAPOURSYNTH=ON' })
if (-not $Out) { $Out = Join-Path $BuildDir 'bin' }

if ($Clean -and (Test-Path (Join-Path $BuildDir 'CMakeCache.txt'))) { Remove-Item (Join-Path $BuildDir 'CMakeCache.txt') }

$configure = @('-G', 'Ninja', '-S', $PSScriptRoot, '-B', $BuildDir,
  "-DCMAKE_MAKE_PROGRAM=$($ninja -replace '\\','/')",
  "-DCMAKE_BUILD_TYPE=$Config",
  '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl',
  "-DCMAKE_ASM_NASM_COMPILER=$($Nasm -replace '\\','/')",
  "-DMVTOOLS_DIST_DIR=$($Out -replace '\\','/')") + $cudaArgs + $vkArgs + $vsArgs + $CMakeArgs

Write-Host "cmake $($configure -join ' ')"
& $cmake @configure
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed ($LASTEXITCODE)" }
$sw = [Diagnostics.Stopwatch]::StartNew()
& $cmake --build $BuildDir --config $Config
if ($LASTEXITCODE -ne 0) { throw "Build failed ($LASTEXITCODE)" }
Write-Host ("Built in {0:N0} s. Plugins in {1}:" -f $sw.Elapsed.TotalSeconds, $Out)
Get-ChildItem $Out -Filter *.dll | ForEach-Object { Write-Host ("  {0,-24} {1,12:N0} bytes" -f $_.Name, $_.Length) }
