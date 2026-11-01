<#
.SYNOPSIS
  Builds every plugin of this repo with CMake + Ninja + MSVC (x64, Release, static CRT) and collects the DLLs in one folder.

.DESCRIPTION
  Output (-Out, default <repo>\build\win-x64\bin): mvtools2.dll, DePan.dll, DePanEstimate.dll, mvtools_vsbridge.dll,
  cudafilters.dll, mvtools_cuda.dll (when a CUDA 12 toolkit and the MSVC 14.4x toolset are found).

  Tools are taken from the parameters, else from the environment, else from the usual install locations:
    MSVC     newest Visual Studio / Build Tools with the C++ workload (vswhere); vcvars64.bat is run here. With CUDA the
             toolset is MSVC 14.4x (v143; -VcvarsVer, default: the newest 14.4x installed): CUDA 12 does not accept the
             VS 2026 STL (14.5x). VS 2026: install the component "MSVC v143 - VS 2022 C++ x64/x86 build tools".
    CMake / Ninja  the copies inside Visual Studio, else the PATH
    NASM     -Nasm, %NASMPATH%, the PATH, C:\dev\tools\nasm, Visual Studio's VC folder
    nvcc     CUDA 12.8 .. 12.x only (12.9 recommended; CUDA 13 has no Pascal / Volta targets): -Nvcc, %CUDA_PATH_V12_9%,
             %CUDA_PATH_V12_8%, the newest 12.x under "NVIDIA GPU Computing Toolkit", %CUDA_PATH%, the PATH;
             -NoCuda skips mvtools_cuda

.EXAMPLE
  .\build_windows.ps1
  .\build_windows.ps1 -BuildDir C:\tmp\b -CudaArch 120-real           # mvtools_cuda for the local RTX 50xx only, faster
  .\build_windows.ps1 -CudaArch 61-virtual       # Pascal code only, JIT-compiled for the local GPU (tests those paths)
  .\build_windows.ps1 -NoCuda                    # CPU plugins + bridge + cudafilters only, about 1 minute
#>
[CmdletBinding()]
param(
  [string]$BuildDir = (Join-Path $PSScriptRoot 'build\win-x64'),
  [string]$Out = '',
  [ValidateSet('Release', 'RelWithDebInfo', 'Debug')][string]$Config = 'Release',
  [string]$Nasm = '',
  [string]$Nvcc = '',
  [string]$CudaArch = '',
  [string]$VcvarsVer = '',
  [switch]$NoCuda,
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
# CUDA release of an nvcc ("release 12.9" -> 12.9), or $null
function Get-NvccVersion([string]$nvcc) {
  if (-not $nvcc -or -not (Test-Path -LiteralPath $nvcc)) { return $null }
  $m = (& $nvcc --version 2>$null | Out-String) | Select-String 'release (\d+)\.(\d+)'
  if ($m) { return [version]"$($m.Matches[0].Groups[1].Value).$($m.Matches[0].Groups[2].Value)" } else { return $null }
}
function Test-Cuda12([version]$v) { return $v -and $v -ge [version]'12.8' -and $v.Major -eq 12 }

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsPath = $null
if (Test-Path $vswhere) {
  $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
}
if (-not $vsPath) { throw 'No Visual Studio / Build Tools with the C++ x64 tools found (vswhere).' }

# --- CUDA 12 (before MSVC: it decides the toolset) --------------------------------------------------------------------
$cudaArgs = @()
if ($NoCuda) {
  $cudaArgs += '-DMVTOOLS_CUDA=OFF'
} else {
  if ($Nvcc) {
    $v = Get-NvccVersion $Nvcc
    if (-not (Test-Cuda12 $v)) { throw "-Nvcc $Nvcc is CUDA $v; mvtools_cuda is built with CUDA 12.8 .. 12.x only (12.9 recommended)" }
  } else {
    $tk = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA'
    $v12 = if (Test-Path $tk) {
      Get-ChildItem $tk -Directory | Where-Object { $_.Name -match '^v12\.\d+$' } |
        Sort-Object { [version]($_.Name.TrimStart('v')) } -Descending | ForEach-Object { Join-Path $_.FullName 'bin\nvcc.exe' }
    }
    $cands = @($env:CUDA_PATH_V12_9, $env:CUDA_PATH_V12_8 | Where-Object { $_ } | ForEach-Object { Join-Path $_ 'bin\nvcc.exe' }) +
      @($v12) + @($(if ($env:CUDA_PATH) { Join-Path $env:CUDA_PATH 'bin\nvcc.exe' }), (Find-OnPath 'nvcc'))
    $Nvcc = $cands | Where-Object { $_ -and (Test-Cuda12 (Get-NvccVersion $_)) } | Select-Object -First 1
  }
  if (-not $Nvcc) {
    Write-Warning 'no CUDA 12.8 .. 12.x nvcc found (install CUDA 12.9, or -Nvcc): mvtools_cuda.dll is not built'
    $cudaArgs += '-DMVTOOLS_CUDA=OFF'
  } elseif (-not $VcvarsVer) {
    # CUDA 12 needs MSVC 14.4x / 14.3x (v143): the newest one installed
    $msvc = Join-Path $vsPath 'VC\Tools\MSVC'
    $t = if (Test-Path $msvc) {
      Get-ChildItem $msvc -Directory | Where-Object { $_.Name -match '^14\.[34]\d\.' } |
        Sort-Object { [version]$_.Name } -Descending | Select-Object -First 1
    }
    if ($t) {
      $VcvarsVer = ([version]$t.Name).ToString(2)
    } else {
      $cur = Get-ChildItem $msvc -Directory -ErrorAction SilentlyContinue | Sort-Object { [version]$_.Name } -Descending | Select-Object -First 1
      if ($cur -and [version]$cur.Name -ge [version]'14.50') {
        Write-Warning (('CUDA 12 needs the MSVC 14.4x toolset (v143), only {0} is installed: mvtools_cuda.dll is not built. ' +
          'Visual Studio installer: add "MSVC v143 - VS 2022 C++ x64/x86 build tools".') -f $cur.Name)
        $Nvcc = ''
        $cudaArgs += '-DMVTOOLS_CUDA=OFF'
      }
    }
  }
  if ($Nvcc) {
    Write-Host "CUDA $(Get-NvccVersion $Nvcc): $Nvcc"
    $cudaArgs += '-DMVTOOLS_CUDA=ON'
    $cudaArgs += "-DCMAKE_CUDA_COMPILER=$($Nvcc -replace '\\','/')"
    if ($CudaArch) { $cudaArgs += "-DMVTOOLS_CUDA_ARCH=$CudaArch" }
  }
}

# --- MSVC environment -----------------------------------------------------------------------------------------------
$vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
if ($VcvarsVer -or -not $env:VCToolsInstallDir -or $env:VSCMD_ARG_TGT_ARCH -ne 'x64') {
  Write-Host "MSVC: $vcvars $(if ($VcvarsVer) { "(toolset $VcvarsVer)" })"
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

$vsArgs = @()
$vsArgs += $(if ($NoVapourSynth) { '-DMVTOOLS_VAPOURSYNTH=OFF' } else { '-DMVTOOLS_VAPOURSYNTH=ON' })
if (-not $Out) { $Out = Join-Path $BuildDir 'bin' }

if ($Clean -and (Test-Path (Join-Path $BuildDir 'CMakeCache.txt'))) { Remove-Item (Join-Path $BuildDir 'CMakeCache.txt') }

$configure = @('-G', 'Ninja', '-S', $PSScriptRoot, '-B', $BuildDir,
  "-DCMAKE_MAKE_PROGRAM=$($ninja -replace '\\','/')",
  "-DCMAKE_BUILD_TYPE=$Config",
  '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl',
  "-DCMAKE_ASM_NASM_COMPILER=$($Nasm -replace '\\','/')",
  "-DMVTOOLS_DIST_DIR=$($Out -replace '\\','/')") + $cudaArgs + $vsArgs + $CMakeArgs

Write-Host "cmake $($configure -join ' ')"
& $cmake @configure
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed ($LASTEXITCODE)" }
$sw = [Diagnostics.Stopwatch]::StartNew()
& $cmake --build $BuildDir --config $Config
if ($LASTEXITCODE -ne 0) { throw "Build failed ($LASTEXITCODE)" }
Write-Host ("Built in {0:N0} s. Plugins in {1}:" -f $sw.Elapsed.TotalSeconds, $Out)
Get-ChildItem $Out -Filter *.dll | ForEach-Object { Write-Host ("  {0,-24} {1,12:N0} bytes" -f $_.Name, $_.Length) }
