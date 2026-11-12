# Development build of mvtools_cuda.dll (sm_120 only by default), with per-unit object caching and the units
# compiled in parallel. A unit is recompiled when one of its sources is newer than its object or the flags changed.
#   .\build_dev.ps1 -Out C:\dev\mvtools\profile\build_w [-Flags '-DMVC_PHASE_PROF']
param(
  [Parameter(Mandatory = $true)][string]$Out,
  [string]$Flags = '',
  # other targets, e.g. '-gencode arch=compute_61,code=compute_61' (PTX 6.1 only: the Pascal code paths, JIT-compiled for
  # the local GPU)
  [string]$Arch = '-gencode arch=compute_120,code=sm_120',
  # toolkit and host compiler as the CMake build: CUDA 12 (MVCUDA_CUDA_PATH, else CUDA_PATH_V12_9, else the default 12.9 folder)
  # with MSVC 14.44 ('default' = the installation's default toolset)
  [string]$VcvarsVer = '14.44'
)
$ErrorActionPreference = 'Stop'
$src = $PSScriptRoot
$obj = Join-Path $Out 'obj_cuda'
New-Item -ItemType Directory -Force $obj | Out-Null
$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$vcArg = if ($VcvarsVer -and $VcvarsVer -ne 'default') { " -vcvars_ver=$VcvarsVer" } else { '' }
$cuda = @($env:MVCUDA_CUDA_PATH, $env:CUDA_PATH_V12_9, 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9') |
  Where-Object { $_ -and (Test-Path (Join-Path $_ 'bin\nvcc.exe')) } | Select-Object -First 1
if (-not $cuda) { throw 'CUDA 12 nvcc not found (install CUDA 12.9 or set MVCUDA_CUDA_PATH)' }
$nvcc = Join-Path $cuda 'bin\nvcc.exe'
$common = "-DMVCUDA_EXPORTS -O3 --std=c++17 $Flags $Arch -Xcompiler `"/MT /O2`""
$hdr = @('mvcuda.h', 'mvcuda_internal.h')
# unit -> extra sources it includes
$units = [ordered]@{
  'mvcuda.cu' = @('mvcuda_nvofa.h'); 'mvcuda_core.cu' = @(); 'mvcuda_frames.cu' = @(); 'mvcuda_resample.cu' = @(); 'mvcuda_temporal.cu' = @(); 'mvcuda_merge.cu' = @(); 'mvcuda_kobs.cu' = @(); 'mvcuda_nnedi3.cu' = @(); 'mvcuda_comp.cu' = @(); 'mvcuda_nvofa.cu' = @('mvcuda_nvofa.h')
  'mvcuda_super_8.cu' = @('mvcuda_super.cu'); 'mvcuda_super_16.cu' = @('mvcuda_super.cu')
  'mvcuda_recalc_8.cu' = @('mvcuda_recalc.cu', 'mvcuda_refine.cuh'); 'mvcuda_recalc_16.cu' = @('mvcuda_recalc.cu', 'mvcuda_refine.cuh')
  'mvcuda_analyse_8.cu' = @('mvcuda_analyse.cu', 'mvcuda_refine.cuh'); 'mvcuda_analyse_16.cu' = @('mvcuda_analyse.cu', 'mvcuda_refine.cuh')
  'mvcuda_analyse_8_p1.cu' = @('mvcuda_analyse.cu', 'mvcuda_refine.cuh'); 'mvcuda_analyse_16_p1.cu' = @('mvcuda_analyse.cu', 'mvcuda_refine.cuh')
  'mvcuda_analyse_8_p2.cu' = @('mvcuda_analyse.cu', 'mvcuda_refine.cuh'); 'mvcuda_analyse_16_p2.cu' = @('mvcuda_analyse.cu', 'mvcuda_refine.cuh')
  'mvcuda_analyse_8_p3.cu' = @('mvcuda_analyse.cu', 'mvcuda_refine.cuh'); 'mvcuda_analyse_16_p3.cu' = @('mvcuda_analyse.cu', 'mvcuda_refine.cuh')
  'mvcuda_nvof_vec_8.cu' = @('mvcuda_nvof_vec.cu', 'mvcuda_nvofa.h'); 'mvcuda_nvof_vec_16.cu' = @('mvcuda_nvof_vec.cu', 'mvcuda_nvofa.h')
  'mvcuda_flow_8.cu' = @('mvcuda_flow.cu'); 'mvcuda_flow_16.cu' = @('mvcuda_flow.cu')
  'mvcuda_splat.cu' = @('mvcuda_splat_ctx.h'); 'mvcuda_splat_exact.cu' = @('mvcuda_splat.cuh', 'mvcuda_splat_ctx.cuh', 'mvcuda_splat_ctx.h', 'mvcuda_splat_me.cuh', 'mvcuda_splat_mo.cuh', '..\..\splatfps\core\splat_const.h', '..\..\splatfps\core\splat_phase_pull.h', '..\..\splatfps\third_party\pocketfft\fftp_pow2.h'); 'mvcuda_splat_fast.cu' = @('mvcuda_splat.cuh', 'mvcuda_splat_ctx.cuh', 'mvcuda_splat_ctx.h', 'mvcuda_splat_me.cuh', 'mvcuda_splat_mo.cuh', '..\..\splatfps\core\splat_const.h', '..\..\splatfps\core\splat_phase_pull.h', '..\..\splatfps\third_party\pocketfft\fftp_pow2.h')
}
# per-unit nvcc flags (as set_source_files_properties in CMakeLists.txt)
$unitFlags = @{ 'mvcuda_splat_exact.cu' = '-fmad=false' }
$stampFile = Join-Path $obj 'flags.txt'
# the toolkit and the host toolset are part of the stamp: objects of another compiler are rebuilt
$stamp = "$nvcc$vcArg $common"
$flagsChanged = -not (Test-Path $stampFile) -or ((Get-Content $stampFile -Raw).Trim() -ne $stamp)
$jobs = @()
foreach ($u in $units.Keys) {
  $o = Join-Path $obj ($u -replace '\.cu$', '.obj')
  $deps = @($u) + $units[$u] + $hdr | ForEach-Object { Get-Item (Join-Path $src $_) }
  $stale = $flagsChanged -or -not (Test-Path $o) -or (($deps | Measure-Object LastWriteTime -Maximum).Maximum -gt (Get-Item $o).LastWriteTime)
  if (-not $stale) { continue }
  $log = "$o.log"
  $bat = "$o.bat"
  Set-Content $bat -Encoding ascii -Value @(
    '@echo off',
    'set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"',
    "call `"$vcvars`"$vcArg >nul || exit /b 1",
    "cd /d `"$src`"",
    "`"$nvcc`" -c -o `"$o`" $u $common $($unitFlags[$u]) > `"$log`" 2>&1")
  $jobs += [pscustomobject]@{ unit = $u; log = $log; p = (Start-Process cmd -ArgumentList '/c', "`"$bat`"" -NoNewWindow -PassThru) }
}
$sw = [Diagnostics.Stopwatch]::StartNew()
$failed = $false
foreach ($j in $jobs) {
  $j.p.WaitForExit()
  if ($j.p.ExitCode -ne 0) { $failed = $true; Write-Host "FAILED $($j.unit):"; Get-Content $j.log | Select-String 'error' | Select-Object -First 10 | ForEach-Object { Write-Host "  $_" } }
}
if ($failed) { exit 1 }
Write-Host ("compiled {0} unit(s) in {1:N0} s: {2}" -f $jobs.Count, $sw.Elapsed.TotalSeconds, (($jobs | ForEach-Object unit) -join ' '))
Set-Content $stampFile $stamp
$objs = ($units.Keys | ForEach-Object { '"' + (Join-Path $obj ($_ -replace '\.cu$', '.obj')) + '"' }) -join ' '
$dll = Join-Path $Out 'mvtools_cuda.dll'
$bat = Join-Path $obj 'link.bat'
Set-Content $bat -Encoding ascii -Value @(
  '@echo off',
  'set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"',
  "call `"$vcvars`"$vcArg >nul || exit /b 1",
  "`"$nvcc`" -shared -o `"$dll`" $objs $Arch -Xcompiler `"/MT /O2`"")
cmd /c "`"$bat`"" 2>&1 | Select-String 'error' | ForEach-Object { Write-Host $_ }
if ($LASTEXITCODE -ne 0) { exit 1 }
Write-Host "built $dll"
