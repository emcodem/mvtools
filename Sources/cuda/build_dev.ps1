# Development build of mvtools_cuda.dll (sm_120 only, like build_dev.bat), with per-unit object caching and the units
# compiled in parallel. A unit is recompiled when one of its sources is newer than its object or the flags changed.
#   .\build_dev.ps1 -Out C:\dev\mvtools\profile\build_w [-Flags '-DMVC_PHASE_PROF']
param(
  [Parameter(Mandatory = $true)][string]$Out,
  [string]$Flags = ''
)
$ErrorActionPreference = 'Stop'
$src = $PSScriptRoot
$obj = Join-Path $Out 'obj_cuda'
New-Item -ItemType Directory -Force $obj | Out-Null
$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$cuda = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4' }
$nvcc = Join-Path $cuda 'bin\nvcc.exe'
$common = "-DMVCUDA_EXPORTS -O3 --std=c++17 $Flags -gencode arch=compute_120,code=sm_120 -Xcompiler `"/MT /O2`""
$hdr = @('mvcuda.h', 'mvcuda_internal.h')
# unit -> extra sources it includes
$units = [ordered]@{
  'mvcuda.cu' = @('mvcuda_nvofa.h'); 'mvcuda_core.cu' = @(); 'mvcuda_frames.cu' = @(); 'mvcuda_resample.cu' = @(); 'mvcuda_temporal.cu' = @(); 'mvcuda_merge.cu' = @(); 'mvcuda_kobs.cu' = @(); 'mvcuda_nnedi3.cu' = @(); 'mvcuda_comp.cu' = @(); 'mvcuda_nvofa.cu' = @('mvcuda_nvofa.h')
  'mvcuda_super_8.cu' = @('mvcuda_super.cu'); 'mvcuda_super_16.cu' = @('mvcuda_super.cu')
  'mvcuda_recalc_8.cu' = @('mvcuda_recalc.cu', 'mvcuda_refine.cuh'); 'mvcuda_recalc_16.cu' = @('mvcuda_recalc.cu', 'mvcuda_refine.cuh')
  'mvcuda_analyse_8.cu' = @('mvcuda_analyse.cu', 'mvcuda_refine.cuh'); 'mvcuda_analyse_16.cu' = @('mvcuda_analyse.cu', 'mvcuda_refine.cuh')
  'mvcuda_nvof_vec_8.cu' = @('mvcuda_nvof_vec.cu', 'mvcuda_nvofa.h'); 'mvcuda_nvof_vec_16.cu' = @('mvcuda_nvof_vec.cu', 'mvcuda_nvofa.h')
  'mvcuda_flow_8.cu' = @('mvcuda_flow.cu'); 'mvcuda_flow_16.cu' = @('mvcuda_flow.cu')
}
$stampFile = Join-Path $obj 'flags.txt'
$flagsChanged = -not (Test-Path $stampFile) -or ((Get-Content $stampFile -Raw).Trim() -ne $common)
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
    "call `"$vcvars`" >nul || exit /b 1",
    "cd /d `"$src`"",
    "`"$nvcc`" -c -o `"$o`" $u $common > `"$log`" 2>&1")
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
Set-Content $stampFile $common
$objs = ($units.Keys | ForEach-Object { '"' + (Join-Path $obj ($_ -replace '\.cu$', '.obj')) + '"' }) -join ' '
$dll = Join-Path $Out 'mvtools_cuda.dll'
$bat = Join-Path $obj 'link.bat'
Set-Content $bat -Encoding ascii -Value @(
  '@echo off',
  'set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"',
  "call `"$vcvars`" >nul || exit /b 1",
  "`"$nvcc`" -shared -o `"$dll`" $objs -gencode arch=compute_120,code=sm_120 -Xcompiler `"/MT /O2`"")
cmd /c "`"$bat`"" 2>&1 | Select-String 'error' | ForEach-Object { Write-Host $_ }
if ($LASTEXITCODE -ne 0) { exit 1 }
Write-Host "built $dll"
