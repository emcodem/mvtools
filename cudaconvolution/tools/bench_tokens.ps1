# CudaConvolution with GPU token frames: throughput of the copy path vs. zero-copy, plus a check that all chains give the same
# frames. Runs from the benchmark package folder (plugins\ holds CudaConvolution.dll and the mvtools GPU plugins of one build).
# Needs AviSynth+ and an ffmpeg with AviSynth input.
#
#   pwsh -File bench_tokens.ps1                                  # synthetic UHD 50p YUV422P10 source
#   pwsh -File bench_tokens.ps1 -Source C:\path\to\uhd50p.mov      # a real UHD 50p file (read with ffms2)
#   pwsh -File bench_tokens.ps1 -FFmpeg C:\path\to\ffmpeg.exe -Frames 2000
#
# Results: printed, and saved to bench_results_<computer>_<date>.txt next to this script.
param(
  [string]$FFmpeg = 'ffmpeg',
  [string]$Source = '',
  [int]$Frames = 1000,        # input frames (output: half, 1080i)
  [int]$Prefetch = 4
)
# Continue, not Stop: Windows PowerShell 5.1 turns every stderr line of a native program into an error record, which
# 'Stop' would make fatal. Failures are checked explicitly below.
$ErrorActionPreference = 'Continue'
$dir = $PSScriptRoot
$plugins = Join-Path $dir 'plugins'
$work = Join-Path $dir 'bench_work'
New-Item -ItemType Directory -Force $work | Out-Null
$log = Join-Path $dir ("bench_results_{0}_{1}.txt" -f $env:COMPUTERNAME, (Get-Date -Format 'yyyy-MM-dd_HHmm'))
function Out-Log([string]$s) { Write-Host $s; Add-Content -Path $log -Value $s }

foreach ($f in 'CudaConvolution.dll', 'cudafilters.dll', 'mvtools_cuda.dll') {
  if (-not (Test-Path (Join-Path $plugins $f))) { throw "plugins\$f missing" }
}
if (-not (Get-Command $FFmpeg -ErrorAction SilentlyContinue)) { throw "ffmpeg not found: pass -FFmpeg <path to ffmpeg.exe>" }

$src = if ($Source) {
  "LoadPlugin(`"$plugins\ffms2.dll`")`nsrc = FFVideoSource(`"$Source`", threads=0).Trim(0, $($Frames - 1)).Prefetch(1)"
} else {
  "src = ColorBars(3840, 2160, pixel_type=`"YUV422P10`").Trim(0, $($Frames - 1)).AssumeFPS(50)"
}
$head = @"
LoadPlugin("$plugins\cudafilters.dll")
LoadPlugin("$plugins\CudaConvolution.dll")
$src
up = CudaUpload(src)

"@

# machine and versions
Out-Log "CudaConvolution token-frame benchmark, $(Get-Date -Format 'yyyy-MM-dd HH:mm'), $env:COMPUTERNAME"
try { Out-Log ("GPU: " + ((& nvidia-smi --query-gpu=name,driver_version,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader) -join '; ')) } catch { Out-Log "GPU: nvidia-smi not available" }
Out-Log ("ffmpeg: " + ((& $FFmpeg -version 2>&1 | Select-Object -First 1)))
Out-Log ("source: " + $(if ($Source) { $Source } else { 'ColorBars 3840x2160 YUV422P10 (synthetic, static)' }) + ", $Frames input frames")
$info = Join-Path $work 'info.txt'
if (Test-Path $info) { Remove-Item $info }
Set-Content "$work\info.avs" ($head + "WriteFileStart(BlankClip(length=1), `"$info`", `"CudaConvolution_Version() + Chr(10) + CudaConvolution_SelfTest()`")")
& $FFmpeg -v error -i "$work\info.avs" -f null - 2>&1 | Out-Null
if (Test-Path $info) { Get-Content $info | ForEach-Object { Out-Log "  $_" } } else { Out-Log "  (CudaConvolution_Version / SelfTest not written: check that AviSynth+ loads the plugins)" }
Out-Log ""

$cases = [ordered]@{
  '1 ordinary frames -> CudaConvolution (reference)'       = 'CudaConvolution(src, width=1920, height=1080)'
  '2 tokens -> CudaDownload -> CudaConvolution (copy)'     = 'CudaConvolution(CudaDownload(up), width=1920, height=1080)'
  '3 tokens -> CudaConvolution (zero-copy)'                = 'CudaConvolution(up, width=1920, height=1080)'
  '4 tokens in + out, copy path'                    = 'CudaUpload(CudaConvolution(CudaDownload(up), width=1920, height=1080)).CudaDownload()'
  '5 tokens in + out, zero-copy (token_out=true)'   = 'CudaConvolution(up, width=1920, height=1080, token_out=true).CudaDownload()'
}

function Run-Md5([string]$avs) {
  $o = & $FFmpeg -v error -i $avs -f framemd5 - 2>&1
  [pscustomobject]@{ exit = $LASTEXITCODE; n = @($o | Where-Object { $_ -match '^0,' }).Count
    h = (($o | Where-Object { $_ -match '^0,' } | ForEach-Object { ($_ -split ',')[-1].Trim() }) -join '')
    err = ($o | Where-Object { $_ -notmatch '^[#0-9]' } | Select-Object -First 2) -join ' | ' }
}

# correctness first: every chain must give the frames of chain 1
Out-Log "Check: identical frames (framemd5 of all output frames)"
$ref = $null
$i = 0
foreach ($k in $cases.Keys) {
  $i++
  Set-Content "$work\c$i.avs" ($head + $cases[$k])
  $r = Run-Md5 "$work\c$i.avs"
  if ($i -eq 1) { $ref = $r }
  $state = if ($r.exit -ne 0 -or $r.n -eq 0) { "FAILED (exit $($r.exit)) $($r.err)" } elseif ($r.h -eq $ref.h) { 'identical' } else { 'DIFFERENT' }
  Out-Log ("  {0,-48} {1,5} frames  {2}" -f $k, $r.n, $state)
}
Out-Log ""

# throughput: warm-up run, then a timed run of all frames
foreach ($pf in @('', ".Prefetch($Prefetch)")) {
  Out-Log ("Throughput, " + $(if ($pf) { "Prefetch($Prefetch) at the end" } else { 'no Prefetch' }) + " (output frames per second)")
  $i = 0
  foreach ($k in $cases.Keys) {
    $i++
    Set-Content "$work\t$i.avs" ($head + $cases[$k] + $pf)
    & $FFmpeg -v error -i "$work\t$i.avs" -frames:v 20 -f null - 2>&1 | Out-Null
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $o = & $FFmpeg -v error -i "$work\t$i.avs" -f null - 2>&1
    $sw.Stop()
    $note = if ($LASTEXITCODE -ne 0) { "  FAILED (exit $LASTEXITCODE) " + (($o | Select-Object -First 1) -join '') } else { '' }
    Out-Log ("  {0,-48} {1,7:N1} fps  ({2,6:N1} s){3}" -f $k, ($Frames / 2 / $sw.Elapsed.TotalSeconds), $sw.Elapsed.TotalSeconds, $note)
  }
  Out-Log ""
}
Out-Log "Saved: $log"
