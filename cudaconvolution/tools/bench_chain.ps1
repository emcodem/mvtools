# Full GPU chain, stage by stage: decode -> QTGMC -> CudaSplatFps -> CudaConvolution, with the settings of the full-chain
# package's example_full_chain.avs (QTGMC Fast, motion vectors from NVOFA, wavefront search where a search is needed,
# NNEDI3 on the CPU, SplatFps with 16 candidates, Prefetch after QTGMC only). Each stage includes the ones before it.
# Runs from the benchmark package folder (QTGMC_gpu_load.avsi, scripts\, plugins\ with CudaConvolution.dll, splatfps.dll,
# LSMASHSource.dll and the GPU QTGMC plugins of one build). Needs AviSynth+ and an ffmpeg with AviSynth input.
#
#   pwsh -File bench_chain.ps1 -Source C:\path\to\pal_1080i25.mxf                 1080i25 -> 1080i29.97 (default)
#   pwsh -File bench_chain.ps1 -Source C:\path\to\ntsc_1080i2997.mxf -Direction 2997to25
#   pwsh -File bench_chain.ps1                         makes a 50 s moving test pattern file first (MPEG-2 4:2:2 MXF)
#
# Real-time factor = source duration / processing time (as ffmpeg's "speed="); >= 1 is faster than real time.
# Results: printed, and saved to bench_chain_<computer>_<date>.txt next to this script.
param(
  [string]$FFmpeg = 'ffmpeg',
  [string]$Source = '',
  [ValidateSet('25to2997', '2997to25')][string]$Direction = '25to2997',
  [ValidateSet('tff', 'bff')][string]$FieldOrder = 'tff',
  [double]$Seconds = 0,          # limit the source to this many seconds (0: all; the test pattern file has 50 s)
  [string]$Preset = 'Fast',      # QTGMC preset
  [switch]$NoNvof,               # QTGMC motion search instead of vectors from the optical flow accelerator (RTX 20+)
  [switch]$EdiOnGpu,             # NNEDI3 on the GPU (only on CPUs where the nnedi3 plugin takes its AVX2 + FMA path)
  [switch]$FullSplat,            # CudaSplatFps with its default 29 candidates instead of 16
  [int]$Threads = 8              # Prefetch after QTGMC
)
# Continue, not Stop: Windows PowerShell 5.1 turns every stderr line of a native program (L-SMASH's indexing progress,
# ffmpeg's messages) into an error record, which 'Stop' would make fatal. Failures are checked explicitly below.
$ErrorActionPreference = 'Continue'
$Nvof = -not $NoNvof
$SplatLight = -not $FullSplat
$dir = $PSScriptRoot
$plugins = Join-Path $dir 'plugins'
$work = Join-Path $dir 'bench_work'
New-Item -ItemType Directory -Force $work | Out-Null
$log = Join-Path $dir ("bench_chain_{0}_{1}.txt" -f $env:COMPUTERNAME, (Get-Date -Format 'yyyy-MM-dd_HHmm'))
function Out-Log([string]$s) { Write-Host $s; Add-Content -Path $log -Value $s }

foreach ($f in 'QTGMC_gpu_load.avsi', 'scripts\QTGMC_gpu.avsi', 'plugins\CudaConvolution.dll', 'plugins\splatfps.dll',
               'plugins\LSMASHSource.dll', 'plugins\mvtools_cuda.dll') {
  if (-not (Test-Path (Join-Path $dir $f))) { throw "$f missing (run this script from the benchmark package folder)" }
}
if ($Nvof -and -not (Select-String -Path (Join-Path $dir 'scripts\QTGMC_gpu.avsi') -Pattern 'QTGMC_gpu_nvof' -SimpleMatch -Quiet)) {
  throw 'scripts\QTGMC_gpu.avsi has no NVOFA switch (QTGMC_gpu_nvof): use the QTGMC script of the full-chain package'
}
if (-not (Get-Command $FFmpeg -ErrorAction SilentlyContinue)) { throw "ffmpeg not found: pass -FFmpeg <path to ffmpeg.exe>" }

$in_rate = if ($Direction -eq '25to2997') { 25.0 } else { 30000.0 / 1001 }
$out = if ($Direction -eq '25to2997') { @{ num = 60000; den = 1001; name = '1080p59.94'; iname = '1080i29.97' } }
       else { @{ num = 50; den = 1; name = '1080p50'; iname = '1080i25' } }

# no source: a 50 s moving pattern, encoded once as an XDCAM-like file, so that decoding costs what it costs in practice
if (-not $Source) {
  $tag = if ($Direction -eq '25to2997') { '1080i25' } else { '1080i2997' }
  $Source = Join-Path $work "pattern_$tag.mxf"
  if (-not (Test-Path $Source)) {
    $fields = [int][Math]::Round(50 * 2 * $in_rate)
    $rate = if ($Direction -eq '25to2997') { 'AssumeFPS(50)' } else { 'AssumeFPS(60000, 1001)' }
    Set-Content "$work\pattern.avs" @"
wide = ColorBarsHD(3840, 1080).Trim(0, -1).Loop($fields)
moving = Animate(0, $($fields - 1), "Crop", wide, 0, 0, 1920, 1080, wide, 1918, 0, 1920, 1080)
moving.ConvertToYUV422(matrix="Rec709").$rate.AssumeTFF().SeparateFields().SelectEvery(4, 0, 3).Weave()
"@
    Write-Host "making the test pattern file $Source (once) ..."
    & $FFmpeg -v error -y -i "$work\pattern.avs" -an -c:v mpeg2video -pix_fmt yuv422p -b:v 50M -minrate 50M -maxrate 50M `
      -bufsize 36M -g 12 -bf 2 -flags +ildct+ilme -f mxf $Source
    if ($LASTEXITCODE) { throw "could not make the test pattern file" }
  }
  $FieldOrder = 'tff'
}

$assume = if ($FieldOrder -eq 'tff') { 'AssumeTFF()' } else { 'AssumeBFF()' }
$trim = if ($Seconds -gt 0) { ".Trim(0, $([int]($Seconds * $in_rate) - 1))" } else { '' }
$splat = if ($SplatLight) { "CudaSplatFps($($out.num), $($out.den), block_peaks=[1,2,3], nearest_block=[0,0,1])" }
         else { "CudaSplatFps($($out.num), $($out.den))" }
$head = @"
LoadPlugin("$plugins\LSMASHSource.dll")
LoadPlugin("$plugins\splatfps.dll")
LoadPlugin("$plugins\CudaConvolution.dll")
Import("$dir\QTGMC_gpu_load.avsi")
v = LWLibavVideoSource("$Source")$trim.$assume
QTGMC_gpu_on()
global QTGMC_gpu_edi = $(([bool]$EdiOnGpu).ToString().ToLower())
global QTGMC_gpu_nvof = $($Nvof.ToString().ToLower())
global QTGMC_gpu_meander = false

"@

# index the source (LWLibavVideoSource writes <source>.lwi the first time) and count its frames
Set-Content "$work\chain_info.avs" ($head + "WriteFileStart(v, `"$work\chain_info.txt`", `"CudaConvolution_Version() + Chr(10) + string(v.FrameCount)`")")
if (Test-Path "$work\chain_info.txt") { Remove-Item "$work\chain_info.txt" }
& $FFmpeg -v error -i "$work\chain_info.avs" -frames:v 1 -f null - 2>&1 | Out-Null
$info = if (Test-Path "$work\chain_info.txt") { @(Get-Content "$work\chain_info.txt") } else { @() }
$srcFrames = if ($info.Count -ge 2) { [int]$info[-1] } else { 0 }
if ($srcFrames -le 0) { throw "the source could not be opened in AviSynth+ (run $work\chain_info.avs in a host that shows the error)" }
$duration = $srcFrames / $in_rate

Out-Log "Full chain benchmark (QTGMC -> CudaSplatFps -> CudaConvolution), $(Get-Date -Format 'yyyy-MM-dd HH:mm'), $env:COMPUTERNAME"
try { Out-Log ("GPU: " + ((& nvidia-smi --query-gpu=name,driver_version,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader) -join '; ')) } catch { Out-Log "GPU: nvidia-smi not available" }
try { Out-Log ("CPU: " + (Get-CimInstance Win32_Processor | Select-Object -First 1).Name.Trim() + ", " + [Environment]::ProcessorCount + " threads") } catch {}
Out-Log ("ffmpeg: " + ((& $FFmpeg -version 2>&1 | Select-Object -First 1)))
Out-Log ("  " + $info[0])
Out-Log ("source: $Source ($FieldOrder), $srcFrames frames = {0:N1} s; $Direction" -f $duration)
Out-Log ("settings: QTGMC Preset=$Preset, NVOFA $Nvof, NNEDI3 on GPU $([bool]$EdiOnGpu), SplatFps light $SplatLight, Prefetch($Threads) after QTGMC")
Out-Log ""

$q = "v.QTGMC(Preset=`"$Preset`").Prefetch($Threads)"
$stages = @(
  @{ name = '1 decode';                                 script = "v.Prefetch($Threads)" }
  @{ name = "2 + QTGMC -> " + $(if ($Direction -eq '25to2997') { '1080p50' } else { '1080p59.94' }); script = $q }
  @{ name = "3 + CudaSplatFps -> $($out.name)";         script = "$q.$splat" }
  @{ name = "4 + CudaConvolution -> $($out.iname) (full chain)"; script = "$q.$splat.CudaConvolution(field_order=`"$FieldOrder`")" }
  # CudaSplatFps and CudaConvolution each in its own thread (Prefetch(1) between them): CudaConvolution no longer waits in series
  @{ name = '5 full chain, pipelined (Prefetch(1))';     script = "$q.$splat.Prefetch(1).CudaConvolution(field_order=`"$FieldOrder`").Prefetch(1)" }
)
Out-Log ("{0,-42} {1,8} {2,10} {3,9} {4,8}" -f 'stage', 'frames', 'fps', 'x rt', 's')
$i = 0
foreach ($s in $stages) {
  $i++
  Set-Content "$work\chain$i.avs" ($head + $s.script)
  $progress = "$work\progress$i.txt"
  if (Test-Path $progress) { Remove-Item $progress }
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $o = & $FFmpeg -v error -progress $progress -i "$work\chain$i.avs" -an -f null - 2>&1
  $sw.Stop()
  $code = $LASTEXITCODE
  $n = 0
  if (Test-Path $progress) {
    $m = Select-String -Path $progress -Pattern '^frame=(\d+)' | Select-Object -Last 1
    if ($m) { $n = [int]$m.Matches[0].Groups[1].Value }
  }
  $t = $sw.Elapsed.TotalSeconds
  $note = if ($code -ne 0) { "  FAILED (exit $code) " + (($o | Select-Object -First 1) -join '') } else { '' }
  Out-Log ("{0,-42} {1,8} {2,10:N1} {3,9:N2} {4,8:N1}{5}" -f $s.name, $n, ($n / $t), ($duration / $t), $t, $note)
}
Out-Log ""
Out-Log "x rt: source duration / processing time (start-up included, as ffmpeg's speed=); >= 1 is faster than real time."
Out-Log "A stage's cost is the time against the line above. Saved: $log"
