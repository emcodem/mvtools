<#
UConvert benchmark: speed of UConvert's stages on generated test clips (MovingPattern), one result file to post.

Each stage runs for a fixed time (-Seconds) on a clip generated in memory before the run, so the source costs
nothing and the whole benchmark takes about a minute on any machine. The speed of a stage is the median of its
frames per second over 1 s windows, after its start-up. Results: results\uconvert_bench_<date>.json (to post) and
.txt (to read), next to this script. They contain no paths and no user or computer names.

  bench_uconvert.cmd [-FFmpeg <ffmpeg.exe>] [-Seconds 5] [-NoCpu] [-Stages deint,frc,...] [-Threads n]

Needs: ffmpeg with AviSynth input (ffmpeg.exe next to this script, in the UConvert folder above it or on the PATH,
else -FFmpeg <path>); AviSynth+ 3.7 or newer, x64, with AviSynth.dll in System32 or next to ffmpeg.exe; for the GPU
stages an NVIDIA GPU with a current driver. Both ffmpeg checks run first, before anything else.
#>
param(
  [string]$FFmpeg = "",
  [double]$Seconds = 5,
  [switch]$NoCpu,
  [string]$Stages = "",
  [int]$Threads = -1,
  [switch]$KeepScripts
)
$ErrorActionPreference = "Stop"
$BenchVersion = 3
$here = $PSScriptRoot
$ucDir = Split-Path $here -Parent
$ucAvsi = Join-Path $ucDir "uconvert.avsi"
$mpDll = Join-Path $here "MovingPattern.dll"

function Fail([string]$msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $ucAvsi)) { Fail "uconvert.avsi not found in $ucDir (the bench folder must sit inside the UConvert folder)" }
if (-not (Test-Path $mpDll)) { Fail "MovingPattern.dll not found next to this script" }

# ---- ffmpeg
if (-not $FFmpeg) {
  foreach ($c in @((Join-Path $here "ffmpeg.exe"), (Join-Path $ucDir "ffmpeg.exe"))) { if (Test-Path $c) { $FFmpeg = $c; break } }
}
if (-not $FFmpeg) {
  $cmd = Get-Command ffmpeg.exe -ErrorAction SilentlyContinue
  if ($cmd) { $FFmpeg = $cmd.Source }
}
if (-not $FFmpeg) {
  Fail ("ffmpeg.exe not found (looked next to this script, in the UConvert folder and on the PATH).`n" +
        "       Tell the benchmark where it is with the -FFmpeg switch, e.g.:`n" +
        "       bench_uconvert.cmd -FFmpeg C:\tools\ffmpeg\bin\ffmpeg.exe")
}
if (-not (Test-Path $FFmpeg -PathType Leaf)) {
  Fail ("ffmpeg not found at the -FFmpeg path: $FFmpeg`n" +
        "       Give the full path of ffmpeg.exe, e.g.: bench_uconvert.cmd -FFmpeg C:\tools\ffmpeg\bin\ffmpeg.exe")
}
$FFmpeg = (Resolve-Path $FFmpeg).Path

$work = Join-Path ([System.IO.Path]::GetTempPath()) ("uconvert_bench_" + $PID)
New-Item -ItemType Directory -Force -Path $work | Out-Null

# hide paths and names in anything that goes into the results
$scrubList = @($work, $here, $ucDir, (Split-Path $FFmpeg -Parent), $env:USERPROFILE, $env:USERNAME, $env:COMPUTERNAME) |
  Where-Object { $_ } | Sort-Object Length -Descending
function Scrub([string]$s) {
  if (-not $s) { return $s }
  foreach ($p in $scrubList) { $s = $s.Replace($p, "<...>") }
  return $s
}

# the error of an ffmpeg run: AviSynth's message (the "[avisynth @ ...]" line and its continuation lines), else
# the last lines; one line, paths and names hidden
function Error-Text([string]$e) {
  $lines = @($e -split "`r?`n" | ForEach-Object { $_.TrimEnd() } | Where-Object { $_ })
  $msg = @()
  for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match '^\[avisynth @ [^\]]*\]\s*(.*)$') {
      $msg += $Matches[1]
      for ($j = $i + 1; $j -lt $lines.Count -and $lines[$j] -notmatch '^\(|^\['; $j++) { $msg += $lines[$j].Trim() }
      break
    }
  }
  if (-not $msg.Count) { $msg = $lines | Select-Object -Last 3 }
  return Scrub (($msg -join ' ') -replace '\s+', ' ').Trim()
}

function Run-FFmpeg([string[]]$a, [int]$timeoutSec = 120) {
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = $FFmpeg
  $psi.Arguments = ($a | ForEach-Object { if ($_ -match '[\s"]') { '"' + $_.Replace('"', '\"') + '"' } else { $_ } }) -join ' '
  $psi.UseShellExecute = $false
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  $psi.CreateNoWindow = $true
  $p = [System.Diagnostics.Process]::Start($psi)
  $out = $p.StandardOutput.ReadToEndAsync()
  $err = $p.StandardError.ReadToEndAsync()
  if (-not $p.WaitForExit($timeoutSec * 1000)) { try { $p.Kill() } catch {} }
  $p.WaitForExit()
  return @{ code = $p.ExitCode; out = $out.Result; err = $err.Result }
}

# ---- checks before anything else: 1. ffmpeg runs, 2. ffmpeg can read AviSynth scripts
try { $ffv = Run-FFmpeg @('-hide_banner', '-version') 30 } catch { $ffv = @{ code = -1; out = ""; err = $_.Exception.Message } }
if ($ffv.code -ne 0 -or $ffv.out -notmatch '^ffmpeg version') {
  Fail ("$FFmpeg does not run as ffmpeg (ffmpeg -version failed).`n" +
        "       Give the path of a working ffmpeg.exe with the -FFmpeg switch, e.g.: bench_uconvert.cmd -FFmpeg C:\tools\ffmpeg\bin\ffmpeg.exe")
}
$ffVersion = (($ffv.out -split "`n")[0] -replace '\s*Copyright.*$', '').Trim()
Write-Host "ffmpeg:    $FFmpeg"

$verAvs = Join-Path $work "version.avs"
Set-Content -Path $verAvs -Encoding ASCII -Value 'Assert(false, "AVSVERSION " + VersionString())'
$r = Run-FFmpeg @('-hide_banner', '-nostdin', '-i', $verAvs) 60
if ($r.err -match 'AVSVERSION ([^\r\n(]+)') { $avsVersion = $Matches[1].Trim() }
else {
  $ffDir = Split-Path $FFmpeg -Parent
  $sys32 = Join-Path $env:WINDIR "System32"
  $dmx = Run-FFmpeg @('-hide_banner', '-demuxers') 30
  if ($dmx.out -notmatch '(?m)^\s*D\S*\s+avisynth\b') {
    Fail ("this ffmpeg cannot read AviSynth scripts at all: it is built without AviSynth input.`n" +
          "       Use an ffmpeg build with AviSynth support (e.g. the gyan.dev or BtbN full builds) and pass it with -FFmpeg <path>.")
  }
  # ffmpeg loads AviSynth.dll from its own folder or System32 (or the PATH); when that fails it only says "Unknown error"
  $nextTo = Join-Path $ffDir "AviSynth.dll"
  $inSys = Join-Path $sys32 "AviSynth.dll"
  $why = if (Test-Path $nextTo) { "The AviSynth.dll next to ffmpeg.exe ($nextTo, used before the one in System32) could not be loaded: 32-bit, too old or broken? Replace or remove it." }
         elseif (Test-Path $inSys) { "The AviSynth.dll in System32 ($inSys) could not be loaded: 32-bit, too old or broken?" }
         else { "AviSynth.dll was found neither in $sys32 nor next to ffmpeg.exe." }
  Fail ("ffmpeg cannot read AviSynth scripts. $why`n" +
        "       AviSynth.dll (AviSynth+ 3.7 or newer, x64) must be either in $sys32 (installed by the AviSynth+ installer)`n" +
        "       or next to ffmpeg.exe in $ffDir (the x64 AviSynth.dll from the AviSynth+ release archive,`n" +
        "       https://github.com/AviSynth/AviSynthPlus/releases). Then run the benchmark again.`n" +
        "       ffmpeg said: " + (($r.err.Trim() -split "`n")[-1]).Trim())
}
Write-Host "AviSynth:  $avsVersion"

# ---- host
Write-Host "UConvert benchmark v${BenchVersion}: collecting host information ..."
$host_ = [ordered]@{}
try {
  $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
  $host_.cpu = ($cpu.Name -replace '\s+', ' ').Trim()
  $host_.cpu_cores = [int](Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum
  $host_.cpu_threads = [int](Get-CimInstance Win32_Processor | Measure-Object NumberOfLogicalProcessors -Sum).Sum
  $host_.cpu_max_mhz = [int]$cpu.MaxClockSpeed
  $host_.cpu_l2_kb = [int]$cpu.L2CacheSize
  $host_.cpu_l3_kb = [int]$cpu.L3CacheSize
  $host_.ram_gb = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
  $os = Get-CimInstance Win32_OperatingSystem
  $host_.os = "$($os.Caption) $($os.Version)".Trim()
} catch { $host_.host_info_error = $_.Exception.Message }
# CPU instruction sets (as Windows reports them; Windows versions that do not know a feature report it as missing,
# so SSSE3 missing on a CPU with SSE3 means "unknown") and the core types of hybrid CPUs (performance / efficiency)
try {
  Add-Type -Namespace UcBench -Name Native -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern bool IsProcessorFeaturePresent(uint feature);
[DllImport("kernel32.dll", SetLastError = true)] public static extern bool GetSystemCpuSetInformation(byte[] info, uint len, out uint ret, System.IntPtr process, uint flags);
'@
  $pf = [ordered]@{ "SSE2" = 10; "SSE3" = 13; "SSSE3" = 36; "SSE4.1" = 37; "SSE4.2" = 38; "AVX" = 39; "AVX2" = 40; "AVX-512F" = 41 }
  $have = @($pf.Keys | Where-Object { [UcBench.Native]::IsProcessorFeaturePresent($pf[$_]) })
  if (($have -contains "SSE3") -and -not ($have -contains "SSSE3")) { $host_.cpu_features = "unknown (Windows too old to report them)" }
  else { $host_.cpu_features = $have -join " " }
  $len = [uint32]0
  [void][UcBench.Native]::GetSystemCpuSetInformation($null, 0, [ref]$len, [IntPtr]::Zero, 0)
  if ($len -gt 0) {
    $buf = New-Object byte[] $len
    if ([UcBench.Native]::GetSystemCpuSetInformation($buf, $len, [ref]$len, [IntPtr]::Zero, 0)) {
      # SYSTEM_CPU_SET_INFORMATION: Size @0, Type @4, Group @12, CoreIndex @15, EfficiencyClass @18
      $cls = @{}
      for ($o = 0; $o -lt $len; $o += [BitConverter]::ToUInt32($buf, $o)) {
        if ([BitConverter]::ToUInt32($buf, $o + 4) -ne 0) { continue }
        $e = [int]$buf[$o + 18]
        if (-not $cls.ContainsKey($e)) { $cls[$e] = @{ threads = 0; cores = @{} } }
        $cls[$e].threads++
        $cls[$e].cores["$([BitConverter]::ToUInt16($buf, $o + 12)):$($buf[$o + 15])"] = 1
      }
      if ($cls.Count -gt 1) {
        $top = [int]($cls.Keys | Measure-Object -Maximum).Maximum
        $host_.cpu_p_cores = $cls[$top].cores.Count
        $host_.cpu_e_cores = [int](($cls.Keys | Where-Object { $_ -ne $top } | ForEach-Object { $cls[$_].cores.Count } | Measure-Object -Sum).Sum)
      }
    }
  }
} catch { $host_.cpu_features_error = $_.Exception.Message }

$gpus = @()
$smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (-not $smi) { $c = Join-Path $env:ProgramFiles "NVIDIA Corporation\NVSMI\nvidia-smi.exe"; if (Test-Path $c) { $smi = Get-Item $c } }
if ($smi) {
  $q = & $smi.Source --query-gpu=name,driver_version,memory.total,pcie.link.gen.max,pcie.link.width.current,power.limit --format=csv,noheader,nounits 2>$null
  foreach ($l in $q) {
    $f = $l -split ',\s*'
    if ($f.Count -ge 3) {
      $gpus += [ordered]@{ name = $f[0]; driver = $f[1]; vram_mb = [int]$f[2]; pcie_gen = $f[3]; pcie_width = $f[4]; power_limit_w = $f[5] }
    }
  }
  # separate query: older drivers do not know these fields (one unknown field fails the whole query)
  $q2 = @(& $smi.Source --query-gpu=compute_cap,clocks.max.sm,clocks.max.mem --format=csv,noheader,nounits 2>$null)
  for ($i = 0; $i -lt $gpus.Count -and $i -lt $q2.Count; $i++) {
    $f = $q2[$i] -split ',\s*'
    if ($f.Count -ge 3 -and $f[0] -match '^\d+\.\d+$') { $gpus[$i].compute_capability = $f[0]; $gpus[$i].max_sm_mhz = $f[1]; $gpus[$i].max_mem_mhz = $f[2] }
  }
  $smiOut = (& $smi.Source 2>$null) | Out-String
  if ($smiOut -match 'CUDA (?:UMD )?Version:\s*([\d.]+)') { $host_.cuda_driver_api = $Matches[1] }
}
if (-not $gpus.Count) {
  foreach ($v in (Get-CimInstance Win32_VideoController)) { $gpus += [ordered]@{ name = $v.Name; driver = $v.DriverVersion } }
}
$host_.gpus = $gpus
$host_.nvidia_gpus = @($gpus | Where-Object { $_.name -match 'NVIDIA' }).Count
try {
  $bat = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
  $host_.on_battery = [bool]($bat -and ($bat | Where-Object { $_.BatteryStatus -eq 1 }))
} catch { $host_.on_battery = $null }
try { $host_.power_plan = ((powercfg /getactivescheme) -replace '^.*\((.*)\).*$', '$1').Trim() } catch {}

$host_.ffmpeg = $ffVersion
$host_.avisynth = $avsVersion

# ---- stages: generated source -> UConvertClip with targets that leave one stage (or the whole chain) to run
# src: MovingPattern arguments; target: UConvertClip arguments after the clip; pre: UConvert settings for the run;
# nvof / nvenc: a comparison stage with motion vectors from the NVIDIA optical flow accelerator / the NVENC encoder's
# motion estimation, skipped where UConvert's plan shows it is not used (GPU without it, nvencmv.dll missing)
$src_i25 = 'width=1920, height=1080, pixel_type="YUV422P10", fps_num=25, fps_den=1, scan="tff"'
$src_p50 = 'width=1920, height=1080, pixel_type="YUV422P10", fps_num=50, fps_den=1, scan="p"'
$src_p5994 = 'width=1920, height=1080, pixel_type="YUV422P10", fps_num=60000, fps_den=1001, scan="p"'
$src_uhd50 = 'width=3840, height=2160, pixel_type="YUV422P10", fps_num=50, fps_den=1, scan="p", period=50'
$all = @(
  [ordered]@{ name = "deint";       desc = "deinterlace 1080i25 -> 1080p50";                src = $src_i25;   target = '1080, 1920, out_rate_nom=50, out_rate_den=1' },
  [ordered]@{ name = "deint_nvof";  desc = "deinterlace 1080i25 -> 1080p50, NVOFA vectors"; src = $src_i25;   target = '1080, 1920, out_rate_nom=50, out_rate_den=1'; pre = 'global uc_qtgmc_nvof = true'; nvof = $true },
  [ordered]@{ name = "deint_wave";  desc = "deinterlace 1080i25 -> 1080p50, wavefront search"; src = $src_i25; target = '1080, 1920, out_rate_nom=50, out_rate_den=1'; pre = 'global uc_qtgmc_wavefront = true' },
  [ordered]@{ name = "deint_nvenc"; desc = "deinterlace 1080i25 -> 1080p50, NVENC vectors"; src = $src_i25;  target = '1080, 1920, out_rate_nom=50, out_rate_den=1'; pre = 'global uc_qtgmc_nvenc = true'; nvenc = $true },
  [ordered]@{ name = "frc";         desc = "frame-rate conversion 1080p50 -> 1080p59.94";   src = $src_p50;   target = '1080, 1920, out_rate_nom=60000, out_rate_den=1001' },
  [ordered]@{ name = "frc_nvof";    desc = "frame-rate conversion 1080p50 -> 1080p59.94, NVOFA vectors"; src = $src_p50; target = '1080, 1920, out_rate_nom=60000, out_rate_den=1001'; pre = 'global uc_frc_nvof = true'; nvof = $true },
  [ordered]@{ name = "frc_nvenc";   desc = "frame-rate conversion 1080p50 -> 1080p59.94, NVENC vectors"; src = $src_p50; target = '1080, 1920, out_rate_nom=60000, out_rate_den=1001'; pre = 'global uc_frc_nvenc = true'; nvenc = $true },
  [ordered]@{ name = "interlace";   desc = "interlace 1080p59.94 -> 1080i29.97";            src = $src_p5994; target = '1080, 1920, out_interlace="tff", out_rate_nom=30000, out_rate_den=1001' },
  [ordered]@{ name = "scale";       desc = "scale 2160p50 -> 1080p50";                      src = $src_uhd50; target = '1080, 1920, out_rate_nom=50, out_rate_den=1' },
  [ordered]@{ name = "chain_gpu";   desc = "whole chain 1080i25 -> 1080i29.97 (GPU)";       src = $src_i25;   target = '1080, 1920, out_interlace="tff", out_rate_nom=30000, out_rate_den=1001' },
  [ordered]@{ name = "chain_cpu";   desc = "whole chain 1080i25 -> 1080i29.97 (CPU only)";  src = $src_i25;   target = '1080, 1920, out_interlace="tff", out_rate_nom=30000, out_rate_den=1001, gpu=false' }
)
$run = $all
if ($Stages) { $want = $Stages -split ','; $run = $all | Where-Object { $want -contains $_.name } }
if ($NoCpu) { $run = $run | Where-Object { $_.name -ne "chain_cpu" } }

function Stage-Script($s, [bool]$info) {
  $t = @(
    "Import(""$ucAvsi"")",
    $(if ($s.pre) { $s.pre } else { "" }),
    "LoadPlugin(""$mpDll"")",
    "src = MovingPattern($($s.src), frames=1000000, storage=""memory"")",
    "c = src.UConvertClip($($s.target), threads=$Threads)"
  )
  if ($info) {
    $t += 'Assert(false, "UCPLAN " + (propGetType(c, "_uc_plan") == 3 ? propGetString(c, "_uc_plan") : "-"))'
  } else { $t += "c" }
  $path = Join-Path $work "$($s.name)$(if ($info) { '_info' }).avs"
  Set-Content -Path $path -Encoding ASCII -Value $t
  return $path
}

# time-boxed run: ffmpeg renders to null and reports progress every 0.25 s; stopped with q after $Seconds of frames
function Time-Stage([string]$avs) {
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = $FFmpeg
  $psi.Arguments = "-hide_banner -nostats -i `"$avs`" -f null - -progress pipe:1 -stats_period 0.25"
  $psi.UseShellExecute = $false
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  $psi.RedirectStandardInput = $true
  $psi.CreateNoWindow = $true
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $p = [System.Diagnostics.Process]::Start($psi)
  try { $p.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::High } catch {}
  $err = $p.StandardError.ReadToEndAsync()
  $samples = New-Object System.Collections.ArrayList
  $frame = 0
  $tFirst = $null
  $quit = $false
  while ($true) {
    $line = $p.StandardOutput.ReadLine()
    if ($null -eq $line) { break }
    if ($line -match '^frame=(\d+)') { $frame = [int]$Matches[1] }
    elseif ($line -match '^progress=') {
      $t = $sw.Elapsed.TotalSeconds
      if ($frame -gt 0) {
        if ($null -eq $tFirst) { $tFirst = $t }
        [void]$samples.Add(@($t, $frame))
        if (-not $quit -and $t - $tFirst -ge $Seconds) { try { $p.StandardInput.Write("q"); $p.StandardInput.Flush() } catch {}; $quit = $true }
      }
      if ($line -eq 'progress=end') { break }
    }
    if ($sw.Elapsed.TotalSeconds -gt $Seconds + 120) { try { $p.Kill() } catch {}; break }
  }
  if (-not $p.WaitForExit(15000)) { try { $p.Kill() } catch {} }
  $p.WaitForExit()
  $res = [ordered]@{ startup_s = $null; fps = $null; fps_min = $null; fps_max = $null; frames = $frame; windows = 0 }
  if ($null -ne $tFirst) { $res.startup_s = [math]::Round($tFirst, 2) }
  # 1 s windows after the first second of frames (warm-up)
  $fps = @()
  $s0 = @($samples | Where-Object { $_[0] -ge $tFirst + 1.0 })
  for ($i = 0; $i -lt $s0.Count; $i++) {
    for ($j = $i + 1; $j -lt $s0.Count; $j++) {
      if ($s0[$j][0] - $s0[$i][0] -ge 1.0) { $fps += ($s0[$j][1] - $s0[$i][1]) / ($s0[$j][0] - $s0[$i][0]); break }
    }
  }
  if ($fps.Count) {
    $sorted = $fps | Sort-Object
    $m = [int][math]::Floor($sorted.Count / 2)
    $med = if ($sorted.Count % 2) { $sorted[$m] } else { ($sorted[$m - 1] + $sorted[$m]) / 2 }
    $res.fps = [math]::Round($med, 1)
    $res.fps_min = [math]::Round($sorted[0], 1)
    $res.fps_max = [math]::Round($sorted[-1], 1)
    $res.windows = $fps.Count
  }
  $e = $err.Result
  $res.ok = ($fps.Count -gt 0) -and ($e -notmatch '(?i)error|exception')
  if (-not $res.ok) { $res.error = Error-Text $e }
  return $res
}

# ---- plans (one information run per stage: the chain UConvert chose, to check that each run measured its stage)
$results = @()
$t0 = Get-Date
foreach ($s in $run) {
  $r = Run-FFmpeg @('-hide_banner', '-nostdin', '-i', (Stage-Script $s $true)) 300
  $plan = if ($r.err -match 'UCPLAN ([^\r\n]+)') { ($Matches[1] -replace '\s*\(.*\.avs, line \d+\)\s*$', '').Trim() } else { $null }
  $res = [ordered]@{ stage = $s.name; what = $s.desc; plan = $plan }
  if (-not $plan) {
    $res.ok = $false
    $res.error = Error-Text $r.err
    Write-Host ("  {0,-11} FAILED: {1}" -f $s.name, $res.error) -ForegroundColor Red
  } elseif (($s.nvof -and $plan -notmatch 'nvof') -or ($s.nvenc -and $plan -notmatch 'nvenc')) {
    $res.ok = $false
    $res.skipped = if ($s.nvof) { "no NVIDIA optical flow accelerator on this GPU (Turing or newer)" }
                   else { "no NVENC motion estimation (nvencmv.dll missing, or a GPU older than Pascal)" }
    Write-Host ("  {0,-11} skipped: {1}" -f $s.name, $res.skipped) -ForegroundColor Yellow
  } else {
    Write-Host ("  {0,-11} {1} ..." -f $s.name, $s.desc)
    $m = Time-Stage (Stage-Script $s $false)
    foreach ($k in $m.Keys) { $res[$k] = $m[$k] }
    if ($res.ok) { Write-Host ("  {0,-11} {1,7:N1} fps   (start-up {2} s)" -f $s.name, $res.fps, $res.startup_s) -ForegroundColor Green }
    else { Write-Host ("  {0,-11} FAILED: {1}" -f $s.name, $res.error) -ForegroundColor Red }
  }
  $results += $res
}

# ---- results
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$doc = [ordered]@{
  benchmark = "uconvert"; version = $BenchVersion; date = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
  seconds_per_stage = $Seconds; threads = $Threads; duration_s = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
  host = $host_; stages = $results
}
$resDir = Join-Path $here "results"
New-Item -ItemType Directory -Force -Path $resDir | Out-Null
$json = Join-Path $resDir "uconvert_bench_$stamp.json"
$utf8 = New-Object System.Text.UTF8Encoding($false)   # no BOM
[System.IO.File]::WriteAllText($json, ($doc | ConvertTo-Json -Depth 6), $utf8)
$gpuLine = "GPU: " + ((@($gpus) | ForEach-Object {
  $cc = if ($_.compute_capability) { ", compute $($_.compute_capability)" } else { "" }
  "$($_.name) (driver $($_.driver)$cc)" }) -join '; ')
if ($host_.cuda_driver_api) { $gpuLine += ", CUDA $($host_.cuda_driver_api)" }
$pe = if ($host_.cpu_p_cores) { " = $($host_.cpu_p_cores) P + $($host_.cpu_e_cores) E" } else { "" }
$txt = @("UConvert benchmark v$BenchVersion  $($doc.date)",
  "CPU: $($host_.cpu) ($($host_.cpu_cores) cores$pe, $($host_.cpu_threads) threads), RAM $($host_.ram_gb) GB",
  "CPU features: $($host_.cpu_features)",
  $gpuLine, "OS: $($host_.os); $($host_.avisynth); $($host_.ffmpeg)", "")
foreach ($r in $results) {
  if ($r.ok) { $txt += ("{0,-11} {1,8:N1} fps   {2}" -f $r.stage, $r.fps, $r.what) }
  elseif ($r.skipped) { $txt += ("{0,-11}  skipped   {1}: {2}" -f $r.stage, $r.what, $r.skipped) }
  else { $txt += ("{0,-11}   FAILED   {1}: {2}" -f $r.stage, $r.what, $r.error) }
}
[System.IO.File]::WriteAllLines([System.IO.Path]::ChangeExtension($json, ".txt"), [string[]]$txt, $utf8)
Write-Host ""
$txt | ForEach-Object { Write-Host $_ }
Write-Host ""
Write-Host "Results: results\$(Split-Path $json -Leaf) - please post this file (it contains no paths or names)."
if (-not $KeepScripts) { Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue }
