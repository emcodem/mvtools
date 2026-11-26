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
$BenchVersion = 5
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
# ProviderPath: a network path (\\server\share\...) as Windows writes it; .Path would carry PowerShell's provider prefix
$FFmpeg = (Resolve-Path $FFmpeg).ProviderPath

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
$smiPath = $null
$smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if ($smi) { $smiPath = $smi.Source }
else { $c = Join-Path $env:ProgramFiles "NVIDIA Corporation\NVSMI\nvidia-smi.exe"; if (Test-Path $c) { $smiPath = $c } }
if ($smiPath) {
  # link maximum here; the link a GPU actually runs at is sampled during each stage (an idle GPU parks its link
  # at gen 1 and its clocks low, so idle readings say nothing about a run)
  $q = & $smiPath --query-gpu=name,driver_version,memory.total,pcie.link.gen.max,pcie.link.width.max,power.limit --format=csv,noheader,nounits 2>$null
  foreach ($l in $q) {
    $f = $l -split ',\s*'
    if ($f.Count -ge 3) {
      $gpus += [ordered]@{ name = $f[0]; driver = $f[1]; vram_mb = [int]$f[2]; pcie_gen_max = $f[3]; pcie_width_max = $f[4]; power_limit_w = $f[5] }
    }
  }
  # separate query: older drivers do not know these fields (one unknown field fails the whole query)
  $q2 = @(& $smiPath --query-gpu=compute_cap,clocks.max.sm,clocks.max.mem --format=csv,noheader,nounits 2>$null)
  for ($i = 0; $i -lt $gpus.Count -and $i -lt $q2.Count; $i++) {
    $f = $q2[$i] -split ',\s*'
    if ($f.Count -ge 3 -and $f[0] -match '^\d+\.\d+$') { $gpus[$i].compute_capability = $f[0]; $gpus[$i].max_sm_mhz = $f[1]; $gpus[$i].max_mem_mhz = $f[2] }
  }
  $smiOut = (& $smiPath 2>$null) | Out-String
  if ($smiOut -match 'CUDA (?:UMD )?Version:\s*([\d.]+)') { $host_.cuda_driver_api = $Matches[1] }
}
# hardware-accelerated GPU scheduling (HAGS): with it off, Windows schedules the GPU work of all programs on the CPU,
# which can keep the GPU work of the parallel frames from running at the same time. The state in force is asked
# from the graphics kernel per adapter (Windows 10 2004 or newer); the registry value (HwSchMode 2 = on, 1 = off)
# only says what was chosen and takes effect after a restart.
try {
  Add-Type -Namespace UcBench -Name Kmt -MemberDefinition @'
[StructLayout(LayoutKind.Sequential)] public struct ADAPTERINFO { public uint hAdapter; public uint LuidLow; public int LuidHigh; public uint NumOfSources; public int bPrecisePresentRegionsPreferred; }
[StructLayout(LayoutKind.Sequential)] public struct ENUMADAPTERS2 { public uint NumAdapters; public System.IntPtr pAdapters; }
[StructLayout(LayoutKind.Sequential)] public struct QUERYADAPTERINFO { public uint hAdapter; public int Type; public System.IntPtr pPrivateDriverData; public uint PrivateDriverDataSize; }
[StructLayout(LayoutKind.Sequential)] public struct CLOSEADAPTER { public uint hAdapter; }
[DllImport("gdi32.dll")] public static extern int D3DKMTEnumAdapters2(ref ENUMADAPTERS2 e);
[DllImport("gdi32.dll")] public static extern int D3DKMTQueryAdapterInfo(ref QUERYADAPTERINFO q);
[DllImport("gdi32.dll")] public static extern int D3DKMTCloseAdapter(ref CLOSEADAPTER c);
// one line per adapter: name | hardware scheduling supported | enabled | enabled by default (or "?" when unknown)
public static string[] HwSch() {
  var e = new ENUMADAPTERS2();
  if (D3DKMTEnumAdapters2(ref e) != 0 || e.NumAdapters == 0) return new string[0];
  int sz = Marshal.SizeOf(typeof(ADAPTERINFO));
  e.pAdapters = Marshal.AllocHGlobal(sz * (int)e.NumAdapters);
  var res = new System.Collections.Generic.List<string>();
  try {
    if (D3DKMTEnumAdapters2(ref e) != 0) return new string[0];
    for (int i = 0; i < e.NumAdapters; i++) {
      var a = (ADAPTERINFO)Marshal.PtrToStructure(new System.IntPtr(e.pAdapters.ToInt64() + i * sz), typeof(ADAPTERINFO));
      string name = "?", caps = "?|?|?";
      System.IntPtr buf = Marshal.AllocHGlobal(4 * 260 * 2);
      try {
        var q = new QUERYADAPTERINFO { hAdapter = a.hAdapter, Type = 8, pPrivateDriverData = buf, PrivateDriverDataSize = 4 * 260 * 2 };  // ADAPTERREGISTRYINFO
        if (D3DKMTQueryAdapterInfo(ref q) == 0) name = Marshal.PtrToStringUni(buf);
        q = new QUERYADAPTERINFO { hAdapter = a.hAdapter, Type = 70, pPrivateDriverData = buf, PrivateDriverDataSize = 4 };  // WDDM_2_7_CAPS
        if (D3DKMTQueryAdapterInfo(ref q) == 0) { int v = Marshal.ReadInt32(buf); caps = (v & 1) + "|" + ((v >> 1) & 1) + "|" + ((v >> 2) & 1); }
      } finally { Marshal.FreeHGlobal(buf); }
      res.Add(name + "|" + caps);
      var c = new CLOSEADAPTER { hAdapter = a.hAdapter };
      D3DKMTCloseAdapter(ref c);
    }
  } finally { Marshal.FreeHGlobal(e.pAdapters); }
  return res.ToArray();
}
'@
  $sch = @()
  foreach ($l in [UcBench.Kmt]::HwSch()) {
    $f = $l -split '\|'
    if ($f[0] -notmatch 'NVIDIA') { continue }
    $st = if ($f[1] -eq '?') { "unknown (Windows too old to report it)" } elseif ($f[1] -eq '0') { "not supported" } elseif ($f[2] -eq '1') { "on" } else { "off" }
    $sch += "$($f[0]): $st"
  }
  $host_.hw_gpu_scheduling = @($sch | Select-Object -Unique)
} catch { $host_.hw_gpu_scheduling_error = $_.Exception.Message }
try {
  $hs = (Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers' -Name HwSchMode -ErrorAction Stop).HwSchMode
  $host_.hw_gpu_scheduling_setting = switch ([int]$hs) { 2 { "on" } 1 { "off" } default { "HwSchMode $hs" } }
} catch { $host_.hw_gpu_scheduling_setting = "not set" }
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
# The worst case of UConvert's 1080 broadcast conversions: 1080i29.97 -> 1080i25. The deinterlacer makes 59.94
# pictures per second (more than for 25i -> 29.97i) and nearly every output picture of the frame-rate conversion is
# a new one. The single stages are the parts of that chain; rate: output frames per second of video (realtime = rate).
$src_i2997 = 'width=1920, height=1080, pixel_type="YUV422P10", fps_num=30000, fps_den=1001, scan="tff"'
$src_p5994 = 'width=1920, height=1080, pixel_type="YUV422P10", fps_num=60000, fps_den=1001, scan="p"'
$src_p50 = 'width=1920, height=1080, pixel_type="YUV422P10", fps_num=50, fps_den=1, scan="p"'
$src_i25sd = 'width=720, height=576, pixel_type="YUV422P10", fps_num=25, fps_den=1, scan="tff"'
$src_uhd50 ='width=3840, height=2160, pixel_type="YUV422P10", fps_num=50, fps_den=1, scan="p", period=50'
$t_deint = '1080, 1920, out_rate_nom=60000, out_rate_den=1001'
$t_frc = '1080, 1920, out_rate_nom=50, out_rate_den=1'
$t_deint_sd = '576, 720, out_rate_nom=50, out_rate_den=1'
$t_i25 = '1080, 1920, out_interlace="tff", out_rate_nom=25, out_rate_den=1'
$all = @(
  [ordered]@{ name = "deint";       rate = 59.94; desc = "deinterlace 1080i29.97 -> 1080p59.94";                    src = $src_i2997; target = $t_deint },
  [ordered]@{ name = "deint_nvof";  rate = 59.94; desc = "deinterlace 1080i29.97 -> 1080p59.94, NVOFA vectors";     src = $src_i2997; target = $t_deint; pre = 'global uc_qtgmc_nvof = true'; nvof = $true },
  [ordered]@{ name = "deint_nvenc"; rate = 59.94; desc = "deinterlace 1080i29.97 -> 1080p59.94, NVENC vectors";     src = $src_i2997; target = $t_deint; pre = 'global uc_qtgmc_nvenc = true'; nvenc = $true },
  # SD: one picture's motion search fills only a small part of a GPU, so the speed depends on the searches of the
  # frames in flight running at the same time
  [ordered]@{ name = "deint_sd";    rate = 50;    desc = "deinterlace 576i25 -> 576p50";                            src = $src_i25sd; target = $t_deint_sd },
  [ordered]@{ name = "frc";         rate = 50;   desc = "frame-rate conversion 1080p59.94 -> 1080p50";             src = $src_p5994; target = $t_frc },
  [ordered]@{ name = "frc_nvof";    rate = 50;    desc = "frame-rate conversion 1080p59.94 -> 1080p50, NVOFA vectors"; src = $src_p5994; target = $t_frc; pre = 'global uc_frc_nvof = true'; nvof = $true },
  [ordered]@{ name = "frc_nvenc";   rate = 50;    desc = "frame-rate conversion 1080p59.94 -> 1080p50, NVENC vectors"; src = $src_p5994; target = $t_frc; pre = 'global uc_frc_nvenc = true'; nvenc = $true },
  [ordered]@{ name = "interlace";   rate = 25;    desc = "interlace 1080p50 -> 1080i25";                            src = $src_p50;   target = $t_i25 },
  [ordered]@{ name = "scale";       rate = 50;    desc = "scale 2160p50 -> 1080p50";                                src = $src_uhd50; target = $t_frc },
  [ordered]@{ name = "chain_gpu";   rate = 25;    desc = "whole chain 1080i29.97 -> 1080i25 (GPU)";                 src = $src_i2997; target = $t_i25 },
  [ordered]@{ name = "chain_gpu_nvof"; rate = 25; desc = "whole chain 1080i29.97 -> 1080i25 (GPU), NVOFA vectors for the deinterlacer"; src = $src_i2997; target = $t_i25; pre = 'global uc_qtgmc_nvof = true'; nvof = $true },
  [ordered]@{ name = "chain_cpu";   rate = 25;    desc = "whole chain 1080i29.97 -> 1080i25 (CPU only)";            src = $src_i2997; target = ($t_i25 + ', gpu=false') }
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

function Median($v) {
  $s = @($v | Sort-Object)
  if (-not $s.Count) { return $null }
  $m = [int][math]::Floor($s.Count / 2)
  if ($s.Count % 2) { return $s[$m] } else { return ($s[$m - 1] + $s[$m]) / 2 }
}
function Range-Text($v, [string]$pre) {
  $lo = ($v | Measure-Object -Minimum).Minimum; $hi = ($v | Measure-Object -Maximum).Maximum
  if ($lo -eq $hi) { return "$pre$lo" } else { return "$pre$lo-$hi" }
}

# GPU state during a stage: nvidia-smi logs every 0.5 s in its own (light) process; only the samples of the timed
# part count. Returns the GPU that worked the most: PCIe link, power state, clocks, load, power, temperature.
$smiFields = "timestamp,index,pcie.link.gen.current,pcie.link.width.current,pstate,clocks.sm,clocks.mem,utilization.gpu,power.draw,temperature.gpu"
function Start-GpuLog {
  if (-not $smiPath) { return $null }
  try {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $smiPath
    $psi.Arguments = "--query-gpu=$smiFields --format=csv,noheader,nounits -lms 500"
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true
    $p = [System.Diagnostics.Process]::Start($psi)
    return @{ p = $p; out = $p.StandardOutput.ReadToEndAsync(); err = $p.StandardError.ReadToEndAsync() }
  } catch { return $null }
}
function Stop-GpuLog($g, $from, $to) {
  if (-not $g) { return $null }
  try { $g.p.Kill() } catch {}
  [void]$g.p.WaitForExit(5000)
  $inv = [System.Globalization.CultureInfo]::InvariantCulture
  $num = { param($x) $d = 0.0; if ([double]::TryParse($x, [System.Globalization.NumberStyles]::Float, $inv, [ref]$d)) { $d } else { $null } }
  $rows = @()
  foreach ($l in ($g.out.Result -split "`r?`n")) {
    $f = $l -split ',\s*'
    if ($f.Count -lt 10) { continue }
    $t = [datetime]::MinValue
    if (-not [datetime]::TryParseExact($f[0].Trim(), 'yyyy/MM/dd HH:mm:ss.fff', $inv, [System.Globalization.DateTimeStyles]::None, [ref]$t)) { continue }
    $rows += [pscustomobject]@{ t = $t; gpu = $f[1]; gen = (& $num $f[2]); width = (& $num $f[3]); pstate = $f[4].Trim()
      sm = (& $num $f[5]); mem = (& $num $f[6]); util = (& $num $f[7]); power = (& $num $f[8]); temp = (& $num $f[9]) }
  }
  $win = @($rows | Where-Object { $from -and $to -and $_.t -ge $from -and $_.t -le $to })
  if ($win.Count -lt 2) { $win = $rows }
  if (-not $win.Count) { return $null }
  $best = $win | Group-Object gpu | Sort-Object { Median @($_.Group | Where-Object { $null -ne $_.util } | ForEach-Object { $_.util }) } | Select-Object -Last 1
  $r = @($best.Group)
  $sel = { param($k) @($r | Where-Object { $null -ne $_.$k } | ForEach-Object { $_.$k }) }
  $o = [ordered]@{ gpu = [int]$best.Name; samples = $r.Count }
  $v = & $sel 'gen';   if ($v.Count) { $o.pcie = (Range-Text $v "gen") + " " + (Range-Text (& $sel 'width') "x") }
  $o.pstate = ($r | Group-Object pstate | Sort-Object Count | Select-Object -Last 1).Name
  $v = & $sel 'sm';    if ($v.Count) { $o.sm_mhz = [int](Median $v) }
  $v = & $sel 'mem';   if ($v.Count) { $o.mem_mhz = [int](Median $v) }
  $v = & $sel 'util';  if ($v.Count) { $o.util_pct = [int](Median $v) }
  $v = & $sel 'power'; if ($v.Count) { $o.power_w_max = [int]($v | Measure-Object -Maximum).Maximum }
  $v = & $sel 'temp';  if ($v.Count) { $o.temp_c_max = [int]($v | Measure-Object -Maximum).Maximum }
  return $o
}

# GPU memory of the ffmpeg process (Windows counters "GPU Process Memory", Windows 10 1709 or newer). "Dedicated
# Usage" is the memory on the GPU; "Non Local Usage" is system RAM the GPU reaches over PCIe: the pinned transfer
# buffers a program allocates on purpose (a few hundred MiB here is normal), plus GPU memory that Windows moved out
# without an error because the GPU's memory was full. Growth far beyond a stage's usual value while the GPU's memory
# is nearly full points to the second, and then the GPU reads part of its working set over PCIe.
$memCat = $null
try {
  if ([System.Diagnostics.PerformanceCounterCategory]::Exists("GPU Process Memory")) { $memCat = New-Object System.Diagnostics.PerformanceCounterCategory("GPU Process Memory") }
} catch {}

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
  $gpuLog = Start-GpuLog
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $p = [System.Diagnostics.Process]::Start($psi)
  try { $p.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::High } catch {}
  $err = $p.StandardError.ReadToEndAsync()
  $samples = New-Object System.Collections.ArrayList
  $frame = 0
  $tFirst = $null
  $quit = $false
  $wFrom = $null; $wTo = $null
  # GPU memory counters of this process: looked up until the timed part starts (they appear when it first uses
  # the GPU), read once per second; peak per GPU (counter instance)
  $memCtr = @{}; $memPeak = @{}; $tMem = 0.0; $memLookup = [bool]$memCat
  while ($true) {
    $line = $p.StandardOutput.ReadLine()
    if ($null -eq $line) { break }
    if ($line -match '^frame=(\d+)') { $frame = [int]$Matches[1] }
    elseif ($line -match '^progress=') {
      $t = $sw.Elapsed.TotalSeconds
      if ($frame -gt 0) {
        if ($null -eq $tFirst) { $tFirst = $t; $wFrom = (Get-Date).AddSeconds(1.0) }
        [void]$samples.Add(@($t, $frame))
        if (-not $quit -and $t - $tFirst -ge $Seconds) { try { $p.StandardInput.Write("q"); $p.StandardInput.Flush() } catch {}; $quit = $true; $wTo = Get-Date }
      }
      if ($t -ge $tMem -and ($memLookup -or $memCtr.Count)) {
        $tMem = $t + 1.0
        if ($memLookup) {
          try {
            foreach ($n in $memCat.GetInstanceNames()) {
              if ($n -like "pid_$($p.Id)_*" -and -not $memCtr.ContainsKey($n)) {
                $memCtr[$n] = @((New-Object System.Diagnostics.PerformanceCounter("GPU Process Memory", "Dedicated Usage", $n, $true)),
                                (New-Object System.Diagnostics.PerformanceCounter("GPU Process Memory", "Non Local Usage", $n, $true)))
                $memPeak[$n] = @(0.0, 0.0)
              }
            }
          } catch { $memLookup = $false }
          if ($null -ne $tFirst -and $t -ge $tFirst + 1.0) { $memLookup = $false }
        }
        foreach ($n in @($memCtr.Keys)) {
          try {
            $d = $memCtr[$n][0].NextValue(); $nl = $memCtr[$n][1].NextValue()
            if ($d -gt $memPeak[$n][0]) { $memPeak[$n][0] = $d }
            if ($nl -gt $memPeak[$n][1]) { $memPeak[$n][1] = $nl }
          } catch { $memCtr.Remove($n) }
        }
      }
      if ($line -eq 'progress=end') { break }
    }
    if ($sw.Elapsed.TotalSeconds -gt $Seconds + 120) { try { $p.Kill() } catch {}; break }
  }
  if (-not $wTo) { $wTo = Get-Date }
  if (-not $p.WaitForExit(15000)) { try { $p.Kill() } catch {} }
  $p.WaitForExit()
  $gpuState = Stop-GpuLog $gpuLog $wFrom $wTo
  $res = [ordered]@{ startup_s = $null; fps = $null; fps_min = $null; fps_max = $null; frames = $frame; windows = 0 }
  if ($gpuState) { $res.gpu_under_load = $gpuState }
  # the GPU this process used the most memory on
  $top = $memPeak.Keys | Sort-Object { $memPeak[$_][0] } | Select-Object -Last 1
  if ($top -and $memPeak[$top][0] -ge 1MB) {
    $res.gpu_memory_mib = [ordered]@{ dedicated_peak = [int]($memPeak[$top][0] / 1MB); non_local_peak = [int]($memPeak[$top][1] / 1MB) }
  }
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
    Write-Host ("  {0,-14} FAILED: {1}" -f $s.name, $res.error) -ForegroundColor Red
  } elseif (($s.nvof -and $plan -notmatch 'nvof') -or ($s.nvenc -and $plan -notmatch 'nvenc')) {
    $res.ok = $false
    $res.skipped = if ($s.nvof) { "no NVIDIA optical flow accelerator on this GPU (Turing or newer)" }
                   else { "no NVENC motion estimation (nvencmv.dll missing, or a GPU older than Pascal)" }
    Write-Host ("  {0,-14} skipped: {1}" -f $s.name, $res.skipped) -ForegroundColor Yellow
  } else {
    Write-Host ("  {0,-14} {1} ..." -f $s.name, $s.desc)
    $m = Time-Stage (Stage-Script $s $false)
    foreach ($k in $m.Keys) { $res[$k] = $m[$k] }
    $res.rate = $s.rate
    if ($res.ok) { $res.realtime = [math]::Round($res.fps / $s.rate, 2) }
    if ($res.ok) { Write-Host ("  {0,-14} {1,7:N1} fps = {2,5:N2}x realtime   (start-up {3} s)" -f $s.name, $res.fps, $res.realtime, $res.startup_s) -ForegroundColor Green }
    else { Write-Host ("  {0,-14} FAILED: {1}" -f $s.name, $res.error) -ForegroundColor Red }
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
  $gpuLine)
if ($host_.hw_gpu_scheduling -or $host_.hw_gpu_scheduling_setting) {
  $hs = if ($host_.hw_gpu_scheduling) { $host_.hw_gpu_scheduling -join '; ' } else { "unknown" }
  $txt += "Hardware-accelerated GPU scheduling: $hs; setting: $($host_.hw_gpu_scheduling_setting)"
}
$txt += @("OS: $($host_.os); $($host_.avisynth); $($host_.ffmpeg)", "")
foreach ($r in $results) {
  if ($r.ok) {
    $txt += ("{0,-14} {1,8:N1} fps {2,6:N2}x realtime   {3}" -f $r.stage, $r.fps, $r.realtime, $r.what)
    $g = $r.gpu_under_load; $mm = $r.gpu_memory_mib
    $parts = @()
    if ($g) {
      $parts += "GPU $($g.gpu): $($g.pcie) $($g.pstate), sm $($g.sm_mhz) / mem $($g.mem_mhz) MHz, load $($g.util_pct) %, max $($g.power_w_max) W $($g.temp_c_max) C"
    }
    if ($mm) { $parts += "GPU memory $($mm.dedicated_peak) MiB, non-local (system RAM) $($mm.non_local_peak) MiB" }
    if ($parts.Count) { $txt += ("{0,-14} {1}" -f "", ($parts -join '; ')) }
  }
  elseif ($r.skipped) { $txt += ("{0,-14}  skipped   {1}: {2}" -f $r.stage, $r.what, $r.skipped) }
  else { $txt += ("{0,-14}   FAILED   {1}: {2}" -f $r.stage, $r.what, $r.error) }
}
[System.IO.File]::WriteAllLines([System.IO.Path]::ChangeExtension($json, ".txt"), [string[]]$txt, $utf8)
Write-Host ""
$txt | ForEach-Object { Write-Host $_ }
Write-Host ""
Write-Host "Results: results\$(Split-Path $json -Leaf) - please post this file (it contains no paths or names)."
if (-not $KeepScripts) { Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue }
