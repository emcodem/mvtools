# Golden scripts with our mvtools2.dll vs DTL's fork (CPU only). Writes raw vector frames (vec) and framemd5 (flow, degrain).
#   .\cmp.ps1 [-Extra 'AreaMode=1'] [-Tag am1] [-Dlls ours,dtl]
param([string]$Extra = '', [string]$Tag = 'base', [string[]]$Dlls = @('ours', 'dtl'))
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$dll = @{ ours = 'C:\dev\mvtools\profile\build_m1\mvtools2.dll'; dtl = "$here\build\mvtools2.dll" }
$fmt = [ordered]@{
  '10_422' = ''
  '8_422' = '.ConvertBits(8, dither=-1)'
  '10_420' = '.ConvertToYUV420()'
  '8_420' = '.ConvertBits(8, dither=-1).ConvertToYUV420()'
  '10_444' = '.ConvertToYUV444()'
}
$ex = if ($Extra) { ", $Extra" } else { '' }
$head = @'
LoadPlugin("C:\dev\uconvert\dlls\LSMASHSource.dll")
LoadPlugin("C:\dev\uconvert\dlls\RgTools.dll")
LoadPlugin("{DLL}")
SEGFILE = "C:\dev\mvtools\profile\clips\seg1_wires_pole.mp4"
m = LWLibavVideoSource(SEGFILE, cachefile=SEGFILE + ".lwi"){FMT}
'@
$frc = @'
pre = m.RemoveGrain(22)
supf = MSuper(pre, pel=2, hpad=16, vpad=16, rfilter=4, mt=false)
sup = MSuper(m, pel=2, hpad=16, vpad=16, rfilter=4, levels=1, mt=false)
bv = MAnalyse(supf, isb=true, blksize=32, overlap=16, search=3, searchparam=2, truemotion=true, chroma=true, mt=false{EX})
fv = MAnalyse(supf, isb=false, blksize=32, overlap=16, search=3, searchparam=2, truemotion=true, chroma=true, mt=false{EX})
bv = MRecalculate(sup, bv, blksize=16, overlap=8, thSAD=100, search=3, searchparam=2, truemotion=true, chroma=true, mt=false{EX})
fv = MRecalculate(sup, fv, blksize=16, overlap=8, thSAD=100, search=3, searchparam=2, truemotion=true, chroma=true, mt=false{EX})
'@
$body = @{
  vec = $frc + 'return Interleave(bv.Trim(10, -1), fv.Trim(10, -1), bv.Trim(50, -1), fv.Trim(50, -1))'
  flow = $frc + "out = MFlowFps(m, sup, bv, fv, num=24000, den=1001, mask=2, ml=200, blend=false)`nreturn out.Trim(18, 25)"
  degrain = @'
src = m.Trim(0, 29)
sup = MSuper(src, pel=2, mt=false)
bv1 = MAnalyse(sup, isb=true, delta=1, blksize=16, overlap=8, chroma=true, mt=false{EX})
fv1 = MAnalyse(sup, isb=false, delta=1, blksize=16, overlap=8, chroma=true, mt=false{EX})
bv2 = MAnalyse(sup, isb=true, delta=2, blksize=16, overlap=8, chroma=true, mt=false{EX})
fv2 = MAnalyse(sup, isb=false, delta=2, blksize=16, overlap=8, chroma=true, mt=false{EX})
d = MDegrain2(src, sup, bv1, fv1, bv2, fv2, thSAD=400, thSCD1=50000)
c = MCompensate(src, sup, bv1, thSCD1=50000)
return StackHorizontal(d, c)
'@
}
foreach ($d in $Dlls) {
  $out = Join-Path $here "out\$Tag\$d"
  New-Item -ItemType Directory -Force $out | Out-Null
  foreach ($f in $fmt.Keys) {
    foreach ($t in 'vec', 'flow', 'degrain') {
      $avs = Join-Path $out "${f}_$t.avs"
      ($head + "`n" + $body[$t]).Replace('{DLL}', $dll[$d]).Replace('{FMT}', $fmt[$f]).Replace('{EX}', $ex) | Set-Content -Encoding ascii $avs
      $dst = if ($t -eq 'vec') { "-f rawvideo `"$([IO.Path]::ChangeExtension($avs, '.raw'))`"" } else { "-f framemd5 `"$([IO.Path]::ChangeExtension($avs, '.md5'))`"" }
      $sw = [Diagnostics.Stopwatch]::StartNew()
      $err = cmd /c "ffmpeg -nostdin -hide_banner -y -loglevel error -i `"$avs`" $dst 2>&1"
      $sw.Stop()
      "{0,-5} {1,-7} {2,-8} rc={3} {4,6:n1}s {5}" -f $d, $f, $t, $LASTEXITCODE, $sw.Elapsed.TotalSeconds, (($err | Out-String).Trim() -replace '\s+', ' ')
    }
  }
}
