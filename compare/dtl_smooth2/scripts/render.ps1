# The commands that produced the report images and numbers, in the order they were run.
# $S = working folder with avs\ (the scripts next to this file), viz\ and vizfix\ for the output.
#   pinterf_sm1_*.avs : our mvtools2.dll (pinterf cdf04a5 + MT patches, CPU), smooth=1 (default)
#   dtl_sm1_flow.avs  : DTL2020 post_2.7.47_822 (c688008) built from source, smooth=1
#   dtl_sm2_*.avs     : same DLL, smooth=2
#   fix_sm2_*.avs     : DTL c688008 + ../smooth2_radius_fix.patch, smooth=2
# DTL build: MSBuild Sources\mvtools.vcxproj /p:Configuration=Release /p:Platform=x64 /p:PlatformToolset=v145, NASMPATH set.
param([string]$S = $PSScriptRoot)
$A = "$S\avs"; $V = "$S\viz"; $F = "$S\vizfix"
New-Item -ItemType Directory -Force $V, $F | Out-Null

# vector frames (bv/fv at frames 10 and 50, raw MVTools vector clip) and MFlowFps frames 18..25 as RGB PNG
foreach ($t in 'pinterf_sm1', 'dtl_sm2', 'fix_sm2') {
  cmd /c "ffmpeg -nostdin -hide_banner -y -loglevel error -i `"$A\${t}_vec.avs`" -f rawvideo `"$V\${t}_vec.raw`" 2>&1"
}
foreach ($p in @(@('ours', 'pinterf_sm1'), @('dtl', 'dtl_sm1'), @('dtlsm2', 'dtl_sm2'), @('fixsm2', 'fix_sm2'))) {
  cmd /c "ffmpeg -nostdin -hide_banner -y -loglevel error -i `"$A\$($p[1])_flow.avs`" -pix_fmt rgb24 `"$V\$($p[0])_%d.png`" 2>&1"
}

# PSNR of each interpolated frame against smooth=1 (ours_N vs dtl_N is inf for all N: identical)
foreach ($b in 'dtl', 'dtlsm2', 'fixsm2') {
  1..8 | ForEach-Object {
    $r = (cmd /c "ffmpeg -nostdin -hide_banner -i `"$V\ours_$_.png`" -i `"$V\${b}_$_.png`" -lavfi psnr -f null - 2>&1" | Select-String 'average:(\S+)').Matches[0].Groups[1].Value
    "$b frame $_ : $r dB"
  }
}

# per-block maps and change statistics (viz.py writes PPM + stats.json)
python "$PSScriptRoot\viz.py" "$V\pinterf_sm1_vec.raw" "$V\dtl_sm2_vec.raw" $V
python "$PSScriptRoot\viz.py" "$V\pinterf_sm1_vec.raw" "$V\fix_sm2_vec.raw" $F
Get-ChildItem "$V\*.ppm" | ForEach-Object { ffmpeg -nostdin -loglevel error -y -i $_.FullName ([IO.Path]::ChangeExtension($_.FullName, '.png')) }
ffmpeg -nostdin -loglevel error -y -i "$F\bv10_change.ppm" "$F\bv10_change.png"

# crops of interpolated frame 5 (pole and wires) and difference x12
$C = 'crop=960:540:2880:240'
Push-Location $V
ffmpeg -nostdin -loglevel error -y -i ours_5.png -vf $C -q:v 2 crop_pinterf.jpg
ffmpeg -nostdin -loglevel error -y -i dtlsm2_5.png -vf $C -q:v 2 crop_dtl.jpg
ffmpeg -nostdin -loglevel error -y -i fixsm2_5.png -vf $C -q:v 2 crop_fix.jpg
ffmpeg -nostdin -loglevel error -y -i ours_5.png -i dtlsm2_5.png -lavfi "[0][1]blend=all_mode=difference,$C,format=gray,lutyuv=y='min(255,val*12)'" -q:v 2 crop_diff.jpg
ffmpeg -nostdin -loglevel error -y -i ours_5.png -i fixsm2_5.png -lavfi "[0][1]blend=all_mode=difference,$C,format=gray,lutyuv=y='min(255,val*12)'" -q:v 2 crop_diff_fix.jpg
ffmpeg -nostdin -loglevel error -y -i ours_5.png -vf "scale=960:-1,drawbox=x=720:y=60:w=240:h=135:color=0xffb000:t=3" -q:v 3 frame_overview.jpg
Pop-Location
