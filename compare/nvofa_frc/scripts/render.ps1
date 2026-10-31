# Renders the comparison of opticalflow.md: the uc_FRC output with the exact GPU search and with NVOFA (fast, medium).
#   .\render.ps1 [-W <work folder>]   (big PNGs go to the work folder, the page images to ..\img)
param([string]$W = "$env:TEMP\nvofa_frc")
$A = "$PSScriptRoot\avs\frc_flow.avs"; $I = "$PSScriptRoot\..\img"
New-Item -ItemType Directory -Force $W, $I | Out-Null
$C = 'crop=960:540:2880:240'           # pole and wires (same crop as compare/dtl_smooth2)
$variants = [ordered]@{ exact = ''; medium = 'medium'; fast = 'fast' }

foreach ($v in $variants.Keys) {
  $env:MVCUDA_NVOFA = $variants[$v]
  # output frames 10..17 of this script = frames 18..25 of the plain MFlowFps output (compare/dtl_smooth2)
  ffmpeg -nostdin -hide_banner -y -loglevel error -i $A -vf "select='between(n,10,17)'" -fps_mode passthrough -pix_fmt rgb24 "$W\${v}_%d.png"
  # the whole crop region as a short H.264 clip for judging motion
  ffmpeg -nostdin -hide_banner -y -loglevel error -i $A -vf "$C,format=yuv420p" -c:v libx264 -crf 14 -preset slow "$W\${v}_crop.mp4"
}
$env:MVCUDA_NVOFA = $null

# crops of the 8 frames, and differences to the exact search x8
Push-Location $W
foreach ($v in $variants.Keys) { 1..8 | ForEach-Object { ffmpeg -nostdin -loglevel error -y -i "${v}_$_.png" -vf $C -q:v 2 "$I\crop_${v}_$_.jpg" } }
foreach ($v in 'medium', 'fast') {
  1..8 | ForEach-Object {
    ffmpeg -nostdin -loglevel error -y -i "exact_$_.png" -i "${v}_$_.png" -lavfi "[0][1]blend=all_mode=difference,$C,format=gray,lutyuv=y='min(255,val*8)'" -q:v 2 "$I\diff_${v}_$_.jpg"
    $r = (cmd /c "ffmpeg -nostdin -hide_banner -i exact_$_.png -i ${v}_$_.png -lavfi psnr -f null - 2>&1" | Select-String 'average:(\S+)').Matches[0].Groups[1].Value
    "$v frame $_ vs exact: PSNR $r dB (whole frame)"
  }
}
# side by side clip: exact | medium | fast (crop region, half size each)
ffmpeg -nostdin -loglevel error -y -i exact_crop.mp4 -i medium_crop.mp4 -i fast_crop.mp4 -lavfi "[0][1][2]hstack=3" -c:v libx264 -crf 18 -preset slow "$I\side_by_side.mp4"
ffmpeg -nostdin -loglevel error -y -i exact_5.png -vf "scale=960:-1,drawbox=x=720:y=60:w=240:h=135:color=0xffb000:t=3" -q:v 3 "$I\frame_overview.jpg"
Pop-Location
