# Crop clips of the lambda variants (README "Cause: truemotion's lambda"): exact search with MAnalyse plevel=0, lambda=4000
# and with lambda=1000, next to the uc_FRC settings and NVOFA medium (exact_crop.mp4 / medium_crop.mp4 from render.ps1),
# then the 2x2 slow-motion clip with the cell grid over the whole run.
#   .\render_lambda.ps1 [-W <work folder of render.ps1>]
param([string]$W = "$env:TEMP\nvofa_frc")
$C = 'crop=960:540:2880:240'
$env:MVCUDA_NVOFA = $null
foreach ($k in 'pl0_l4000', 'l1000') {
  ffmpeg -nostdin -hide_banner -y -loglevel error -i "$PSScriptRoot\avs\frc_flow_$k.avs" -vf "$C,format=yuv420p" -c:v libx264 -crf 14 -preset slow "$W\${k}_crop.mp4"
}
python -I "$PSScriptRoot\slowmo.py" $W 0 4.7 4 --variants "exact:uc_FRC (lambda 16000),pl0_l4000:plevel 0 lambda 4000,l1000:lambda 1000,medium:NVOFA medium" --cols 2 --name slowmo_lambda
