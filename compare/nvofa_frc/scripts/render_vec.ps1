# Vector dumps for the maps of the README (output frames 41 / 42): MRecalculate and MAnalyse vectors of the exact search
# and of NVOFA medium / fast as raw files, then vecviz.py draws the maps (..\img\vec_*.png) and prints the statistics.
#   .\render_vec.ps1 [-W <work folder>]
param([string]$W = "$env:TEMP\nvofa_frc")
New-Item -ItemType Directory -Force $W | Out-Null
foreach ($v in @(@('exact', ''), @('medium', 'medium'), @('fast', 'fast'))) {
  $env:MVCUDA_NVOFA = $v[1]
  foreach ($kind in 'recalc', 'analyse') {
    ffmpeg -nostdin -hide_banner -y -loglevel error -i "$PSScriptRoot\avs\frc_vec_$kind.avs" -f rawvideo "$W\vec_${kind}_$($v[0]).raw"
  }
}
$env:MVCUDA_NVOFA = $null
python -I "$PSScriptRoot\vecviz.py" $W
