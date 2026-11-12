# Release build + tests + package: dist\CudaConvolution-<version>\ and dist\CudaConvolution-<version>-win64.zip
# usage: pwsh -File tools\package.ps1 [-Cuda <CUDA 12.x toolkit root>]
# CUDA 12.x on purpose: the DLL then runs on NVIDIA drivers R525+ (production A4000 machines run R535).
param([string]$Cuda = $env:CUDA_PATH)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$version = (Select-String -Path "$root\CMakeLists.txt" -Pattern 'project\(CudaConvolution VERSION ([0-9.]+)').Matches[0].Groups[1].Value
$nvcc = Join-Path $Cuda "bin\nvcc.exe"
if (-not (Test-Path $nvcc)) { throw "CUDA toolkit not found at $Cuda (need bin\nvcc.exe)" }
if (-not ((& $nvcc --version) -match 'release 12\.')) { throw "use a CUDA 12.x toolkit for releases (driver R525+ compatibility)" }

# build + all tests (unit, GPU == CPU, self-test vectors, dependency check)
pwsh -NoProfile -File "$root\tools\build.ps1" -Cuda $Cuda -BuildDir build-release -Config Release
if ($LASTEXITCODE) { throw "build or tests failed" }

$dist = "$root\dist\CudaConvolution-$version"
if (Test-Path $dist) { Remove-Item -Recurse -Force $dist }
New-Item -ItemType Directory -Force "$dist\examples" | Out-Null
Copy-Item "$root\build-release\CudaConvolution.dll" $dist
Copy-Item "$root\docs\usage.md" "$dist\README.md"
Copy-Item "$root\CHANGELOG.md", "$root\THIRD_PARTY.md" $dist
Copy-Item "$root\docs\validation.md" "$dist\VALIDATION.md"
Copy-Item "$root\examples\*.avs" "$dist\examples"
New-Item -ItemType Directory -Force "$dist\presets" | Out-Null
Copy-Item "$root\presets\*.json" "$dist\presets"  # template for own preset files (built-in copy is in the DLL)
$hash = (Get-FileHash "$dist\CudaConvolution.dll" -Algorithm SHA256).Hash.ToLower()
"$hash  CudaConvolution.dll" | Set-Content "$dist\SHA256SUMS.txt"
$zip = "$root\dist\CudaConvolution-$version-win64.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path "$dist\*" -DestinationPath $zip
"packaged $zip"
"CudaConvolution.dll $([int]((Get-Item "$dist\CudaConvolution.dll").Length / 1KB)) KB, sha256 $hash"
