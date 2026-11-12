# Fails if the DLL imports anything besides Windows system DLLs (self-contained deployment check).
# usage: pwsh -File check_deps.ps1 <CudaConvolution.dll>
param([Parameter(Mandatory)][string]$Dll)
$ErrorActionPreference = 'Stop'

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$dumpbin = Get-ChildItem "$vs\VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe" | Select-Object -Last 1
$deps = & $dumpbin.FullName /nologo /dependents $Dll | Where-Object { $_ -match '^\s+\S+\.dll\s*$' } | ForEach-Object { $_.Trim() }

$allowed = '^(KERNEL32|USER32|ADVAPI32|SHELL32|OLE32|OLEAUT32|WS2_32|bcrypt|ntdll|VERSION|api-ms-win-core-.*)\.dll$'
$bad = $deps | Where-Object { $_ -notmatch $allowed }
"dependencies: $($deps -join ', ')"
if ($bad) { "NOT self-contained: $($bad -join ', ')"; exit 1 }
"OK: only system DLLs"
