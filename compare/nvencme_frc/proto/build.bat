@echo off
rem Builds probe.exe and nvencme_proto.dll (MSVC, /MT) of the NVENC ME-only prototype.
rem   build.bat <folder of nvEncodeAPI.h> [probe|dll|dx12]
rem nvEncodeAPI.h: FFmpeg/nv-codec-headers tag n13.0.19.1 (MIT), include\ffnvcodec. Output next to this file.
setlocal
cd /d "%~dp0"
if "%~1"=="" (echo usage: build.bat ^<folder of nvEncodeAPI.h^> [probe^|dll] & exit /b 1)
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set NVCH=%~1
set MV=%~dp0..\..\..\Sources
if "%2"=="dll" goto dll
if "%2"=="dx12" goto dx12
cl /nologo /O2 /MT /EHsc /std:c++17 /I"%NVCH%" probe.cpp /Fe:probe.exe || exit /b 1
if "%2"=="probe" exit /b 0
:dll
cl /nologo /O2 /MT /EHsc /std:c++17 /LD /I"%NVCH%" /I"%MV%" /I"%MV%\include" plugin.cpp /Fe:nvencme_proto.dll || exit /b 1
if "%2"=="dll" exit /b 0
:dx12
cl /nologo /O2 /MT /EHsc /std:c++17 dx12probe.cpp /Fe:dx12probe.exe || exit /b 1
