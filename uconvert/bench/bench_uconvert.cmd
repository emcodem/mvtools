@echo off
rem UConvert benchmark (README.md in this folder). Arguments are passed on, e.g. -FFmpeg C:\tools\ffmpeg.exe -NoCpu
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0bench_uconvert.ps1" %*
if "%~1"=="" pause
