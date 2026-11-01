@echo off
rem build.bat with the k_search phase counters (-DMVC_PHASE_PROF): x64\Prof\mvtools_cuda.dll, same toolkit and architecture
rem rules as build.bat.
setlocal
set "MVCUDA_OUT=x64\Prof"
set "MVCUDA_NVCC_FLAGS=-DMVC_PHASE_PROF %MVCUDA_NVCC_FLAGS%"
call "%~dp0build.bat"
