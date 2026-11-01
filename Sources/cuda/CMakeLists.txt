# mvtools_cuda: the optional CUDA library that mvtools2 and cudafilters load at run time (mvtools_cuda.dll /
# libmvtools_cuda.so, next to the plugins). Static CUDA runtime: only the NVIDIA driver is needed at run time, and the
# library has no link-time dependency on the driver (libcuda / nvcuda is opened by the runtime itself), so a machine
# without a driver just reports "no device" and the plugins use the CPU.
#
# The _8 / _16 units include the shared sources (mvcuda_analyse.cu etc.) with the sample size defined; the shared sources
# are not compiled on their own.

set(MVCUDA_SOURCES
  mvcuda.cu mvcuda_core.cu mvcuda_frames.cu mvcuda_resample.cu mvcuda_temporal.cu mvcuda_merge.cu mvcuda_kobs.cu
  mvcuda_nnedi3.cu mvcuda_comp.cu mvcuda_nvofa.cu
  mvcuda_super_8.cu mvcuda_super_16.cu mvcuda_recalc_8.cu mvcuda_recalc_16.cu mvcuda_analyse_8.cu mvcuda_analyse_16.cu
  mvcuda_flow_8.cu mvcuda_flow_16.cu mvcuda_nvof_vec_8.cu mvcuda_nvof_vec_16.cu)

add_library(mvtools_cuda SHARED ${MVCUDA_SOURCES})

# Release: the device code runs on compute capability 6.0 and newer (cp.async / acquire-release paths for 8.0+ / 7.0+,
# fallbacks below, mvcuda_internal.h MVC_ARCH_BELOW). With CUDA 12.x (12.8+ for 12.0; 12.9 recommended): machine code
# for 6.1 (GTX 10xx), 7.0, 7.5, 8.0, 8.6, 8.9, 9.0, 12.0, PTX 6.0 (P100, JIT) and PTX 8.0 (newer architectures get the
# 8.0 code JIT-compiled). CUDA 13.x dropped Pascal and Volta: there the default is 7.5 .. 12.0 + PTX 8.0, and cards
# below 7.5 find no code in the library (no usable device). On Windows, CUDA 12.x needs the VS 2022 (v143) host
# toolset; the VS 2026 STL (14.5x) does not compile with it.
# Development: -DMVTOOLS_CUDA_ARCH=120-real builds for the local RTX 50xx only, several times faster.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13)
  set(_mvcuda_arch_default "61-real;70-real;75-real;80-real;86-real;89-real;90-real;120-real;60-virtual;80-virtual")
else()
  set(_mvcuda_arch_default "75-real;80-real;86-real;89-real;90-real;120-real;80-virtual")
endif()
set(MVTOOLS_CUDA_ARCH "${_mvcuda_arch_default}" CACHE STRING "CUDA architectures of mvtools_cuda")
# Extra nvcc flags, e.g. -DMVC_PHASE_PROF (k_search phase timing, was build_prof.bat)
set(MVTOOLS_CUDA_FLAGS "" CACHE STRING "extra nvcc flags for mvtools_cuda")

set_target_properties(mvtools_cuda PROPERTIES
  CUDA_ARCHITECTURES "${MVTOOLS_CUDA_ARCH}"
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  CUDA_RUNTIME_LIBRARY Static
  CUDA_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(mvtools_cuda PRIVATE MVCUDA_EXPORTS)
separate_arguments(_mvcuda_extra NATIVE_COMMAND "${MVTOOLS_CUDA_FLAGS}")
target_compile_options(mvtools_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 ${_mvcuda_extra}>)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13)
  # CUDA 12 warns about every target below 7.5 (deprecated, still supported)
  target_compile_options(mvtools_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
endif()

if(WIN32)
  # same output name and runtime as the former build.bat: mvtools_cuda.dll, static CRT, /O2 host code
  set_target_properties(mvtools_cuda PROPERTIES PREFIX "" MSVC_RUNTIME_LIBRARY "MultiThreaded")
  target_compile_options(mvtools_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/O2>)
else()
  # dlopen / dladdr in dyn_lib.h (NVOFA loader)
  target_link_libraries(mvtools_cuda PRIVATE ${CMAKE_DL_LIBS})
endif()

include(GNUInstallDirs)
install(TARGETS mvtools_cuda
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/avisynth"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
