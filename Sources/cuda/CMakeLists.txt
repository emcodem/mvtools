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

# Release: machine code for 8.0 / 8.6 / 8.9 / 9.0 / 12.0 plus PTX 8.0 for newer cards (the code needs cp.async, 8.0+).
# Development: -DMVTOOLS_CUDA_ARCH=120-real builds for the local RTX 50xx only, several times faster.
set(MVTOOLS_CUDA_ARCH "80-real;86-real;89-real;90-real;120-real;80-virtual" CACHE STRING "CUDA architectures of mvtools_cuda")
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
