# cudafilters: AviSynth+ GPU filters (CudaUpload, CudaBob, CudaNVOFAnalyse, ...). Plain C++, no CUDA toolkit needed:
# all device code is in mvtools_cuda, which this plugin opens at run time from its own folder.

file(GLOB CUDAFILTERS_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_library(cudafilters SHARED ${CUDAFILTERS_SOURCES})
target_include_directories(cudafilters PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../Sources/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../Sources/cuda")

if(MSVC)
  set_target_properties(cudafilters PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded")
  target_compile_definitions(cudafilters PRIVATE _CRT_SECURE_NO_WARNINGS)
elseif(MINGW)
  set_target_properties(cudafilters PROPERTIES PREFIX "")
endif()
if(NOT WIN32)
  target_link_libraries(cudafilters PRIVATE ${CMAKE_DL_LIBS})
endif()

include(GNUInstallDirs)
install(TARGETS cudafilters
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/avisynth"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
