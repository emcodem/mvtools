# cudafilters: AviSynth+ GPU filters (CudaUpload, CudaBob, CudaNVOFAnalyse, ...). Plain C++, no CUDA toolkit needed:
# all device code is in mvtools_cuda, which this plugin opens at run time from its own folder.

file(GLOB CUDAFILTERS_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_library(cudafilters SHARED ${CUDAFILTERS_SOURCES})
target_include_directories(cudafilters PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../Sources/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../Sources/cuda")

if(MSVC)
  # cudafilters.vcxproj, Release|x64: Unicode, static CRT
  set_target_properties(cudafilters PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  target_compile_definitions(cudafilters PRIVATE _CRT_SECURE_NO_WARNINGS _WINDOWS _USRDLL UNICODE _UNICODE)
  mv_msvc_release_flags(cudafilters)
endif()
mv_windows_names(cudafilters)
if(NOT WIN32)
  target_link_libraries(cudafilters PRIVATE ${CMAKE_DL_LIBS})
endif()

include(GNUInstallDirs)
install(TARGETS cudafilters
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/avisynth"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
