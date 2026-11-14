# nvencmv: CudaNVENCAnalyse, motion vectors from NVIDIA's video encoder (NVENC ME-only) as an MVTools vector clip; its
# own plugin, plain C++. The device code (grey pictures, NVENC sessions, candidate selection, block vectors) is in
# mvtools_cuda, which the plugin opens at run time (as cudafilters and splatfps do).
add_library(nvencmv SHARED "${CMAKE_CURRENT_SOURCE_DIR}/NvencMV.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/VapourSynthEntry.cpp")
target_include_directories(nvencmv PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../Sources/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../Sources/cuda")
set_target_properties(nvencmv PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
if(MSVC)
  target_compile_definitions(nvencmv PRIVATE _CRT_SECURE_NO_WARNINGS _WINDOWS _USRDLL UNICODE _UNICODE)
  mv_msvc_release_flags(nvencmv)
else()
  target_link_libraries(nvencmv PRIVATE ${CMAKE_DL_LIBS})
endif()
mv_windows_names(nvencmv)

include(GNUInstallDirs)
install(TARGETS nvencmv
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/avisynth"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
