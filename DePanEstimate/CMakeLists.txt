# DePanEstimate: DePanEstimate.dll (MSVC, MinGW) / libdepanestimate.so. MSVC reproduces DePanEstimate/DePanEstimate.vcxproj, Release|x64 (multibyte, static CRT).

include("Files.cmake")

add_library(depanestimate SHARED ${DePanEstimate_Sources})
if(WIN32)
  set_target_properties(depanestimate PROPERTIES OUTPUT_NAME "DePanEstimate")
  target_sources(depanestimate PRIVATE depanestimate.rc)
else()
  set_target_properties(depanestimate PROPERTIES OUTPUT_NAME "depanestimate")
endif()
mv_windows_names(depanestimate)

target_include_directories(depanestimate PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(MSVC)
  target_compile_definitions(depanestimate PRIVATE WIN32 _WINDOWS _USRDLL DePanEstimate_EXPORTS _MBCS)
  mv_msvc_release_flags(depanestimate)
endif()
# GCC / Clang: SSE2 base (upstream 2.7.48) with INTEL_INTRINSICS, or NEON_INTRINSICS, from the top level
if(NOT WIN32)
  target_link_libraries(depanestimate PRIVATE ${CMAKE_DL_LIBS})
endif()

INSTALL(TARGETS depanestimate
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/avisynth"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
