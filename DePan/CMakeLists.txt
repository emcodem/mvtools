# DePan: DePan.dll (MSVC, MinGW) / libdepan.so. MSVC reproduces DePan/DePan.vcxproj, Release|x64 (multibyte, static CRT).

include("Files.cmake")

add_library(depan SHARED ${DePan_Sources})
if(WIN32)
  set_target_properties(depan PROPERTIES OUTPUT_NAME "DePan")
  target_sources(depan PRIVATE depan.rc)
else()
  set_target_properties(depan PROPERTIES OUTPUT_NAME "depan")
endif()
mv_windows_names(depan)

target_include_directories(depan PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(MSVC)
  target_compile_definitions(depan PRIVATE WIN32 _WINDOWS _USRDLL DePan_EXPORTS _MBCS)
  mv_msvc_release_flags(depan)
endif()
# GCC / Clang: SSE2 base (upstream 2.7.48) with INTEL_INTRINSICS, or NEON_INTRINSICS, from the top level
if(NOT WIN32)
  target_link_libraries(depan PRIVATE ${CMAKE_DL_LIBS})
endif()

INSTALL(TARGETS depan
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/avisynth"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
