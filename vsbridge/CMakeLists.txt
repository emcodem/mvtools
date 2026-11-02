# mvtools_vsbridge: AviSynth+ host for VapourSynth. Compiles avisynth.h in core mode
# (BUILDING_AVSCORE): the AviSynth+ member functions live here and reach the plugins as their AVS_Linkage table, so this
# must stay a library of its own, never linked into a plugin.

add_library(mvtools_vsbridge SHARED
  avs_interface.cpp
  avs_cpuid.cpp
  bridge.cpp
  bridge_core.h
  mt_port.cpp
  mt_port.h)
# mt_port.cpp emulates AviSynth+ Expr's float32 arithmetic: no fused multiply-add (MSVC: /fp:precise without
# /fp:contract does not contract)
if(MSVC)
  set_source_files_properties(mt_port.cpp PROPERTIES COMPILE_OPTIONS "/fp:precise")
else()
  set_source_files_properties(mt_port.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()
target_include_directories(mvtools_vsbridge PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../Sources/include")
target_compile_definitions(mvtools_vsbridge PRIVATE BUILDING_AVSCORE)
set_target_properties(mvtools_vsbridge PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)

if(MSVC)
  set_target_properties(mvtools_vsbridge PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  target_compile_definitions(mvtools_vsbridge PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
  target_compile_options(mvtools_vsbridge PRIVATE /bigobj /utf-8)
  mv_msvc_release_flags(mvtools_vsbridge)
endif()
mv_windows_names(mvtools_vsbridge)
if(NOT WIN32)
  find_package(Threads REQUIRED)
  target_link_libraries(mvtools_vsbridge PRIVATE Threads::Threads)
  if(NOT APPLE)
    target_link_options(mvtools_vsbridge PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
    set_property(TARGET mvtools_vsbridge APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
  endif()
endif()

include(GNUInstallDirs)
install(TARGETS mvtools_vsbridge
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/vapoursynth"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
