# mvtools_vsbridge: AviSynth+ host for VapourSynth (VAPOURSYNTH_PLAN.md, decision 5). Compiles avisynth.h in core mode
# (BUILDING_AVSCORE): the AviSynth+ member functions live here and reach the plugins as their AVS_Linkage table, so this
# must stay a library of its own, never linked into a plugin.

add_library(mvtools_vsbridge SHARED
  avs_interface.cpp
  avs_cpuid.cpp
  bridge.cpp
  bridge_core.h)
target_include_directories(mvtools_vsbridge PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../Sources/include")
target_compile_definitions(mvtools_vsbridge PRIVATE BUILDING_AVSCORE)
set_target_properties(mvtools_vsbridge PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)

if(MSVC)
  set_target_properties(mvtools_vsbridge PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded")
  target_compile_definitions(mvtools_vsbridge PRIVATE _CRT_SECURE_NO_WARNINGS)
elseif(MINGW)
  set_target_properties(mvtools_vsbridge PROPERTIES PREFIX "")
endif()
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
