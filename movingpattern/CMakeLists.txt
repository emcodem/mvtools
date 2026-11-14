# MovingPattern: synthetic moving test clips (AviSynth+ source filter), plain C++
add_library(movingpattern SHARED "${CMAKE_CURRENT_SOURCE_DIR}/MovingPattern.cpp")
target_include_directories(movingpattern PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../Sources/include")
set_target_properties(movingpattern PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
if(MSVC)
  target_compile_definitions(movingpattern PRIVATE _CRT_SECURE_NO_WARNINGS _WINDOWS _USRDLL UNICODE _UNICODE)
  mv_msvc_release_flags(movingpattern)
else()
  target_compile_definitions(movingpattern PRIVATE _stricmp=strcasecmp)
  find_package(Threads REQUIRED)
  target_link_libraries(movingpattern PRIVATE Threads::Threads)
endif()
mv_windows_names(movingpattern)
if(WIN32)
  set_target_properties(movingpattern PROPERTIES OUTPUT_NAME "MovingPattern")
endif()

include(GNUInstallDirs)
install(TARGETS movingpattern
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/avisynth"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
