FILE(GLOB DePan_Sources RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*.c"
  "*.cpp"
  "*.hpp"
  "*.h"
  "include/*.h"
  "include/avs/*.h"
)


IF( MSVC OR MINGW )
    # Export definitions in general are not needed on x64 and only cause warnings,
    # unfortunately we still must need a .def file for some COM functions.
    # NO C interface for this plugin
    # if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    #  LIST(APPEND DePan_Sources "DePan64.def")
    # else()
    #  LIST(APPEND DePan_Sources "DePan.def")
    # endif() 
ENDIF()
# the version resource (.rc) is added by CMakeLists.txt for Windows builds
