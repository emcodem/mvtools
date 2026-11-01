# Cross compile the Windows x64 DLLs on Linux with MinGW-w64 (GCC, posix threads):
#
#   cmake -G Ninja -S . -B build/mingw -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-mingw64.cmake
#   cmake --build build/mingw          -> build/mingw/bin: mvtools2.dll, DePan.dll, DePanEstimate.dll,
#                                         mvtools_vsbridge.dll, cudafilters.dll
#
# Not built: mvtools_cuda.dll (nvcc has no Linux-hosted Windows target; build it natively with build_windows.ps1 and put it
# next to the cross-compiled DLLs, the C API between them is plain C). The MinGW mvtools2.dll uses the intrinsic code
# paths instead of the NASM sources of the MSVC build, and has no avstp support (def.h), so MAnalyse runs single-sliced
# under AviSynth+ (as on Linux): use the MSVC build for production speed.
#
# Compiler: the Debian / Ubuntu packages g++-mingw-w64-x86-64-posix + binutils-mingw-w64-x86-64 (any prefix: without
# root, `apt download` them plus gcc-mingw-w64-x86-64-posix{,-runtime}, gcc-mingw-w64-base, mingw-w64-common,
# mingw-w64-x86-64-dev, binutils-mingw-w64-base and unpack each with `dpkg -x <deb> ~/opt/mingw`, then pass
# -DMINGW_ROOT=~/opt/mingw/usr). The posix thread model is required (std::thread, std::mutex).
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(MINGW_TRIPLE "x86_64-w64-mingw32" CACHE STRING "MinGW-w64 target triple")
set(MINGW_ROOT "" CACHE PATH "prefix of the MinGW-w64 tools (contains bin/); empty: the PATH")
if(NOT MINGW_ROOT AND DEFINED ENV{MINGW_ROOT})
  set(MINGW_ROOT "$ENV{MINGW_ROOT}")
endif()
# try_compile projects read this file again: hand them the prefix
set(ENV{MINGW_ROOT} "${MINGW_ROOT}")
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES MINGW_ROOT MINGW_TRIPLE)
set(_mgw_hints "")
if(MINGW_ROOT)
  set(_mgw_hints HINTS "${MINGW_ROOT}/bin" NO_DEFAULT_PATH)
endif()

find_program(CMAKE_C_COMPILER NAMES ${MINGW_TRIPLE}-gcc-posix ${MINGW_TRIPLE}-gcc ${_mgw_hints} REQUIRED)
find_program(CMAKE_CXX_COMPILER NAMES ${MINGW_TRIPLE}-g++-posix ${MINGW_TRIPLE}-g++ ${_mgw_hints} REQUIRED)
find_program(CMAKE_RC_COMPILER NAMES ${MINGW_TRIPLE}-windres windres ${_mgw_hints} REQUIRED)
# windres preprocesses with "<triple>-gcc", which the -posix packages do not provide under that name
set(CMAKE_RC_FLAGS_INIT "--preprocessor=${CMAKE_C_COMPILER} --preprocessor-arg=-E --preprocessor-arg=-xc --preprocessor-arg=-DRC_INVOKED")
find_program(CMAKE_ARNAMES ${MINGW_TRIPLE}-gcc-ar-posix ${MINGW_TRIPLE}-ar ${_mgw_hints})
find_program(CMAKE_RANLIB NAMES ${MINGW_TRIPLE}-gcc-ranlib-posix ${MINGW_TRIPLE}-ranlib ${_mgw_hints})

# headers and libraries only from the MinGW sysroot, programs from the host
if(MINGW_ROOT)
  set(CMAKE_FIND_ROOT_PATH "${MINGW_ROOT}/${MINGW_TRIPLE}")
else()
  set(CMAKE_FIND_ROOT_PATH "/usr/${MINGW_TRIPLE}")
endif()
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
