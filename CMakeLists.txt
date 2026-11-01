# mvtools2 + DePan + DePanEstimate (AviSynth+ plugins), their VapourSynth bridge and the optional GPU libraries.
#
# Builds (README "Build Instructions"):
#   Windows, MSVC + Ninja (x64): build_windows.ps1, or by hand from a "x64 Native Tools" prompt:
#     cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_ASM_NASM_COMPILER=C:/path/nasm.exe
#     cmake --build build
#   Linux (GCC / Clang): build_linux.sh, or cmake -G Ninja -S . -B build && cmake --build build
#   Windows from Linux (MinGW-w64, CPU plugins + bridge + cudafilters): cmake/toolchain-mingw64.cmake
#   Windows on ARM64 (llvm-mingw, NEON, CPU plugins): cmake/toolchains/llvm-mingw-aarch64.cmake
#
# Targets and outputs:
#   mvtools2        mvtools2.dll / libmvtools2.so          (MSVC: with the NASM sources of the .vcxproj)
#   depan           DePan.dll / libdepan.so
#   depanestimate   DePanEstimate.dll / libdepanestimate.so
#   mvtools_vsbridge  VapourSynth host for the plugins above (MVTOOLS_VAPOURSYNTH)
#   mvtools_cuda, cudafilters   GPU part (MVTOOLS_GPU; mvtools_cuda needs CUDA 12, on Windows with MSVC 14.4x)
# This CMake build is the one we maintain and test. The Visual Studio solution (mvtools.sln / .slnx) is upstream's,
# unchanged: it builds upstream's CPU plugins only (without the GPU and VapourSynth parts).
cmake_minimum_required(VERSION 3.18...3.31)

project("MvTools2" LANGUAGES C CXX)
include(GNUInstallDirs)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release, RelWithDebInfo, Debug)" FORCE)
endif()

# Avoid uselessly linking to unused libraries
set(CMAKE_C_STANDARD_LIBRARIES "" CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD_LIBRARIES "" CACHE STRING "" FORCE)
if(WIN32)
  # what the default list provided and the plugins use
  set(CMAKE_C_STANDARD_LIBRARIES "kernel32.lib user32.lib gdi32.lib advapi32.lib ole32.lib uuid.lib" CACHE STRING "" FORCE)
  set(CMAKE_CXX_STANDARD_LIBRARIES "${CMAKE_C_STANDARD_LIBRARIES}" CACHE STRING "" FORCE)
  if(MINGW)
    set(CMAKE_C_STANDARD_LIBRARIES "-lkernel32 -luser32 -lgdi32 -ladvapi32 -lole32 -luuid" CACHE STRING "" FORCE)
    set(CMAKE_CXX_STANDARD_LIBRARIES "${CMAKE_C_STANDARD_LIBRARIES}" CACHE STRING "" FORCE)
  endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS FALSE)

# Detect Intel and AArch64 processors and turn Intel SIMD or NEON SIMD on automatically.
message(STATUS "Target processor: ${CMAKE_SYSTEM_PROCESSOR}")
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" ARCHID)
set(INTEL_SIMD "OFF")
set(NEON_SIMD "OFF")
if( ("${ARCHID}" STREQUAL "x86") OR
    ("${ARCHID}" STREQUAL "x64") OR
    ("${ARCHID}" STREQUAL "i686") OR
    ("${ARCHID}" STREQUAL "amd64") OR
    ("${ARCHID}" STREQUAL "x86_64") )
  set(INTEL_SIMD "ON")
elseif( ("${ARCHID}" STREQUAL "aarch64") OR
        ("${ARCHID}" STREQUAL "arm64") )
  set(NEON_SIMD "ON")
endif()

option(ENABLE_INTEL_SIMD "Enable SIMD intrinsics for Intel processors" "${INTEL_SIMD}")
option(ENABLE_NEON_SIMD "Enable SIMD intrinsics for AArch64 processors" "${NEON_SIMD}")
if(ENABLE_INTEL_SIMD AND ENABLE_NEON_SIMD)
  message(FATAL_ERROR "ENABLE_INTEL_SIMD and ENABLE_NEON_SIMD are mutually exclusive")
endif()

if(CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_CONFIGURATION_TYPES Debug Release RelWithDebInfo)
  set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING "Reset the configurations to what we need" FORCE)
endif()

if(MSVC)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "clang-cl is not supported by this CMake build (MSVC cl, GCC or Clang)")
  endif()
  if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "Only x64 is supported with MSVC by this CMake build")
  endif()
  # Static CRT for every DLL (no Visual C++ redistributable needed). Note: the .vcxproj of mvtools2 uses /MD.
  option(MVTOOLS_STATIC_CRT "MSVC: link the C runtime statically (/MT)" ON)
  if(MVTOOLS_STATIC_CRT)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  else()
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
  endif()
  add_compile_definitions($<$<COMPILE_LANGUAGE:C,CXX>:_CRT_SECURE_NO_WARNINGS>)
endif()

# Release code generation of the Visual Studio projects (mvtools.vcxproj Release|x64): full optimisation, favour speed,
# intrinsics, frame-pointer omission, fiber-safe TLS, string pooling, function-level linking, no buffer checks, whole
# program optimisation + link-time code generation, /OPT:REF /OPT:ICF. Per target, so that the CUDA target keeps its own.
function(mv_msvc_release_flags target)
  if(NOT MSVC)
    return()
  endif()
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:C,CXX>:/W3 /permissive- /Zc:inline /EHsc>
    $<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<NOT:$<CONFIG:Debug>>>:/O2 /Ob2 /Oi /Ot /Oy /GT /GF /Gy /GS- /GL>)
  target_link_options(${target} PRIVATE
    $<$<NOT:$<CONFIG:Debug>>:/LTCG /OPT:REF /OPT:ICF /INCREMENTAL:NO>)
endfunction()

if(NOT MSVC)
  if(ENABLE_INTEL_SIMD)
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-msse2>")
    add_compile_definitions(INTEL_INTRINSICS)
  elseif(ENABLE_NEON_SIMD)
    add_compile_definitions(NEON_INTRINSICS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      # GCC (GNU mode) contracts a*b+c to fmadd also across inlined calls, Clang's default
      # (-ffp-contract=on) only within one expression. The NEON float paths fuse like GCC's C;
      # with this the C reference is the same under GCC and Clang.
      add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-ffp-contract=fast>")
    endif()
  endif()
  if(WIN32)
    # MinGW: GCC does not align the stack to 32 bytes on Win64 (GCC bug 54412), so AVX spills with aligned moves crash;
    # let the assembler turn them into unaligned moves
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-Wa,-muse-unaligned-vector-move>")
    # MinGW: no runtime DLLs of GCC next to the plugins
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--enable-stdcall-fixup -static-libgcc -static-libstdc++ -static")
  elseif(APPLE)
    # macOS uses Clang's linker, doesn't like --no-undefined
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-undefined,error")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # make sure there are no undefined symbols
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")
  endif()
endif()

# ELF platforms: export only the plugin entry points (cmake/plugin_exports.map). Otherwise plugin globals are interposed
# by same-named symbols of the host or other plugins (AVS_linkage of libavisynth.so: crash in AvisynthPluginInit3).
function(mv_plugin_exports target)
  if(NOT WIN32 AND NOT APPLE)
    target_link_options(${target} PRIVATE "-Wl,--version-script=${PROJECT_SOURCE_DIR}/cmake/plugin_exports.map")
    set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS "${PROJECT_SOURCE_DIR}/cmake/plugin_exports.map")
  endif()
endfunction()

# MinGW: plugin DLLs without the lib prefix, like the MSVC ones
function(mv_windows_names target)
  if(MINGW)
    set_target_properties(${target} PROPERTIES PREFIX "" IMPORT_PREFIX "")
  endif()
endfunction()

add_subdirectory("Sources")
add_subdirectory("DePan")
add_subdirectory("DePanEstimate")
mv_plugin_exports(mvtools2)
mv_plugin_exports(depan)
mv_plugin_exports(depanestimate)

# VapourSynth: mvtools_vsbridge, an AviSynth+ host for VapourSynth that runs the plugins unchanged (VAPOURSYNTH_PLAN.md).
# The plugins only gain a VapourSynthPluginInit2 export that opens it at run time; no VapourSynth installation is needed to
# build (VapourSynth4.h is vendored in Sources/include).
option(MVTOOLS_VAPOURSYNTH "Build mvtools_vsbridge (VapourSynth support)" ON)
if(MVTOOLS_VAPOURSYNTH)
  add_subdirectory("vsbridge")
endif()

# GPU part: mvtools_cuda (needs the CUDA toolkit) and the cudafilters plugin. mvtools2 itself never links against it:
# it opens mvtools_cuda at run time and uses the CPU when the library, the driver or a device is missing. cudafilters is
# plain C++ and is built whenever MVTOOLS_GPU is on, also without a CUDA compiler (it then uses its CPU fallbacks).
# mvtools_cuda is built with CUDA 12 only (12.8 or newer for the RTX 50xx code, 12.9 recommended): CUDA 13 has no
# Pascal / Volta targets. On Windows, CUDA 12 needs the MSVC 14.4x host toolset (v143, VS 2022 or the "MSVC v143"
# component of the VS 2026 installer); build_windows.ps1 selects it. Any other toolkit or host compiler: a warning, and
# the build goes on without mvtools_cuda.
option(MVTOOLS_GPU "Build cudafilters and (MVTOOLS_CUDA) mvtools_cuda" ON)
option(MVTOOLS_CUDA "Build mvtools_cuda (needs CUDA 12.8 .. 12.x: CMAKE_CUDA_COMPILER)" ON)
if(MVTOOLS_GPU)
  set(_mv_cuda_ok OFF)
  if(MVTOOLS_CUDA AND MINGW)
    message(STATUS "MVTOOLS_GPU: mvtools_cuda needs nvcc with MSVC; MinGW builds only cudafilters (CPU fallbacks)")
  elseif(MVTOOLS_CUDA AND MSVC AND MSVC_VERSION GREATER_EQUAL 1950)
    message(WARNING "MVTOOLS_GPU: CUDA 12 needs the MSVC 14.4x host toolset (v143), this is MSVC "
      "${CMAKE_CXX_COMPILER_VERSION}; building cudafilters without mvtools_cuda. build_windows.ps1 selects 14.4x when it is "
      "installed (Visual Studio installer: \"MSVC v143 - VS 2022 C++ x64/x86 build tools\")")
  elseif(MVTOOLS_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
      execute_process(COMMAND "${CMAKE_CUDA_COMPILER}" --version OUTPUT_VARIABLE _mv_nvcc_out ERROR_QUIET)
      string(REGEX MATCH "release ([0-9]+)\\.([0-9]+)" _mv_nvcc_rel "${_mv_nvcc_out}")
      set(_mv_nvcc_ver "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
      if(_mv_nvcc_ver VERSION_GREATER_EQUAL 12.8 AND _mv_nvcc_ver VERSION_LESS 13)
        set(_mv_cuda_ok ON)
      else()
        message(WARNING "MVTOOLS_GPU: ${CMAKE_CUDA_COMPILER} is CUDA ${_mv_nvcc_ver}; mvtools_cuda is built with CUDA "
          "12.8 .. 12.x only (12.9 recommended). Building cudafilters without mvtools_cuda; point CMAKE_CUDA_COMPILER at a "
          "CUDA 12 nvcc (and delete CMakeCache.txt)")
      endif()
    else()
      message(WARNING "MVTOOLS_GPU: no CUDA compiler found (set CMAKE_CUDA_COMPILER to a CUDA 12 nvcc); building cudafilters "
        "without mvtools_cuda")
    endif()
  endif()
  if(_mv_cuda_ok)
    enable_language(CUDA)
    add_subdirectory("Sources/cuda")
  endif()
  add_subdirectory("cudafilters")
  mv_plugin_exports(cudafilters)
endif()

# Everything the plugins load from their own folder, in one place: <build>/bin (copied after each build).
set(MVTOOLS_DIST_DIR "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Folder that receives all plugin libraries after the build")
set(_mv_dist_targets mvtools2 depan depanestimate mvtools_vsbridge cudafilters mvtools_cuda)
set(_mv_dist_cmds "")
set(_mv_dist_deps "")
foreach(t ${_mv_dist_targets})
  if(TARGET ${t})
    list(APPEND _mv_dist_cmds COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:${t}>" "${MVTOOLS_DIST_DIR}/")
    list(APPEND _mv_dist_deps ${t})
  endif()
endforeach()
add_custom_target(dist ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory "${MVTOOLS_DIST_DIR}"
  ${_mv_dist_cmds}
  DEPENDS ${_mv_dist_deps}
  COMMENT "Collecting the plugin libraries in ${MVTOOLS_DIST_DIR}"
  VERBATIM)

# uninstall target
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
