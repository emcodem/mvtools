# mvtools2 + DePan + DePanEstimate (AviSynth+ plugins), their VapourSynth bridge and the optional GPU libraries.
#
# Builds (README "Build Instructions"):
#   Windows, MSVC + Ninja (x64): build_windows.ps1, or by hand from a "x64 Native Tools" prompt:
#     cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_ASM_NASM_COMPILER=C:/path/nasm.exe
#     cmake --build build
#   Linux (GCC / Clang): build_linux.sh, or cmake -G Ninja -S . -B build && cmake --build build
#   Windows from Linux (MinGW-w64, CPU plugins + bridge + Vulkan): cmake/toolchain-mingw64.cmake
#
# Targets and outputs:
#   mvtools2        mvtools2.dll / libmvtools2.so          (MSVC: with the NASM sources of the .vcxproj)
#   depan           DePan.dll / libdepan.so
#   depanestimate   DePanEstimate.dll / libdepanestimate.so
#   mvtools_vsbridge  VapourSynth host for the plugins above (MVTOOLS_VAPOURSYNTH)
#   mvtools_cuda, cudafilters   GPU part (MVTOOLS_GPU, needs a CUDA compiler for mvtools_cuda)
#   mvtools_vulkan  Vulkan backend (MVTOOLS_VULKAN, needs glslangValidator or prebuilt SPIR-V)
# The Visual Studio solution (mvtools.sln) still works and builds the same AviSynth+ DLLs.
cmake_minimum_required(VERSION 3.18)

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

if(CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_CONFIGURATION_TYPES Debug Release RelWithDebInfo)
  set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING "Reset the configurations to what we need" FORCE)
endif()

message(STATUS "Target processor: ${CMAKE_SYSTEM_PROCESSOR}")
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" ARCHID)
if(ARCHID MATCHES "^(x86|x64|i686|amd64|x86_64)$")
  set(INTEL_SIMD "ON")
else()
  set(INTEL_SIMD "OFF")
endif()
option(ENABLE_INTEL_SIMD "Enable SIMD intrinsics for Intel processors" "${INTEL_SIMD}")

if(MSVC)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "clang-cl is not supported by this CMake build (the Visual Studio solution has LLVM configurations)")
  endif()
  if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "Only x64 is supported by the CMake build (use the Visual Studio solution for Win32)")
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
# it opens mvtools_cuda (or mvtools_vulkan) at run time and uses the CPU when the library, the driver or a device is
# missing. cudafilters is plain C++ and is built whenever MVTOOLS_GPU is on, also without a CUDA compiler (it then works
# with mvtools_vulkan or its CPU fallbacks).
option(MVTOOLS_GPU "Build cudafilters and (MVTOOLS_CUDA) mvtools_cuda" ON)
option(MVTOOLS_CUDA "Build mvtools_cuda (needs a CUDA compiler: CMAKE_CUDA_COMPILER)" ON)
if(MVTOOLS_GPU)
  if(MVTOOLS_CUDA AND NOT MINGW)
    include(CheckLanguage)
    check_language(CUDA)
  endif()
  if(MVTOOLS_CUDA AND CMAKE_CUDA_COMPILER AND NOT MINGW)
    enable_language(CUDA)
    add_subdirectory("Sources/cuda")
  elseif(MINGW)
    message(STATUS "MVTOOLS_GPU: mvtools_cuda needs nvcc with MSVC; MinGW builds only cudafilters (Vulkan / CPU)")
  elseif(MVTOOLS_CUDA)
    message(WARNING "MVTOOLS_GPU: no CUDA compiler found (set CMAKE_CUDA_COMPILER); building cudafilters without mvtools_cuda")
  endif()
  add_subdirectory("cudafilters")
  mv_plugin_exports(cudafilters)
endif()

# Vulkan backend: mvtools_vulkan, the same C API as mvtools_cuda on any Vulkan 1.2 GPU (NVIDIA incl. Pascal / GTX 10xx, AMD,
# Intel). Needs glslangValidator at build time only (MVTOOLS_GLSLANG); mvtools2 / cudafilters pick it at run time when
# mvtools_cuda has no usable device (MVTOOLS_GPU_BACKEND=cuda|vulkan|auto).
option(MVTOOLS_VULKAN "Build mvtools_vulkan (needs glslangValidator)" ON)
if(MVTOOLS_VULKAN)
  add_subdirectory("Sources/vulkan")
endif()

# Everything the plugins load from their own folder, in one place: <build>/bin (copied after each build).
set(MVTOOLS_DIST_DIR "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Folder that receives all plugin libraries after the build")
set(_mv_dist_targets mvtools2 depan depanestimate mvtools_vsbridge cudafilters mvtools_cuda mvtools_vulkan)
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
