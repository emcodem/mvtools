# mvtools2: mvtools2.dll (MSVC, MinGW) / libmvtools2.so.
#
# MSVC (x64) follows upstream's Sources/mvtools.vcxproj, Release|x64: the same C++ sources (plus VapourSynthEntry.cpp),
# /arch:AVX2 for the *_avx2.cpp files only, the NASM sources with the per-file defines and include paths of the project,
# the version resource. Other compilers (GCC, Clang, MinGW, llvm-mingw) use the intrinsic code paths instead of the
# external assembler (def.h): x86 with an SSE2 base (the SSE4.1 kernels carry target attributes) and per-file ISA flags,
# AArch64 with the *_neon.cpp files. Files.cmake (upstream's) drops the files of the other architecture.

include("Files.cmake")

add_library(mvtools2 SHARED ${MvTools2_Sources})
set_target_properties(mvtools2 PROPERTIES OUTPUT_NAME "mvtools2")
mv_windows_names(mvtools2)

target_include_directories(mvtools2 PRIVATE
  "$<$<COMPILE_LANGUAGE:C,CXX>:${CMAKE_CURRENT_SOURCE_DIR}>"
  "$<$<COMPILE_LANGUAGE:C,CXX>:${CMAKE_CURRENT_SOURCE_DIR}/include>")

# Automatically group source files according to directory structure (IDE generators)
foreach(FILE ${MvTools2_Sources})
  get_filename_component(PARENT_DIR "${FILE}" PATH)
  string(REGEX REPLACE "(\\./)" "" GROUP "${PARENT_DIR}")
  string(REPLACE "/" "\\" GROUP "${GROUP}")
  if("${FILE}" MATCHES ".*\\.cpp")
    set(GROUP "Source Files\\${GROUP}")
  elseif("${FILE}" MATCHES ".*\\.h")
    set(GROUP "Header Files\\${GROUP}")
  endif()
  source_group("${GROUP}" FILES "${FILE}")
endforeach()

file(GLOB SRCS_AVX2 RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_avx2.cpp")

if(MSVC)
  # --- mvtools.vcxproj, Release|x64 -----------------------------------------------------------------------------------
  target_compile_definitions(mvtools2 PRIVATE
    $<$<COMPILE_LANGUAGE:C,CXX>:_SILENCE_CXX17_ALLOCATOR_VOID_DEPRECATION_WARNING;_SILENCE_CXX17_OLD_ALLOCATOR_MEMBERS_DEPRECATION_WARNING;FORCE_INLINE;WIN32;_WINDOWS;_USRDLL;MVTOOLS_EXPORTS;_CRT_SECURE_NO_WARNINGS;_MBCS>)
  mv_msvc_release_flags(mvtools2)
  set_source_files_properties(${SRCS_AVX2} PROPERTIES COMPILE_OPTIONS "/arch:AVX2")

  # NASM (x264 / mvtools asm). nasm.exe: on the PATH, in NASMPATH, or -DCMAKE_ASM_NASM_COMPILER=...
  if(NOT CMAKE_ASM_NASM_COMPILER)
    find_program(CMAKE_ASM_NASM_COMPILER nasm HINTS ENV NASMPATH "C:/dev/tools/nasm" PATHS "$ENV{VCINSTALLDIR}")
  endif()
  enable_language(ASM_NASM)
  if(NOT CMAKE_ASM_NASM_COMPILER_LOADED)
    message(FATAL_ERROR "mvtools2 needs NASM with MSVC (set CMAKE_ASM_NASM_COMPILER or NASMPATH)")
  endif()
  set(_asm "${CMAKE_CURRENT_SOURCE_DIR}/asm")
  set(_inc_both "-I${_asm}/" "-I${_asm}/include/")
  # file | defines | include dirs (the project's per-file NASM settings for Release|x64)
  set(MV_NASM_FILES
    "Bilinear_x64.asm|ARCH_X86_64=1,NO_PREFIX|both"
    "CopyCode-a.asm|ARCH_X86_64=1,NO_FUNCTION_PREFIX|asm"
    "fdct_mmx_x64.asm|PREFIX,ARCH_X86_64=1,NO_PREFIX|both"
    "Overlap-a.asm|ARCH_X86_64=1,NO_FUNCTION_PREFIX|both"
    "pixel-a.asm|private_prefix=x264,ARCH_X86_64=1,HIGH_BIT_DEPTH=0,BIT_DEPTH=8|both"
    "sad-a.asm|private_prefix=x264,ARCH_X86_64=1,BIT_DEPTH=8,HIGH_BIT_DEPTH=0|both"
    "const-a.asm|private_prefix=x264,ARCH_X86_64=1,BIT_DEPTH=8,HIGH_BIT_DEPTH=0|both"
    "cpu-a.asm|private_prefix=x264,ARCH_X86_64=1|include"
    "Variance-a.asm|ARCH_X86_64=1,BIT_DEPTH=8,HIGH_BIT_DEPTH=0,NO_FUNCTION_PREFIX|both"
    "const16-a.asm|PREFIX,private_prefix=x264_10,ARCH_X86_64=1,BIT_DEPTH=10,HIGH_BIT_DEPTH=1|both")
  # not built for x64 by the project: Bilinear.asm, fdct_mmx.asm (32 bit), sad16-a.asm, include/x86inc.asm
  foreach(entry ${MV_NASM_FILES})
    string(REPLACE "|" ";" parts "${entry}")
    list(GET parts 0 f)
    list(GET parts 1 defs)
    list(GET parts 2 incs)
    string(REPLACE "," ";" defs "${defs}")
    if(incs STREQUAL "both")
      set(opts ${_inc_both})
    elseif(incs STREQUAL "asm")
      set(opts "-I${_asm}/")
    else()
      set(opts "-I${_asm}/include/")
    endif()
    list(APPEND opts "-w-macro-params-legacy")
    set_source_files_properties("asm/${f}" PROPERTIES
      LANGUAGE ASM_NASM
      COMPILE_DEFINITIONS "${defs}"
      COMPILE_OPTIONS "${opts}"
      OBJECT_DEPENDS "${_asm}/include/x86inc.asm;${_asm}/include/x86util.asm")
    target_sources(mvtools2 PRIVATE "asm/${f}")
  endforeach()

  target_link_libraries(mvtools2 PRIVATE winmm)
else()
  # GCC / Clang (Linux, MinGW, llvm-mingw): -msse2 + INTEL_INTRINSICS or NEON_INTRINSICS come from the top level
  if(ENABLE_INTEL_SIMD)
    set_source_files_properties(${SRCS_AVX2} PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  elseif(ENABLE_NEON_SIMD)
    file(GLOB SRCS_NEON RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_neon.cpp")
    set_source_files_properties(${SRCS_NEON} PROPERTIES COMPILE_OPTIONS "-march=armv8-a")
  endif()
  if(WIN32)
    target_link_libraries(mvtools2 PRIVATE winmm)
  else()
    target_link_libraries(mvtools2 PRIVATE ${CMAKE_DL_LIBS})
  endif()
endif()

if(WIN32)
  target_sources(mvtools2 PRIVATE mvtools.rc)
  set_source_files_properties(mvtools.rc PROPERTIES COMPILE_DEFINITIONS "NDEBUG")
endif()

INSTALL(TARGETS mvtools2
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/avisynth"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
