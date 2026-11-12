# CudaConvolution: AviSynth+ plugin, progressive -> interlaced with integer downscale (README.md). Windows x64 + MSVC.
#
# Two ways to build it:
#   - as part of mvtools (root CMakeLists.txt, option MVTOOLS_CUDACONVOLUTION): the root decides about CUDA (its CUDA 12
#     + MSVC 14.4x checks); the CUDA engine is built when the root builds mvtools_cuda, else the CPU-only DLL
#   - standalone from this folder (tools/build.ps1, tools/package.ps1): CUDACONVOLUTION_CUDA = AUTO / ON / OFF
# Either way the DLL has its own CUDA engine and runtime (cudart_static); mvtools_cuda.dll is only opened at run time
# for GPU token frames (src/avs/token_interop.cpp).
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(CUDACONVOLUTION_STANDALONE ON)
  cmake_minimum_required(VERSION 3.27)
  # Static MSVC runtime (/MT): no VC++ redistributable needed. Must be set before project().
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  cmake_policy(SET CMP0091 NEW)
else()
  set(CUDACONVOLUTION_STANDALONE OFF)
endif()

project(CudaConvolution VERSION 0.10.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(CUDACONVOLUTION_TESTS "Build the internal test executables (downloads Catch2)" ${CUDACONVOLUTION_STANDALONE})

# --- CUDA (optional; without it only the CPU engine is built) -------------------------------------------------------
set(CUDACONVOLUTION_HAVE_CUDA OFF)
if(CUDACONVOLUTION_STANDALONE)
  set(CUDACONVOLUTION_CUDA "AUTO" CACHE STRING "Build the CUDA engine: AUTO, ON, OFF")
  set_property(CACHE CUDACONVOLUTION_CUDA PROPERTY STRINGS AUTO ON OFF)
  if(NOT CUDACONVOLUTION_CUDA STREQUAL "OFF")
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
      enable_language(CUDA)
      set(CUDACONVOLUTION_HAVE_CUDA ON)
    elseif(CUDACONVOLUTION_CUDA STREQUAL "ON")
      message(FATAL_ERROR "CUDACONVOLUTION_CUDA=ON but no CUDA compiler found (CUDA Toolkit >= 12.8 required)")
    else()
      message(WARNING "No CUDA Toolkit found: building the CPU engine only")
    endif()
  endif()
elseif(_mv_cuda_ok)  # root CMakeLists.txt: CUDA 12.8+ with a matching MSVC toolset, CUDA language enabled
  set(CUDACONVOLUTION_HAVE_CUDA ON)
endif()

if(CUDACONVOLUTION_HAVE_CUDA)
  find_package(CUDAToolkit 12.8 REQUIRED)
  # SASS for Pascal (6.0 P100, 6.1 GTX 10xx / Quadro P), Volta, Turing, Ampere (A4000), Ada, Hopper, Blackwell (RTX 50xx)
  # + PTX for newer GPUs. CUDA 13 has no Pascal / Volta targets: there the list starts at 7.5.
  # MVTOOLS_CUDA_ARCH_SET nightly (root CMakeLists.txt): one per generation, a GPU runs the nearest lower version of its
  # generation (6.1 -> 6.0, 7.5 -> 7.0, 8.6 / 8.9 -> 8.0); CUDA 13 keeps 7.5 (no 7.0 target there)
  if(MVTOOLS_CUDA_ARCH_SET STREQUAL "nightly")
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
      set(_cc_default_arch "75-real;80-real;90-real;120-real;120-virtual")
    else()
      set(_cc_default_arch "60-real;70-real;80-real;90-real;120-real;120-virtual")
    endif()
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    set(_cc_default_arch "75-real;80-real;86-real;89-real;90-real;120-real;120-virtual")
  else()
    set(_cc_default_arch "60-real;61-real;70-real;75-real;80-real;86-real;89-real;90-real;120-real;120-virtual")
  endif()
  # Development: 61-virtual / 70-virtual builds only the Pascal / Volta PTX, which the driver JIT-compiles for the local
  # GPU (runs the code of those targets on a newer GPU); 120-real only the local RTX 50xx
  set(CUDACONVOLUTION_CUDA_ARCH "" CACHE STRING "CUDA architectures of CudaConvolution (empty: MVTOOLS_CUDA_ARCH_SET)")
  if(CUDACONVOLUTION_CUDA_ARCH)
    set(_cc_arch "${CUDACONVOLUTION_CUDA_ARCH}")
  else()
    set(_cc_arch "${_cc_default_arch}")
  endif()
  # oldest architecture of the build and a readable list, for the run-time device check (src/cuda/device.cu)
  set(_cc_min 9999)
  set(_cc_text "")
  foreach(a IN LISTS _cc_arch)
    if(a MATCHES "^([0-9]+)(-real|-virtual)?$")
      set(_n ${CMAKE_MATCH_1})
      if(_n LESS _cc_min)
        set(_cc_min ${_n})
      endif()
      if(CMAKE_MATCH_2 STREQUAL "-virtual")
        list(APPEND _cc_text "PTX ${_n}")
      else()
        list(APPEND _cc_text "sm_${_n}")
      endif()
    else()
      message(FATAL_ERROR "CUDACONVOLUTION_CUDA_ARCH: '${a}' is not of the form 61, 61-real or 61-virtual")
    endif()
  endforeach()
  list(JOIN _cc_text ", " _cc_text)
  message(STATUS "CudaConvolution: CUDA engine for ${_cc_text}")
else()
  message(STATUS "CudaConvolution: CPU engine only")
endif()

# /fp:precise keeps float semantics IEEE (no reassociation); FMA only where written explicitly (std::fma / intrinsics).
# Quoted wherever used: the list separators belong inside the generator expression.
set(_cc_cxx_flags "$<$<COMPILE_LANGUAGE:CXX>:/W4;/permissive-;/fp:precise;/utf-8;/Zc:__cplusplus>")

# --- embedded presets ------------------------------------------------------------------------------------------------
set(_cc_gen ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(GLOB CUDACONVOLUTION_PRESET_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/presets/*.json)
set(CUDACONVOLUTION_PRESETS_CPP ${_cc_gen}/embedded_presets.cpp)
add_custom_command(
  OUTPUT ${CUDACONVOLUTION_PRESETS_CPP}
  COMMAND ${CMAKE_COMMAND} -DOUT=${CUDACONVOLUTION_PRESETS_CPP} "-DFILES=${CUDACONVOLUTION_PRESET_FILES}"
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_presets.cmake
  DEPENDS ${CUDACONVOLUTION_PRESET_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_presets.cmake
  COMMENT "Embedding presets")
set(CUDACONVOLUTION_SELFTEST_CPP ${_cc_gen}/embedded_selftest.cpp)
add_custom_command(
  OUTPUT ${CUDACONVOLUTION_SELFTEST_CPP}
  COMMAND ${CMAKE_COMMAND} -DOUT=${CUDACONVOLUTION_SELFTEST_CPP} -DFILES=${CMAKE_CURRENT_SOURCE_DIR}/selftest/vectors.json
          -DFUNC=embedded_selftest -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_presets.cmake
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/selftest/vectors.json ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_presets.cmake
  COMMENT "Embedding self-test vectors")

# --- core: pure C++, no CUDA ----------------------------------------------------------------------------------------
add_library(cudaconvolution_core STATIC
  src/core/presets.cpp
  src/core/preset.cpp
  src/core/engine_ref.cpp
  src/core/engine_avx2.cpp
  src/core/md5.cpp
  src/core/format.cpp
  src/core/selftest.cpp
  ${CUDACONVOLUTION_PRESETS_CPP}
  ${CUDACONVOLUTION_SELFTEST_CPP})
target_include_directories(cudaconvolution_core PUBLIC src third_party)
target_compile_options(cudaconvolution_core PRIVATE "${_cc_cxx_flags}")
# only this file uses AVX2 (runtime dispatch in engine_ref.cpp); /fp:precise: no FMA contraction by the compiler
set_source_files_properties(src/core/engine_avx2.cpp PROPERTIES COMPILE_OPTIONS "$<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>")
target_compile_definitions(cudaconvolution_core PUBLIC CUDACONVOLUTION_VERSION="${PROJECT_VERSION}")

# --- cuda engine ----------------------------------------------------------------------------------------------------
if(CUDACONVOLUTION_HAVE_CUDA)
  add_library(cudaconvolution_cuda STATIC src/cuda/device.cu src/cuda/engine_cuda.cu)
  target_link_libraries(cudaconvolution_cuda PUBLIC cudaconvolution_core CUDA::cudart_static version)
  set_target_properties(cudaconvolution_cuda PROPERTIES
    CUDA_ARCHITECTURES "${_cc_arch}"
    CUDA_RUNTIME_LIBRARY Static)
  # --fmad=false: nvcc must not contract a*b+c on its own; FMA only where written as __fmaf_rn.
  # CUDA 12 warns about every target below 7.5 (deprecated, still supported).
  target_compile_options(cudaconvolution_cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-Wno-deprecated-gpu-targets>")
  target_compile_definitions(cudaconvolution_cuda PRIVATE CUDACONVOLUTION_MIN_CC=${_cc_min}
                             "CUDACONVOLUTION_BUILD_ARCHS=\"${_cc_text}\"")
  target_compile_definitions(cudaconvolution_core PUBLIC CUDACONVOLUTION_HAVE_CUDA=1)
endif()

# --- AviSynth+ plugin -----------------------------------------------------------------------------------------------
add_library(CudaConvolution SHARED src/avs/plugin.cpp src/avs/token_interop.cpp)
target_compile_options(CudaConvolution PRIVATE "${_cc_cxx_flags}")
target_link_libraries(CudaConvolution PRIVATE cudaconvolution_core)
if(CUDACONVOLUTION_HAVE_CUDA)
  target_link_libraries(CudaConvolution PRIVATE cudaconvolution_cuda)
  set_target_properties(CudaConvolution PROPERTIES CUDA_RUNTIME_LIBRARY Static)
endif()

# --- tests ----------------------------------------------------------------------------------------------------------
if(CUDACONVOLUTION_TESTS)
  include(FetchContent)
  FetchContent_Declare(Catch2
    URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.7.1.tar.gz
    URL_HASH SHA256=c991b247a1a0d7bb9c39aa35faf0fe9e19764213f28ffba3109388e62ee0269c
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
  FetchContent_MakeAvailable(Catch2)
  add_executable(cudaconvolution_tests tests/test_presets.cpp tests/test_engine.cpp tests/test_formats.cpp tests/test_selftest.cpp tests/test_probe.cpp)
  target_compile_options(cudaconvolution_tests PRIVATE "${_cc_cxx_flags}")
  target_link_libraries(cudaconvolution_tests PRIVATE cudaconvolution_core Catch2::Catch2WithMain)
  if(CUDACONVOLUTION_HAVE_CUDA)
    target_sources(cudaconvolution_tests PRIVATE tests/test_cuda.cpp tests/bench_cuda.cu)
    set_target_properties(cudaconvolution_tests PROPERTIES CUDA_RUNTIME_LIBRARY Static
                          CUDA_ARCHITECTURES "${_cc_arch}")
    target_compile_options(cudaconvolution_tests PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>")
    target_link_libraries(cudaconvolution_tests PRIVATE cudaconvolution_cuda)
  endif()
  target_compile_definitions(cudaconvolution_tests PRIVATE CUDACONVOLUTION_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}" _CRT_SECURE_NO_WARNINGS)
  enable_testing()
  add_test(NAME unit COMMAND cudaconvolution_tests "~[probe]")
  # needs CUDACONVOLUTION_DATA (decoded probe, see tools/pio.py); skipped otherwise
  add_test(NAME probe COMMAND cudaconvolution_tests "[probe]")
  set_tests_properties(probe PROPERTIES SKIP_REGULAR_EXPRESSION "SKIPPED")
  # the built DLL inside AviSynth+ (format conversion, 8 bit, limits, errors); needs AviSynth+ installed
  add_executable(cudaconvolution_avs_tests tests/test_avisynth.cpp)
  target_compile_options(cudaconvolution_avs_tests PRIVATE "${_cc_cxx_flags}")
  target_include_directories(cudaconvolution_avs_tests PRIVATE third_party)
  target_link_libraries(cudaconvolution_avs_tests PRIVATE Catch2::Catch2WithMain)
  # CUDACONVOLUTION_TOKEN_TEST_DIR (run time): folder with cudafilters.dll + mvtools_cuda.dll for the [tokens] test
  target_compile_definitions(cudaconvolution_avs_tests PRIVATE CUDACONVOLUTION_DLL="$<TARGET_FILE:CudaConvolution>" _CRT_SECURE_NO_WARNINGS)
  add_dependencies(cudaconvolution_avs_tests CudaConvolution)
  add_test(NAME avisynth COMMAND cudaconvolution_avs_tests)
  add_test(NAME dll_dependencies
           COMMAND pwsh -NoProfile -File ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_deps.ps1 $<TARGET_FILE:CudaConvolution>)
endif()
