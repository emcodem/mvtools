# splatfps: motion-compensated frame-rate conversion (SplatFps / CudaSplatFps), its own plugin DLL. Plain C++ here;
# the device code goes into mvtools_cuda, which the plugin opens at run time (as cudafilters does).
#
# splat_core: the exact CPU path of the stages. It needs the float32 arithmetic in source order: no contraction of
# a * b + c into FMA, no reassociation (MSVC /fp:precise is the default and does not contract; GCC / Clang need
# -ffp-contract=off).

#
# The fast path (splat_fast, SplatFps(exact=false)) keeps the same rules; its *_avx2.cpp kernels are compiled for
# AVX2 + FMA (only those files, x86 only) and called after a run-time CPU check.

file(GLOB SPLAT_CORE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/core/*.h")
file(GLOB SPLAT_CORE_AVX2 CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/core/*_avx2.cpp")
if(MSVC OR ENABLE_INTEL_SIMD)
  set(SPLAT_AVX2 ON)
else()
  set(SPLAT_AVX2 OFF)
  list(REMOVE_ITEM SPLAT_CORE_SOURCES ${SPLAT_CORE_AVX2})
endif()
add_library(splat_core STATIC ${SPLAT_CORE_SOURCES})
set_target_properties(splat_core PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
if(MSVC)
  target_compile_options(splat_core PRIVATE /fp:precise)
  mv_msvc_release_flags(splat_core)
else()
  target_compile_options(splat_core PRIVATE -ffp-contract=off)
  find_package(Threads REQUIRED)
  target_link_libraries(splat_core PUBLIC Threads::Threads)
endif()
if(SPLAT_AVX2)
  target_compile_definitions(splat_core PRIVATE SPLAT_HAVE_AVX2=1)
  if(MSVC)
    set_source_files_properties(${SPLAT_CORE_AVX2} PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  else()
    set_source_files_properties(${SPLAT_CORE_AVX2} PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  endif()
endif()

# splatcore_test: C ABI over splat_core for the stage parity tests (test/check_fast.py, check_motion.py, check_cuda.py).
option(SPLATFPS_TESTS "Build splatcore_test (stage tests, not shipped)" ON)
if(SPLATFPS_TESTS)
  add_library(splatcore_test SHARED "${CMAKE_CURRENT_SOURCE_DIR}/test/splatcore_capi.cpp")
  target_link_libraries(splatcore_test PRIVATE splat_core)
  set_target_properties(splatcore_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
  if(MSVC)
    target_compile_options(splatcore_test PRIVATE /fp:precise)
    mv_msvc_release_flags(splatcore_test)
  else()
    target_compile_options(splatcore_test PRIVATE -ffp-contract=off)
  endif()
  mv_windows_names(splatcore_test)
endif()

# the plugin: SplatFps (CPU); the GPU filter joins it with the device pipeline
add_library(splatfps SHARED "${CMAKE_CURRENT_SOURCE_DIR}/SplatFps.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/SplatAnalyse.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/VapourSynthEntry.cpp")
target_link_libraries(splatfps PRIVATE splat_core)
target_include_directories(splatfps PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../Sources/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../Sources/cuda")
set_target_properties(splatfps PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
if(MSVC)
  target_compile_definitions(splatfps PRIVATE _CRT_SECURE_NO_WARNINGS _WINDOWS _USRDLL UNICODE _UNICODE)
  target_compile_options(splatfps PRIVATE /fp:precise)
  mv_msvc_release_flags(splatfps)
else()
  target_compile_options(splatfps PRIVATE -ffp-contract=off)
  target_link_libraries(splatfps PRIVATE ${CMAKE_DL_LIBS})
endif()
mv_windows_names(splatfps)

include(GNUInstallDirs)
install(TARGETS splatfps
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/avisynth"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
