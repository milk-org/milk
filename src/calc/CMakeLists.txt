set(LIBNAME "milkcalc")

set(SOURCEFILES
    mc_ext_registry.c
    mc_ext_loader.c
    mc_ext_builtin.c
    mc_ext_exec.c
    mc_func.c
    mc_lex.c
    mc_parse.c
    mc_resolve.c
    mc_scalar_eval.c
    mc_reduce.c
    mc_prestage.c
    mc_scalar_ops.c
    mc_cc.c
    mc_views.c
    mc_out.c
    mc_plan.c
    mc_emit.c
    mc_exec.c
    mc_prog.c
    mc_history.c
    mc_explain.c
    mc_eval.c
    linalg/mc_linalg_blas.c
    linalg/mc_linalg_lapack.c
    linalg/mc_fft_stack.c)

set(INCLUDEFILES
    mcalc.h
    mcalc_host.h
    milk_calc_ext.h)

add_library(${LIBNAME} SHARED ${SOURCEFILES})
milk_apply_extensions(${LIBNAME})

# Scalar evaluation keeps the IEEE NaN semantics of the JIT kernels (mc_cc.c): under -Ofast,
# -ffinite-math-only folds `nan == nan` to 1 and `nan >= 1` to !(nan < 1). libmilkbash compiles
# mc_scalar_ops.c with the same flag. The reduction kernels test NaN bit by bit (arith_isnan_d) and keep
# -Ofast's finite math: it makes icentroid 8% and istd 3% faster.
if(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
  target_compile_options(${LIBNAME} PRIVATE -fno-finite-math-only)
  set_source_files_properties(mc_reduce.c PROPERTIES COMPILE_OPTIONS -ffinite-math-only)
endif()

find_package(OpenMP)
if(OpenMP_C_FOUND)
  target_link_libraries(${LIBNAME} PUBLIC OpenMP::OpenMP_C)
endif()

find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(FFTW QUIET fftw3)
  pkg_check_modules(FFTWF QUIET fftw3f)
  if(FFTW_FOUND AND FFTWF_FOUND)
    target_compile_definitions(${LIBNAME} PRIVATE HAVE_FFTW3=1)
    target_include_directories(${LIBNAME} PRIVATE ${FFTW_INCLUDE_DIRS} ${FFTWF_INCLUDE_DIRS})
    target_link_directories(${LIBNAME} PRIVATE ${FFTW_LIBRARY_DIRS} ${FFTWF_LIBRARY_DIRS})
    target_link_libraries(${LIBNAME} PRIVATE ${FFTW_LIBRARIES} ${FFTWF_LIBRARIES})
    # With the threads libraries, FFT plans run on the calculator's thread team
    find_library(FFTW_THREADS_LIB fftw3_threads HINTS ${FFTW_LIBRARY_DIRS})
    find_library(FFTWF_THREADS_LIB fftw3f_threads HINTS ${FFTWF_LIBRARY_DIRS})
    if(FFTW_THREADS_LIB AND FFTWF_THREADS_LIB)
      target_compile_definitions(${LIBNAME} PRIVATE HAVE_FFTW3_THREADS=1)
      target_link_libraries(${LIBNAME} PRIVATE ${FFTW_THREADS_LIB} ${FFTWF_THREADS_LIB})
    endif()
  endif()
endif()

target_compile_definitions(
  ${LIBNAME}
  PRIVATE MILK_NO_CLI
          MILK_DEFAULT_CC="${CMAKE_C_COMPILER}"
          MILK_CC_FLAGS="${MILK_CC_FLAGS}")

target_include_directories(
  ${LIBNAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/units>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
         $<INSTALL_INTERFACE:include/milkcalc>
         $<INSTALL_INTERFACE:include>
  PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine>
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/ImageStreamIO>
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libmilkcommon>
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libmilkdata>
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libprocessinfo>
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libcomputeunit>
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/coremods>
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/coremods/COREMOD_memory>
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/coremods/COREMOD_arith>
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/prop>)

target_link_libraries(
  ${LIBNAME}
  PUBLIC milkprop
         milkCOREMODarith
         milkCOREMODmemory
         ImageStreamIO
         milkcommon
         milkdata
         milkprocessinfo
         milkfps
         milkcomputeunit
         dl
         rt
         m
         ${CMAKE_THREAD_LIBS_INIT})

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/milkcalc)
install(FILES ${INCLUDEFILES} DESTINATION include)
install(FILES milk_calc_ext.h DESTINATION include/milkscript)

add_executable(milk-fpsexec-milk-calc units/milk_unit_calc.c)
target_compile_definitions(milk-fpsexec-milk-calc PRIVATE MILK_NO_CLI)
target_include_directories(
  milk-fpsexec-milk-calc PRIVATE $<TARGET_PROPERTY:${LIBNAME},INCLUDE_DIRECTORIES>)
milk_configure_fpsexec(milk-fpsexec-milk-calc PREFIX milk)
target_link_libraries(milk-fpsexec-milk-calc PUBLIC ${LIBNAME})
