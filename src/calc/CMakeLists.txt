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
    mc_scalar.c
    mc_reduce.c
    mc_prestage.c
    mc_local.c
    mc_cc.c
    mc_views.c
    mc_out.c
    mc_plan.c
    mc_emit.c
    mc_exec.c
    mc_prog.c
    mc_hist.c
    mc_eval.c
    linalg/mc_linalg_blas.c
    linalg/mc_linalg_lapack.c
    linalg/mc_fft_stack.c
    milk_argval.c
    units/milk_unit.c)

set(INCLUDEFILES
    mcalc.h
    mcalc_host.h
    milk_calc_ext.h
    milk_argval.h
    units/milk_unit.h)

add_library(${LIBNAME} SHARED ${SOURCEFILES})
milk_apply_extensions(${LIBNAME})

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
  endif()
endif()

target_compile_definitions(
  ${LIBNAME}
  PRIVATE MILK_NO_CLI
          MILK_DEFAULT_CC="${CMAKE_C_COMPILER}")

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
