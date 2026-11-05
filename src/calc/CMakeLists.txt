set(LIBNAME "milkcalc")

set(SOURCEFILES
    mc_ext_registry.c
    mc_ext_builtin.c
    mc_func.c
    mc_lex.c
    mc_parse.c
    mc_resolve.c
    mc_scalar.c
    mc_local.c
    mc_eval.c
    milk_argval.c
    units/milk_unit.c)

set(INCLUDEFILES
    mcalc.h
    mcalc_host.h
    milk_calc_ext.h
    milk_argval.h
    units/milk_unit.h)

add_library(${LIBNAME} SHARED ${SOURCEFILES})

target_compile_definitions(${LIBNAME} PRIVATE MILK_NO_CLI)

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
