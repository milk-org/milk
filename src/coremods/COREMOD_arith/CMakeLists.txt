# library name
set(LIBNAME "milkCOREMODarith")
set(SRCNAME "COREMOD_arith")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

set(SOURCEFILES
    ${SRCNAME}.c
    image_crop.c
    image_crop2D.c
    image_multicrop2D.c
    image_cropmask.c
    image_merge3D.c
    image_norm.c
    image_pixremap.c
    image_pixunmap.c
    image_set_1Dpixrange.c
    image_set_2Dpix.c
    image_set_3Daxes.c
    image_set_col.c
    image_set_row.c
    image_setzero.c
    image_slicenormalize.c
    image_stats.c
    image_total.c
    image_unfold.c
    image_vecmult.c
    image_dxdy.c
    imfunctions.c
    im_complex.c
    im_math_unary.c
    im_math_basic.c
    im_math_logic.c
    mathfuncs.c
    image_arith__im__im.c
    image_arith__im_im__im.c
    image_arith__Cim_Cim__Cim.c
    image_arith__im_f__im.c
    image_arith__im_f_f__im.c
    execute_arith.c
    execute_arith_engines.c)

set(INCLUDEFILES
    ${SRCNAME}.h
    image_crop.h
    image_crop2D.h
    image_multicrop2D.h
    image_cropmask.h
    image_merge3D.h
    image_norm.h
    image_pixremap.h
    image_pixunmap.h
    image_set_1Dpixrange.h
    image_set_2Dpix.h
    image_set_3Daxes.h
    image_set_col.h
    image_set_row.h
    image_setzero.h
    image_slicenormalize.h
    image_stats.h
    image_total.h
    image_unfold.h
    image_vecmult.h
    image_dxdy.h
    imfunctions.h
    mathfuncs.h
    image_arith__im__im.h
    image_arith__im_im__im.h
    image_arith__Cim_Cim__Cim.h
    image_arith__im_f__im.h
    image_arith__im_f_f__im.h
    execute_arith.h)

set(SCRIPTS scripts/milk-fpslistadd-WFSCROPMASK
            scripts/milk-fpslistadd-SCICROPMASK scripts/milk-stream-crop)

# TESTING

include(CTest)

# test module is loaded
set(TESTNAME "CLIfunc-COREMOD_arith-module")
add_test(NAME "${TESTNAME}" COMMAND milk-exec "-T" "m?")
set_property(TEST "${TESTNAME}" PROPERTY LABELS "CLImodule")
set_property(TEST "${TESTNAME}" PROPERTY TIMEOUT 1)
set_property(TEST "${TESTNAME}" PROPERTY PASS_REGULAR_EXPRESSION
                                         "COREMOD_arith")

# test that commands are registered

list(
  APPEND
  commandlist
  "extractim"
  "extract3Dim"
  "setpix"
  "setpix1Drange"
  "setrow"
  "setcol"
  "imzero"
  "imtrunc"
  "cropmask")

foreach(CLIcmdname IN LISTS commandlist)

  set(TESTCLIFUNC "${CLIcmdname}")
  set(TESTNAME "CLIfunc-COREMOD_arith-${TESTCLIFUNC}")
  add_test(NAME "${TESTNAME}" COMMAND milk-exec "-T" "cmd? arith.${CLIcmdname}")
  set_property(TEST "${TESTNAME}" PROPERTY LABELS "CLIfunc")
  set_property(TEST "${TESTNAME}" PROPERTY TIMEOUT 1)
  set_property(TEST "${TESTNAME}" PROPERTY PASS_REGULAR_EXPRESSION "src:")

endforeach()

# DEFAULT SETTINGS Do not change unless needed
# =====================================================================

project(lib_${LIBNAME}_project)

# Library can be compiled from multiple source files Convention: the main souce
# file is named <libname>.c
#
add_library(${LIBNAME} SHARED ${SOURCEFILES})

find_package(PkgConfig REQUIRED)
if(USE_CFITSIO)
  pkg_check_modules(CFITSIO cfitsio)
  if(${CFITSIO_FOUND})
    link_directories(${CFITSIO_LIBRARY_DIRS})
    target_compile_definitions(${LIBNAME} PUBLIC USE_CFITSIO=1)
    target_include_directories(
      ${LIBNAME}
      PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
             $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
             $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
             $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
             $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
             $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
             ${CFITSIO_INCLUDE_DIRS})
    target_link_directories(${LIBNAME} PUBLIC ${CFITSIO_LIBRARY_DIRS})
  endif()
endif()

target_include_directories(
  ${LIBNAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
         $<INSTALL_INTERFACE:include>)
target_link_libraries(
  ${LIBNAME} PUBLIC m ${CFITSIO_LIBRARIES} milkfps
                    $<$<BOOL:${USE_CFITSIO}>:milkCOREMODiofits>)

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})

install(PROGRAMS ${SCRIPTS} DESTINATION bin)

# Compute-only library for standalone builds (no CLI code)
set(LIBNAME_COMPUTE "${LIBNAME}_compute")
add_library(${LIBNAME_COMPUTE} SHARED ${SOURCEFILES})
target_compile_definitions(${LIBNAME_COMPUTE} PRIVATE MILK_NO_CLI)
if(USE_CFITSIO AND CFITSIO_FOUND)
  target_compile_definitions(${LIBNAME_COMPUTE} PUBLIC USE_CFITSIO=1)
  target_include_directories(
    ${LIBNAME_COMPUTE}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           ${CFITSIO_INCLUDE_DIRS})
  target_link_directories(${LIBNAME_COMPUTE} PUBLIC ${CFITSIO_LIBRARY_DIRS})
endif()
target_include_directories(
  ${LIBNAME_COMPUTE}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
         $<INSTALL_INTERFACE:include>)
target_link_libraries(
  ${LIBNAME_COMPUTE} PUBLIC m ${CFITSIO_LIBRARIES} milkfps
                            $<$<BOOL:${USE_CFITSIO}>:milkCOREMODiofits_compute>)
install(
  TARGETS ${LIBNAME_COMPUTE}
  EXPORT milkTargets
  DESTINATION lib)

# Static archive for standalone LTO builds
if(USE_STATIC_LTO)
  set(LIBNAME_COMPUTE_STATIC "${LIBNAME_COMPUTE}_static")
  add_library(${LIBNAME_COMPUTE_STATIC} STATIC ${SOURCEFILES})
  set_target_properties(${LIBNAME_COMPUTE_STATIC} PROPERTIES OUTPUT_NAME
                                                             ${LIBNAME_COMPUTE})
  target_compile_definitions(${LIBNAME_COMPUTE_STATIC} PRIVATE MILK_NO_CLI)
  if(USE_CFITSIO AND CFITSIO_FOUND)
    target_compile_definitions(${LIBNAME_COMPUTE_STATIC} PUBLIC USE_CFITSIO=1)
    target_include_directories(${LIBNAME_COMPUTE_STATIC}
                               PUBLIC ${CFITSIO_INCLUDE_DIRS})
    target_link_directories(${LIBNAME_COMPUTE_STATIC} PUBLIC
                            ${CFITSIO_LIBRARY_DIRS})
  endif()
  target_include_directories(
    ${LIBNAME_COMPUTE_STATIC}
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/streamCTRL>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps/fpsCTRL>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libprocessinfo>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libmilkdata>
           $<INSTALL_INTERFACE:include>)
  if(USE_CFITSIO AND CFITSIO_LIBRARIES)
    target_link_libraries(
      ${LIBNAME_COMPUTE_STATIC}
      PUBLIC m ${CFITSIO_LIBRARIES} milkfps_static milkcommon
             milkCOREMODiofits_compute_static)
  else()
    target_link_libraries(${LIBNAME_COMPUTE_STATIC} PUBLIC m milkfps_static
                                                           milkcommon)
  endif()
endif()

add_milk_standalone(arith-crop2D image_crop2D.c)
add_milk_standalone(arith-multicrop2D image_multicrop2D.c)
add_milk_standalone(arith-immerge image_merge3D.c)
add_milk_standalone(arith-setrow image_set_row.c)
add_milk_standalone(arith-setcol image_set_col.c)
add_milk_standalone(arith-setpix image_set_2Dpix.c)
add_milk_standalone(arith-imsetzero image_setzero.c)
add_milk_standalone(arith-setpix1Drange image_set_1Dpixrange.c)
add_milk_standalone(arith-set3Daxes image_set_3Daxes.c)
add_milk_standalone(arith-normslice image_norm.c)

add_milk_standalone(arith-cropmask image_cropmask.c)
add_milk_standalone(arith-unfold image_unfold.c)
add_milk_standalone(arith-pixremap image_pixremap.c)
add_milk_standalone(arith-slicenormalize image_slicenormalize.c)
add_milk_standalone(arith-pixunmap image_pixunmap.c)
add_milk_standalone(arith-vecmult image_vecmult.c)
add_milk_standalone(arith-imtrunc image_arith__im_f_f__im.c)
