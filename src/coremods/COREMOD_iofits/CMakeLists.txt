# library name
set(LIBNAME "milkCOREMODiofits")

# list source files (.c) other than modulename.c
set(SOURCEFILES
    breakcube.c
    check_fitsio_status.c
    data_type_code.c
    file_exists.c
    images2cube.c
    is_fits_file.c
    loadfits.c
    loadmemstream.c
    read_keyword.c
    savefits.c)

# list include files (.h) that should be installed on system
set(INCLUDEFILES
    COREMOD_iofits_common.h
    breakcube.h
    check_fitsio_status.h
    data_type_code.h
    file_exists.h
    images2cube.h
    is_fits_file.h
    loadfits.h
    loadmemstream.h
    read_keyword.h
    savefits.h)

# list scripts that should be installed on system
set(SCRIPTFILES)

# TESTING

include(CTest)

# test module is loaded
set(TESTNAME "CLIfunc-COREMOD_iofits-module")
add_test(NAME "${TESTNAME}" COMMAND milk-exec "-T" "m?")
set_property(TEST "${TESTNAME}" PROPERTY LABELS "CLImodule")
set_property(TEST "${TESTNAME}" PROPERTY TIMEOUT 1)
set_property(TEST "${TESTNAME}" PROPERTY PASS_REGULAR_EXPRESSION
                                         "COREMOD_iofits")

# test that commands are registered

list(APPEND commandlist "loadfits" "saveFITS" "breakcube" "imgs2cube")

foreach(CLIcmdname IN LISTS commandlist)

  set(TESTCLIFUNC "${CLIcmdname}")
  set(TESTNAME "CLIfunc-COREMOD_iofits-${TESTCLIFUNC}")
  add_test(NAME "${TESTNAME}" COMMAND milk-exec "-T"
                                      "cmd? iofits.${CLIcmdname}")
  set_property(TEST "${TESTNAME}" PROPERTY LABELS "CLIfunc")
  set_property(TEST "${TESTNAME}" PROPERTY TIMEOUT 1)
  set_property(TEST "${TESTNAME}" PROPERTY PASS_REGULAR_EXPRESSION "src:")

endforeach()

# DEFAULT SETTINGS Do not change unless needed
# =====================================================================

# SRCNAME is current directory (last subdir)
set(NAME0 "${CMAKE_CURRENT_SOURCE_DIR}")
string(REPLACE "/" " " NAME1 ${NAME0})
string(REGEX MATCH "[a-zA-Z0-9_]+$" SRCNAME "${NAME0}")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

project(lib_${LIBNAME}_project)

add_library(${LIBNAME} SHARED ${SRCNAME}.c ${SOURCEFILES})

find_package(PkgConfig REQUIRED)
if(USE_CFITSIO)
  pkg_check_modules(CFITSIO cfitsio)
  if(${CFITSIO_FOUND})
    message("Found cfitsio")
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
target_link_libraries(${LIBNAME} PUBLIC m ${CFITSIO_LIBRARIES} milkfps)

message(" CFITSIO_LIBRARIES : ${CFITSIO_LIBRARIES}")

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${SRCNAME}.h ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTFILES} DESTINATION bin)

# Compute-only library for standalone builds (no CLI code)
set(LIBNAME_COMPUTE "${LIBNAME}_compute")
add_library(${LIBNAME_COMPUTE} SHARED ${SRCNAME}.c ${SOURCEFILES})
target_compile_definitions(${LIBNAME_COMPUTE} PRIVATE MILK_NO_CLI)
if(${CFITSIO_FOUND})
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
target_link_libraries(${LIBNAME_COMPUTE} PUBLIC m ${CFITSIO_LIBRARIES} milkfps)
install(
  TARGETS ${LIBNAME_COMPUTE}
  EXPORT milkTargets
  DESTINATION lib)

# Static archive for standalone LTO builds
if(USE_STATIC_LTO)
  set(LIBNAME_COMPUTE_STATIC "${LIBNAME_COMPUTE}_static")
  add_library(${LIBNAME_COMPUTE_STATIC} STATIC ${SRCNAME}.c ${SOURCEFILES})
  set_target_properties(${LIBNAME_COMPUTE_STATIC} PROPERTIES OUTPUT_NAME
                                                             ${LIBNAME_COMPUTE})
  target_compile_definitions(${LIBNAME_COMPUTE_STATIC} PRIVATE MILK_NO_CLI)
  if(${CFITSIO_FOUND})
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
  target_link_libraries(${LIBNAME_COMPUTE_STATIC}
                        PUBLIC m ${CFITSIO_LIBRARIES} milkfps_static milkcommon)
endif()

add_milk_standalone(iofits-saveFITS savefits.c)
add_milk_standalone(iofits-loadfits loadfits.c)
add_milk_standalone(iofits-imgs2cube images2cube.c)
