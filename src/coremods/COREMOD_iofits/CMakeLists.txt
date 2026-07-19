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
target_link_libraries(${LIBNAME} PUBLIC m milkfps)

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${SRCNAME}.h ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTFILES} DESTINATION bin)

# Static archive for USE_STATIC_LTO builds
if(USE_STATIC_LTO)
  set(LIBNAME_STATIC "${LIBNAME}_static")
  add_library(${LIBNAME_STATIC} STATIC ${SRCNAME}.c ${SOURCEFILES})
  set_target_properties(${LIBNAME_STATIC} PROPERTIES OUTPUT_NAME ${LIBNAME})
  target_compile_definitions(${LIBNAME_STATIC} PRIVATE MILK_NO_CLI)
  target_include_directories(
    ${LIBNAME_STATIC}
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libprocessinfo>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libmilkdata>
           $<INSTALL_INTERFACE:include>)
  if(HAVE_CFITSIO)
    target_link_libraries(${LIBNAME_STATIC} PUBLIC m ${CFITSIO_LIBRARIES}
                                                   milkfps_static milkcommon)
  else()
    target_link_libraries(${LIBNAME_STATIC} PUBLIC m milkfps_static milkcommon)
  endif()
  install(
    TARGETS ${LIBNAME_STATIC}
    EXPORT milkTargets
    DESTINATION lib)
endif()

milk_apply_extensions(${LIBNAME})
if(TARGET ${LIBNAME_STATIC})
  milk_apply_extensions(${LIBNAME_STATIC})
endif()

add_milk_standalone(iofits-saveFITS savefits.c)
add_milk_standalone(iofits-loadfits loadfits.c)
add_milk_standalone(iofits-imgs2cube images2cube.c)
