# library name
set(LIBNAME "milkCOREMODtools")
set(SRCNAME "COREMOD_tools")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

set(SOURCEFILES
	${SRCNAME}.c
	fileutils.c
	imdisplay3d.c
	linregress.c
	logfunc.c
	mvprocCPUset.c
	statusstat.c
	stringutils.c
)

set(INCLUDEFILES
	${SRCNAME}.h
	fileutils.h
	imdisplay3d.h
	linregress.h
	logfunc.h
	mvprocCPUset.h
	statusstat.h
	stringutils.h
)

# TESTING

include(CTest)

# test module is loaded
set(TESTNAME "CLIfunc-COREMOD_tools-module")
add_test (NAME "${TESTNAME}" COMMAND milk-exec "-T" "m?")
set_property (TEST "${TESTNAME}" PROPERTY LABELS "CLImodule")
set_property (TEST "${TESTNAME}" PROPERTY TIMEOUT 1)
set_property (TEST "${TESTNAME}" PROPERTY PASS_REGULAR_EXPRESSION "COREMOD_tools")

# test that commands are registered

list(APPEND commandlist "rtprio" "tsetpmove" "tsetpmoveext" "csetpmove" "csetandprioext" "writef2file" "dispim3d" "ctsmstats")

foreach(CLIcmdname IN LISTS commandlist)

set(TESTCLIFUNC "${CLIcmdname}")
set(TESTNAME "CLIfunc-COREMOD_tools-${TESTCLIFUNC}")
add_test (NAME "${TESTNAME}" COMMAND milk-exec "-T" "cmd? tools.${CLIcmdname}")
set_property (TEST "${TESTNAME}" PROPERTY LABELS "CLIfunc")
set_property (TEST "${TESTNAME}" PROPERTY TIMEOUT 1)
set_property (TEST "${TESTNAME}" PROPERTY PASS_REGULAR_EXPRESSION "src:")

endforeach()

# DEFAULT SETTINGS
# Do not change unless needed
# =====================================================================

project(lib_${LIBNAME}_project)

# Library can be compiled from multiple source files
# Convention: the main souce file is named <libname>.c
#
add_library(${LIBNAME} SHARED ${SOURCEFILES})

find_package(PkgConfig REQUIRED)
pkg_check_modules(CFITSIO cfitsio)
if(${CFITSIO_FOUND})
  link_directories(${CFITSIO_LIBRARY_DIRS})
  target_compile_definitions(${LIBNAME} PUBLIC USE_CFITSIO=1)
  target_include_directories(${LIBNAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore> ${CFITSIO_INCLUDE_DIRS})
  target_link_directories(${LIBNAME} PUBLIC ${CFITSIO_LIBRARY_DIRS})
endif()

target_include_directories(${LIBNAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore> $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
											$<INSTALL_INTERFACE:include>)
target_link_libraries(${LIBNAME} PUBLIC m milkfps)

install(TARGETS ${LIBNAME} EXPORT milkTargets DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})

# Compute-only library for standalone builds (no CLI code)
set(LIBNAME_COMPUTE "${LIBNAME}_compute")
add_library(${LIBNAME_COMPUTE} SHARED ${SOURCEFILES})
target_compile_definitions(${LIBNAME_COMPUTE} PRIVATE MILK_NO_CLI)
target_include_directories(${LIBNAME_COMPUTE} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(${LIBNAME_COMPUTE} PUBLIC m milkfps)
install(TARGETS ${LIBNAME_COMPUTE} EXPORT milkTargets
    DESTINATION lib)

# Static archive for standalone LTO builds
if(USE_STATIC_LTO)
  set(LIBNAME_COMPUTE_STATIC "${LIBNAME_COMPUTE}_static")
  add_library(${LIBNAME_COMPUTE_STATIC} STATIC
      ${SOURCEFILES})
  set_target_properties(${LIBNAME_COMPUTE_STATIC}
      PROPERTIES OUTPUT_NAME ${LIBNAME_COMPUTE})
  target_compile_definitions(${LIBNAME_COMPUTE_STATIC}
      PRIVATE MILK_NO_CLI)
  target_include_directories(${LIBNAME_COMPUTE_STATIC}
      PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
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
endif()
