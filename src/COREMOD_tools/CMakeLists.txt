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
	quicksort.c
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
	quicksort.h
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
add_test (NAME "${TESTNAME}" COMMAND milk-exec "-T" "cmd? ${CLIcmdname}")
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
  target_include_directories(${LIBNAME} PUBLIC ${CFITSIO_INCLUDE_DIRS})
  target_link_directories(${LIBNAME} PUBLIC ${CFITSIO_LIBRARY_DIRS})
endif()

target_include_directories(${LIBNAME} PUBLIC ${PROJECT_SOURCE_DIR}/..)
target_link_libraries(${LIBNAME} PUBLIC ${CFITSIO_LIBRARIES})

install(TARGETS ${LIBNAME} DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
