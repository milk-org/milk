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
endif()

target_include_directories(${LIBNAME} PUBLIC ${PROJECT_SOURCE_DIR}/..)
target_link_libraries(${LIBNAME} PUBLIC ${CFITSIO_LIBRARIES})

install(TARGETS ${LIBNAME} DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
