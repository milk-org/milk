# library name
set(LIBNAME "milkCOREMODtools")
set(SRCNAME "COREMOD_tools")


set(SOURCEFILES
	${SRCNAME}.c
	fileutils.c
	imdisplay3d.c
	logfunc.c
	mvprocCPUset.c
	quicksort.c
	statusstat.c
	stringutils.c
	timeutils.c
)

set(INCLUDEFILES
	${SRCNAME}.h
	fileutils.h
	imdisplay3d.h
	logfunc.h
	mvprocCPUset.h
	quicksort.h
	statusstat.h
	stringutils.h
	timeutils.h
)


# DEFAULT SETTINGS 
# Do not change unless needed
# =====================================================================

project(lib_${LIBNAME}_project)

include_directories ("${PROJECT_SOURCE_DIR}/src")
include_directories ("${PROJECT_SOURCE_DIR}/..")


# Library can be compiled from multiple source files
# Convention: the main souce file is named <libname>.c
# 
add_library(${LIBNAME} SHARED ${SOURCEFILES})

install(TARGETS ${LIBNAME} DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})

