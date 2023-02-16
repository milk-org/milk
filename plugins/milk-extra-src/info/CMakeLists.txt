# library name
set(LIBNAME "milkinfo")
set(SRCNAME "info")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

set(SOURCEFILES
	${SRCNAME}.c
	cubeMatchMatrix.c
	cubestats.c
	image_stats.c
	imagemon.c
	improfile.c
	kbdhit.c
	percentile.c
	print_header.c
	streamtiming_stats.c
	timediff.c
)

set(INCLUDEFILES
	${SRCNAME}.h
	cubeMatchMatrix.h
	cubestats.h
	image_stats.h
	imagemon.h
	improfile.h
	kbdhit.h
	percentile.h
	print_header.h
	streamtiming_stats.h
	timediff.h
)

set(SCRIPTS
	scripts/milk-shmimmon
)

set(LINKLIBS
	CLIcore
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

target_include_directories(${LIBNAME} PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/..
	${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${LIBNAME} PRIVATE ${LINKLIBS})

install(TARGETS ${LIBNAME} DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})

install(PROGRAMS ${SCRIPTS} DESTINATION bin)
