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
	stream_monproc.c
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
	stream_monproc.h
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
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
	$<INSTALL_INTERFACE:include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
	$<INSTALL_INTERFACE:include/${SRCNAME}>
)

target_link_libraries(${LIBNAME} PRIVATE ${LINKLIBS})

install(TARGETS ${LIBNAME} EXPORT milkTargets DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})

install(PROGRAMS ${SCRIPTS} DESTINATION bin)

# Standalone runner for stream_monproc
add_executable(stream-monproc-runner stream-monproc-runner.c)
target_link_libraries(stream-monproc-runner ${LIBNAME} CLIcore ImageStreamIO)
install(TARGETS stream-monproc-runner DESTINATION bin)

# Standalone display for stream_monproc
add_executable(stream-monproc-disp stream-monproc-disp.c)
target_link_libraries(stream-monproc-disp ${LIBNAME} CLIcore ImageStreamIO ncurses)
install(TARGETS stream-monproc-disp DESTINATION bin)
