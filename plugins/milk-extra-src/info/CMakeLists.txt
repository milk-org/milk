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
    milkTUI
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
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
	$<INSTALL_INTERFACE:include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
	$<INSTALL_INTERFACE:include/${SRCNAME}>
)

target_link_libraries(${LIBNAME} PRIVATE ${LINKLIBS})

install(TARGETS ${LIBNAME} EXPORT milkTargets DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})

install(PROGRAMS ${SCRIPTS} DESTINATION bin)

# Compute-only variant: no CLI code, no CLIcore dependency
# imagemon.c excluded — depends on TUItools.h (ncurses)
set(COMPUTE_SOURCEFILES ${SOURCEFILES})
list(REMOVE_ITEM COMPUTE_SOURCEFILES imagemon.c)
set(LIBNAME_COMPUTE "${LIBNAME}_compute")
add_library(${LIBNAME_COMPUTE} SHARED ${COMPUTE_SOURCEFILES})
target_compile_definitions(${LIBNAME_COMPUTE}
    PRIVATE MILK_NO_CLI)
if(USE_CFITSIO AND CFITSIO_FOUND)
  target_compile_definitions(${LIBNAME_COMPUTE}
      PUBLIC USE_CFITSIO=1)
  target_include_directories(${LIBNAME_COMPUTE}
      PUBLIC ${CFITSIO_INCLUDE_DIRS})
  target_link_directories(${LIBNAME_COMPUTE}
      PUBLIC ${CFITSIO_LIBRARY_DIRS})
endif()
target_include_directories(${LIBNAME_COMPUTE} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/coremods>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/${SRCNAME}>)
target_link_libraries(${LIBNAME_COMPUTE} PUBLIC
    m milkfps)
install(TARGETS ${LIBNAME_COMPUTE} EXPORT milkTargets
    DESTINATION lib)

# Standalone runner for stream_monproc (needs CLI)
if(USE_CLI)
add_executable(stream-monproc-runner stream-monproc-runner.c)
target_link_libraries(stream-monproc-runner ${LIBNAME} CLIcore ImageStreamIO)
install(TARGETS stream-monproc-runner DESTINATION bin)
endif()

# Standalone display for stream_monproc (needs CLI+ncurses)
if(USE_CLI)
add_executable(stream-monproc-disp stream-monproc-disp.c)
target_link_libraries(stream-monproc-disp ${LIBNAME} CLIcore ImageStreamIO ncurses)
install(TARGETS stream-monproc-disp DESTINATION bin)
endif()

# imagemon is a TUI app — legitimately needs milkTUI + ncurses
add_milk_standalone(info-imagemon imagemon.c)
target_link_libraries(milk-fpsexec-info-imagemon
    PUBLIC ${LIBNAME} milkTUI ncurses)

add_milk_standalone(info-strmonproc stream_monproc.c)
