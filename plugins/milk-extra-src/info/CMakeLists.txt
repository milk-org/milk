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
    improfile.c
    kbdhit.c
    percentile.c
    print_header.c
    stream_monproc.c
    stream_monproc_shm.c
    streamtiming_stats.c
    timediff.c)

set(INCLUDEFILES
    ${SRCNAME}.h
    cubeMatchMatrix.h
    cubestats.h
    image_stats.h
    improfile.h
    kbdhit.h
    percentile.h
    print_header.h
    stream_monproc.h
    streamtiming_stats.h
    timediff.h)

# milk-shmimmon is now a compiled C binary — no scripts to install here

set(LINKLIBS CLIcore)

# DEFAULT SETTINGS Do not change unless needed
# =====================================================================

project(lib_${LIBNAME}_project)

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/..")

# Library can be compiled from multiple source files Convention: the main souce
# file is named <libname>.c
#
add_library(${LIBNAME} SHARED ${SOURCEFILES})

target_include_directories(
  ${LIBNAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
         $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<INSTALL_INTERFACE:include/${SRCNAME}>)

target_link_libraries(${LIBNAME} PRIVATE ${LINKLIBS})

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})

install(PROGRAMS ${SCRIPTS} DESTINATION bin)

# Standalone display for stream_monproc (no ncurses, no CLIcore, no milkinfo)
if(USE_CLI)
  add_executable(stream-monproc-disp stream-monproc-disp.c stream_monproc_shm.c)
  target_link_libraries(stream-monproc-disp milkdata milkfps ImageStreamIO m)
  target_compile_definitions(stream-monproc-disp PRIVATE MILK_NO_CLI)
  target_include_directories(
    stream-monproc-disp PRIVATE ${CMAKE_SOURCE_DIR}/src/engine/libfps
                                ${CMAKE_SOURCE_DIR}/src/cli/CLIcore)
  install(TARGETS stream-monproc-disp DESTINATION bin)
endif()

# milk-fpsexec-info-imagemon retired — use milk-shmimmon instead

add_milk_standalone(info-strmonproc stream_monproc.c)
target_sources(milk-fpsexec-info-strmonproc PRIVATE stream_monproc_shm.c)

# milk-shmimmon: lightweight stream monitor Links only ImageStreamIO + libc/libm
# — no CLIcore, no FPS, no ncurses
if(USE_CLI)
  add_executable(milk-shmimmon milk-shmimmon.c)
  target_include_directories(
    milk-shmimmon
    PRIVATE ${CMAKE_SOURCE_DIR}/src/engine
            ${CMAKE_SOURCE_DIR}/src/engine/libfps ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(milk-shmimmon PRIVATE ImageStreamIO m)
  install(TARGETS milk-shmimmon DESTINATION bin)
endif()
