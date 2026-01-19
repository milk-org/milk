cmake_minimum_required(VERSION 3.10)
project(milkprocessinfo)

find_package(PkgConfig REQUIRED)
pkg_check_modules(NCURSES REQUIRED ncurses)

set(SOURCES
    processinfo_compute_status.c
    processinfo_exec_end.c
    processinfo_exec_start.c
    processinfo_globals.c
    processinfo_loopstep.c
    processinfo_procdirname.c
    processinfo_setup.c
    processinfo_shm_close.c
    processinfo_shm_create.c
    processinfo_shm_link.c
    processinfo_shm_list_create.c
    processinfo_SIGexit.c
    processinfo_signals.c
    processinfo_update_output_stream.c
    processinfo_WriteMessage.c
    processtools_trigger.c
    # procCTRL tools are now part of the library
    procCTRL/procCTRL_GetCPUloads.c
    procCTRL/procCTRL_GetNumberCPUs.c
    procCTRL/procCTRL_PIDcollectSystemInfo.c
    procCTRL/procCTRL_processinfo_scan.c
    procCTRL/procCTRL_TUI.c
)

add_library(milkprocessinfo SHARED ${SOURCES})

# Include directories
target_include_directories(milkprocessinfo PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/procCTRL>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/ImageStreamIO>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(milkprocessinfo ImageStreamIO rt ${NCURSES_LIBRARIES} pthread m)

install(TARGETS milkprocessinfo EXPORT milkTargets DESTINATION lib)

install(FILES 
    processinfo.h
    processtools.h
    processtools_trigger.h
    processinfo_exec_end.h
    processinfo_exec_start.h
    processinfo_loopstep.h
    processinfo_procdirname.h
    processinfo_setup.h
    processinfo_shm_close.h
    processinfo_shm_create.h
    processinfo_shm_link.h
    processinfo_shm_list_create.h
    processinfo_SIGexit.h
    processinfo_signals.h
    processinfo_update_output_stream.h
    processinfo_WriteMessage.h
    DESTINATION include)

# ==========================================
# milk-procCTRL executable
# ==========================================

add_executable(milk-procCTRL procCTRL/milk-procCTRL.c)
target_link_libraries(milk-procCTRL PUBLIC 
    milkprocessinfo 
    ImageStreamIO
)

install(TARGETS milk-procCTRL DESTINATION bin)
