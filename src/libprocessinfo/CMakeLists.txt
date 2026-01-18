cmake_minimum_required(VERSION 3.10)
project(processinfo)

set(SOURCES
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
)

add_library(processinfo SHARED ${SOURCES})

# Include directories
target_include_directories(processinfo PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/ImageStreamIO>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(processinfo ImageStreamIO rt)

install(TARGETS processinfo EXPORT milkTargets DESTINATION lib)

install(FILES 
    processinfo.h
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
