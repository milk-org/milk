cmake_minimum_required(VERSION 3.10)
project(milkprocessinfo)

option(USE_NCURSES "Enable ncurses and TUI support" ON)

if(USE_NCURSES)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(NCURSES REQUIRED ncurses)
endif()

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
        quicksort.c
        timeutils.c
    )
    
    # Core library (no TUI/ncurses dependency)
    add_library(milkprocessinfo SHARED ${SOURCES})
    
    if(USE_NCURSES)
        set(SOURCES_TUI
            procCTRL/procCTRL_GetCPUloads.c
            procCTRL/procCTRL_GetNumberCPUs.c
            procCTRL/procCTRL_PIDcollectSystemInfo.c
            procCTRL/procCTRL_processinfo_scan.c
            procCTRL/procCTRL_TUI.c
        )
        
        add_library(milkprocessinfoTUI SHARED ${SOURCES_TUI})
        target_compile_definitions(milkprocessinfoTUI PUBLIC USE_NCURSES)
        target_include_directories(milkprocessinfoTUI PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/procCTRL>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        )
        target_link_libraries(milkprocessinfoTUI PUBLIC milkprocessinfo milkTUI ${NCURSES_LIBRARIES})
        install(TARGETS milkprocessinfoTUI EXPORT milkTargets DESTINATION lib)
    endif()
    
    
    # Include directories
    target_include_directories(milkprocessinfo PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/procCTRL>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/ImageStreamIO>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/libmilkTUI>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/CommandLineInterface>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/CommandLineInterface/streamCTRL>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/CommandLineInterface/CLIcore>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/CommandLineInterface/procCTRL>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/libfps>
        $<INSTALL_INTERFACE:include>
    )
    
    # Link dependencies
    target_link_libraries(milkprocessinfo ImageStreamIO rt pthread m)
        
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
    quicksort.h
    timeutils.h
    DESTINATION include)

if(USE_NCURSES)
# ==========================================
# milk-procCTRL executable
# ==========================================

add_executable(milk-procCTRL procCTRL/milk-procCTRL.c)
target_link_libraries(milk-procCTRL PUBLIC 
    milkprocessinfoTUI
    milkprocessinfo 
    ImageStreamIO
    milkTUI
    ${NCURSES_LIBRARIES}
    pthread
    m
)

add_executable(milk-procCTRL-scan procCTRL/milk-procCTRL-scan.c)
target_link_libraries(milk-procCTRL-scan PUBLIC
    milkprocessinfoTUI
    milkprocessinfo
    ImageStreamIO
    pthread
    m
)

install(TARGETS milk-procCTRL DESTINATION bin)
install(TARGETS milk-procCTRL-scan DESTINATION bin)
endif()
