cmake_minimum_required(VERSION 3.10)
project(milkprocessinfo)

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
    quicksort_multiarray.c
    timeutils.c)

# Core library (no TUI/ncurses dependency)
add_library(milkprocessinfo SHARED ${SOURCES})
# Include directories
target_include_directories(
  milkprocessinfo
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/procCTRL>
         $<INSTALL_INTERFACE:include>)

# Link dependencies
target_link_libraries(milkprocessinfo PUBLIC ImageStreamIO milkcommon rt
                                             pthread m)

install(
  TARGETS milkprocessinfo
  EXPORT milkTargets
  DESTINATION lib)

# Static archive for standalone LTO builds
if(USE_STATIC_LTO)
  add_library(milkprocessinfo_static STATIC ${SOURCES})
  set_target_properties(milkprocessinfo_static PROPERTIES OUTPUT_NAME
                                                          milkprocessinfo)
  target_include_directories(
    milkprocessinfo_static
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/procCTRL>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libmilkcommon>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libmilkdata>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli>
           $<INSTALL_INTERFACE:include>)
  target_link_libraries(milkprocessinfo_static PUBLIC ImageStreamIO_static
                                                      milkcommon rt pthread m)
endif()

# Add milk-procinfo-help executable
add_executable(milk-procinfo-help milk-procinfo-help.c)
install(TARGETS milk-procinfo-help DESTINATION bin)

add_executable(milk-procinfo-rm milk-procinfo-rm.c)
target_include_directories(
  milk-procinfo-rm
  PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)
target_link_libraries(milk-procinfo-rm PUBLIC milkprocessinfo ImageStreamIO
                                              pthread m)
install(TARGETS milk-procinfo-rm DESTINATION bin)

add_executable(milk-procinfo-list procCTRL/milk-procinfo-list.c)
target_include_directories(
  milk-procinfo-list
  PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)
target_link_libraries(milk-procinfo-list PUBLIC milkprocessinfo ImageStreamIO
                                                pthread m)
install(TARGETS milk-procinfo-list DESTINATION bin)

install(
  FILES processinfo.h
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

# ==========================================
# milk-procCTRL executable
# ==========================================

add_executable(
  milk-procCTRL
  procCTRL/milk-procCTRL.c
  procCTRL/procCTRL_GetCPUloads.c
  procCTRL/procCTRL_GetNumberCPUs.c
  procCTRL/procCTRL_PIDcollectSystemInfo.c
  procCTRL/procCTRL_processinfo_scan.c
  procCTRL/procCTRL_TUI.c
  procCTRL/procCTRL_TUI_render.c
  procCTRL/procCTRL_TUI_input.c
  procCTRL/procCTRL_TUI_sort.c
  procCTRL/procCTRL_TUIcompat.c)
target_link_libraries(milk-procCTRL PUBLIC milkprocessinfo ImageStreamIO
                                           pthread m)
target_include_directories(
  milk-procCTRL
  PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)

add_executable(milk-procCTRL-scan procCTRL/milk-procCTRL-scan.c
                                  procCTRL/procCTRL_PIDcollectSystemInfo.c)
target_link_libraries(milk-procCTRL-scan PUBLIC milkprocessinfo ImageStreamIO
                                                pthread m)
target_include_directories(
  milk-procCTRL-scan
  PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)

install(TARGETS milk-procCTRL DESTINATION bin)
install(TARGETS milk-procCTRL-scan DESTINATION bin)
