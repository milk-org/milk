set(LIBNAME "milkprocessinfo")
set(SRCNAME "libprocessinfo")

set(SOURCEFILES
    processinfo_compute_status.c
    processinfo_exec_end.c
    processinfo_exec_start.c
    processinfo_globals.c
    processinfo_labels.c
    processinfo_list.c
    processinfo_loopstep.c
    processinfo_procdirname.c
    processinfo_setup.c
    processinfo_shm_close.c
    processinfo_shm_create.c
    processinfo_shm_link.c
    processinfo_shm_list_create.c
    processinfo_timing.c
    processinfo_SIGexit.c
    processinfo_signals.c
    processinfo_update_output_stream.c
    processinfo_WriteMessage.c
    processtools_trigger.c
    quicksort.c
    quicksort_multiarray.c
    timeutils.c)

# Core library (no TUI/ncurses dependency)
add_library(${LIBNAME} SHARED ${SOURCEFILES})
# Single translation unit
set_target_properties(${LIBNAME} PROPERTIES UNITY_BUILD ON
                                            UNITY_BUILD_BATCH_SIZE 0)

# Include directories
target_include_directories(
  ${LIBNAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<INSTALL_INTERFACE:include>)

# Link dependencies
target_link_libraries(${LIBNAME} PUBLIC ImageStreamIO milkcommon rt
                                             pthread m)

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)

# Static archive for standalone LTO builds
if(USE_STATIC_LTO)
  add_library(${LIBNAME}_static STATIC ${SOURCEFILES})
  set_target_properties(
    ${LIBNAME}_static PROPERTIES OUTPUT_NAME ${LIBNAME} UNITY_BUILD ON
                                 UNITY_BUILD_BATCH_SIZE 0)
  target_include_directories(
    ${LIBNAME}_static
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libmilkcommon>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libmilkdata>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli>
           $<INSTALL_INTERFACE:include>)
  target_link_libraries(${LIBNAME}_static PUBLIC ImageStreamIO_static
                                                      milkcommon rt pthread m)
endif()

if(BUILD_TESTING)
  add_executable(test-processinfo-helpers tests/test_processinfo_helpers.c)
  target_link_libraries(test-processinfo-helpers PRIVATE milkprocessinfo)
  add_test(NAME processinfo_unit_helpers COMMAND test-processinfo-helpers)
  add_executable(test-processinfo-signals tests/test_processinfo_signals.c)
  target_link_libraries(test-processinfo-signals PRIVATE milkprocessinfo)
  add_test(NAME processinfo_unit_signals COMMAND test-processinfo-signals)
endif()

# Add milk-procinfo-help executable
add_executable(milk-procinfo-help milk-procinfo-help.c)
install(TARGETS milk-procinfo-help DESTINATION bin)

add_executable(milk-procinfo-rm milk-procinfo-rm.c)
target_include_directories(
  milk-procinfo-rm
  PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)
target_link_libraries(milk-procinfo-rm PUBLIC ${LIBNAME} ImageStreamIO
                                              pthread m)
install(TARGETS milk-procinfo-rm DESTINATION bin)

add_executable(milk-procinfo-list milk-procinfo-list.c)
target_include_directories(
  milk-procinfo-list
  PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)
target_link_libraries(milk-procinfo-list PUBLIC ${LIBNAME} ImageStreamIO
                                                pthread m)
install(TARGETS milk-procinfo-list DESTINATION bin)

set(PROCESSINFO_HEADERS
    processinfo.h
    processinfo_labels.h
    processinfo_list.h
    processinfo_timing.h
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
    timeutils.h)
install(FILES ${PROCESSINFO_HEADERS} DESTINATION include)
install(FILES ${PROCESSINFO_HEADERS} DESTINATION include/libprocessinfo)
