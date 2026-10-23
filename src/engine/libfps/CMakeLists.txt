set(LIBNAME "milkfps")
set(SRCNAME "libfps")

set(SOURCEFILES
    cmd/fps_scan.c
    core/fps_connect.c
    core/fps_connectExternalFPS.c
    core/fps_disconnect.c
    core/fps_isvalid.c
    core/fps_shmdirname.c
    core/fps_globals.c
    core/fps_struct_create.c
    core/fps_ID.c
    io/fps_save2disk.c
    io/fps_WriteParameterToDisk.c
    io/fps_loadstream.c
    io/fps_outlog.c
    io/fps_load.c
    io/fps_loadmemstream_lite.c
    io/fps_streamname_parse.c
    io/fps_local_store.c
    params/fps_add_entry.c
    params/fps_SetParamCLIindex.c
    params/fps_print_info.c
    params/fps_checkparameter.c
    params/fps_FPSremove.c
    params/fps_GetFileName.c
    params/fps_GetParamIndex.c
    params/fps_GetTypeString.c
    params/fps_paramvalue.c
    params/fps_printlist.c
    params/fps_PrintParameterInfo.c
    params/fps_printparameter_valuestring.c
    params/fps_userinputsetparamvalue.c
    imgid_slice.c)

add_library(${LIBNAME} SHARED ${SOURCEFILES})
# Single translation unit
set_target_properties(${LIBNAME} PROPERTIES UNITY_BUILD ON
                                            UNITY_BUILD_BATCH_SIZE 0)

target_include_directories(
  ${LIBNAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
         $<INSTALL_INTERFACE:include>)

target_link_libraries(${LIBNAME} PUBLIC ImageStreamIO m rt milkprocessinfo
                                        milkcommon)

if(USE_STATIC_LTO)
  add_library(${LIBNAME}_static STATIC ${SOURCEFILES})
  set_target_properties(
    ${LIBNAME}_static PROPERTIES OUTPUT_NAME ${LIBNAME} UNITY_BUILD ON
                                 UNITY_BUILD_BATCH_SIZE 0)
  target_include_directories(
    ${LIBNAME}_static
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libprocessinfo>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libmilkdata>
           $<INSTALL_INTERFACE:include>)
  target_link_libraries(
    ${LIBNAME}_static PUBLIC ImageStreamIO_static m rt milkprocessinfo_static
                             milkcommon)
endif()

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)

file(GLOB LIBFPS_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(
  FILES ${LIBFPS_HEADERS}
  DESTINATION include/libfps)

# ==========================================
# Standalone FPS tools (no CLIcore needed)
# ==========================================

# --- Help / listing ---

add_executable(milk-fpsexec-help tools/milk-fpsexec-help.c)
install(TARGETS milk-fpsexec-help DESTINATION bin)

# milk-fpsexec-list: C replacement for bash script
add_executable(milk-fpsexec-list tools/milk-fpsexec-list.c)
target_include_directories(
  milk-fpsexec-list
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)
install(TARGETS milk-fpsexec-list DESTINATION bin)

install(PROGRAMS scripts/milk-demopipeline DESTINATION bin)
install(PROGRAMS scripts/milk-latency-audit DESTINATION bin)

add_executable(milk-fps-help tools/milk-fps-help.c)
install(TARGETS milk-fps-help DESTINATION bin)

# --- FPS query tools ---

add_executable(milk-fps-list tools/milk-fps-list.c)
target_link_libraries(milk-fps-list PUBLIC ${LIBNAME} pthread m)
install(TARGETS milk-fps-list DESTINATION bin)

add_executable(milk-fps-search tools/milk-fps-search.c)
target_link_libraries(milk-fps-search PUBLIC ${LIBNAME} pthread m)
install(TARGETS milk-fps-search DESTINATION bin)

add_executable(
  milk-fps-info
  tools/milk-fps-info.c
  ${CMAKE_SOURCE_DIR}/src/cli/overview/data/overview_data.c
  ${CMAKE_SOURCE_DIR}/src/cli/overview/data/overview_data_cache.c
  ${CMAKE_SOURCE_DIR}/src/cli/overview/data/overview_data_cache_fps.c
  ${CMAKE_SOURCE_DIR}/src/cli/overview/data/overview_data_graph.c
  ${CMAKE_SOURCE_DIR}/src/cli/overview/data/overview_data_loops.c
  ${CMAKE_SOURCE_DIR}/src/cli/overview/data/overview_data_loops_names.c
  ${CMAKE_SOURCE_DIR}/src/cli/overview/data/overview_data_loops_reg.c
  ${CMAKE_SOURCE_DIR}/src/cli/overview/data/overview_data_scan.c
  ${CMAKE_SOURCE_DIR}/src/cli/overview/data/overview_data_scan_streams.c
  ${CMAKE_SOURCE_DIR}/src/cli/overview/data/overview_data_scan_fps.c
  ${CMAKE_SOURCE_DIR}/src/cli/overview/data/overview_data_scan_procs.c
  ${CMAKE_SOURCE_DIR}/src/cli/overview/data/overview_data_scan_tmux.c
  ${CMAKE_SOURCE_DIR}/src/cli/overview/data/overview_data_sort.c
  ${CMAKE_SOURCE_DIR}/src/cli/overview/data/overview_data_sort_streams.c
  ${CMAKE_SOURCE_DIR}/src/cli/overview/data/overview_data_sort_procs.c
  ${CMAKE_SOURCE_DIR}/src/cli/overview/data/overview_data_sort_fps.c
  ${CMAKE_SOURCE_DIR}/src/cli/overview/data/overview_data_sys.c
  ${CMAKE_SOURCE_DIR}/src/cli/overview/data/overview_data_sys_proc.c
  ${CMAKE_SOURCE_DIR}/src/cli/overview/data/overview_data_sys_perf.c)
target_compile_definitions(milk-fps-info PRIVATE FPS_INFO_CONNECTIONS)
target_include_directories(
  milk-fps-info
  PRIVATE ${CMAKE_SOURCE_DIR}/src/engine/ImageStreamIO
          ${CMAKE_SOURCE_DIR}/src/engine/libprocessinfo
          ${CMAKE_SOURCE_DIR}/src/cli/overview
          ${CMAKE_SOURCE_DIR}/src/cli/overview/ansi
          ${CMAKE_SOURCE_DIR}/src/cli/overview/data)
target_link_libraries(milk-fps-info PUBLIC ${LIBNAME} ImageStreamIO
                                           milkprocessinfo pthread m rt)
install(TARGETS milk-fps-info DESTINATION bin)

add_executable(milk-fps-rm tools/milk-fps-rm.c)
target_link_libraries(milk-fps-rm PUBLIC ${LIBNAME} pthread m)
install(TARGETS milk-fps-rm DESTINATION bin)

# --- milk-fps-set ---

add_executable(milk-fps-set tools/milk-fps-set.c)
target_link_libraries(milk-fps-set PUBLIC ${LIBNAME} pthread m)
install(TARGETS milk-fps-set DESTINATION bin)

# --- milk-fps-track ---

add_executable(milk-fps-track tools/milk-fps-track.c)
target_link_libraries(milk-fps-track PUBLIC ${LIBNAME} pthread m)
install(TARGETS milk-fps-track DESTINATION bin)
