set(LIBNAME "milkfps")
set(SRCNAME "libfps")

option(USE_NCURSES "Enable ncurses and TUI support" ON)

if(USE_NCURSES)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(NCURSES REQUIRED ncurses)
endif()

set(SOURCEFILES
    fps_add_entry.c
    fps_SetParamCLIindex.c
    fps_print_info.c
    fps_checkparameter.c
    fps_connect.c
    fps_connectExternalFPS.c
    fps_processinfo_entries.c
    fps_disconnect.c
    fps_execFPScmd.c
    fps_FPCONFsetup.c
    fps_FPCONFexit.c
    fps_FPCONFloopstep.c
    fps_RUNexit.c
    fps_RUNstart.c
    fps_isvalid.c
    fps_RUNstop.c
    fps_CONFstart.c
    fps_CONFstop.c
    fps_FPSremove.c
    fps_GetFileName.c
    fps_getFPSargs.c
    fps_GetParamIndex.c
    fps_GetTypeString.c
    fps_paramvalue.c
    fps_printlist.c
    fps_PrintParameterInfo.c
    fps_printparameter_valuestring.c
    fps_save2disk.c
    fps_shmdirname.c
    fps_WriteParameterToDisk.c
    fps_loadstream.c
    fps_outlog.c
    fps_globals.c
    fps_struct_create.c
    fps_ID.c
    fps_load.c
    fps_scan.c
    fps_tmux.c
    fps_processinfo.c
    fps_processcmdline_interactive.c
    fps_processcmdline_file.c
    fps_process_fpsCMDarray.c
    fps_read_fpsCMD_fifo.c
    fps_userinputsetparamvalue.c
    fps_loadmemstream_lite.c
    fps_streamname_parse.c
    fps_local_store.c
    imgid_slice.c
    fps_cli_init.c
    fps_cli_function_registry.c
)

# Core library (no TUI/ncurses dependency)
add_library(${LIBNAME} SHARED ${SOURCEFILES})

target_include_directories(${LIBNAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/fpsCTRL>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    # CLIcore path (src/cli/CLIcore) no longer needed in PUBLIC interface
    # after extracting CLICMDARGDEF to engine tier.
    $<INSTALL_INTERFACE:include>
)
# src/cli and src/coremods: only for guarded .c files
target_include_directories(${LIBNAME} PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/coremods>
)

target_link_libraries(${LIBNAME} PUBLIC ImageStreamIO m rt milkprocessinfo milkcommon)

# ==========================================
# milkfpsStandalone — standalone FPS framework
# No CLI/readline dependency.
# ==========================================

set(SOURCEFILES_STANDALONE
    fps_cli_sync.c
    fps_cli_init.c
    fps_lifecycle.c
    fps_apply_process_settings.c
)

add_library(milkfpsStandalone SHARED ${SOURCEFILES_STANDALONE})
target_compile_definitions(milkfpsStandalone PRIVATE FPS_STANDALONE)
target_include_directories(milkfpsStandalone PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(milkfpsStandalone PUBLIC ${LIBNAME} milkdata)
install(TARGETS milkfpsStandalone EXPORT milkTargets DESTINATION lib)

# Static archives for standalone LTO builds
if(USE_STATIC_LTO)
  add_library(${LIBNAME}_static STATIC ${SOURCEFILES})
  set_target_properties(${LIBNAME}_static
      PROPERTIES OUTPUT_NAME ${LIBNAME})
  target_include_directories(${LIBNAME}_static PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/fpsCTRL>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/coremods>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libprocessinfo>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libmilkdata>
      $<INSTALL_INTERFACE:include>)

  add_library(milkfpsStandalone_static STATIC
      ${SOURCEFILES_STANDALONE})
  set_target_properties(milkfpsStandalone_static
      PROPERTIES OUTPUT_NAME milkfpsStandalone)
  target_compile_definitions(milkfpsStandalone_static
      PRIVATE FPS_STANDALONE)
  target_include_directories(milkfpsStandalone_static
      PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libprocessinfo>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libmilkdata>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli>
      $<INSTALL_INTERFACE:include>)
endif()

# ==========================================
# milkfpsCLI — CLI-dependent FPS framework
# Links to both milkfps and CLIcore.
# ==========================================

set(SOURCEFILES_CLI
    fps_cli_sync.c
    fps_cli_function.c
    fps_cli_query.c
    fps_lifecycle.c
    fps_apply_process_settings.c
)

add_library(milkfpsCLI SHARED ${SOURCEFILES_CLI})
target_include_directories(milkfpsCLI PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(milkfpsCLI PUBLIC ${LIBNAME} CLIcore)
install(TARGETS milkfpsCLI EXPORT milkTargets DESTINATION lib)


# milkfpsTUI removed — milk-fpsCTRL links the same fpsCTRL sources directly
# and uses the ANSI TUIcompat path; no external consumers of milkfpsTUI exist.


install(TARGETS ${LIBNAME} EXPORT milkTargets DESTINATION lib)
install(FILES
    fps.h
    fps_core.h
    fps_types.h
    fps_internal.h
    fps_globals.h
    fps_processcmdline.h
    fps_cli_binding.h
    fps_cli_function.h
    fps_cli_function_registry.h
    fps_procinfo_macros.h
    fps_apply_process_settings.h
    imgid_slice.h
    IMGID.h
    milk_help.h
    DESTINATION include/libfps)

# ==========================================
# milk-fpsCTRL executable
# ==========================================

set(SOURCEFILES_FPSCTRL
    fpsCTRL/milk-fpsCTRL.c
    fpsCTRL/fpsCTRL_FPSdisplay.c
    fpsCTRL/fpsCTRL_TUI.c
    fpsCTRL/fpsCTRL_TUIcompat.c
    fpsCTRL/fpsCTRL_TUI_process_user_key.c
    fpsCTRL/level0node_summary.c
    fpsCTRL/print_nodeinfo.c
    fpsCTRL/scheduler_display.c
    fpsCTRL/fpsCTRL_globals.c
)

add_executable(milk-fpsCTRL ${SOURCEFILES_FPSCTRL})
target_include_directories(milk-fpsCTRL PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/fpsCTRL
)
target_link_libraries(milk-fpsCTRL PUBLIC
    milkfpsseq
    milkfps
    milkprocessinfo
    ImageStreamIO
    pthread
    m
)

install(TARGETS milk-fpsCTRL DESTINATION bin)

# ==========================================
# Standalone FPS tools (no CLIcore needed)
# ==========================================

# --- Help / listing ---

add_executable(milk-fpsexec-help milk-fpsexec-help.c)
install(TARGETS milk-fpsexec-help DESTINATION bin)

# milk-fpsexec-list: C replacement for bash script
add_executable(milk-fpsexec-list milk-fpsexec-list.c)
target_include_directories(milk-fpsexec-list PRIVATE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)
install(TARGETS milk-fpsexec-list DESTINATION bin)

install(PROGRAMS scripts/milk-demopipeline DESTINATION bin)
install(PROGRAMS scripts/milk-latency-audit DESTINATION bin)

add_executable(milk-fps-help milk-fps-help.c)
install(TARGETS milk-fps-help DESTINATION bin)

# --- FPS query tools ---

add_executable(milk-fps-list milk-fps-list.c)
target_link_libraries(milk-fps-list PUBLIC
    milkfpsStandalone
    pthread
    m
)
install(TARGETS milk-fps-list DESTINATION bin)

add_executable(milk-fps-search milk-fps-search.c)
target_link_libraries(milk-fps-search PUBLIC
    milkfpsStandalone
    pthread
    m
)
install(TARGETS milk-fps-search DESTINATION bin)

add_executable(milk-fps-info
    milk-fps-info.c
    ${CMAKE_SOURCE_DIR}/src/cli/overview/overview_data.c
    ${CMAKE_SOURCE_DIR}/src/cli/overview/overview_data_sys.c
    ${CMAKE_SOURCE_DIR}/src/cli/overview/overview_data_cache.c
    ${CMAKE_SOURCE_DIR}/src/cli/overview/overview_data_scan.c
    ${CMAKE_SOURCE_DIR}/src/cli/overview/overview_data_graph.c
    ${CMAKE_SOURCE_DIR}/src/cli/overview/overview_data_sort.c
)
target_compile_definitions(milk-fps-info PRIVATE
    FPS_INFO_CONNECTIONS)
target_include_directories(milk-fps-info PRIVATE
    ${CMAKE_SOURCE_DIR}/src/engine/ImageStreamIO
    ${CMAKE_SOURCE_DIR}/src/engine/libprocessinfo
    ${CMAKE_SOURCE_DIR}/src/cli/overview
)
target_link_libraries(milk-fps-info PUBLIC
    milkfpsStandalone
    ImageStreamIO milkprocessinfo
    pthread m rt
)
install(TARGETS milk-fps-info DESTINATION bin)

add_executable(milk-fps-rm milk-fps-rm.c)
target_link_libraries(milk-fps-rm PUBLIC
    milkfpsStandalone
    pthread
    m
)
install(TARGETS milk-fps-rm DESTINATION bin)

# --- FPS control commands ---

set(FPS_GENERIC_CMDS
    milk-fps-confstart
    milk-fps-confstop
    milk-fps-runstart
    milk-fps-runstop
    milk-fps-confstep
)

foreach(cmd ${FPS_GENERIC_CMDS})
    add_executable(${cmd} milk-fps-cmd.c)
    target_link_libraries(${cmd} PUBLIC
        milkfpsStandalone
        pthread
        m
    )
    install(TARGETS ${cmd} DESTINATION bin)
endforeach()

# --- milk-fps-set ---

add_executable(milk-fps-set milk-fps-set.c)
target_link_libraries(milk-fps-set PUBLIC
    milkfpsStandalone
    pthread
    m
)
install(TARGETS milk-fps-set DESTINATION bin)

# --- milk-fps-track ---

add_executable(milk-fps-track milk-fps-track.c)
target_link_libraries(milk-fps-track PUBLIC
    milkfpsStandalone
    pthread
    m
)
install(TARGETS milk-fps-track DESTINATION bin)
