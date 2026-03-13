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
    fps_processcmdline.c
    fps_process_fpsCMDarray.c
    fps_read_fpsCMD_fifo.c
    fps_userinputsetparamvalue.c
    fps_loadmemstream_lite.c
    fps_streamname_parse.c
    fps_local_store.c
    fps_cli_init.c
    fps_cli_function_registry.c
)

# Core library (no TUI/ncurses dependency)
add_library(${LIBNAME} SHARED ${SOURCEFILES})

target_include_directories(${LIBNAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/fpsCTRL>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/coremods>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(${LIBNAME} PUBLIC ImageStreamIO m rt milkprocessinfo)

# ==========================================
# milkfpsStandalone — standalone FPS framework
# No CLI/readline dependency.
# ==========================================

set(SOURCEFILES_STANDALONE
    fps_cli_sync.c
    fps_cli_init.c
    fps_lifecycle.c
)

add_library(milkfpsStandalone SHARED ${SOURCEFILES_STANDALONE})
target_compile_definitions(milkfpsStandalone PRIVATE FPS_STANDALONE)
target_include_directories(milkfpsStandalone PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(milkfpsStandalone PUBLIC ${LIBNAME} milkdata)
install(TARGETS milkfpsStandalone EXPORT milkTargets DESTINATION lib)

# ==========================================
# milkfpsCLI — CLI-dependent FPS framework
# Links to both milkfps and CLIcore.
# ==========================================

set(SOURCEFILES_CLI
    fps_cli_sync.c
    fps_cli_function.c
    fps_cli_query.c
    fps_lifecycle.c
)

add_library(milkfpsCLI SHARED ${SOURCEFILES_CLI})
target_include_directories(milkfpsCLI PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(milkfpsCLI PUBLIC ${LIBNAME} CLIcore)
install(TARGETS milkfpsCLI EXPORT milkTargets DESTINATION lib)


if(USE_NCURSES AND USE_CLI)
    set(SOURCEFILES_TUI
        # fpsCTRL sources
        fpsCTRL/fpsCTRL_FPSdisplay.c
        fpsCTRL/fpsCTRL_TUI.c
        fpsCTRL/fpsCTRL_TUI_process_user_key.c
        fpsCTRL/level0node_summary.c
        fpsCTRL/print_nodeinfo.c
        fpsCTRL/scheduler_display.c
        fpsCTRL/fpsCTRL_globals.c
    )
    
    add_library(milkfpsTUI SHARED ${SOURCEFILES_TUI})
    target_compile_definitions(milkfpsTUI PUBLIC USE_NCURSES)
    target_include_directories(milkfpsTUI PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/fpsCTRL>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    )
    target_link_libraries(milkfpsTUI PUBLIC ${LIBNAME} milkTUI ${NCURSES_LIBRARIES})
    install(TARGETS milkfpsTUI EXPORT milkTargets DESTINATION lib)
endif()


install(TARGETS ${LIBNAME} EXPORT milkTargets DESTINATION lib)
install(FILES fps.h fps_core.h fps_types.h fps_internal.h fps_globals.h fps_cli_binding.h fps_cli_function.h fps_cli_function_registry.h ../libprocessinfo/timeutils.h DESTINATION include/libfps)

if(USE_NCURSES AND USE_CLI)
# ==========================================
# milk-fpsCTRL executable (requires TUI)
# ==========================================

add_executable(milk-fpsCTRL fpsCTRL/milk-fpsCTRL.c)
target_include_directories(milk-fpsCTRL PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/fpsCTRL
)
target_link_libraries(milk-fpsCTRL PUBLIC 
    milkfpsTUI
    milkfps
    milkprocessinfo 
    ImageStreamIO
    CLIcore
    ${NCURSES_LIBRARIES}
    pthread
    m
)

install(TARGETS milk-fpsCTRL DESTINATION bin)
endif()

# ==========================================
# Standalone FPS tools (no CLIcore needed)
# ==========================================

# --- Help / listing ---

add_executable(milk-fpsexec-help milk-fpsexec-help.c)
install(TARGETS milk-fpsexec-help DESTINATION bin)

install(PROGRAMS scripts/milk-fpsexec-list DESTINATION bin)

add_executable(milk-fps-help milk-fps-help.c)
install(TARGETS milk-fps-help DESTINATION bin)

# --- FPS query tools ---

add_executable(milk-fps-list milk-fps-list.c)
target_link_libraries(milk-fps-list PUBLIC 
    milkfps
    milkfpsStandalone
    milkdata
    milkprocessinfo 
    ImageStreamIO
    pthread
    m
)
install(TARGETS milk-fps-list DESTINATION bin)

add_executable(milk-fps-search milk-fps-search.c)
target_link_libraries(milk-fps-search PUBLIC 
    milkfps
    milkfpsStandalone
    milkdata
    milkprocessinfo 
    ImageStreamIO
    pthread
    m
)
install(TARGETS milk-fps-search DESTINATION bin)

add_executable(milk-fps-info milk-fps-info.c)
target_link_libraries(milk-fps-info PUBLIC 
    milkfps
    milkfpsStandalone
    milkdata
    milkprocessinfo 
    ImageStreamIO
    pthread
    m
)
install(TARGETS milk-fps-info DESTINATION bin)

add_executable(milk-fps-rm milk-fps-rm.c)
target_link_libraries(milk-fps-rm PUBLIC 
    milkfps
    milkfpsStandalone
    milkdata
    milkprocessinfo 
    ImageStreamIO
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
        milkfps
        milkfpsStandalone
        milkdata
        milkprocessinfo 
        ImageStreamIO
        pthread
        m
    )
    install(TARGETS ${cmd} DESTINATION bin)
endforeach()

# --- milk-fps-set ---

add_executable(milk-fps-set milk-fps-set.c)
target_link_libraries(milk-fps-set PUBLIC 
    milkfps
    milkfpsStandalone
    milkdata
    milkprocessinfo 
    ImageStreamIO
    pthread
    m
)
install(TARGETS milk-fps-set DESTINATION bin)

# --- milk-fps-track ---

add_executable(milk-fps-track milk-fps-track.c)
target_link_libraries(milk-fps-track PUBLIC 
    milkfps
    milkfpsStandalone
    milkdata
    milkprocessinfo 
    ImageStreamIO
    pthread
    m
)
install(TARGETS milk-fps-track DESTINATION bin)

