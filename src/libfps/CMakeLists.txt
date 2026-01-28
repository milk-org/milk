set(LIBNAME "milkfps")
set(SRCNAME "libfps")

option(USE_NCURSES "Enable ncurses and TUI support" ON)

if(USE_NCURSES)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(NCURSES REQUIRED ncurses)
endif()

set(SOURCEFILES
    fps_add_entry.c
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
)

# Core library (no TUI/ncurses dependency)
add_library(${LIBNAME} SHARED ${SOURCEFILES})

target_include_directories(${LIBNAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/fpsCTRL>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(${LIBNAME} PUBLIC ImageStreamIO milkcore m rt milkprocessinfo)


if(USE_NCURSES)
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
install(FILES fps.h fps_internal.h fps_globals.h ../libmilkcore/timeutils.h DESTINATION include/libfps)

if(USE_NCURSES)
# ==========================================
# milk-fpsCTRL executable
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

# ==========================================
# milk-fps-list executable
# ==========================================

add_executable(milk-fps-list milk-fps-list.c)
target_link_libraries(milk-fps-list PUBLIC 
    milkfps
    milkprocessinfo 
    ImageStreamIO
    CLIcore
    ${NCURSES_LIBRARIES}
    pthread
    m
)

install(TARGETS milk-fps-list DESTINATION bin)

# ==========================================
# milk-fps-print executable
# ==========================================

add_executable(milk-fps-print milk-fps-print.c)
target_link_libraries(milk-fps-print PUBLIC 
    milkfps
    milkprocessinfo 
    ImageStreamIO
    CLIcore
    ${NCURSES_LIBRARIES}
    pthread
    m
)

install(TARGETS milk-fps-print DESTINATION bin)
endif()
