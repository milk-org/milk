set(LIBNAME "fps")
set(SRCNAME "libfps")

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
    fps_processcmdline.c
    fps_process_fpsCMDarray.c
    fps_read_fpsCMD_fifo.c
    fps_userinputsetparamvalue.c
    fps_loadmemstream_lite.c
    TUItools_stubs.c
    timeutils.c
)

add_library(${LIBNAME} SHARED ${SOURCEFILES})

target_include_directories(${LIBNAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>
)

# Standalone libfps dependencies
# We removed milkCOREMODiofits and milkCOREMODtools to break dependency on CLIcore
target_link_libraries(fps ImageStreamIO m rt processinfo)

install(TARGETS ${LIBNAME} EXPORT milkTargets DESTINATION lib)
install(FILES fps.h fps_internal.h fps_globals.h timeutils.h DESTINATION include/libfps)