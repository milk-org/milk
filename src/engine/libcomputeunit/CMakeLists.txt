set(LIBNAME "milkcomputeunit")
set(SRCNAME "libcomputeunit")

set(SOURCEFILES
    cmd/fps_execFPScmd.c
    cmd/fps_tmux.c
    cmd/fps_processcmdline_interactive.c
    cmd/fps_processcmdline_file.c
    cmd/fps_process_fpsCMDarray.c
    cli/fps_cli_init.c
    cli/fps_cli_sync.c
    core/fps_lifecycle.c
    core/fps_standalone_main.c
    core/milk_runnable.c
    core/milk_runnable_main.c
    io/fps_read_fpsCMD_fifo.c
    params/fps_modulevars_resync.c
    procinfo/fps_processinfo.c
    procinfo/fps_processinfo_entries.c
    procinfo/fps_apply_process_settings.c
    state/fps_FPCONFsetup.c
    state/fps_FPCONFexit.c
    state/fps_FPCONFloopstep.c
    state/fps_RUNexit.c
    state/fps_RUNstart.c
    state/fps_RUNstop.c
    state/fps_CONFstart.c
    state/fps_CONFstop.c)

add_library(${LIBNAME} SHARED ${SOURCEFILES})
# Single translation unit
set_target_properties(${LIBNAME} PROPERTIES UNITY_BUILD ON
                                            UNITY_BUILD_BATCH_SIZE 0)

target_include_directories(
  ${LIBNAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                    $<INSTALL_INTERFACE:include>)
target_link_libraries(${LIBNAME} PUBLIC milkfps milkdata milkprocessinfo
                                        milkcommon ImageStreamIO m rt)
install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)

file(GLOB LIBCOMPUTEUNIT_HEADERS CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${LIBCOMPUTEUNIT_HEADERS} DESTINATION include/${SRCNAME})

# Static archive for -DUSE_STATIC_LTO
if(USE_STATIC_LTO)
  add_library(${LIBNAME}_static STATIC ${SOURCEFILES})
  set_target_properties(
    ${LIBNAME}_static PROPERTIES OUTPUT_NAME ${LIBNAME} UNITY_BUILD ON
                                 UNITY_BUILD_BATCH_SIZE 0)
  target_include_directories(
    ${LIBNAME}_static
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libprocessinfo>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libmilkdata>
           $<INSTALL_INTERFACE:include>)
  target_link_libraries(${LIBNAME}_static PUBLIC milkfps_static milkdata_static
                                                 milkcommon)
endif()

# Generic FPS lifecycle command, compiled once per binary name
set(FPS_GENERIC_CMDS milk-fps-confstart milk-fps-confstop milk-fps-runstart
                     milk-fps-runstop milk-fps-confstep)

foreach(cmd ${FPS_GENERIC_CMDS})
  add_executable(${cmd} tools/milk-fps-cmd.c)
  target_link_libraries(${cmd} PUBLIC ${LIBNAME} pthread m)
  install(TARGETS ${cmd} DESTINATION bin)
endforeach()
