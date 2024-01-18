cmake_minimum_required(VERSION 3.8)


set(SRCNAME "CommandLineInterface")


option(python_module "Compile Python Wrappers" OFF)

project(CLIcore C)

find_package(PkgConfig REQUIRED)
pkg_check_modules(NCURSES REQUIRED ncurses)
pkg_check_modules(FFTW fftw3)
pkg_check_modules(FFTWF fftw3f)
find_package(OpenMP)
find_package(GSL)

pkg_check_modules(HWLOC hwloc)

pkg_check_modules(CFITSIO cfitsio)

find_package(BISON)
find_package(FLEX)
# find_package(Curses REQUIRED) # already performed in main build dir
find_package(Threads REQUIRED)
bison_target(MilkBison calc_bison.y ${PROJECT_SOURCE_DIR}/calc_bison.c)
flex_target(MilkFlex calc_flex.l ${PROJECT_SOURCE_DIR}/calc_flex.c)
add_flex_bison_dependency(MilkFlex MilkBison)
set(CORE_FILES
    CLIcore.c
    CLIcore/CLIcore_utils.c
    CLIcore/CLIcore_UI.c
    CLIcore/CLIcore_checkargs.c
    CLIcore/CLIcore_datainit.c
    CLIcore/CLIcore_help.c
    CLIcore/CLIcore_memory.c
    CLIcore/CLIcore_modules.c
    CLIcore/CLIcore_setSHMdir.c
    CLIcore/CLIcore_signals.c
    ${BISON_MilkBison_OUTPUTS}
    ${FLEX_MilkFlex_OUTPUTS})

link_directories(${NCURSES_LIBRARY_DIRS} ${HWLOC_LIBRARY_DIRS})

# main
add_library(CLIcore
            SHARED
            fpsCTRL/fpsCTRL_TUI.c
            fpsCTRL/fpsCTRL_TUI_process_user_key.c
            fpsCTRL/fpsCTRL_FPSdisplay.c
            fpsCTRL/print_nodeinfo.c
            fpsCTRL/level0node_summary.c
            fpsCTRL/scheduler_display.c
            timeutils.c
            fps/fps_add_entry.c
            fps/fps_checkparameter.c
            fps/fps_connect.c
            fps/fps_connectExternalFPS.c
            fps/fps_CONFstart.c
            fps/fps_CONFstop.c
            fps/fps_disconnect.c
            fps/fps_execFPScmd.c
            fps/fps_FPCONFexit.c
            fps/fps_FPCONFloopstep.c
            fps/fps_FPCONFsetup.c
            fps/fps_FPSremove.c
            fps/fps_GetFileName.c
            fps/fps_getFPSargs.c
            fps/fps_GetParamIndex.c
            fps/fps_GetTypeString.c
            fps/fps_load.c
            fps/fps_loadstream.c
            fps/fps_outlog.c
            fps/fps_paramvalue.c
            fps/fps_printlist.c
            fps/fps_PrintParameterInfo.c
            fps/fps_printparameter_valuestring.c
            fps/fps_processcmdline.c
            fps/fps_process_fpsCMDarray.c
            fps/fps_processinfo_entries.c
            fps/fps_read_fpsCMD_fifo.c
            fps/fps_RUNexit.c
            fps/fps_RUNstart.c
            fps/fps_RUNstop.c
            fps/fps_save2disk.c
            fps/fps_scan.c
            fps/fps_shmdirname.c
            fps/fps_tmux.c
            fps/fps_userinputsetparamvalue.c
            fps/fps_WriteParameterToDisk.c
            procCTRL/procCTRL_TUI.c
            procCTRL/procCTRL_processinfo_scan.c
            procCTRL/procCTRL_GetCPUloads.c
            procCTRL/procCTRL_GetNumberCPUs.c
            procCTRL/procCTRL_PIDcollectSystemInfo.c
            processtools_trigger.c
            processinfo/processinfo_procdirname.c
            processinfo/processinfo_exec_start.c
            processinfo/processinfo_exec_end.c
            processinfo/processinfo_loopstep.c
            processinfo/processinfo_setup.c
            processinfo/processinfo_shm_close.c
            processinfo/processinfo_shm_create.c
            processinfo/processinfo_shm_link.c
            processinfo/processinfo_shm_list_create.c
            processinfo/processinfo_signals.c
            processinfo/processinfo_SIGexit.c
            processinfo/processinfo_update_output_stream.c
            processinfo/processinfo_WriteMessage.c
            streamCTRL/streamCTRL_TUI.c
            streamCTRL/streamCTRL_find_streams.c
            streamCTRL/streamCTRL_print_inode.c
            streamCTRL/streamCTRL_print_procpid.c
            streamCTRL/streamCTRL_print_trace.c
            streamCTRL/streamCTRL_scan.c
            streamCTRL/streamCTRL_utilfuncs.c
            TUItools.c
            ${CORE_FILES})

if(${CFITSIO_FOUND})
  link_directories(${CFITSIO_LIBRARY_DIRS})
  target_compile_definitions(CLIcore PUBLIC USE_CFITSIO=1)
  target_include_directories(CLIcore PUBLIC ${CFITSIO_INCLUDE_DIRS})
endif()

if(${GSL_FOUND})
  target_compile_definitions(CLIcore PUBLIC USE_GSL=1)
  target_include_directories(CLIcore PUBLIC ${GSL_INCLUDE_DIRS})
endif()

if(${FFTW_FOUND})
  target_compile_definitions(CLIcore PUBLIC USE_FFTW3=1)
  target_include_directories(CLIcore PUBLIC
                             ${FFTW_INCLUDE_DIRS}
                             ${FFTWF_INCLUDE_DIRS})
  target_compile_options(CLIcore PUBLIC 
                         ${FFTW_CFLAGS_OTHER} 
                         ${FFTWF_CFLAGS_OTHER})
endif()

if (OPENMP_C_FOUND)
  message("Found OpenMP")
  target_link_libraries(CLIcore PUBLIC OpenMP::OpenMP_C)
endif()


target_link_libraries(CLIcore
                      PUBLIC m
                              readline
                              dl
                              rt
                              ${CFITSIO_LIBRARIES}
                              ${GSL_LIBRARIES}
                              ${FFTW_LIBRARIES}
                              ${FFTWF_LIBRARIES}
                              ${CMAKE_THREAD_LIBS_INIT}
                              ${NCURSES_LIBRARIES}
                              milkCOREMODarith milkCOREMODiofits milkCOREMODmemory milkCOREMODtools)

if(${HWLOC_FOUND})
  target_compile_definitions(CLIcore PUBLIC USE_HWLOC)
endif()
target_include_directories(CLIcore
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                  ${FFTW_INCLUDE_DIRS} ${FFTWF_INCLUDE_DIRS}
                                  ${HWLOC_INCLUDE_DIR} ${NCURSES_INCLUDE_DIR})
target_compile_options(CLIcore PUBLIC)

target_link_directories(CLIcore
                        PUBLIC ${NCURSES_LIBRARY_DIRS} ${HWLOC_LIBRARY_DIRS})

target_link_libraries(CLIcore
                      PUBLIC ${CMAKE_THREAD_LIBS_INIT} ${HWLOC_LIBRARIES}
                             ${NCURSES_LIBRARIES} )

if(${python_module})
  add_subdirectory(python_module)
endif()

install(TARGETS CLIcore DESTINATION lib)


install(FILES CLIcore.h
              IMGID.h
              TUItools.h
              calc.h
              calc_bison.h
              cmdsettings.h
              function_parameters.h
              milkDebugTools.h
              processinfo.h
              processtools.h
              processtools_trigger.h
              standalone_dependencies.h
              timeutils.h
              DESTINATION include/${SRCNAME})
              
install(FILES CLIcore/CLIcore_UI.h
              CLIcore/CLIcore_checkargs.h
              CLIcore/CLIcore_datainit.h
              CLIcore/CLIcore_help.h
              CLIcore/CLIcore_memory.h
              CLIcore/CLIcore_modules.h
              CLIcore/CLIcore_setSHMdir.h
              CLIcore/CLIcore_signals.h
              CLIcore/CLIcore_utils.h
              DESTINATION include/${SRCNAME}/CLIcore)
              
install(FILES fps/fps_CONFstart.h
              fps/fps_CONFstop.h
              fps/fps_FPCONFexit.h
              fps/fps_FPCONFloopstep.h
              fps/fps_FPCONFsetup.h
              fps/fps_FPSremove.h
              fps/fps_GetFileName.h
              fps/fps_GetParamIndex.h
              fps/fps_GetTypeString.h
              fps/fps_PrintParameterInfo.h
              fps/fps_RUNexit.h
              fps/fps_RUNstart.h
              fps/fps_RUNstop.h
              fps/fps_WriteParameterToDisk.h
              fps/fps_add_entry.h
              fps/fps_checkparameter.h
              fps/fps_connect.h
              fps/fps_connectExternalFPS.h
              fps/fps_disconnect.h
              fps/fps_execFPScmd.h
              fps/fps_getFPSargs.h
              fps/fps_load.h
              fps/fps_loadstream.h
              fps/fps_outlog.h
              fps/fps_paramvalue.h
              fps/fps_printparameter_valuestring.h
              fps/fps_process_fpsCMDarray.h
              fps/fps_processcmdline.h
              fps/fps_processinfo_entries.h
              fps/fps_read_fpsCMD_fifo.h
              fps/fps_save2disk.h
              fps/fps_scan.h
              fps/fps_shmdirname.h
              fps/fps_tmux.h
              fps/fps_userinputsetparamvalue.h
              DESTINATION include/${SRCNAME}/fps)
              
install(FILES fpsCTRL/fpsCTRL_FPSdisplay.h
              fpsCTRL/fpsCTRL_TUI.h
              fpsCTRL/fpsCTRL_TUI_process_user_key.h
              fpsCTRL/level0node_summary.h
              fpsCTRL/print_nodeinfo.h
              fpsCTRL/scheduler_display.h
              DESTINATION include/${SRCNAME}/fpsCTRL)
              
install(FILES procCTRL/procCTRL_GetCPUloads.h
              procCTRL/procCTRL_GetNumberCPUs.h
              procCTRL/procCTRL_PIDcollectSystemInfo.h
              procCTRL/procCTRL_TUI.h
              procCTRL/procCTRL_processinfo_scan.h
              DESTINATION include/${SRCNAME}/procCTRL)
              
install(FILES processinfo/processinfo_SIGexit.h
              processinfo/processinfo_WriteMessage.h
              processinfo/processinfo_exec_end.h
              processinfo/processinfo_exec_start.h
              processinfo/processinfo_loopstep.h
              processinfo/processinfo_procdirname.h
              processinfo/processinfo_setup.h
              processinfo/processinfo_shm_close.h
              processinfo/processinfo_shm_create.h
              processinfo/processinfo_shm_link.h
              processinfo/processinfo_shm_list_create.h
              processinfo/processinfo_signals.h
              processinfo/processinfo_update_output_stream.h              
              DESTINATION include/${SRCNAME}/processinfo)
              
install(FILES streamCTRL/streamCTRL_TUI.h
              DESTINATION include/${SRCNAME}/streamCTRL)



# TESTING

include(CTest)

set (failRegex "ERROR" "FAIL")

add_test (NAME milklistim COMMAND milk-exec "listim")
set_property (TEST milklistim PROPERTY LABELS "CLI")
set_tests_properties(milklistim PROPERTIES TIMEOUT 1)
set_property (TEST milklistim PROPERTY PASS_REGULAR_EXPRESSION "0 image")




