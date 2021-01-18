cmake_minimum_required(VERSION 3.8)


set(SRCNAME "CommandLineInterface")


option(python_module "Compile Python Wrappers" OFF)

project(CLIcore C)

find_package(PkgConfig REQUIRED)
find_package(GSL QUIET REQUIRED)
pkg_check_modules(FFTW REQUIRED fftw3)
pkg_check_modules(FFTWF REQUIRED fftw3f)
pkg_check_modules(NCURSES REQUIRED ncurses)

pkg_check_modules(HWLOC hwloc)

pkg_check_modules(CFITSIO cfitsio)


include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/..")

find_package(BISON)
find_package(FLEX)
# find_package(Curses REQUIRED) # already performed in main build dir
find_package(Threads REQUIRED)
bison_target(MilkBison calc_bison.y ${PROJECT_SOURCE_DIR}/calc_bison.c)
flex_target(MilkFlex calc_flex.l ${PROJECT_SOURCE_DIR}/calc_flex.c)
add_flex_bison_dependency(MilkFlex MilkBison)
set(CORE_FILES
    CLIcore.c
    CLIcore_utils.c
    CLIcore_UI.c
    CLIcore_checkargs.c
    CLIcore_datainit.c
    CLIcore_help.c
    CLIcore_memory.c
    CLIcore_modules.c
    CLIcore_setSHMdir.c
    CLIcore_signals.c
    ${BISON_MilkBison_OUTPUTS}
    ${FLEX_MilkFlex_OUTPUTS})

link_directories(${NCURSES_LIBRARY_DIRS} ${HWLOC_LIBRARY_DIRS})

# main
add_library(CLIcore
            SHARED
            processtools.c
            processtools_trigger.c
            streamCTRL.c
            timeutils.c
            fps_add_entry.c
            fps_checkparameter.c
            fps_connect.c
            fps_connectExternalFPS.c
            fps_CONFstart.c
            fps_CONFstop.c
            fps_CTRLscreen.c
            fps_disconnect.c
            fps_execFPScmd.c
            fps_FPCONFexit.c
            fps_FPCONFloopstep.c
            fps_FPCONFsetup.c
            fps_FPSremove.c
            fps_GetFileName.c
            fps_getFPSargs.c
            fps_GetParamIndex.c
            fps_GetTypeString.c
            fps_load.c
            fps_loadstream.c
            fps_outlog.c
            fps_paramvalue.c
            fps_printlist.c
            fps_PrintParameterInfo.c
            fps_printparameter_valuestring.c
            fps_processcmdline.c
            fps_process_fpsCMDarray.c
            fps_process_user_key.c
            fps_processinfo_entries.c
            fps_read_fpsCMD_fifo.c
            fps_RUNexit.c
            fps_RUNstart.c
            fps_RUNstop.c
            fps_save2disk.c
            fps_scan.c
            fps_shmdirname.c
            fps_tmux.c
            fps_userinputsetparamvalue.c
            fps_WriteParameterToDisk.c
            TUItools.c
            ${CORE_FILES})

if(${CFITSIO_FOUND})
  link_directories(${CFITSIO_LIBRARY_DIRS})
  target_compile_definitions(CLIcore PUBLIC USE_CFITSIO=1)
  target_include_directories(CLIcore PUBLIC ${CFITSIO_INCLUDE_DIRS})
endif()

target_include_directories(CLIcore
                PUBLIC ${PROJECT_SOURCE_DIR}/..
                      ${PROJECT_SOURCE_DIR}/src
                      ${GSL_INCLUDE_DIRS}
                      ${FFTW_INCLUDE_DIRS}
                      ${FFTWF_INCLUDE_DIRS})

target_compile_options(CLIcore
                        PUBLIC ${FFTW_CFLAGS_OTHER} ${FFTWF_CFLAGS_OTHER})

target_link_libraries(CLIcore
                      PUBLIC m
                              readline
                              ${CFITSIO_LIBRARIES}
                              dl
                              rt
                              ${GSL_LIBRARIES}
                              ${FFTW_LIBRARIES}
                              ${FFTWF_LIBRARIES}
                              ${CMAKE_THREAD_LIBS_INIT}
                              ${NCURSES_LIBRARIES}
                              milkCOREMODarith milkCOREMODiofits milkCOREMODmemory milkCOREMODtools)

if(${HWLOC_FOUND})
  target_compile_definitions(CLIcore PRIVATE USE_HWLOC)
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
                             ${NCURSES_LIBRARIES})

if(${python_module})
  add_subdirectory(python_module)
endif()

install(TARGETS CLIcore DESTINATION lib)

install(FILES CLIcore.h
              CLIcore_utils.h
              CLIcore_UI.h
              CLIcore_checkargs.h
              CLIcore_datainit.h
              CLIcore_help.h
              CLIcore_memory.h
              CLIcore_modules.h
              CLIcore_setSHMdir.h
              CLIcore_signals.h
              milkDebugTools.h
              processinfo.h
              processtools.h
              processtools_trigger.h
              streamCTRL.h
              timeutils.h
              function_parameters.h
              fps_add_entry.h
              fps_checkparameter.h
              fps_CONFstart.h
              fps_CONFstop.h
              fps_connect.h
              fps_connectExternalFPS.h
              fps_CTRLscreen.h
              fps_disconnect.h
              fps_execFPScmd.h
              fps_FPCONFexit.h
              fps_FPCONFloopstep.h
              fps_FPCONFsetup.h
              fps_FPSremove.h
              fps_GetFileName.h
              fps_getFPSargs.h
              fps_GetParamIndex.h
              fps_GetTypeString.h
              fps_load.h
              fps_loadstream.h
              fps_outlog.h
              fps_paramvalue.h
              fps_PrintParameterInfo.h
              fps_printparameter_valuestring.h
              fps_processcmdline.h
              fps_process_fpsCMDarray.h
              fps_processinfo_entries.h
              fps_process_user_key.h
              fps_read_fpsCMD_fifo.h
              fps_RUNexit.h
              fps_RUNstart.h
              fps_RUNstop.h
              fps_save2disk.h
              fps_scan.h
              fps_shmdirname.h
              fps_tmux.h
              fps_userinputsetparamvalue.h
              IMGID.h
              DESTINATION include/${SRCNAME})
