cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0069 NEW)

set(SRCNAME "CommandLineInterface")

option(python_module "Compile Python Wrappers" OFF)

project(CLIcore C)

find_package(PkgConfig REQUIRED)

if(USE_NCURSES)
    pkg_check_modules(NCURSES REQUIRED ncurses)
endif()
if(USE_READLINE)
    pkg_check_modules(READLINE REQUIRED readline)
endif()
find_package(OpenMP)



pkg_check_modules(CFITSIO cfitsio)



find_package(Threads REQUIRED)

set(CORE_FILES
    CLIcore.c
    CLIcore/CLIcore_help_tui.c
    CLIcore/CLIcore_UI_bookmark.c
    CLIcore/CLIcore_UI_builtins.c
    CLIcore/CLIcore_UI_completion.c
    CLIcore/CLIcore_UI_highlight.c
    CLIcore/CLIcore_UI_hintarea.c
    CLIcore/CLIcore_UI_history.c
    CLIcore/CLIcore_UI_prompt.c)
    
    link_directories(${NCURSES_LIBRARY_DIRS})
    
        # main
    
        add_library(CLIcore
    
                    SHARED
    
        ../streamCTRL/streamCTRL_TUI.c
        ../streamCTRL/streamCTRL_find_streams.c
        ../streamCTRL/streamCTRL_print_inode.c
        ../streamCTRL/streamCTRL_print_procpid.c
        ../streamCTRL/streamCTRL_print_trace.c
        ../streamCTRL/streamCTRL_scan.c
        ../streamCTRL/streamCTRL_utilfuncs.c
        ${CORE_FILES})

# Add milk-help executable
add_executable(milk-help milk-help.c)
target_link_libraries(milk-help PUBLIC CLIcore)
install(TARGETS milk-help DESTINATION bin)

# Add milk-cli-help executable
add_executable(milk-cli-help milk-cli-help.c)
target_link_libraries(milk-cli-help PUBLIC CLIcore)
install(TARGETS milk-cli-help DESTINATION bin)

# Add milk-cli-help-synchro executable
add_executable(milk-cli-help-synchro milk-cli-help-synchro.c)
target_link_libraries(milk-cli-help-synchro PUBLIC CLIcore)
install(TARGETS milk-cli-help-synchro DESTINATION bin)

if(${CFITSIO_FOUND})
  link_directories(${CFITSIO_LIBRARY_DIRS})
  target_compile_definitions(CLIcore PUBLIC USE_CFITSIO=1)
  target_include_directories(CLIcore PUBLIC ${CFITSIO_INCLUDE_DIRS})
endif()


if (OPENMP_C_FOUND)
  message("Found OpenMP")
  target_link_libraries(CLIcore PUBLIC OpenMP::OpenMP_C)
endif()

if(USE_READLINE)
    target_compile_definitions(CLIcore PUBLIC USE_READLINE)
endif()

if(USE_NCURSES)
    target_compile_definitions(CLIcore PUBLIC USE_NCURSES)
endif()

target_link_libraries(CLIcore
                      PUBLIC m
                              milkscript
                              $<$<BOOL:${USE_READLINE}>:readline>
                              ${CFITSIO_LIBRARIES}
                              dl
                              rt

                              ${CMAKE_THREAD_LIBS_INIT}
                              $<$<BOOL:${USE_NCURSES}>:${NCURSES_LIBRARIES}>
                              $<$<BOOL:${USE_NCURSES}>:milkTUI>
                              $<$<BOOL:${USE_NCURSES}>:milkprocessinfoTUI>
                              $<$<BOOL:${USE_NCURSES}>:milkfpsTUI>
                              milkCOREMODarith $<$<BOOL:${USE_CFITSIO}>:milkCOREMODiofits> milkCOREMODmemory milkCOREMODtools milkprocessinfo milkfps milkfpsseq milkdata)


target_include_directories(CLIcore
                           PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
                                  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli>
                                  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/libmilkscript>
                                  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../streamCTRL>
                                  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/CLIcore>
                                  $<INSTALL_INTERFACE:include/${SRCNAME}>
                                  $<$<BOOL:${USE_NCURSES}>:${NCURSES_INCLUDE_DIR}>)
target_compile_options(CLIcore PUBLIC)

target_link_directories(CLIcore
                        PUBLIC $<$<BOOL:${USE_NCURSES}>:${NCURSES_LIBRARY_DIRS}>)

target_link_libraries(CLIcore
                      PUBLIC ${CMAKE_THREAD_LIBS_INIT}
                             $<$<BOOL:${USE_NCURSES}>:${NCURSES_LIBRARIES}> )

if(${python_module})
  add_subdirectory(python_module)
endif()

install(TARGETS CLIcore EXPORT milkTargets DESTINATION lib)


install(FILES CLIcore.h
              IMGID.h
              ../libmilkTUI/TUItools.h
              ../libmilkscript/cli_calc_parser.h
              ../libmilkscript/cli_calc_tokenizer.h
              cmdsettings.h
              ../../engine/libmilkdata/milkDebugTools.h
              ../libmilkscript/standalone_dependencies.h
              ../../engine/libprocessinfo/timeutils.h
              DESTINATION include/${SRCNAME})

install(FILES ../libmilkscript/CLIcore_UI_execute.h
              ../libmilkscript/CLIcore_checkargs.h
              ../libmilkscript/CLIcore_datainit.h
              ../libmilkscript/CLIcore_help.h
              ../libmilkscript/CLIcore_memory.h
              ../libmilkscript/CLIcore_modules.h
              ../libmilkscript/CLIcore_setSHMdir.h
              ../libmilkscript/CLIcore_script.h
              ../libmilkscript/CLIcore_signals.h
              ../libmilkscript/CLIcore_utils.h
              DESTINATION include/${SRCNAME}/CLIcore)



add_executable(milk-streamCTRL ../streamCTRL/milk-streamCTRL.c)
target_link_libraries(milk-streamCTRL PUBLIC CLIcore ImageStreamIO)

install(TARGETS milk-streamCTRL DESTINATION bin)

add_executable(milk-termview termview/milk-termview.c termview/termview.c)
target_link_libraries(milk-termview PUBLIC CLIcore ImageStreamIO)

install(TARGETS milk-termview DESTINATION bin)

# TESTING

include(CTest)

set (failRegex "ERROR" "FAIL")

add_test (NAME milklistim COMMAND milk-exec "mem.listim")
set_property (TEST milklistim PROPERTY LABELS "CLI")
set_tests_properties(milklistim PROPERTIES TIMEOUT 1)
set_property (TEST milklistim PROPERTY PASS_REGULAR_EXPRESSION "0 image")
