cmake_minimum_required(VERSION 3.17)

project(milkscript C)

set(SRCNAME "milkscript")

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP)

if(USE_CFITSIO)
  pkg_check_modules(CFITSIO cfitsio)
endif()

# -------------------------------------------------------
# libmilkscript — the readline-free scripting engine
# -------------------------------------------------------
# This library contains the command parser, dispatcher, flow control, variable
# expansion, math evaluator, module loader, signal handling, and help system.
#
# It does NOT link against readline or ncurses.
# -------------------------------------------------------

set(MILKSCRIPT_SOURCES
    CLIcore_UI_execute.c
    CLIcore_UI_execute_debug.c
    CLIcore_UI_execute_preproc.c
    CLIcore_UI_execute_redir.c
    CLIcore_UI_alias.c
    CLIcore_checkargs.c
    CLIcore_checkargs_fps.c
    CLIcore_datainit.c
    CLIcore_help.c
    CLIcore_help_command.c
    CLIcore_help_topics.c
    CLIcore_memory.c
    CLIcore_modules.c
    CLIcore_setSHMdir.c
    CLIcore_script.c
    CLIcore_script_intercept.c
    CLIcore_script_intercept_process.c
    CLIcore_script_intercept_environment.c
    CLIcore_script_intercept_env_trap.c
    CLIcore_script_intercept_execution.c
    CLIcore_script_intercept_exec_waitany.c
    CLIcore_script_intercept_flow.c
    CLIcore_script_intercept_utils.c
    CLIcore_script_builtin.c
    CLIcore_script_traps.c
    CLIcore_script_cmd_builtins.c
    CLIcore_script_cmd_io.c
    CLIcore_script_cmd_inspect.c
    CLIcore_script_cmd_defer.c
    CLIcore_script_case.c
    CLIcore_script_expand_arith.c
    CLIcore_script_expand_fps.c
    CLIcore_script_expand_test.c
    CLIcore_script_expand_env.c
    CLIcore_script_flow.c
    CLIcore_script_func.c
    CLIcore_script_var.c
    CLIcore_signals.c
    CLIcore_utils.c
    cli_calc_tokenizer.c
    cli_calc_parser.c
    cli_calc_functions.c
    cli_calc_binops.c
    cli_calc_eval.c
    milkscript_stubs.c
    milkscript_api.c)

add_library(milkscript SHARED ${MILKSCRIPT_SOURCES})

if(USE_CFITSIO AND ${CFITSIO_FOUND})
  link_directories(${CFITSIO_LIBRARY_DIRS})
  target_compile_definitions(milkscript PUBLIC USE_CFITSIO=1)
  target_include_directories(milkscript PUBLIC ${CFITSIO_INCLUDE_DIRS})
endif()

if(OPENMP_C_FOUND)
  target_link_libraries(milkscript PUBLIC OpenMP::OpenMP_C)
endif()

# NO readline, NO ncurses — this is the key constraint
target_link_libraries(
  milkscript
  PUBLIC m
         dl
         rt
         ${CMAKE_THREAD_LIBS_INIT}
         ${CFITSIO_LIBRARIES}
         milkCOREMODarith
         $<$<BOOL:${USE_CFITSIO}>:milkCOREMODiofits>
         milkCOREMODmemory
         milkCOREMODtools
         milkprocessinfo
         milkfps
         milkfpsseq
         milkdata)

target_include_directories(
  milkscript
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/streamCTRL>
         $<INSTALL_INTERFACE:include/${SRCNAME}>)

install(
  TARGETS milkscript
  EXPORT milkTargets
  DESTINATION lib)

# Install public headers
install(
  FILES CLIcore_UI_execute.h
        CLIcore_checkargs.h
        CLIcore_datainit.h
        CLIcore_help.h
        CLIcore_memory.h
        CLIcore_modules.h
        CLIcore_setSHMdir.h
        CLIcore_script.h
        CLIcore_signals.h
        CLIcore_utils.h
        cli_calc_parser.h
        cli_calc_tokenizer.h
        standalone_dependencies.h
        script_includes.h
        milkscript.h
  DESTINATION include/${SRCNAME})

add_executable(milk-script milkscript_main.c)
target_link_libraries(milk-script PRIVATE milkscript milkprocessinfo milkdata)
target_include_directories(
  milk-script PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                     ${CMAKE_CURRENT_SOURCE_DIR}/../CLIcore)

install(TARGETS milk-script DESTINATION bin)
