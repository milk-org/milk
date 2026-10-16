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
    core/CLIcore_UI_execute.c
    core/CLIcore_UI_execute_debug.c
    core/CLIcore_checkargs.c
    core/CLIcore_checkargs_fps.c
    core/CLIcore_datainit.c
    core/CLIcore_help.c
    core/CLIcore_help_command.c
    core/CLIcore_help_topics.c
    core/CLIcore_inspect.c
    core/CLIcore_memory.c
    core/CLIcore_modules.c
    core/CLIcore_setSHMdir.c
    core/CLIcore_signals.c
    core/CLIcore_utils.c
    core/CLIcore_vars.c
    calc/cli_calc_tokenizer.c
    calc/cli_calc_parser.c
    calc/cli_calc_functions.c
    calc/cli_calc_binops.c
    calc/cli_calc_eval.c
    calc/cli_calc_slice.c
    shm_prop/milk_shm_prop.c
    shm_prop/milk_shm_prop_stream.c
    shm_prop/milk_shm_prop_fps.c
    shm_prop/milk_shm_prop_proc_seq.c
    ipc/milk_ipc_common.c
    ipc/milk_client_conn.c
    ipc/milk_client_ops.c
    ipc/milk_srvlog_shm.c
    ipc/milk_srv_manager.c
    server/milk_server.c
    server/milk_server_vars.c
    server/milk_server_stream.c
    server/milk_server_io.c
    server/milk_server_dispatch.c
    bash/milk_bash_translate.c
    bash/milk_shell_lex.c
    bash/milk_bash_coproc.c
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
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/core>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/calc>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/shm_prop>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ipc>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/server>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/bash>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/streamCTRL>
         $<INSTALL_INTERFACE:include/${SRCNAME}>)

set_target_properties(milkscript PROPERTIES INSTALL_RPATH "$ORIGIN")
target_link_options(milkscript PRIVATE "-Wl,--disable-new-dtags")

install(
  TARGETS milkscript
  EXPORT milkTargets
  DESTINATION lib)

# Install public headers
install(
  FILES core/CLIcore_UI_execute.h
        core/CLIcore_checkargs.h
        core/CLIcore_datainit.h
        core/CLIcore_help.h
        core/CLIcore_memory.h
        core/CLIcore_modules.h
        core/CLIcore_setSHMdir.h
        core/CLIcore_script.h
        core/CLIcore_signals.h
        core/CLIcore_utils.h
        calc/cli_calc_parser.h
        calc/cli_calc_tokenizer.h
        shm_prop/milk_shm_prop.h
        ipc/milk_proto.h
        ipc/milk_srvlog.h
        ipc/milk_client.h
        server/milk_server.h
        bash/milk_bash_engine.h
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

# Loadable GNU Bash builtin module (enable -f libmilkbash.so ...)
add_library(milkbash SHARED ipc/milk_bash_builtins.c)
target_link_libraries(milkbash PRIVATE milkscript dl)
target_include_directories(
  milkbash PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                  ${CMAKE_CURRENT_SOURCE_DIR}/../CLIcore)
set_target_properties(milkbash PROPERTIES INSTALL_RPATH "$ORIGIN")
target_link_options(milkbash PRIVATE "-Wl,--disable-new-dtags")
install(TARGETS milkbash DESTINATION lib)

# Standalone helper executable (milk-call)
add_executable(milk-call ipc/milk_call_main.c)
target_link_libraries(milk-call PRIVATE milkscript milkprocessinfo milkdata)
target_include_directories(
  milk-call PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                   ${CMAKE_CURRENT_SOURCE_DIR}/../CLIcore)
set_target_properties(milk-call PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
install(TARGETS milk-call DESTINATION bin)
