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
    core/CLIcore_fps_function.c
    core/CLIcore_fps_query.c
    core/CLIcore_datainit.c
    core/CLIcore_help.c
    core/CLIcore_help_command.c
    core/CLIcore_help_topics.c
    core/CLIcore_inspect.c
    core/CLIcore_memory.c
    core/CLIcore_modules.c
    core/CLIcore_setSHMdir.c
    core/CLIcore_signals.c
    core/CLIcore_vars.c
    core/milk_catalog.c
    core/milk_catalog_build.c
    core/milk_cmd_args.c
    core/milk_cmd_index.c
    core/milk_keywords.c
    core/milk_renames.c
    core/milk_builtin_module.c
    ipc/milk_ipc_common.c
    client/milk_client_conn.c
    client/milk_client_ops.c
    client/milk_client_stream.c
    client/milk_client_args.c
    ipc/milk_waitfor.c
    ipc/milk_srvlog.c
    client/milk_sessions.c
    server/milk_server.c
    server/milk_server_vars.c
    server/milk_server_stream.c
    server/milk_server_complete.c
    server/milk_server_dispatch.c
    server/milk_server_eval.c
    server/milk_server_admin.c
    bash/milk_bash_translate.c
    bash/milk_shell_lex.c
    bash/milk_shell_lex_code.c
    bash/milk_bash_coproc.c
    bash/milk_bash_exec.c
    bash/milk_bash_cmdfile.c
    milkscript_stubs.c
    milkscript_api.c)

add_library(milkscript SHARED ${MILKSCRIPT_SOURCES})
milk_apply_extensions(milkscript)

# NaN comparisons and isfinite() must keep IEEE semantics, as in libmilkcalc (src/calc/CMakeLists.txt).
if(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
  target_compile_options(milkscript PRIVATE -fno-finite-math-only)
endif()

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
         milkcalc
         milkprop
         milkCOREMODarith
         $<$<BOOL:${USE_CFITSIO}>:milkCOREMODiofits>
         milkCOREMODmemory
         milkCOREMODtools
         milkprocessinfo
         milkfps
         milkcomputeunit
         milkdata)

target_include_directories(
  milkscript
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/core>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/prop>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/calc>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/calc/units>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ipc>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/client>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/server>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/bash>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
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
        core/CLIcore_fps_function.h
        core/CLIcore_fps_getFPSargs.h
        core/CLIcore_fps_query.h
        core/CLIcore_help.h
        core/CLIcore_memory.h
        core/CLIcore_modules.h
        core/CLIcore_setSHMdir.h
        core/CLIcore_script.h
        core/CLIcore_signals.h
        core/CLIcore_utils.h
        ipc/milk_proto.h
        ipc/milk_srvlog.h
        client/milk_client.h
        server/milk_server.h
        bash/milk_bash.h
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

# Loadable GNU Bash builtin module (enable -f libmilkbash.so ...): the thin client every user shell
# loads. IPC, shm properties, @{...} translation and the in-shell scalar calculator, without the server
# (no cfitsio, OpenMP or module code); it exports only the builtins.
add_library(
  milkbash SHARED
  bash/milk_bash_builtins.c
  bash/milk_bash_hooks.c
  bash/milk_bash_defer.c
  bash/milk_bash_translate.c
  bash/milk_shell_lex.c
  bash/milk_shell_lex_code.c
  bash/milk_bash_cmdfile.c
  client/milk_client_conn.c
  client/milk_client_ops.c
  client/milk_client_stream.c
  client/milk_client_args.c
  ipc/milk_ipc_common.c
  ipc/milk_waitfor.c
  ipc/milk_srvlog.c
  client/milk_sessions.c
  core/milk_keywords.c
  ../../calc/ext/mc_ext_registry.c
  ../../calc/ext/mc_ext_loader.c
  ../../calc/mc_func.c
  ../../calc/front/mc_lex.c
  ../../calc/run/mc_scalar_ops.c
  core/milk_renames.c
  core/CLIcore_inspect.c)
target_include_directories(
  milkbash PRIVATE $<TARGET_PROPERTY:milkscript,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_definitions(milkbash PRIVATE MILK_BASH_CLIENT)
if(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
  target_compile_options(milkbash PRIVATE -fopenmp-simd -fno-finite-math-only)
endif()
target_link_libraries(
  milkbash PRIVATE milkprop milkfps milkcomputeunit milkprocessinfo milkdata m dl rt
                   ${CMAKE_THREAD_LIBS_INIT})
set_target_properties(milkbash PROPERTIES C_VISIBILITY_PRESET hidden INSTALL_RPATH "$ORIGIN")
target_link_options(milkbash PRIVATE "-Wl,--disable-new-dtags" "-Wl,--no-undefined")
install(TARGETS milkbash DESTINATION lib)

# Compute units without C code (milk.calc, milk.script): fpsexec binaries that run calculator statements or
# a bash snippet on each frame; the calculator and milk.bash come from libmilkscript.
foreach(_unit script)
  add_executable(milk-fpsexec-milk-${_unit} units/milk_unit_${_unit}.c)
  target_compile_definitions(milk-fpsexec-milk-${_unit} PRIVATE MILK_NO_CLI)
  milk_configure_fpsexec(milk-fpsexec-milk-${_unit} PREFIX milk)
  target_link_libraries(milk-fpsexec-milk-${_unit} PUBLIC milkscript)
endforeach()

# Standalone helper executable (milk-call)
add_executable(milk-call client/milk_call_main.c)
target_link_libraries(milk-call PRIVATE milkscript milkprocessinfo milkdata)
target_include_directories(
  milk-call PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                   ${CMAKE_CURRENT_SOURCE_DIR}/../CLIcore)
set_target_properties(milk-call PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
install(TARGETS milk-call DESTINATION bin)
