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
    core/milk_argval.c
    core/milk_cmd_index.c
    core/milk_lexicon.c
    core/milk_renames.c
    core/milk_session.c
    calc/core/cli_calc_tokenizer.c
    calc/core/cli_calc_parser.c
    calc/core/cli_calc_parser_assign.c
    calc/core/cli_calc_parser_state.c
    calc/core/cli_calc_functions.c
    calc/core/cli_calc_functions_red_str.c
    calc/core/cli_calc_binops.c
    calc/core/cli_calc_eval.c
    calc/core/cli_calc_slice.c
    calc/core/cli_calc_complex.c
    calc/core/cli_calc_scalar.c
    calc/core/cli_calc_scalar_temporal.c
    calc/fuse/cli_calc_fuse.c
    calc/fuse/cli_calc_fuse_views.c
    calc/fuse/cli_calc_fuse_ops_spatial.c
    calc/fuse/cli_calc_fuse_ops_axis.c
    calc/fuse/cli_calc_fuse_exec.c
    calc/fuse/cli_calc_fuse_exec_spatial.c
    calc/fuse/cli_calc_fuse_prestage.c
    calc/fuse/cli_calc_fuse_prestage_seg.c
    calc/fuse/cli_calc_fuse_reduce.c
    calc/fuse/cli_calc_fuse_kept.c
    calc/fuse/cli_calc_fuse_merge.c
    calc/linalg/cli_calc_linalg_construct.c
    calc/linalg/cli_calc_linalg_blas.c
    calc/linalg/cli_calc_linalg_kept.c
    calc/linalg/cli_calc_lapack_triangular.c
    calc/linalg/cli_calc_lapack_solve.c
    calc/linalg/cli_calc_lapack_svd_qr.c
    calc/prog/cli_calc_prog.c
    calc/prog/cli_calc_prog_parse.c
    calc/prog/cli_calc_prog_history.c
    calc/prog/cli_calc_engines.c
    calc/ext/cli_calc_ext.c
    calc/ext/cli_calc_fuse_user_kern.c
    calc/ext/cli_calc_fuse_user_op.c
    shm_prop/milk_shm_prop.c
    shm_prop/milk_shm_prop_stream.c
    shm_prop/milk_shm_prop_fps.c
    shm_prop/milk_shm_prop_proc.c
    shm_prop/milk_shm_cache.c
    ipc/milk_ipc_common.c
    ipc/milk_client_conn.c
    ipc/milk_client_ops.c
    ipc/milk_client_ext.c
    ipc/milk_client_args.c
    ipc/milk_waitfor.c
    ipc/milk_srvlog_shm.c
    ipc/milk_srv_manager.c
    server/milk_server.c
    server/milk_server_vars.c
    server/milk_server_stream.c
    server/milk_server_io.c
    server/milk_server_dispatch.c
    server/milk_server_eval.c
    server/milk_server_session.c
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

# The fused evaluator's float32 kernels: under -Ofast, GCC vectorizes float division and 1/sqrt with
# approximate reciprocals (RCPPS/RSQRTPS + Newton step), which turn x/0 into NaN instead of inf.
# -mno-recip keeps IEEE division and square root there (-mrecip=none is not honoured by GCC 15), and
# unlike -fno-finite-math-only it keeps __FAST_MATH__, so libm calls still vectorize with libmvec.
# Applied to the whole library: the shared inline kernels are instantiated in several calc sources.
if(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_compile_options(milkscript PRIVATE -mno-recip)
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
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/calc>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/shm_prop>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ipc>
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
        calc/cli_calc_parser.h
        calc/cli_calc_tokenizer.h
        ../../coremods/COREMOD_arith/milk_calc_ext.h
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

install(FILES ../../coremods/COREMOD_arith/milk_calc_ext.h DESTINATION include)

add_executable(milk-script milkscript_main.c)
target_link_libraries(milk-script PRIVATE milkscript milkprocessinfo milkdata)
target_include_directories(
  milk-script PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                     ${CMAKE_CURRENT_SOURCE_DIR}/../CLIcore)

install(TARGETS milk-script DESTINATION bin)

# Loadable GNU Bash builtin module (enable -f libmilkbash.so ...): the thin client every user shell
# loads. IPC, shm properties, @{...} translation and the in-shell scalar calculator, without the engine
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
  ipc/milk_client_conn.c
  ipc/milk_client_ops.c
  ipc/milk_client_ext.c
  ipc/milk_client_args.c
  ipc/milk_ipc_common.c
  ipc/milk_waitfor.c
  ipc/milk_srvlog_shm.c
  ipc/milk_srv_manager.c
  shm_prop/milk_shm_prop.c
  shm_prop/milk_shm_prop_stream.c
  shm_prop/milk_shm_prop_fps.c
  shm_prop/milk_shm_prop_proc.c
  shm_prop/milk_shm_cache.c
  core/milk_lexicon.c
  core/milk_argval.c
  core/milk_renames.c
  core/CLIcore_inspect.c
  calc/prog/cli_calc_engines.c
  calc/core/cli_calc_scalar.c
  calc/core/cli_calc_scalar_temporal.c)
target_include_directories(
  milkbash PRIVATE $<TARGET_PROPERTY:milkscript,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_definitions(milkbash PRIVATE MILK_BASH_CLIENT)
if(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
  target_compile_options(milkbash PRIVATE -fopenmp-simd)
endif()
target_link_libraries(
  milkbash PRIVATE milkfps milkcomputeunit milkprocessinfo milkdata m dl rt
                   ${CMAKE_THREAD_LIBS_INIT})
set_target_properties(milkbash PROPERTIES C_VISIBILITY_PRESET hidden INSTALL_RPATH "$ORIGIN")
target_link_options(milkbash PRIVATE "-Wl,--disable-new-dtags" "-Wl,--no-undefined")
install(TARGETS milkbash DESTINATION lib)

# Compute units without C code (milk.calc, milk.script): fpsexec binaries that run calculator statements or
# a bash snippet on each frame; the calculator and milk.bash come from libmilkscript.
foreach(_unit calc script)
  add_executable(milk-fpsexec-milk-${_unit} units/milk_unit_${_unit}.c units/milk_unit.c)
  target_compile_definitions(milk-fpsexec-milk-${_unit} PRIVATE MILK_NO_CLI)
  milk_configure_fpsexec(milk-fpsexec-milk-${_unit} PREFIX milk)
  target_link_libraries(milk-fpsexec-milk-${_unit} PUBLIC milkscript)
endforeach()

# Standalone helper executable (milk-call)
add_executable(milk-call ipc/milk_call_main.c)
target_link_libraries(milk-call PRIVATE milkscript milkprocessinfo milkdata)
target_include_directories(
  milk-call PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                   ${CMAKE_CURRENT_SOURCE_DIR}/../CLIcore)
set_target_properties(milk-call PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
install(TARGETS milk-call DESTINATION bin)
