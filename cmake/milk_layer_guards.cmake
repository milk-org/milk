# Layer guards: what each build layer may use. Included by the top-level CMakeLists.txt after the
# engine targets; CTest runs this same file in script mode (-DSRC_DIR=<path/to/src>) as the check.
#   - Includes: src/engine, src/coremods and src/prop never #include a header that exists only under
#     src/calc or src/cli (except the MILK_NO_CLI stub CLIcore_standalone.h); src/calc never
#     #includes one that exists only under src/cli.
#   - Engine libraries: they spawn no shell and drive no tmux, and only the files in
#     MILK_ENGINE_NONPOSIX_SOURCES go beyond POSIX.1-2024 with XSI (docs/arch/unit_lifecycle.md,
#     U13). That list only shrinks: a file leaves it in the change that removes its last use.

# Conditional includes in engine/coremods that predate the guard (not active in engine/core builds)
set(MILK_LAYER_GUARD_EXCEPTIONS
    "engine/libfps/tools/milk-fps-info.c:overview_data.h"
    "coremods/COREMOD_memory/stream_TCP_receive.c:CLIcore.h"
    "coremods/COREMOD_memory/stream_UDP_receive.c:CLIcore.h")

# The MILK_STRICT_POSIX loop below names the same libraries by target
set(MILK_ENGINE_DIRS libmilkcommon libprocessinfo libfps libcomputeunit libmilkdata)

# Engine files that use GNU, BSD or Linux calls; with MILK_STRICT_POSIX=ON every other engine source
# compiles with -D_XOPEN_SOURCE=800 -U_GNU_SOURCE
set(MILK_ENGINE_NONPOSIX_SOURCES
    libcomputeunit/core/fps_lifecycle.c
    libcomputeunit/state/fps_FPCONFloopstep.c
    libfps/io/fps_WriteParameterToDisk.c
    libfps/io/fps_save2disk.c
    libfps/tools/milk-fpsexec-list.c
    libmilkcommon/process/milk_syscalls_linux.c
    libprocessinfo/processinfo_loopstep.c
    libprocessinfo/processtools_trigger.c)

if(CMAKE_SCRIPT_MODE_FILE)
  if(NOT SRC_DIR)
    message(FATAL_ERROR "milk_layer_guards.cmake requires -DSRC_DIR=<path/to/src>")
  endif()

  set(_lower_basenames "CLIcore_standalone.h")
  foreach(_ldir engine coremods prop)
    file(GLOB_RECURSE _lhdrs "${SRC_DIR}/${_ldir}/*.h")
    foreach(_h ${_lhdrs})
      get_filename_component(_bn "${_h}" NAME)
      list(APPEND _lower_basenames "${_bn}")
    endforeach()
  endforeach()
  list(REMOVE_DUPLICATES _lower_basenames)

  set(_calc_basenames "")
  file(GLOB_RECURSE _chdrs "${SRC_DIR}/calc/*.h")
  foreach(_h ${_chdrs})
    get_filename_component(_bn "${_h}" NAME)
    list(APPEND _calc_basenames "${_bn}")
  endforeach()
  list(REMOVE_DUPLICATES _calc_basenames)

  set(_cli_basenames "")
  file(GLOB_RECURSE _clihdrs "${SRC_DIR}/cli/*.h")
  foreach(_h ${_clihdrs})
    get_filename_component(_bn "${_h}" NAME)
    list(APPEND _cli_basenames "${_bn}")
  endforeach()
  list(REMOVE_DUPLICATES _cli_basenames)

  set(_problems "")

  # 1. Check src/engine, src/coremods, src/prop against src/calc and src/cli
  foreach(_dir engine coremods prop)
    file(GLOB_RECURSE _files RELATIVE "${SRC_DIR}" "${SRC_DIR}/${_dir}/*.c"
         "${SRC_DIR}/${_dir}/*.h")
    foreach(_f ${_files})
      file(STRINGS "${SRC_DIR}/${_f}" _inc_lines REGEX "^[ \t]*#[ \t]*include[ \t]+[\"<][^\">]+[\">]")
      foreach(_line ${_inc_lines})
        string(REGEX REPLACE "^[ \t]*#[ \t]*include[ \t]+[\"<]([^\">]+)[\">].*$" "\\1" _inc
                             "${_line}")
        get_filename_component(_bn "${_inc}" NAME)
        list(FIND _lower_basenames "${_bn}" _in_lower)
        list(FIND _calc_basenames "${_bn}" _in_calc)
        list(FIND _cli_basenames "${_bn}" _in_cli)
        list(FIND MILK_LAYER_GUARD_EXCEPTIONS "${_f}:${_bn}" _is_exc)
        if(_in_lower EQUAL -1 AND _is_exc EQUAL -1 AND (NOT _in_calc EQUAL -1 OR NOT _in_cli EQUAL -1))
          string(APPEND _problems
                 "  src/${_f}: includes '${_inc}' (exists only in src/calc or src/cli)\n")
        endif()
      endforeach()
    endforeach()
  endforeach()

  # 2. Check src/calc against src/cli
  file(GLOB_RECURSE _cfiles RELATIVE "${SRC_DIR}" "${SRC_DIR}/calc/*.c" "${SRC_DIR}/calc/*.h")
  foreach(_f ${_cfiles})
    file(STRINGS "${SRC_DIR}/${_f}" _inc_lines REGEX "^[ \t]*#[ \t]*include[ \t]+[\"<][^\">]+[\">]")
    foreach(_line ${_inc_lines})
      string(REGEX REPLACE "^[ \t]*#[ \t]*include[ \t]+[\"<]([^\">]+)[\">].*$" "\\1" _inc
                           "${_line}")
      get_filename_component(_bn "${_inc}" NAME)
      list(FIND _lower_basenames "${_bn}" _in_lower)
      list(FIND _calc_basenames "${_bn}" _in_calc)
      list(FIND _cli_basenames "${_bn}" _in_cli)
      if(_in_lower EQUAL -1 AND _in_calc EQUAL -1 AND NOT _in_cli EQUAL -1)
        string(APPEND _problems "  src/${_f}: includes '${_inc}' (exists only in src/cli)\n")
      endif()
    endforeach()
  endforeach()

  # 3. Check the engine libraries: no shell, no tmux, _GNU_SOURCE only in the listed files
  set(_engine "${SRC_DIR}/engine")
  foreach(_dir ${MILK_ENGINE_DIRS})
    file(GLOB_RECURSE _files RELATIVE "${_engine}" "${_engine}/${_dir}/*.c" "${_engine}/${_dir}/*.h")
    foreach(_f ${_files})
      file(READ "${_engine}/${_f}" _text)
      # milkDebugTools.h defines EXECUTE_SYSTEM_COMMAND for coremods and plugins
      if(NOT _f STREQUAL "libmilkcommon/milkDebugTools.h")
        string(REGEX MATCH "(^|[^A-Za-z0-9_])(system|popen)\\(|EXECUTE_SYSTEM_COMMAND|\"tmux" _hit
                     "${_text}")
        if(_hit)
          string(APPEND _problems "  src/engine/${_f}: spawns a shell or names tmux\n")
        endif()
      endif()
      string(REGEX MATCH "#[ \t]*define[ \t]+_GNU_SOURCE" _gnu "${_text}")
      list(FIND MILK_ENGINE_NONPOSIX_SOURCES "${_f}" _allowed)
      if(_gnu AND _allowed EQUAL -1)
        string(APPEND _problems
               "  src/engine/${_f}: defines _GNU_SOURCE (not in MILK_ENGINE_NONPOSIX_SOURCES)\n")
      endif()
    endforeach()
  endforeach()
  foreach(_f ${MILK_ENGINE_NONPOSIX_SOURCES})
    if(NOT EXISTS "${_engine}/${_f}")
      string(APPEND _problems "  src/engine/${_f}: listed but gone, remove it from its list\n")
    endif()
  endforeach()

  if(_problems)
    message(FATAL_ERROR "Layer guards (cmake/milk_layer_guards.cmake):\n${_problems}")
  endif()
  message(STATUS "Layer guards: OK")
  return()
endif()

option(MILK_STRICT_POSIX "Compile engine sources as POSIX.1-2024 with XSI only (CI check)" OFF)
if(MILK_STRICT_POSIX)
  if(CMAKE_VERSION VERSION_LESS 3.18)
    message(FATAL_ERROR "MILK_STRICT_POSIX needs CMake 3.18 or newer")
  endif()
  set(_engine_dir "${CMAKE_SOURCE_DIR}/src/engine")
  foreach(_tgt milkcommon milkprocessinfo milkfps milkcomputeunit milkdata)
    get_target_property(_srcs ${_tgt} SOURCES)
    get_target_property(_sdir ${_tgt} SOURCE_DIR)
    foreach(_src ${_srcs})
      get_filename_component(_abs "${_src}" ABSOLUTE BASE_DIR "${_sdir}")
      file(RELATIVE_PATH _rel "${_engine_dir}" "${_abs}")
      list(FIND MILK_ENGINE_NONPOSIX_SOURCES "${_rel}" _allowed)
      if(_abs MATCHES "\\.c$" AND _allowed EQUAL -1)
        set_source_files_properties("${_abs}" TARGET_DIRECTORY ${_tgt}
                                    PROPERTIES COMPILE_OPTIONS "-D_XOPEN_SOURCE=800;-U_GNU_SOURCE")
      endif()
    endforeach()
  endforeach()
endif()

if(BUILD_TESTING)
  add_test(NAME layer_guards COMMAND ${CMAKE_COMMAND} -DSRC_DIR=${CMAKE_SOURCE_DIR}/src -P
                                     ${CMAKE_CURRENT_LIST_FILE})
endif()
