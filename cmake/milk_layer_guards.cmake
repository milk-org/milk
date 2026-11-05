# Layer guards (calc_one_engine_plan, D1 / Phase 2).
# Enforces strict upward-only header dependencies across architectural layers:
#   - src/engine, src/coremods, src/prop must never #include a header that exists only under
#     src/calc or src/cli (except the MILK_NO_CLI stub CLIcore_standalone.h).
#   - src/calc must never #include a header that exists only under src/cli.
# Included by the top-level CMakeLists.txt; CTest runs this file in script mode (-DSRC_DIR=...).

# Pre-existing conditional includes in engine/coremods (not active in engine/core builds)
set(MILK_LAYER_GUARD_EXCEPTIONS
    "engine/libfps/tools/milk-fps-info.c:overview_data.h"
    "coremods/COREMOD_memory/stream_TCP_receive.c:CLIcore.h"
    "coremods/COREMOD_memory/stream_UDP_receive.c:CLIcore.h")

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

  if(_problems)
    message(FATAL_ERROR "Layer guards (cmake/milk_layer_guards.cmake):\n${_problems}")
  endif()
  message(STATUS "Layer guards: OK")
  return()
endif()

if(BUILD_TESTING)
  add_test(NAME layer_guards COMMAND ${CMAKE_COMMAND} -DSRC_DIR=${CMAKE_SOURCE_DIR}/src -P
                                     ${CMAKE_CURRENT_LIST_FILE})
endif()
