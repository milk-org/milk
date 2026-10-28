# Engine guards (unit-lifecycle plan, U13). The engine libraries spawn no shell and drive no tmux,
# and only the files listed below go beyond POSIX.1-2024 with XSI. The list only shrinks: a file
# leaves it in the change that removes its last use. Included by the top-level CMakeLists.txt;
# CTest runs this same file in script mode (-DENGINE_DIR=...) as the check.

set(MILK_ENGINE_DIRS libmilkcommon libprocessinfo libfps libcomputeunit)

# Files that use GNU, BSD or Linux calls; with MILK_STRICT_POSIX=ON every other engine source
# compiles with -D_XOPEN_SOURCE=800 -U_GNU_SOURCE
set(MILK_ENGINE_NONPOSIX_SOURCES
    libcomputeunit/core/fps_lifecycle.c
    libcomputeunit/state/fps_FPCONFloopstep.c
    libfps/io/fps_WriteParameterToDisk.c
    libfps/io/fps_save2disk.c
    libfps/tools/milk-fpsexec-list.c
    libmilkcommon/milk_compiler.c
    libmilkcommon/milk_os_linux.c
    libmilkcommon/milk_spawn.c
    libprocessinfo/processinfo_loopstep.c
    libprocessinfo/processtools_trigger.c)

if(CMAKE_SCRIPT_MODE_FILE)
  set(_problems "")
  foreach(_dir ${MILK_ENGINE_DIRS})
    file(GLOB_RECURSE _files RELATIVE "${ENGINE_DIR}" "${ENGINE_DIR}/${_dir}/*.c"
         "${ENGINE_DIR}/${_dir}/*.h")
    foreach(_f ${_files})
      file(READ "${ENGINE_DIR}/${_f}" _text)
      # milkDebugTools.h defines EXECUTE_SYSTEM_COMMAND for coremods and plugins
      if(NOT _f STREQUAL "libmilkcommon/milkDebugTools.h")
        string(REGEX MATCH "(^|[^A-Za-z0-9_])(system|popen)\\(|EXECUTE_SYSTEM_COMMAND|\"tmux" _hit
                     "${_text}")
        if(_hit)
          string(APPEND _problems "  ${_f}: spawns a shell or names tmux\n")
        endif()
      endif()
      string(REGEX MATCH "#[ \t]*define[ \t]+_GNU_SOURCE" _gnu "${_text}")
      list(FIND MILK_ENGINE_NONPOSIX_SOURCES "${_f}" _allowed)
      if(_gnu AND _allowed EQUAL -1)
        string(APPEND _problems "  ${_f}: defines _GNU_SOURCE (not in MILK_ENGINE_NONPOSIX_SOURCES)\n")
      endif()
    endforeach()
  endforeach()
  foreach(_f ${MILK_ENGINE_NONPOSIX_SOURCES})
    if(NOT EXISTS "${ENGINE_DIR}/${_f}")
      string(APPEND _problems "  ${_f}: listed but gone, remove it from its list\n")
    endif()
  endforeach()
  if(_problems)
    message(FATAL_ERROR "Engine guards (cmake/milk_engine_guards.cmake):\n${_problems}")
  endif()
  message(STATUS "Engine guards: OK")
  return()
endif()

option(MILK_STRICT_POSIX "Compile engine sources as POSIX.1-2024 with XSI only (CI check)" OFF)
if(MILK_STRICT_POSIX)
  if(CMAKE_VERSION VERSION_LESS 3.18)
    message(FATAL_ERROR "MILK_STRICT_POSIX needs CMake 3.18 or newer")
  endif()
  set(_engine_dir "${CMAKE_SOURCE_DIR}/src/engine")
  foreach(_tgt milkcommon milkprocessinfo milkfps milkcomputeunit)
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
  add_test(NAME engine_guards COMMAND ${CMAKE_COMMAND} -DENGINE_DIR=${CMAKE_SOURCE_DIR}/src/engine -P
                                      ${CMAKE_CURRENT_LIST_FILE})
endif()
