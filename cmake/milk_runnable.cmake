# ─────────────────────────────────────────────────────────────────────────────
# milk_runnable.cmake — Modules made of MILK_RUNNABLE units, compiled once
# ─────────────────────────────────────────────────────────────────────────────
#
# See src/engine/libcomputeunit/milk_runnable.h for the C side.
#
# milk_add_runnable_module(<LIBNAME> SOURCES <files...>)
#
#   Compiles the sources ONCE (OBJECT library <LIBNAME>_objs, PIC, -DMILK_NO_CLI)
#   and builds from those objects
#     - the shared library <LIBNAME> (loaded by the CLI, linked by the fpsexecs),
#     - <LIBNAME>_static when USE_STATIC_LTO is ON.
#   Per-module PGO / LTO flags apply to the objects.
#
# milk_add_fpsexec_stubs(<LIBNAME>)
#
#   Scans the module sources for lines such as
#       EXPORT_FPSEXEC(milk, "myunit", runnable_myunit);
#   and, for each, generates a minimal main() in the build tree and builds the
#   executable <prefix>-fpsexec-<id> (prefix: milk or cacao) linked to the module.
#   The macro must be on a single line. Ids of sources dropped by an unmet
#   MILK_CMAKE_MANDATE_* directive get no executable (a [SKIP] line is printed).
#
# A source dropped by an unmet mandate also vanishes from <LIBNAME>: the module registry
# (MILK_MODULE_RUNNABLES) must list its runnables through MILK_RUNNABLE_WEAK_DECL, and the file
# holding the registry must not carry a mandate itself (FATAL_ERROR otherwise).
# See docs/developer/dependency_system.md.
#
# Call both after milk_standalone.cmake and milk_static_lto.cmake are included.

# Libraries every runnable module needs (include paths + DT_NEEDED of the .so)
set(_MILK_RUNNABLE_LIBS
    milkfps
    milkcomputeunit
    milkprocessinfo
    ImageStreamIO
    milkdata
    milkCOREMODmemory
    milkCOREMODtools
    milkCOREMODarith
    m)

set(_MILK_FPSEXEC_STUB_TEMPLATE [=[
#include "milk_runnable.h"

extern const MILK_RUNNABLE @UNIT@;

int main(int argc, char *argv[])
{
    MILK_EMBED_BUILD_TAG();
    return MILK_FPSEXEC_MAIN(argc, argv, &@UNIT@);
}
]=])

function(milk_add_runnable_module LIBNAME)
  cmake_parse_arguments(ARG "" "" "SOURCES" ${ARGN})

  set(_objs ${LIBNAME}_objs)
  add_library(${_objs} OBJECT ${ARG_SOURCES})
  set_target_properties(${_objs} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(${_objs} PRIVATE MILK_NO_CLI)
  target_include_directories(
    ${_objs}
    PRIVATE ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/src/cli/CLIcore
            ${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/..)
  if(TARGET CLIcore)
    target_include_directories(
      ${_objs} PRIVATE $<TARGET_PROPERTY:CLIcore,INTERFACE_INCLUDE_DIRECTORIES>)
  endif()
  target_link_libraries(${_objs} PUBLIC ${_MILK_RUNNABLE_LIBS})
  milk_apply_extensions(${_objs})
  milk_pgo_target(${_objs})

  # The registry file cannot be dropped: it would leave an empty module
  get_target_property(_srcdir ${_objs} SOURCE_DIR)
  get_target_property(_objsrcs ${_objs} SOURCES)
  foreach(_src IN LISTS _objsrcs)
    if(NOT IS_ABSOLUTE "${_src}")
      set(_src "${_srcdir}/${_src}")
    endif()
    if(NOT EXISTS "${_src}")
      continue()
    endif()
    get_source_file_property(_hfo "${_src}" HEADER_FILE_ONLY)
    if(_hfo)
      file(STRINGS "${_src}" _reg REGEX "^[ \t]*MILK_MODULE(_RUNNABLES)?[ \t]*\\(")
      if(_reg)
        message(FATAL_ERROR "${_src} defines the module registry but is dropped by an unmet "
                            "MILK_CMAKE_MANDATE_*; move the mandate to another file")
      endif()
    endif()
  endforeach()

  add_library(${LIBNAME} SHARED $<TARGET_OBJECTS:${_objs}>)
  # Link requirements (including those added by milk_apply_extensions) of the objects
  get_target_property(_obj_libs ${_objs} LINK_LIBRARIES)
  get_target_property(_obj_ilibs ${_objs} INTERFACE_LINK_LIBRARIES)
  foreach(_l IN LISTS _obj_libs _obj_ilibs)
    if(_l)
      target_link_libraries(${LIBNAME} PUBLIC ${_l})
    endif()
  endforeach()
  milk_pgo_target(${LIBNAME})

  milk_add_static_lto(${LIBNAME} OBJECTS_FROM ${_objs})
  if(USE_STATIC_LTO AND TARGET ${LIBNAME}_static)
    # milk_apply_extensions() cannot scan an archive built from objects: forward the extension
    # libraries (not the shared framework libs, the archive links the _static ones).
    set(_ext_libs ${_obj_libs} ${_obj_ilibs})
    list(REMOVE_ITEM _ext_libs ${_MILK_RUNNABLE_LIBS})
    foreach(_l IN LISTS _ext_libs)
      if(_l)
        target_link_libraries(${LIBNAME}_static PUBLIC ${_l})
      endif()
    endforeach()
  endif()

  install(TARGETS ${LIBNAME} EXPORT milkTargets DESTINATION lib)
endfunction()


function(milk_add_fpsexec_stubs LIBNAME)
  set(_objs ${LIBNAME}_objs)
  get_target_property(_sources ${_objs} SOURCES)
  get_target_property(_srcdir ${_objs} SOURCE_DIR)

  foreach(_src IN LISTS _sources)
    if(NOT IS_ABSOLUTE "${_src}")
      set(_src "${_srcdir}/${_src}")
    endif()
    if(NOT EXISTS "${_src}")
      continue()
    endif()
    get_source_file_property(_hfo "${_src}" HEADER_FILE_ONLY)
    # Re-run CMake when a source (and thus its EXPORT_FPSEXEC lines) changes
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_src}")

    # file(READ) + MATCHALL: file(STRINGS) would split lines at the ';' ending the macro
    file(READ "${_src}" _text)
    string(REGEX MATCHALL "(^|\n)[ \t]*EXPORT_FPSEXEC[ \t]*\\([^\n]*\\)" _hits "${_text}")
    foreach(_line IN LISTS _hits)
      if(NOT _line MATCHES
         "EXPORT_FPSEXEC[ \t]*\\([ \t]*([A-Za-z0-9_]+)[ \t]*,[ \t]*\"([^\"]+)\"[ \t]*,[ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*\\)"
      )
        message(FATAL_ERROR "Malformed EXPORT_FPSEXEC in ${_src}: ${_line}")
      endif()
      set(_prefix ${CMAKE_MATCH_1})
      set(_id ${CMAKE_MATCH_2})
      set(_unit ${CMAKE_MATCH_3})
      if(NOT _prefix STREQUAL "milk" AND NOT _prefix STREQUAL "cacao")
        message(FATAL_ERROR "EXPORT_FPSEXEC prefix must be milk or cacao: ${_line}")
      endif()

      set(_exe ${_prefix}-fpsexec-${_id})
      if(_hfo)
        get_filename_component(_src_name "${_src}" NAME)
        message(STATUS "[SKIP] fpsexec ${_exe}: ${_src_name} (unmet MANDATE_*)")
        continue()
      endif()
      if(TARGET ${_exe})
        message(FATAL_ERROR "Duplicate fpsexec target ${_exe} (${_src})")
      endif()

      string(REPLACE "@UNIT@" "${_unit}" _content "${_MILK_FPSEXEC_STUB_TEMPLATE}")
      set(_stub ${CMAKE_CURRENT_BINARY_DIR}/fpsexec_stubs/${_exe}.c)
      file(GENERATE OUTPUT ${_stub} CONTENT "${_content}")

      add_executable(${_exe} ${_stub})
      target_include_directories(${_exe} PRIVATE ${CMAKE_SOURCE_DIR}/src)
      if(_prefix STREQUAL "cacao")
        target_include_directories(
          ${_exe} PRIVATE ${CMAKE_SOURCE_DIR}/plugins/milk-extra-src)
      endif()
      if(USE_STATIC_LTO)
        target_link_libraries(${_exe} PUBLIC ${_MILK_STANDALONE_STATIC_LIBS})
      else()
        target_link_libraries(${_exe} PUBLIC ${_MILK_STANDALONE_LIBS})
      endif()
      milk_link_module(${_exe} ${LIBNAME})
      milk_pgo_target(${_exe})
      milk_lto_target(${_exe})
      milk_build_tag_target(${_exe})
      install(TARGETS ${_exe} DESTINATION bin)
    endforeach()
  endforeach()
endfunction()
