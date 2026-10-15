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

# ── Public API ───────────────────────────────────────────────────────────────

function(milk_add_runnable_module LIBNAME)
  cmake_parse_arguments(ARG "" "" "SOURCES;LINK_LIBS" ${ARGN})
  set(_objs ${LIBNAME}_objs)

  _milk_runnable_make_objects(${_objs} ${ARG_SOURCES})
  _milk_runnable_check_registry(${_objs})

  add_library(${LIBNAME} SHARED $<TARGET_OBJECTS:${_objs}>)
  set_target_properties(${LIBNAME} PROPERTIES MILK_OBJECTS_TARGET ${_objs})
  _milk_forward_link_libs(${_objs} ${LIBNAME})
  milk_pgo_target(${LIBNAME})

  milk_add_static_lto(${LIBNAME} OBJECTS_FROM ${_objs} LINK_LIBS ${ARG_LINK_LIBS})
  if(USE_STATIC_LTO AND TARGET ${LIBNAME}_static)
    # The archive links the _static framework libs, not the shared ones
    _milk_forward_link_libs(${_objs} ${LIBNAME}_static EXCLUDE ${_MILK_RUNNABLE_LIBS})
  endif()

  install(TARGETS ${LIBNAME} EXPORT milkTargets DESTINATION lib)
endfunction()


function(milk_add_fpsexec_stubs LIBNAME)
  get_target_property(_objs ${LIBNAME} MILK_OBJECTS_TARGET)
  _milk_target_source_files(${_objs} _sources)

  foreach(_src IN LISTS _sources)
    # Re-run CMake when a source (and thus its EXPORT_FPSEXEC lines) changes
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_src}")
    get_source_file_property(_dropped "${_src}" HEADER_FILE_ONLY)

    _milk_parse_fpsexec_exports("${_src}" _exports)
    foreach(_export IN LISTS _exports)
      string(REPLACE "|" ";" _export "${_export}")
      list(GET _export 0 _prefix)
      list(GET _export 1 _id)
      list(GET _export 2 _unit)
      if(_dropped)
        get_filename_component(_src_name "${_src}" NAME)
        message(STATUS "[SKIP] fpsexec ${_prefix}-fpsexec-${_id}: ${_src_name} (unmet MANDATE_*)")
        continue()
      endif()
      _milk_add_fpsexec_stub(${LIBNAME} ${_prefix} ${_id} ${_unit} "${_src}")
    endforeach()
  endforeach()
endfunction()


# ── Private helpers ──────────────────────────────────────────────────────────

# Existing sources of <target> as absolute paths
function(_milk_target_source_files target out)
  get_target_property(_srcdir ${target} SOURCE_DIR)
  get_target_property(_srcs ${target} SOURCES)
  set(_files "")
  foreach(_src IN LISTS _srcs)
    if(NOT IS_ABSOLUTE "${_src}")
      set(_src "${_srcdir}/${_src}")
    endif()
    if(EXISTS "${_src}")
      list(APPEND _files "${_src}")
    endif()
  endforeach()
  set(${out} "${_files}" PARENT_SCOPE)
endfunction()


# OBJECT library <objs>: PIC, -DMILK_NO_CLI, framework libs, extensions, PGO
function(_milk_runnable_make_objects objs)
  add_library(${objs} OBJECT ${ARGN})
  set_target_properties(${objs} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(${objs} PRIVATE MILK_NO_CLI)
  target_include_directories(
    ${objs}
    PRIVATE ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/src/cli/CLIcore
            ${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/..)
  if(TARGET CLIcore)
    target_include_directories(
      ${objs} PRIVATE $<TARGET_PROPERTY:CLIcore,INTERFACE_INCLUDE_DIRECTORIES>)
  endif()
  target_link_libraries(${objs} PUBLIC ${_MILK_RUNNABLE_LIBS})
  milk_apply_extensions(${objs})
  milk_pgo_target(${objs})
endfunction()


# The registry file cannot be dropped by an unmet mandate: it would leave an empty module
function(_milk_runnable_check_registry objs)
  _milk_target_source_files(${objs} _sources)
  foreach(_src IN LISTS _sources)
    get_source_file_property(_dropped "${_src}" HEADER_FILE_ONLY)
    if(NOT _dropped)
      continue()
    endif()
    file(STRINGS "${_src}" _reg REGEX "^[ \t]*MILK_MODULE(_RUNNABLES)?[ \t]*\\(")
    if(_reg)
      message(FATAL_ERROR "${_src} defines the module registry but is dropped by an unmet "
                          "MILK_CMAKE_MANDATE_*; move the mandate to another file")
    endif()
  endforeach()
endfunction()


# Link requirements of <from> (including those added by milk_apply_extensions) applied to <to>;
# an archive built from objects cannot be scanned by milk_apply_extensions itself.
function(_milk_forward_link_libs from to)
  cmake_parse_arguments(ARG "" "" "EXCLUDE" ${ARGN})
  get_target_property(_libs ${from} LINK_LIBRARIES)
  get_target_property(_ilibs ${from} INTERFACE_LINK_LIBRARIES)
  set(_all ${_libs} ${_ilibs})
  if(ARG_EXCLUDE)
    list(REMOVE_ITEM _all ${ARG_EXCLUDE})
  endif()
  foreach(_l IN LISTS _all)
    if(_l)
      target_link_libraries(${to} PUBLIC ${_l})
    endif()
  endforeach()
endfunction()


# Lists "<prefix>|<id>|<unit>" for each EXPORT_FPSEXEC line of <src>
function(_milk_parse_fpsexec_exports src out)
  # file(READ) + MATCHALL: file(STRINGS) would split lines at the ';' ending the macro
  file(READ "${src}" _text)
  string(REGEX MATCHALL "(^|\n)[ \t]*EXPORT_FPSEXEC[ \t]*\\([^\n]*\\)" _hits "${_text}")
  set(_exports "")
  foreach(_line IN LISTS _hits)
    if(NOT _line MATCHES
       "EXPORT_FPSEXEC[ \t]*\\([ \t]*([A-Za-z0-9_]+)[ \t]*,[ \t]*\"([^\"]+)\"[ \t]*,[ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*\\)"
    )
      message(FATAL_ERROR "Malformed EXPORT_FPSEXEC in ${src}: ${_line}")
    endif()
    if(NOT CMAKE_MATCH_1 STREQUAL "milk" AND NOT CMAKE_MATCH_1 STREQUAL "cacao")
      message(FATAL_ERROR "EXPORT_FPSEXEC prefix must be milk or cacao: ${_line}")
    endif()
    list(APPEND _exports "${CMAKE_MATCH_1}|${CMAKE_MATCH_2}|${CMAKE_MATCH_3}")
  endforeach()
  set(${out} "${_exports}" PARENT_SCOPE)
endfunction()


# Generates the main() stub of <unit> and builds <prefix>-fpsexec-<id> linked to <LIBNAME>
function(_milk_add_fpsexec_stub LIBNAME prefix id unit src)
  set(_exe ${prefix}-fpsexec-${id})
  if(TARGET ${_exe})
    message(FATAL_ERROR "Duplicate fpsexec target ${_exe} (${src})")
  endif()

  string(REPLACE "@UNIT@" "${unit}" _content "${_MILK_FPSEXEC_STUB_TEMPLATE}")
  set(_stub ${CMAKE_CURRENT_BINARY_DIR}/fpsexec_stubs/${_exe}.c)
  file(GENERATE OUTPUT ${_stub} CONTENT "${_content}")

  add_executable(${_exe} ${_stub})
  milk_configure_fpsexec(${_exe} PREFIX ${prefix} MODULE ${LIBNAME})
endfunction()
