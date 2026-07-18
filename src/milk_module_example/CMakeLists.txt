# ═══════════════════════════════════════════
# milk_module_example — Module template
# ═══════════════════════════════════════════
#
# Copy this CMakeLists.txt as the starting point for new modules. See also
# examplefunc_fps_cli_poc.c for the compute unit template.

set(LIBNAME "milk_module_example")
set(SRCNAME "milk_module_example") # <- if you <module.c> file has a different name

message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

# ── Source files ─────────────────────────────────
# Dual-mode files: compiled into ${LIBNAME} for CLI use,
# AND as standalone executables (with -DFPS_STANDALONE).
# Single-mode files should NOT appear in add_*_standalone() calls below.
# Collect all the h files and c files
file(GLOB SOURCEFILES "*.c")
file(GLOB INCLUDEFILES "*.h")

# Scripts installed to ${prefix}/bin/
# All the files in scripts/* are considered to be executable scripts
file(GLOB SCRIPTS "scripts/*")

# ── Library ──────────────────────────────────────
# Shared library for CLI use (loaded by milk-cli).

project(lib_${LIBNAME}_project)

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/..")

add_library(${LIBNAME} SHARED ${SOURCEFILES})
target_include_directories(
  ${LIBNAME} PRIVATE $<TARGET_PROPERTY:CLIcore,INTERFACE_INCLUDE_DIRECTORIES>)

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTFILES} DESTINATION bin)

# ── Standalone executables ───────────────────────
# V2 pattern: each is a single source file compiled with -DFPS_STANDALONE.
# add_milk_standalone() handles linking, include paths, and install.
# ${LIBNAME} is linked by default.

add_milk_standalone(clitest examplefunc_fps_cli_poc.c)
add_milk_standalone(imsum2 examplefunc2_FPS.c)
add_milk_standalone(streamprocess examplefunc4_streamprocess.c)

# In case of optional dependency management, this block handles removing
# the standalones that don't meet their needed dependencies.
get_property(_stdalones DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  PROPERTY BUILDSYSTEM_TARGETS)
foreach(_t IN LISTS _stdalones)
  get_target_property(_type ${_t} TYPE)
  if(_type STREQUAL "EXECUTABLE")
    target_link_libraries(${_t} PUBLIC ${LIBNAME})
  endif()
endforeach()


# ── Tests ────────────────────────────────────────
# Smoke tests: verify template standalones link correctly and run without segfault.

add_test(NAME template-clitest-h1 COMMAND milk-fpsexec-clitest -h1)
add_test(NAME template-imsum2-h1 COMMAND milk-fpsexec-imsum2 -h1)
add_test(NAME template-streamprocess-h1 COMMAND milk-fpsexec-streamprocess -h1)
