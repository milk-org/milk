# ═══════════════════════════════════════════
#  milk_module_example — Module template
# ═══════════════════════════════════════════
#
# Copy this CMakeLists.txt as the starting point
# for new modules. See also examplefunc_fps_cli_poc.c
# for the compute unit template.
#

set(LIBNAME "milk_module_example")

# ── Source files ─────────────────────────────────
# Dual-mode files: compiled into ${LIBNAME} for CLI
# use, AND as standalone executables (with
# -DFPS_STANDALONE). Single-mode files should NOT
# appear in add_*_standalone() calls below.

set(SOURCEFILES
	examplefunc1.c
	examplefunc2_FPS.c
	examplefunc3_updatestreamloop.c
	examplefunc4_streamprocess.c
	examplefunc_fps_cli_poc.c
)

# Headers installed to ${prefix}/include/${SRCNAME}/
set(INCLUDEFILES
)

# Scripts installed to ${prefix}/bin/
set(SCRIPTFILES
	scripts/milk-tutorial
	scripts/milk-streampoke
	scripts/milk-semtestspeed
)

set(LINKLIBS
	CLIcore
)


# ── Library ──────────────────────────────────────
# Shared library for CLI use (loaded by milk-cli).

set(NAME0 "${CMAKE_CURRENT_SOURCE_DIR}")
string(REPLACE "/" " " NAME1 ${NAME0})
string(REGEX MATCH "[a-zA-Z0-9_]+$" SRCNAME "${NAME0}")
message(" SRCNAME = ${SRCNAME}")

project(lib_${LIBNAME}_project)

add_library(${LIBNAME} SHARED ${SRCNAME}.c ${SOURCEFILES})
target_include_directories(${LIBNAME}
    PRIVATE ${PROJECT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${LIBNAME}
    PRIVATE ${LINKLIBS} milkfpsCLI)

install(TARGETS ${LIBNAME} EXPORT milkTargets DESTINATION lib)
install(FILES ${SRCNAME}.h ${INCLUDEFILES}
    DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTFILES} DESTINATION bin)


# ── Standalone executables ───────────────────────
# V2 pattern: each is a single source file compiled
# with -DFPS_STANDALONE. add_milk_standalone()
# handles linking, include paths, and install.
#
# ${LIBNAME} is NOT linked by default. If a
# standalone needs module-lib symbols, add:
#   target_link_libraries(milk-fpsexec-… PUBLIC ${LIBNAME})

add_milk_standalone(clitest examplefunc_fps_cli_poc.c)
add_milk_standalone(imsum2 examplefunc2_FPS.c)
add_milk_standalone(streamprocess examplefunc4_streamprocess.c)


# ── Tests ────────────────────────────────────────
# Smoke tests: verify template standalones link
# correctly and run without segfault.

add_test(NAME template-clitest-h1
         COMMAND milk-fpsexec-clitest -h1)
add_test(NAME template-imsum2-h1
         COMMAND milk-fpsexec-imsum2 -h1)
add_test(NAME template-streamprocess-h1
         COMMAND milk-fpsexec-streamprocess -h1)
