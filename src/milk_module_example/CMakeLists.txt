# ===========================================
# milk_module_example — Module template
# ===========================================
#
# Copy this CMakeLists.txt as the starting point for new modules.

set(LIBNAME "milk_module_example") # the name of the target: lib${LIBNAME}.so
set(SRCNAME "milk_module_example") # the name of the main module source: ${SRCNAME.c}

message("Module CMakeLists.txt ---- SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

# CUSTOM ===========
# Module-specific external dependencies, e.g.:
# find_package(PkgConfig REQUIRED)
# pkg_check_modules(FFTW REQUIRED fftw3)
# ==================

# -- Source files ---------------------------------
# Collect all the h files and c/cpp files at the source of the project
# auxiliaries are better included in an internal/ subfolder.
# Extra executables with their own main() belong in bin/ (not globbed).
file(GLOB SOURCEFILES "*.c" "*.cpp")
file(GLOB INCLUDEFILES "*.h" "*.hpp")

# Scripts installed to ${prefix}/bin/
# All the files in scripts/* are considered to be executable scripts
file(GLOB SCRIPTS "scripts/*")

# -- Library --------------------------------------
project(lib_${LIBNAME}_project)

include_directories("${PROJECT_SOURCE_DIR}/src") # milk/src
include_directories("${PROJECT_SOURCE_DIR}/..") # milk/src or milk/plugins

# Sources are compiled once (-DMILK_NO_CLI) into the shared library, which the CLI loads and the
# generated fpsexec stubs link. Requests/mandates are resolved by milk_apply_extensions().
milk_add_runnable_module(${LIBNAME} SOURCES ${SOURCEFILES})

# One milk-fpsexec-<id> per EXPORT_FPSEXEC(milk, "<id>", unit) line in the sources
milk_add_fpsexec_stubs(${LIBNAME})

# CUSTOM ===========
# Module-specific compile definitions and external libraries, e.g.:
# set_target_properties(
#   ${LIBNAME} PROPERTIES COMPILE_FLAGS
#   "-DFFTCONFIGDIR=\\\"${PROJECT_SOURCE_DIR}/config\\\"")
# target_link_libraries(${LIBNAME} PUBLIC ${FFTW_LIBRARIES} ${FFTWF_LIBRARIES})
# ==================

install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTS} DESTINATION bin)

# -- Executables ----------------------------------
# milk-fpsexec-clitest, milk-fpsexec-imsum2 and milk-fpsexec-streamprocess are generated
# from the EXPORT_FPSEXEC lines of the sources (see cmake/milk_runnable.cmake).

# CUSTOM ===========
# Extra executables (sources in bin/) and per-executable links, e.g.:
# add_executable(milk-mytool bin/milk-mytool.c)
# target_link_libraries(milk-mytool PRIVATE ImageStreamIO)
# install(TARGETS milk-mytool DESTINATION bin)
# target_link_libraries(milk-fpsexec-clitest PUBLIC otherlib)
# ==================


# -- Tests ----------------------------------------
# Smoke tests: verify template standalones link correctly and run without segfault.
add_test(NAME template-clitest-h1 COMMAND milk-fpsexec-clitest -h1)
add_test(NAME template-imsum2-h1 COMMAND milk-fpsexec-imsum2 -h1)
add_test(NAME template-streamprocess-h1 COMMAND milk-fpsexec-streamprocess -h1)
