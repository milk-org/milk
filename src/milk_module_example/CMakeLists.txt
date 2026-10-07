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

add_library(${LIBNAME} SHARED ${SOURCEFILES})
target_include_directories(
  ${LIBNAME} PRIVATE $<TARGET_PROPERTY:CLIcore,INTERFACE_INCLUDE_DIRECTORIES>)

# Process the MILK dependency system: resolve MILK_CMAKE_REQUEST and MILK_CMAKE_MANDATE
# And apply appropriate linkage
milk_apply_extensions(${LIBNAME})

# CUSTOM ===========
# Module-specific compile definitions and external libraries, e.g.:
# set_target_properties(
#   ${LIBNAME} PROPERTIES COMPILE_FLAGS
#   "-DFFTCONFIGDIR=\\\"${PROJECT_SOURCE_DIR}/config\\\"")
# target_link_libraries(${LIBNAME} PUBLIC ${FFTW_LIBRARIES} ${FFTWF_LIBRARIES})
# ==================

# Static archive for USE_STATIC_LTO builds (see cmake/milk_static_lto.cmake).
# The function is a no-op unless USE_STATIC_LTO is ON.
# milk_add_static_lto(${LIBNAME}
#   LINK_LIBS <static deps>
#   CFITSIO_LINK_LIBS <static deps needing CFITSIO>)

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTS} DESTINATION bin)

# -- Standalone executables -----------------------
# V2 pattern: each is a single source file compiled with -DFPS_STANDALONE.
# add_milk_standalone() handles linking, include paths, and install.
# ${LIBNAME} is linked by the loop below.

add_milk_standalone(clitest examplefunc_fps_cli_poc.c)
add_milk_standalone(imsum2 examplefunc2_FPS.c)
add_milk_standalone(streamprocess examplefunc4_streamprocess.c)

# Link the module library into every standalone of this directory.
get_property(_stdalones DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  PROPERTY BUILDSYSTEM_TARGETS)
foreach(_t IN LISTS _stdalones)
  get_target_property(_type ${_t} TYPE)
  if(_type STREQUAL "EXECUTABLE")
    target_link_libraries(${_t} PUBLIC ${LIBNAME})
  endif()
endforeach()

# CUSTOM ===========
# Extra executables (sources in bin/) and per-standalone links, e.g.:
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
