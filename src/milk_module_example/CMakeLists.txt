# ===========================================
# milk_module_example — Module template
# ===========================================
#
# Copy this CMakeLists.txt as the starting point for new modules.

set(LIBNAME "milk_module_example") # the name of the target: lib${LIBNAME}.so
set(SRCNAME "milk_module_example") # the name of the main module source: ${SRCNAME.c}

message("Module CMakeLists.txt ---- SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

# -- Source files ---------------------------------
# Collect all the h files and c files at the source of the project
# auxiliaries are better included in an internal/ subfolder.
file(GLOB SOURCEFILES "*.c")
file(GLOB INCLUDEFILES "*.h")

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

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTS} DESTINATION bin)

# -- Standalone executables -----------------------
# V2 pattern: each is a single source file compiled with -DFPS_STANDALONE.
# add_milk_standalone() handles linking, include paths, and install.
# ${LIBNAME} is linked by default.

add_milk_standalone(clitest examplefunc_fps_cli_poc.c)
add_milk_standalone(imsum2 examplefunc2_FPS.c)
add_milk_standalone(streamprocess examplefunc4_streamprocess.c)

# In case of optional dependency management, this block handles removing
# the standalones that don't meet their mandated dependencies.
get_property(_stdalones DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  PROPERTY BUILDSYSTEM_TARGETS)
foreach(_t IN LISTS _stdalones)
  get_target_property(_type ${_t} TYPE)
  if(_type STREQUAL "EXECUTABLE")
    target_link_libraries(${_t} PUBLIC ${LIBNAME})
  endif()
endforeach()


# -- Tests ----------------------------------------
# Smoke tests: verify template standalones link correctly and run without segfault.
add_test(NAME template-clitest-h1 COMMAND milk-fpsexec-clitest -h1)
add_test(NAME template-imsum2-h1 COMMAND milk-fpsexec-imsum2 -h1)
add_test(NAME template-streamprocess-h1 COMMAND milk-fpsexec-streamprocess -h1)
