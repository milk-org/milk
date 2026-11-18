# milk engine libraries.
#
# Added by milk's root CMakeLists.txt and by pyMilk's BUNDLED build
# (python/pyMilk/bindings/CMakeLists.txt). It must therefore configure without
# anything from the root CMakeLists besides options (USE_STATIC_LTO, USE_CUDA, ...).
# USE_STATIC_LTO also builds the <lib>_static archives; pyMilk BUNDLED links those.

get_filename_component(MILK_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." REALPATH)

include(${MILK_SOURCE_DIR}/cmake/milk_version.cmake)
if(NOT MILK_GIT_COMMIT)
  set(MILK_GIT_COMMIT "unknown")
endif()

# Configure a header file to pass some of the CMake settings to the source code.
# USE THE BUILD TREE -- allows to preserve multiple builds with different configs.
configure_file("${MILK_SOURCE_DIR}/src/milk_config.h.in"
               "${CMAKE_BINARY_DIR}/config/milk_config.h")

# Includes the engine sources rely on. milk's root CMakeLists owns them for its
# whole tree; they are only set here when the engine is built without it.
if(NOT PROJECT_NAME STREQUAL "milk")
  include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/libcomputeunit
    ${MILK_SOURCE_DIR}/src/cli/CLIcore # CLIcore_standalone.h (MILK_NO_CLI)
    ${CMAKE_BINARY_DIR}/config)
endif()

if(NOT TARGET ImageStreamIO)
  add_subdirectory(ImageStreamIO)
endif()
add_subdirectory(libmilkcommon)
add_subdirectory(libfps)
add_subdirectory(libcomputeunit)
add_subdirectory(libfpsseq)
add_subdirectory(libmilkdata)
add_subdirectory(libprocessinfo)
