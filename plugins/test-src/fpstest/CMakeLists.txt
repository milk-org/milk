# library name
set(LIBNAME "milkfpstest")
set(SRCNAME "fpstest")

file(GLOB SOURCEFILES "*.c")
file(GLOB INCLUDEFILES "*.h")
file(GLOB SCRIPTS "scripts/*")

# DEFAULT SETTINGS Do not change unless needed
# =====================================================================

project(lib_${LIBNAME}_project)

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/..")

add_library(${LIBNAME} SHARED ${SOURCEFILES})
target_include_directories(
  ${LIBNAME} PRIVATE $<TARGET_PROPERTY:CLIcore,INTERFACE_INCLUDE_DIRECTORIES>)

milk_apply_extensions(${LIBNAME})
# Library can be compiled from multiple source files Convention: the main souce
# file is named <libname>.c
#

message("---- module fpstest ------------------")

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTS} DESTINATION bin)

add_milk_standalone(fpstest-fpssynctest fps_sync_test.c)
