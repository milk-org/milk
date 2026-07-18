# library name
set(LIBNAME "milkcompilertest")
set(SRCNAME "compilertest")

set(SOURCEFILES ${SRCNAME}.c mandate_cuda.c request_cuda.c)

set(INCLUDEFILES ${SRCNAME}.h mandate_cuda.h request_cuda.h)

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

message("---- module compilertest ------------------")

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTS} DESTINATION bin)

add_milk_standalone(compilertest-requestcuda request_cuda.c)
add_milk_standalone(compilertest-mandatecuda mandate_cuda.c)
add_milk_standalone(compilertest-mandatelapacke mandate_lapacke.c)
