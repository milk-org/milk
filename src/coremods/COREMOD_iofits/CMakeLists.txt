set(LIBNAME "milkCOREMODiofits")
set(SRCNAME "COREMOD_iofits")

message("Module CMakeLists.txt ---- SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

file(GLOB SOURCEFILES "*.c" "*.cpp")
file(GLOB INCLUDEFILES "*.h" "*.hpp")
file(GLOB SCRIPTS "scripts/*")

project(lib_${LIBNAME}_project)

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/..")

milk_add_runnable_module(${LIBNAME} SOURCES ${SOURCEFILES} COREMOD)
milk_add_fpsexec_stubs(${LIBNAME})

# CUSTOM ===========
# CLIcore does not exist at this tier: reuse the include dirs of libfps
target_include_directories(
  ${LIBNAME}_objs PRIVATE $<TARGET_PROPERTY:milkfps,INCLUDE_DIRECTORIES>)

milk_runnable_module_usage(
  ${LIBNAME} INCLUDES $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
  $<INSTALL_INTERFACE:include>)
# ==================

install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTS} DESTINATION bin)

# CUSTOM ===========
include(CTest)

set(TESTNAME "CLIfunc-COREMOD_iofits-module")
add_test(NAME "${TESTNAME}" COMMAND milk-exec "-T" "m?")
set_property(TEST "${TESTNAME}" PROPERTY LABELS "CLImodule")
set_property(TEST "${TESTNAME}" PROPERTY TIMEOUT 1)
set_property(TEST "${TESTNAME}" PROPERTY PASS_REGULAR_EXPRESSION
                                         "COREMOD_iofits")

list(APPEND commandlist "loadfits" "saveFITS" "breakcube" "imgs2cube")

foreach(CLIcmdname IN LISTS commandlist)

  set(TESTCLIFUNC "${CLIcmdname}")
  set(TESTNAME "CLIfunc-COREMOD_iofits-${TESTCLIFUNC}")
  add_test(NAME "${TESTNAME}" COMMAND milk-exec "-T"
                                      "cmd? iofits.${CLIcmdname}")
  set_property(TEST "${TESTNAME}" PROPERTY LABELS "CLIfunc")
  set_property(TEST "${TESTNAME}" PROPERTY TIMEOUT 1)
  set_property(TEST "${TESTNAME}" PROPERTY PASS_REGULAR_EXPRESSION "src:")

endforeach()
# ==================
