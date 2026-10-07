set(LIBNAME "milkCOREMODiofits")
set(SRCNAME "COREMOD_iofits")

message("Module CMakeLists.txt ---- SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

file(GLOB SOURCEFILES "*.c" "*.cpp")
file(GLOB INCLUDEFILES "*.h" "*.hpp")
file(GLOB SCRIPTS "scripts/*")

project(lib_${LIBNAME}_project)

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/..")

add_library(${LIBNAME} SHARED ${SOURCEFILES})
# CLIcore does not exist at this tier: reuse the include dirs of libfps
target_include_directories(
  ${LIBNAME} PRIVATE $<TARGET_PROPERTY:milkfps,INCLUDE_DIRECTORIES>)
milk_apply_extensions(${LIBNAME})

# CUSTOM ===========
target_include_directories(
  ${LIBNAME} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
                    $<INSTALL_INTERFACE:include>)
target_link_libraries(${LIBNAME} PUBLIC m milkfps)

milk_add_static_lto(${LIBNAME})
# ==================

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTS} DESTINATION bin)

add_milk_standalone(iofits-saveFITS savefits.c)
add_milk_standalone(iofits-loadfits loadfits.c)
add_milk_standalone(iofits-imgs2cube images2cube.c)

get_property(_stdalones DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  PROPERTY BUILDSYSTEM_TARGETS)
foreach(_t IN LISTS _stdalones)
  get_target_property(_type ${_t} TYPE)
  if(_type STREQUAL "EXECUTABLE")
    target_link_libraries(${_t} PUBLIC ${LIBNAME})
  endif()
endforeach()

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
