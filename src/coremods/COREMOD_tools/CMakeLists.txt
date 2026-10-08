set(LIBNAME "milkCOREMODtools")
set(SRCNAME "COREMOD_tools")

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
find_package(PkgConfig REQUIRED)
if(USE_CFITSIO)
  pkg_check_modules(CFITSIO cfitsio)
endif()
if(USE_CFITSIO AND ${CFITSIO_FOUND})
  link_directories(${CFITSIO_LIBRARY_DIRS})
  target_compile_definitions(${LIBNAME} PUBLIC USE_CFITSIO=1)
  target_include_directories(${LIBNAME} PUBLIC ${CFITSIO_INCLUDE_DIRS})
  target_link_directories(${LIBNAME} PUBLIC ${CFITSIO_LIBRARY_DIRS})
endif()

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

# CUSTOM ===========
include(CTest)

set(TESTNAME "CLIfunc-COREMOD_tools-module")
add_test(NAME "${TESTNAME}" COMMAND milk-exec "-T" "m?")
set_property(TEST "${TESTNAME}" PROPERTY LABELS "CLImodule")
set_property(TEST "${TESTNAME}" PROPERTY TIMEOUT 1)
set_property(TEST "${TESTNAME}" PROPERTY PASS_REGULAR_EXPRESSION
                                         "COREMOD_tools")

list(
  APPEND
  commandlist
  "rtprio"
  "tsetpmove"
  "tsetpmoveext"
  "csetpmove"
  "csetandprioext"
  "writef2file"
  "dispim3d"
  "ctsmstats")

foreach(CLIcmdname IN LISTS commandlist)

  set(TESTCLIFUNC "${CLIcmdname}")
  set(TESTNAME "CLIfunc-COREMOD_tools-${TESTCLIFUNC}")
  add_test(NAME "${TESTNAME}" COMMAND milk-exec "-T" "cmd? tools.${CLIcmdname}")
  set_property(TEST "${TESTNAME}" PROPERTY LABELS "CLIfunc")
  set_property(TEST "${TESTNAME}" PROPERTY TIMEOUT 1)
  set_property(TEST "${TESTNAME}" PROPERTY PASS_REGULAR_EXPRESSION "src:")

endforeach()

add_executable(milk-makecsetandrt bin/milk-makecsetandrt.c)
target_include_directories(
  milk-makecsetandrt
  PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)
target_link_libraries(milk-makecsetandrt PRIVATE milkcommon)
install(
  TARGETS milk-makecsetandrt
  DESTINATION bin
  PERMISSIONS
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE
    SETUID)
# ==================
