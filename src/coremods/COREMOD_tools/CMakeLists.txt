set(LIBNAME "milkCOREMODtools")
set(SRCNAME "COREMOD_tools")

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

find_package(PkgConfig REQUIRED)
if(USE_CFITSIO)
  pkg_check_modules(CFITSIO cfitsio)
endif()
if(USE_CFITSIO AND ${CFITSIO_FOUND})
  milk_runnable_module_usage(
    ${LIBNAME}
    DEFINITIONS USE_CFITSIO=1
    INCLUDES ${CFITSIO_INCLUDE_DIRS}
    LINK_DIRS ${CFITSIO_LIBRARY_DIRS})
endif()

milk_runnable_module_usage(
  ${LIBNAME} INCLUDES $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
  $<INSTALL_INTERFACE:include>)
# ==================

install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTS} DESTINATION bin)

# CUSTOM ===========
include(CTest)

set(TESTNAME "CLIfunc-COREMOD_tools-module")
add_test(NAME "${TESTNAME}" COMMAND milk-exec "-T" "@help tools")
set_property(TEST "${TESTNAME}" PROPERTY LABELS "CLImodule")
set_property(TEST "${TESTNAME}" PROPERTY TIMEOUT 1)
set_property(TEST "${TESTNAME}" PROPERTY PASS_REGULAR_EXPRESSION
                                         "COREMOD_tools")

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
