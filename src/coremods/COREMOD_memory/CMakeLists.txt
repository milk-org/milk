set(LIBNAME "milkCOREMODmemory")
set(SRCNAME "COREMOD_memory")

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
  if(${CFITSIO_FOUND})
    link_directories(${CFITSIO_LIBRARY_DIRS})
    milk_runnable_module_usage(
      ${LIBNAME}
      DEFINITIONS USE_CFITSIO=1
      INCLUDES ${CFITSIO_INCLUDE_DIRS}
      LINK_DIRS ${CFITSIO_LIBRARY_DIRS})
  endif()
endif()

milk_runnable_module_usage(
  ${LIBNAME}
  INCLUDES
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(
  ${LIBNAME}
  PUBLIC
    m ${CFITSIO_LIBRARIES} milkfps milkdata
    $<$<AND:$<BOOL:${USE_CFITSIO}>,$<BOOL:${CFITSIO_FOUND}>>:milkCOREMODiofits>)

set(SCRIPTS
    scripts/milk-nettransmit
    scripts/milk-shmimave
    scripts/milk-shmimcopy-semtrig
    scripts/milk-shmimpoke
    scripts/milk-shmimpoke-semtrig
    scripts/milk-semloopspeed
    scripts/milk-streamFITSlog)
# ==================

install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTS} DESTINATION bin)

# CUSTOM ===========
include(CTest)

set(TESTNAME "CLIfunc-COREMOD_memory-module")
add_test(NAME "${TESTNAME}" COMMAND milk-exec "-T" "@help mem")
set_property(TEST "${TESTNAME}" PROPERTY LABELS "CLImodule")
set_property(TEST "${TESTNAME}" PROPERTY TIMEOUT 1)
set_property(TEST "${TESTNAME}" PROPERTY PASS_REGULAR_EXPRESSION
                                         "COREMOD_memory")

add_executable(milk-fpsexec-cnt2push bin/cnt2push_main.c)
target_include_directories(milk-fpsexec-cnt2push
                           PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>)
target_link_libraries(milk-fpsexec-cnt2push ImageStreamIO m)
install(TARGETS milk-fpsexec-cnt2push DESTINATION bin)

add_executable(milk-stream-list bin/milk-stream-list.c)
target_include_directories(
  milk-stream-list
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)
target_link_libraries(milk-stream-list ImageStreamIO)
install(TARGETS milk-stream-list DESTINATION bin)

add_executable(milk-stream-create bin/milk-stream-create.c)
target_include_directories(
  milk-stream-create
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)
target_link_libraries(milk-stream-create ImageStreamIO)
install(TARGETS milk-stream-create DESTINATION bin)

add_executable(milk-stream-rm bin/milk-stream-rm.c)
target_include_directories(
  milk-stream-rm
  PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/libmilkscript>
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)
target_link_libraries(milk-stream-rm ImageStreamIO)
install(TARGETS milk-stream-rm DESTINATION bin)

add_executable(milk-streamCTRL-cli bin/milk-streamCTRL-cli.c)
target_include_directories(
  milk-streamCTRL-cli
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)
target_link_libraries(milk-streamCTRL-cli ImageStreamIO)
install(TARGETS milk-streamCTRL-cli DESTINATION bin)

add_executable(milk-shmimpurge bin/milk-shmimpurge.c)
target_include_directories(
  milk-shmimpurge
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)
target_link_libraries(milk-shmimpurge PRIVATE ImageStreamIO)
install(TARGETS milk-shmimpurge DESTINATION bin)

add_executable(milk-logshim-ctrl bin/milk-logshim-ctrl.c)
target_include_directories(
  milk-logshim-ctrl
  PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libmilkcommon>)
target_link_libraries(milk-logshim-ctrl PRIVATE milkcommon)
install(TARGETS milk-logshim-ctrl DESTINATION bin)

add_executable(milk-stream-link bin/milk-stream-link.c)
target_include_directories(
  milk-stream-link
  PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/ImageStreamIO>
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>)
target_link_libraries(milk-stream-link PRIVATE ImageStreamIO)
install(TARGETS milk-stream-link DESTINATION bin)
# ==================
