set(LIBNAME "milkCOREMODmemory")
set(SRCNAME "COREMOD_memory")

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
  if(${CFITSIO_FOUND})
    link_directories(${CFITSIO_LIBRARY_DIRS})
    target_compile_definitions(${LIBNAME} PUBLIC USE_CFITSIO=1)
    target_include_directories(${LIBNAME} PUBLIC ${CFITSIO_INCLUDE_DIRS})
    target_link_directories(${LIBNAME} PUBLIC ${CFITSIO_LIBRARY_DIRS})
  endif()
endif()

target_include_directories(
  ${LIBNAME}
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/streamCTRL>
         $<INSTALL_INTERFACE:include>)
target_link_libraries(
  ${LIBNAME}
  PUBLIC
    m ${CFITSIO_LIBRARIES} milkfps
    $<$<AND:$<BOOL:${USE_CFITSIO}>,$<BOOL:${CFITSIO_FOUND}>>:milkCOREMODiofits>)

set(SCRIPTS
    scripts/milk-nettransmit
    scripts/milk-shmimave
    scripts/milk-shmimcopy-semtrig
    scripts/milk-shmimpoke
    scripts/milk-shmimpoke-semtrig
    scripts/milk-semloopspeed
    scripts/milk-streamFITSlog)

milk_add_static_lto(${LIBNAME} CFITSIO_LINK_LIBS milkCOREMODiofits_static)
# ==================

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTS} DESTINATION bin)

add_milk_standalone(mem-streamave stream_ave.c)
add_milk_standalone(mem-streampoke stream_poke.c)
add_milk_standalone(mem-streamcopy stream_copy.c)
add_milk_standalone(mem-streamdelay stream_delay.c)
add_milk_standalone(imkwadd image_keyword_add.c)

get_property(_stdalones DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  PROPERTY BUILDSYSTEM_TARGETS)
foreach(_t IN LISTS _stdalones)
  get_target_property(_type ${_t} TYPE)
  if(_type STREQUAL "EXECUTABLE")
    milk_link_module(${_t} ${LIBNAME})
  endif()
endforeach()

# CUSTOM ===========
include(CTest)

set(TESTNAME "CLIfunc-COREMOD_memory-module")
add_test(NAME "${TESTNAME}" COMMAND milk-exec "-T" "m?")
set_property(TEST "${TESTNAME}" PROPERTY LABELS "CLImodule")
set_property(TEST "${TESTNAME}" PROPERTY TIMEOUT 1)
set_property(TEST "${TESTNAME}" PROPERTY PASS_REGULAR_EXPRESSION
                                         "COREMOD_memory")

list(APPEND commandlist "mk2Dim" "mk3Dim" "listim" "rmall")

foreach(CLIcmdname IN LISTS commandlist)

  set(TESTCLIFUNC "${CLIcmdname}")
  set(TESTNAME "CLIfunc-COREMOD_memory-${TESTCLIFUNC}")
  add_test(NAME "${TESTNAME}" COMMAND milk-exec "-T" "cmd? mem.${CLIcmdname}")
  set_property(TEST "${TESTNAME}" PROPERTY LABELS "CLIfunc")
  set_property(TEST "${TESTNAME}" PROPERTY TIMEOUT 1)
  set_property(TEST "${TESTNAME}" PROPERTY PASS_REGULAR_EXPRESSION "src:")

endforeach()

set(TESTNAME "milksemloopspeed")
add_test(NAME "${TESTNAME}"
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/milk-semloopspeed "-D"
                 "123456")
set_property(TEST "${TESTNAME}" PROPERTY LABELS "perf")
set_property(TEST "${TESTNAME}" PROPERTY TIMEOUT 5)
set_property(TEST "${TESTNAME}" PROPERTY PASS_REGULAR_EXPRESSION
                                         "cnt0 = 123457")

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
  PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)
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
