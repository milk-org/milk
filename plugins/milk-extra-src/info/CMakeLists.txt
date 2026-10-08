set(LIBNAME "milkinfo")
set(SRCNAME "info")

message("Module CMakeLists.txt ---- SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

file(GLOB SOURCEFILES "*.c" "*.cpp")
file(GLOB INCLUDEFILES "*.h" "*.hpp")
file(GLOB SCRIPTS "scripts/*")

project(lib_${LIBNAME}_project)

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/..")

add_library(${LIBNAME} SHARED ${SOURCEFILES})
target_include_directories(
  ${LIBNAME} PRIVATE $<TARGET_PROPERTY:CLIcore,INTERFACE_INCLUDE_DIRECTORIES>)
milk_apply_extensions(${LIBNAME})

# CUSTOM ===========
if(USE_NCURSES)
  target_compile_definitions(${LIBNAME} PRIVATE USE_NCURSES)
  target_include_directories(${LIBNAME} PRIVATE ${NCURSES_INCLUDE_DIR})
  target_link_libraries(${LIBNAME} PRIVATE ${NCURSES_LIBRARIES})
endif()

milk_add_static_lto(${LIBNAME})
# ==================

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTS} DESTINATION bin)

add_milk_standalone(info-strmonproc stream_monproc.c)

get_property(_stdalones DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  PROPERTY BUILDSYSTEM_TARGETS)
foreach(_t IN LISTS _stdalones)
  get_target_property(_type ${_t} TYPE)
  if(_type STREQUAL "EXECUTABLE")
    milk_link_module(${_t} ${LIBNAME})
  endif()
endforeach()

# CUSTOM ===========
target_sources(milk-fpsexec-info-strmonproc PRIVATE stream_monproc_shm.c)

if(USE_CLI)
  add_executable(stream-monproc-disp bin/stream-monproc-disp.c
                                     stream_monproc_shm.c)
  target_link_libraries(stream-monproc-disp milkdata milkfps ImageStreamIO m)
  target_compile_definitions(stream-monproc-disp PRIVATE MILK_NO_CLI)
  target_include_directories(
    stream-monproc-disp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                ${CMAKE_SOURCE_DIR}/src/engine/libfps
                                ${CMAKE_SOURCE_DIR}/src/cli/CLIcore)
  install(TARGETS stream-monproc-disp DESTINATION bin)

  add_executable(milk-shmimmon bin/milk-shmimmon.c)
  target_include_directories(
    milk-shmimmon
    PRIVATE ${CMAKE_SOURCE_DIR}/src/engine
            ${CMAKE_SOURCE_DIR}/src/engine/libfps ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(milk-shmimmon PRIVATE ImageStreamIO pthread m)
  install(TARGETS milk-shmimmon DESTINATION bin)
endif()
# ==================
