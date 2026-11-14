set(LIBNAME "milkinfo")
set(SRCNAME "info")

message("Module CMakeLists.txt ---- SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

file(GLOB SOURCEFILES "*.c" "*.cpp")
file(GLOB INCLUDEFILES "*.h" "*.hpp")
file(GLOB SCRIPTS "scripts/*")

project(lib_${LIBNAME}_project)

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/..")

milk_add_runnable_module(${LIBNAME} SOURCES ${SOURCEFILES})
milk_add_fpsexec_stubs(${LIBNAME})

install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTS} DESTINATION bin)

if(USE_CLI)
  add_executable(stream-monproc-disp bin/stream-monproc-disp.c
                                     stream_monproc_shm.c)
  target_link_libraries(stream-monproc-disp milkdata milkfps milkcomputeunit ImageStreamIO m)
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
