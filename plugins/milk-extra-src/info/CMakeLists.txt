set(LIBNAME "milkinfo")
set(SRCNAME "info")

message("Module CMakeLists.txt ---- SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

file(GLOB SOURCEFILES "*.c" "*.cpp")
file(GLOB INCLUDEFILES "*.h" "*.hpp")
file(GLOB SCRIPTS "scripts/*")

project(lib_${LIBNAME}_project)

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/..")

# CUSTOM ===========
# NCURSES libs also feed the static LTO archive, which is built from the objects.
# CLIcore is the only other place that resolves NCURSES, and it is skipped with USE_CLI=OFF.
if(USE_NCURSES AND NOT NCURSES_LIBRARIES)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(NCURSES REQUIRED ncurses)
endif()

set(_info_link_libs)
if(USE_NCURSES)
  set(_info_link_libs ${NCURSES_LIBRARIES})
endif()
# ==================

milk_add_runnable_module(${LIBNAME} SOURCES ${SOURCEFILES} LINK_LIBS ${_info_link_libs})
milk_add_fpsexec_stubs(${LIBNAME})

# CUSTOM ===========
if(USE_NCURSES)
  target_compile_definitions(${LIBNAME}_objs PRIVATE USE_NCURSES)
  target_include_directories(${LIBNAME}_objs PRIVATE ${NCURSES_INCLUDE_DIR})
  target_link_libraries(${LIBNAME} PRIVATE ${NCURSES_LIBRARIES})
endif()
# ==================

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
