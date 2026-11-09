set(LIBNAME "milkcommon")

add_library(
  ${LIBNAME} SHARED milk_compiler.c milk_omp_team.c milk_rt.c milk_path.c milk_spawn.c
                    milk_syscalls_linux.c milk_detach.c milk_sdnotify.c)
set_target_properties(${LIBNAME} PROPERTIES UNITY_BUILD ON)
# milk_syscalls_linux.c defines _GNU_SOURCE before its first include: it needs its own translation unit
set_source_files_properties(milk_syscalls_linux.c PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

target_include_directories(
  ${LIBNAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libmilkdata>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/ImageStreamIO>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libprocessinfo>
         $<INSTALL_INTERFACE:include>)
target_link_libraries(${LIBNAME} PUBLIC pthread)

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES milkDebugTools.h milk_compiler.h milk_omp_team.h pixel_dispatch.h milk_types.h
              multiselect_parse.h milk_rt.h milk_path.h milk_spawn.h milk_syscalls.h
              milk_detach.h milk_sdnotify.h
        DESTINATION include/libmilkcommon)

if(BUILD_TESTING)
  add_executable(test-milk-spawn test_milk_spawn.c)
  target_link_libraries(test-milk-spawn PRIVATE ${LIBNAME})
  add_test(NAME milkcommon-spawn COMMAND test-milk-spawn)
  add_executable(test-milk-syscalls test_milk_syscalls.c)
  target_link_libraries(test-milk-syscalls PRIVATE ${LIBNAME})
  add_test(NAME milkcommon-syscalls COMMAND test-milk-syscalls)
  add_executable(test-milk-detach test_milk_detach.c)
  target_link_libraries(test-milk-detach PRIVATE ${LIBNAME})
  add_test(NAME milkcommon-detach COMMAND test-milk-detach)
  add_executable(test-milk-sdnotify test_milk_sdnotify.c)
  target_link_libraries(test-milk-sdnotify PRIVATE ${LIBNAME})
  add_test(NAME milkcommon-sdnotify COMMAND test-milk-sdnotify)
endif()
