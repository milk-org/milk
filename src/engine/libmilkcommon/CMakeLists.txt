set(LIBNAME "milkcommon")

add_library(
  ${LIBNAME} SHARED milk_compiler.c milk_rt.c milk_path.c milk_spawn.c
                    milk_os_linux.c milk_detach.c milk_notify.c)

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
install(FILES milkDebugTools.h milk_compiler.h pixel_dispatch.h milk_types.h
              multiselect_parse.h milk_rt.h milk_path.h milk_spawn.h milk_os.h
              milk_detach.h milk_notify.h
        DESTINATION include/libmilkcommon)

if(BUILD_TESTING)
  add_executable(test-milk-spawn test_milk_spawn.c)
  target_link_libraries(test-milk-spawn PRIVATE ${LIBNAME})
  add_test(NAME milkcommon-spawn COMMAND test-milk-spawn)
  add_executable(test-milk-os test_milk_os.c)
  target_link_libraries(test-milk-os PRIVATE ${LIBNAME})
  add_test(NAME milkcommon-os COMMAND test-milk-os)
  add_executable(test-milk-detach test_milk_detach.c)
  target_link_libraries(test-milk-detach PRIVATE ${LIBNAME})
  add_test(NAME milkcommon-detach COMMAND test-milk-detach)
  add_executable(test-milk-notify test_milk_notify.c)
  target_link_libraries(test-milk-notify PRIVATE ${LIBNAME})
  add_test(NAME milkcommon-notify COMMAND test-milk-notify)
endif()
