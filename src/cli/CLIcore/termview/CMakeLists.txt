# milk-termview: standalone terminal stream viewer (links ImageStreamIO and libm only)

# Flags shared by the viewer and its tests. -Ofast implies -ffinite-math-only, which would fold the
# NaN/Inf guards away; the guards also use bit tests, but keep IEEE semantics for this code.
add_library(termview_flags INTERFACE)
target_compile_options(termview_flags INTERFACE -fno-finite-math-only -Wall
                                                -Wextra -Wimplicit-fallthrough)
if(USE_SANITIZERS)
  target_compile_options(termview_flags INTERFACE -fsanitize=float-cast-overflow
                                                  -fno-sanitize-recover=all)
  target_link_options(termview_flags INTERFACE -fsanitize=float-cast-overflow
                      -fno-sanitize-recover=all)
endif()

add_library(termview_core OBJECT termview.c tv_cmap.c tv_flush.c tv_input.c
                                 tv_stream.c tv_term.c)
target_link_libraries(termview_core PUBLIC termview_flags ImageStreamIO m)
target_include_directories(
  termview_core PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)

add_executable(milk-termview milk-termview.c)
target_link_libraries(milk-termview PRIVATE termview_core)
install(TARGETS milk-termview DESTINATION bin)

if(BUILD_TESTING)
  add_executable(test_termview_unit tests/test_termview_unit.c)
  target_link_libraries(test_termview_unit PRIVATE termview_core)
  add_test(NAME termview_unit COMMAND test_termview_unit)

  add_executable(tv-test-mkstream tests/tv-test-mkstream.c)
  target_link_libraries(tv-test-mkstream PRIVATE termview_flags ImageStreamIO m)

  find_package(
    Python3
    COMPONENTS Interpreter
    QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(
      NAME termview_pty
      COMMAND
        ${Python3_EXECUTABLE} -I
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/termview_pty_test.py --termview
        $<TARGET_FILE:milk-termview> --mkstream $<TARGET_FILE:tv-test-mkstream>)
    set_tests_properties(termview_pty PROPERTIES TIMEOUT 300)
    if(NOT USE_SANITIZERS)
      add_test(
        NAME termview_perf
        COMMAND
          ${Python3_EXECUTABLE} -I
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/termview_pty_test.py --perf
          --termview $<TARGET_FILE:milk-termview> --mkstream
          $<TARGET_FILE:tv-test-mkstream>)
      set_tests_properties(termview_perf PROPERTIES TIMEOUT 120 RUN_SERIAL TRUE
                                                    LABELS "termview;perf")
    endif()
  endif()
  set_tests_properties(termview_unit PROPERTIES LABELS termview)
  if(TEST termview_pty)
    set_tests_properties(termview_pty PROPERTIES LABELS termview)
  endif()
endif()
