# milk-termview: standalone terminal stream viewer (links ImageStreamIO and libm only)

# Flags shared by the viewer and its tests. -Ofast implies -ffinite-math-only, which would fold the
# NaN/Inf guards away; the guards also use bit tests, but keep IEEE semantics for this code.
add_library(termview_flags INTERFACE)
target_compile_options(termview_flags INTERFACE -fno-finite-math-only -Wall
                                                -Wextra -Wimplicit-fallthrough)
if(USE_SANITIZERS)
  target_compile_definitions(termview_flags INTERFACE TV_KEEP_STDERR)
  target_compile_options(termview_flags INTERFACE -fsanitize=float-cast-overflow
                                                  -fno-sanitize-recover=all)
  target_link_options(termview_flags INTERFACE -fsanitize=float-cast-overflow
                      -fno-sanitize-recover=all)
endif()

add_library(
  termview_core OBJECT
  termview.c
  tv_cmap.c
  tv_flush.c
  tv_image.c
  tv_input.c
  tv_stream.c
  tv_term.c
  tv_ui.c)
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
  set_tests_properties(termview_unit PROPERTIES LABELS termview)
endif()
