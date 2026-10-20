# milk-CTRL unit tests (CTest milkctrl_unit_ansi, milkctrl_unit_model).
# Included from src/cli/CLIcore/CMakeLists.txt: relative paths resolve against that directory.
if(BUILD_TESTING)
  add_executable(
    milkctrl-unit-ansi
    ../overview/tests/test_ov_ansi.c ../overview/ansi/overview_ansi_term.c
    ../overview/ansi/overview_ansi_buf.c ../overview/ansi/overview_ansi_delta.c
    ../overview/ansi/overview_ansi_utf8.c ../overview/ansi/overview_ansi_key.c)
  target_include_directories(milkctrl-unit-ansi
                             PRIVATE ${CMAKE_SOURCE_DIR}/src/cli/overview/ansi)
  add_test(NAME milkctrl_unit_ansi COMMAND milkctrl-unit-ansi)

  add_executable(milkctrl-unit-model ../overview/tests/test_ov_model.c)
  target_link_libraries(milkctrl-unit-model PRIVATE milkctrl_model)
  add_test(NAME milkctrl_unit_model COMMAND milkctrl-unit-model)
endif()
