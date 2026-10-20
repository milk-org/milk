# milk-CTRL unit tests (CTest milkctrl_unit_ansi, _model, _graph, _view, _opts, milkctrl_keys_doc).
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

  add_executable(milkctrl-unit-graph ../overview/tests/test_ov_graph.c)
  target_link_libraries(milkctrl-unit-graph PRIVATE milkctrl_model)
  add_test(NAME milkctrl_unit_graph COMMAND milkctrl-unit-graph)

  add_executable(
    milkctrl-unit-view
    ../overview/tests/test_ov_view.c ../overview/render/overview_render_view.c
    ../overview/render/overview_render_select.c ../overview/render/overview_render_relate.c
    ../overview/render/overview_render_filter.c ../overview/overview_cmdlog.c
    ../overview/render/overview_render_derived.c ../overview/render/overview_render_units.c
    ../overview/render/overview_columns.c ../overview/render/overview_columns_fps.c
    ../overview/render/overview_columns_procs.c ../overview/render/overview_columns_streams.c
    ../overview/render/overview_render_utils.c ../overview/overview_view.c ../overview/overview_keymap.c
    ../overview/input/overview_input_keys.c
    ../overview/render/overview_theme.c ../overview/render/overview_theme_palettes.c
    ../overview/render/overview_theme_palettes_extra.c ../overview/ansi/overview_ansi_buf.c
    ../overview/ansi/overview_ansi_delta.c ../overview/ansi/overview_ansi_utf8.c
    ../overview/ansi/overview_ansi_term.c ../overview/ansi/overview_ansi_key.c)
  target_include_directories(
    milkctrl-unit-view PRIVATE ${CMAKE_SOURCE_DIR}/src/cli/overview/render
                               ${CMAKE_SOURCE_DIR}/src/cli/overview/ansi ${CMAKE_SOURCE_DIR}/src/cli/overview/input)
  target_link_libraries(milkctrl-unit-view PRIVATE milkctrl_model)
  add_test(NAME milkctrl_unit_view COMMAND milkctrl-unit-view)

  add_executable(
    milkctrl-unit-ui ../overview/tests/test_ov_ui.c ../overview/overview_view.c ../overview/overview_keymap.c
                     ../overview/input/overview_input_keys.c ../overview/render/overview_render_units.c)
  target_include_directories(
    milkctrl-unit-ui PRIVATE ${CMAKE_SOURCE_DIR}/src/cli/overview/render ${CMAKE_SOURCE_DIR}/src/cli/overview/ansi
                             ${CMAKE_SOURCE_DIR}/src/cli/overview/input)
  target_link_libraries(milkctrl-unit-ui PRIVATE milkctrl_model)
  add_test(NAME milkctrl_unit_ui COMMAND milkctrl-unit-ui)

  add_executable(
    milkctrl-unit-opts
    ../overview/tests/test_ov_opts.c ../overview/tools/milkCTRL_opts_parse.c ../overview/overview_view.c
    ../overview/render/overview_theme.c ../overview/render/overview_theme_palettes.c
    ../overview/render/overview_theme_palettes_extra.c ../overview/ansi/overview_ansi_buf.c
    ../overview/ansi/overview_ansi_delta.c ../overview/ansi/overview_ansi_utf8.c
    ../overview/ansi/overview_ansi_term.c ../overview/ansi/overview_ansi_key.c)
  target_include_directories(
    milkctrl-unit-opts PRIVATE ${CMAKE_SOURCE_DIR}/src/cli/overview/render
                               ${CMAKE_SOURCE_DIR}/src/cli/overview/ansi ${CMAKE_SOURCE_DIR}/src/cli/overview/tools)
  target_link_libraries(milkctrl-unit-opts PRIVATE milkctrl_model)
  add_test(NAME milkctrl_unit_opts COMMAND milkctrl-unit-opts)

  add_test(NAME milkctrl_keys_doc
           COMMAND bash ${CMAKE_SOURCE_DIR}/src/cli/overview/tests/check_keys_doc.sh $<TARGET_FILE:milk-CTRL>
                   ${CMAKE_SOURCE_DIR}/docs/guide/milk-CTRL.md)
endif()
