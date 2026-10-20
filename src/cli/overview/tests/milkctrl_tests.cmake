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

  add_executable(
    milkctrl-unit-model
    ../overview/tests/test_ov_model.c
    ../overview/data/overview_data.c
    ../overview/data/overview_data_loops.c
    ../overview/data/overview_data_loops_names.c
    ../overview/data/overview_data_loops_reg.c
    ../overview/data/overview_data_sys.c
    ../overview/data/overview_data_sys_proc.c
    ../overview/data/overview_data_sys_perf.c
    ../overview/data/overview_data_cache.c
    ../overview/data/overview_data_cache_fps.c
    ../overview/data/overview_data_scan.c
    ../overview/data/overview_data_scan_streams.c
    ../overview/data/overview_data_scan_fps.c
    ../overview/data/overview_data_scan_procs.c
    ../overview/data/overview_data_scan_tmux.c
    ../overview/data/overview_data_graph.c
    ../overview/data/overview_data_sort.c
    ../overview/data/overview_data_sort_streams.c
    ../overview/data/overview_data_sort_procs.c
    ../overview/data/overview_data_sort_fps.c
    ../overview/data/stream_graph.c
    ../overview/data/stream_graph_lineage.c
    ../overview/data/stream_graph_tree.c)
  target_include_directories(
    milkctrl-unit-model
    PRIVATE ${CMAKE_SOURCE_DIR}/src/engine/ImageStreamIO
            ${CMAKE_SOURCE_DIR}/src/engine/libprocessinfo
            ${CMAKE_SOURCE_DIR}/src/engine/libfps
            ${CMAKE_SOURCE_DIR}/src/cli/overview
            ${CMAKE_SOURCE_DIR}/src/cli/overview/ansi)
  target_link_libraries(milkctrl-unit-model PRIVATE ImageStreamIO milkprocessinfo
                                                    milkfps m rt pthread)
  add_test(NAME milkctrl_unit_model COMMAND milkctrl-unit-model)
endif()
