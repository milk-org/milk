# milkctrl_model: the milk-CTRL data layer (scan, model, graph, loops, filters), shared by milk-CTRL
# and its companion tools. Included from src/cli/CLIcore/CMakeLists.txt.
set(OV_DIR ${CMAKE_SOURCE_DIR}/src/cli/overview)
add_library(
  milkctrl_model STATIC
  ${OV_DIR}/data/overview_data.c
  ${OV_DIR}/data/overview_data_adj.c
  ${OV_DIR}/data/overview_data_cache.c
  ${OV_DIR}/data/overview_data_cache_fps.c
  ${OV_DIR}/data/overview_data_filter.c
  ${OV_DIR}/data/overview_data_graph.c
  ${OV_DIR}/data/overview_data_index.c
  ${OV_DIR}/data/overview_data_pid.c
  ${OV_DIR}/data/overview_data_loops.c
  ${OV_DIR}/data/overview_data_loops_names.c
  ${OV_DIR}/data/overview_data_loops_reg.c
  ${OV_DIR}/data/overview_data_order.c
  ${OV_DIR}/data/overview_data_scan.c
  ${OV_DIR}/data/overview_data_scan_streams.c
  ${OV_DIR}/data/overview_data_scan_fps.c
  ${OV_DIR}/data/overview_data_scan_procs.c
  ${OV_DIR}/data/overview_data_scan_tmux.c
  ${OV_DIR}/data/overview_data_sort.c
  ${OV_DIR}/data/overview_data_sort_streams.c
  ${OV_DIR}/data/overview_data_sort_procs.c
  ${OV_DIR}/data/overview_data_sort_fps.c
  ${OV_DIR}/data/overview_data_sys.c
  ${OV_DIR}/data/overview_data_trace.c
  ${OV_DIR}/data/overview_data_stream_err.c
  ${OV_DIR}/data/overview_data_inspect.c
  ${OV_DIR}/data/overview_data_holders.c
  ${OV_DIR}/data/overview_data_sys_proc.c
  ${OV_DIR}/data/overview_data_sys_perf.c
  ${OV_DIR}/data/overview_scan.c
  ${OV_DIR}/data/overview_sig.c
  ${OV_DIR}/data/stream_graph.c
  ${OV_DIR}/data/stream_graph_lineage.c
  ${OV_DIR}/data/stream_graph_tree.c)
target_include_directories(
  milkctrl_model
  PUBLIC ${CMAKE_SOURCE_DIR}/src/engine/ImageStreamIO
         ${CMAKE_SOURCE_DIR}/src/engine/libprocessinfo
         ${CMAKE_SOURCE_DIR}/src/engine/libfps
         ${OV_DIR}
         ${OV_DIR}/data)
target_compile_options(milkctrl_model PRIVATE -Wmissing-prototypes)
target_link_libraries(milkctrl_model PUBLIC ImageStreamIO milkprocessinfo milkfps milkdata m rt pthread)
