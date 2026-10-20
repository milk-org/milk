cmake_minimum_required(VERSION 3.17)
cmake_policy(SET CMP0069 NEW)

set(SRCNAME "CommandLineInterface")

option(python_module "Compile Python Wrappers" OFF)

project(CLIcore C)

find_package(PkgConfig REQUIRED)

if(USE_NCURSES)
  pkg_check_modules(NCURSES REQUIRED ncurses)
endif()
if(USE_READLINE)
  pkg_check_modules(READLINE REQUIRED readline)
endif()
find_package(OpenMP)

pkg_check_modules(CFITSIO cfitsio)

find_package(Threads REQUIRED)

# Tree-sitter: optional full-line syntax highlighting
if(USE_TREESITTER)
  # Some systems only have the versioned .so.0 symlink (no -dev package).
  # Temporarily extend suffixes.
  set(_saved_suffixes ${CMAKE_FIND_LIBRARY_SUFFIXES})
  list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES ".so.0" ".so.0.22" ".so.0.24")
  find_library(TREESITTER_LIB NAMES tree-sitter)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_saved_suffixes})
  if(TREESITTER_LIB)
    message(STATUS "Found tree-sitter: ${TREESITTER_LIB}")
    set(TREESITTER_FOUND TRUE)
  else()
    message(STATUS "tree-sitter library not found; "
                   "falling back to legacy highlighting")
    set(TREESITTER_FOUND FALSE)
  endif()
else()
  set(TREESITTER_FOUND FALSE)
endif()

set(CORE_FILES
    CLIcore.c
    CLIcore_cmd_registry.c
    CLIcore/CLIcore_help_tui.c
    CLIcore/CLIcore_UI_builtins.c
    CLIcore/CLIcore_UI_builtins_fifo.c
    CLIcore/CLIcore_UI_completion.c
    CLIcore/CLIcore_UI_completion_tab.c
    CLIcore/CLIcore_UI_completion_menu.c
    CLIcore/CLIcore_UI_highlight.c
    CLIcore/CLIcore_UI_hintarea.c
    CLIcore/CLIcore_UI_history.c
    CLIcore/CLIcore_UI_history_display.c
    CLIcore/CLIcore_UI_history_cmds.c
    CLIcore/CLIcore_UI_prompt.c
    treesitter/cli_treesitter.c
    treesitter/cli_treesitter_lexical.c
    treesitter/cli_treesitter_diag.c
    treesitter/cli_treesitter_format.c)

# Add vendored parser when tree-sitter is available
if(TREESITTER_FOUND)
  list(APPEND CORE_FILES treesitter/milkcli_parser.c)
endif()

link_directories(${NCURSES_LIBRARY_DIRS})

# main

add_library(CLIcore SHARED ${CORE_FILES})

# Add milk-help executable
add_executable(milk-help milk-help.c)
target_link_libraries(milk-help PUBLIC CLIcore)
install(TARGETS milk-help DESTINATION bin)

# Add milk-cli-help executable
add_executable(milk-cli-help milk-cli-help.c)
target_link_libraries(milk-cli-help PUBLIC CLIcore)
install(TARGETS milk-cli-help DESTINATION bin)

# Add milk-stream-help executable
add_executable(milk-stream-help milk-stream-help.c)
target_link_libraries(milk-stream-help PUBLIC CLIcore)
install(TARGETS milk-stream-help DESTINATION bin)

# Add milk-cli-help-synchro executable
add_executable(milk-cli-help-synchro milk-cli-help-synchro.c)
target_link_libraries(milk-cli-help-synchro PUBLIC CLIcore)
install(TARGETS milk-cli-help-synchro DESTINATION bin)

if(USE_CFITSIO AND ${CFITSIO_FOUND})
  link_directories(${CFITSIO_LIBRARY_DIRS})
  target_compile_definitions(CLIcore PUBLIC USE_CFITSIO=1)
  target_include_directories(CLIcore PUBLIC ${CFITSIO_INCLUDE_DIRS})
endif()

if(OPENMP_C_FOUND)
  message("Found OpenMP")
  target_link_libraries(CLIcore PUBLIC OpenMP::OpenMP_C)
endif()

if(USE_READLINE)
  target_compile_definitions(CLIcore PUBLIC USE_READLINE)
endif()

if(USE_NCURSES)
  target_compile_definitions(CLIcore PUBLIC USE_NCURSES)
endif()

if(TREESITTER_FOUND)
  target_compile_definitions(CLIcore PUBLIC USE_TREESITTER)
  target_include_directories(CLIcore
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/treesitter)
  target_link_libraries(CLIcore PUBLIC ${TREESITTER_LIB})
endif()

target_link_libraries(
  CLIcore
  PUBLIC m
         milkscript
         $<$<BOOL:${USE_READLINE}>:readline>
         ${CFITSIO_LIBRARIES}
         dl
         rt
         ${CMAKE_THREAD_LIBS_INIT}
         $<$<BOOL:${USE_NCURSES}>:${NCURSES_LIBRARIES}>
         milkcommon
         milkCOREMODarith
         $<$<BOOL:${USE_CFITSIO}>:milkCOREMODiofits>
         milkCOREMODmemory
         milkCOREMODtools
         milkprocessinfo
         milkfps
         milkfpsseq
         milkdata)

target_include_directories(
  CLIcore
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/libmilkscript>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/CLIcore>
         $<INSTALL_INTERFACE:include/${SRCNAME}>
         $<$<BOOL:${USE_NCURSES}>:${NCURSES_INCLUDE_DIR}>)
target_compile_options(CLIcore PUBLIC)

target_link_directories(CLIcore PUBLIC
                        $<$<BOOL:${USE_NCURSES}>:${NCURSES_LIBRARY_DIRS}>)

target_link_libraries(
  CLIcore PUBLIC ${CMAKE_THREAD_LIBS_INIT}
                 $<$<BOOL:${USE_NCURSES}>:${NCURSES_LIBRARIES}>)

if(${python_module})
  add_subdirectory(python_module)
endif()

install(
  TARGETS CLIcore
  EXPORT milkTargets
  DESTINATION lib)

install(
  FILES CLIcore.h
        ../libmilkscript/calc/cli_calc_parser.h
        ../libmilkscript/calc/cli_calc_tokenizer.h
        cmdsettings.h
        ../../engine/libmilkcommon/milkDebugTools.h
        ../libmilkscript/standalone_dependencies.h
        ../../engine/libprocessinfo/timeutils.h
  DESTINATION include/${SRCNAME})

install(
  FILES ../libmilkscript/core/CLIcore_UI_execute.h
        ../libmilkscript/core/CLIcore_checkargs.h
        ../libmilkscript/core/CLIcore_datainit.h
        ../libmilkscript/core/CLIcore_help.h
        ../libmilkscript/core/CLIcore_memory.h
        ../libmilkscript/core/CLIcore_modules.h
        ../libmilkscript/core/CLIcore_setSHMdir.h
        ../libmilkscript/core/CLIcore_script.h
        ../libmilkscript/core/CLIcore_signals.h
        ../libmilkscript/core/CLIcore_utils.h
  DESTINATION include/${SRCNAME}/CLIcore)

# milk-streamCTRL is milk-CTRL started under that name (it opens F3): a link, installed with milk-CTRL below

add_subdirectory(termview)

include(${CMAKE_SOURCE_DIR}/src/cli/overview/data/milkctrl_model.cmake)

# milkCTRL: unified system dashboard TUI
add_executable(
  milk-CTRL
  ../overview/tools/milkCTRL.c
  ../overview/tools/milkCTRL_opts.c
  ../overview/tools/milkCTRL_opts_parse.c
  ../overview/overview_layout.c
  ../overview/overview_view.c
  ../overview/overview_keymap.c
  ../overview/ansi/overview_ansi_term.c
  ../overview/ansi/overview_ansi_buf.c
  ../overview/ansi/overview_ansi_delta.c
  ../overview/ansi/overview_ansi_utf8.c
  ../overview/ansi/overview_ansi_key.c
  ../overview/overview_cmdlog.c
  ../overview/render/overview_theme.c
  ../overview/render/overview_theme_palettes.c
  ../overview/render/overview_theme_palettes_extra.c
  ../overview/render/overview_help_data.c
  ../overview/render/overview_render_utils.c
  ../overview/render/overview_render_filter.c
  ../overview/render/overview_render_relate.c
  ../overview/render/overview_render_select.c
  ../overview/render/overview_render_view.c
  ../overview/render/overview_render_header.c
  ../overview/render/overview_render_tabs.c
  ../overview/render/overview_render_subtabs.c
  ../overview/render/overview_render_derived.c
  ../overview/render/overview_render_units.c
  ../overview/render/overview_render_bar.c
  ../overview/render/overview_render_insp.c
  ../overview/render/overview_render_insp_pages.c
  ../overview/input/overview_input_nav_insp.c
  ../overview/render/overview_render_graph_health.c
  ../overview/render/overview_render_popup.c
  ../overview/render/overview_render_sync.c
  ../overview/render/overview_render_status.c
  ../overview/render/overview_render_cmdlog.c
  ../overview/render/overview_render_preview.c
  ../overview/render/overview_render_graph.c
  ../overview/render/overview_render_streams.c
  ../overview/render/overview_render_streams_hdr.c
  ../overview/render/overview_render_streams_row.c
  ../overview/render/overview_render_procs.c
  ../overview/render/overview_render_procs_hdr.c
  ../overview/render/overview_render_procs_row.c
  ../overview/render/overview_render_fps.c
  ../overview/render/overview_render_columns.c
  ../overview/render/overview_columns.c
  ../overview/render/overview_columns_fps.c
  ../overview/render/overview_columns_procs.c
  ../overview/render/overview_columns_streams.c
  ../overview/render/overview_render_fps_hdr.c
  ../overview/render/overview_render_fps_params.c
  ../overview/render/overview_render_fps_params_info.c
  ../overview/render/overview_render_loops.c
  ../overview/render/overview_render_loops_panel.c
  ../overview/render/overview_render_detail.c
  ../overview/render/overview_render_detail_stream.c
  ../overview/render/overview_render_detail_proc.c
  ../overview/render/overview_render_detail_fps.c
  ../overview/render/overview_render_resources.c
  ../overview/render/overview_render_resources_perf.c
  ../overview/render/overview_render_help.c
  ../overview/render/overview_render_help_search.c
  ../overview/render/overview_render_help_nav.c
  ../overview/render/overview_render_help_detail.c
  ../overview/render/overview_render_help_intro.c
  ../overview/render/overview_render_help_list.c
  ../overview/render/overview_render.c
  ../overview/input/overview_ctrl.c
  ../overview/input/overview_ctrl_proc.c
  ../overview/input/overview_fps_edit.c
  ../overview/input/overview_input.c
  ../overview/input/overview_input_actions.c
  ../overview/input/overview_input_help.c
  ../overview/input/overview_input_history.c
  ../overview/input/overview_input_hittest.c
  ../overview/input/overview_input_hittest_cols.c
  ../overview/input/overview_input_modal.c
  ../overview/input/overview_input_confirm.c
  ../overview/input/overview_input_modal_filter.c
  ../overview/input/overview_input_mouse.c
  ../overview/input/overview_input_mouse_drag.c
  ../overview/input/overview_input_mouse_panels.c
  ../overview/input/overview_input_nav.c
  ../overview/input/overview_input_subtab.c
  ../overview/input/overview_input_keys.c
  ../overview/input/overview_input_nav_fps.c
  ../overview/input/overview_input_nav_lineage.c
  ../overview/input/overview_input_sort_keys.c
  ../overview/input/overview_input_toggles.c)
target_include_directories(
  milk-CTRL
  PRIVATE ${CMAKE_SOURCE_DIR}/src/engine/ImageStreamIO
          ${CMAKE_SOURCE_DIR}/src/engine/libprocessinfo
          ${CMAKE_SOURCE_DIR}/src/engine/libfps
          ${CMAKE_SOURCE_DIR}/src/cli/overview
          ${CMAKE_SOURCE_DIR}/src/cli/overview/ansi
          ${CMAKE_SOURCE_DIR}/src/cli/overview/data
          ${CMAKE_SOURCE_DIR}/src/cli/overview/input
          ${CMAKE_SOURCE_DIR}/src/cli/overview/render
          ${CMAKE_SOURCE_DIR}/src/cli/overview/tools)
target_link_libraries(milk-CTRL PRIVATE milkctrl_model)
install(TARGETS milk-CTRL DESTINATION bin)
# milk-streamCTRL: the profile of milk-CTRL that opens F3 (argv[0]); a link in the build tree and at install
add_custom_command(
  TARGET milk-CTRL
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink milk-CTRL $<TARGET_FILE_DIR:milk-CTRL>/milk-streamCTRL)
install(CODE "file(CREATE_LINK milk-CTRL \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin/milk-streamCTRL\" SYMBOLIC)")
# render/ and input/ must not cast away the const of the scanned model
file(GLOB OV_UI_SRC ${CMAKE_SOURCE_DIR}/src/cli/overview/render/*.c ${CMAKE_SOURCE_DIR}/src/cli/overview/input/*.c)
set_source_files_properties(${OV_UI_SRC} PROPERTIES COMPILE_OPTIONS -Wcast-qual)
install(PROGRAMS ../overview/milk-setup-caps DESTINATION bin)

# milk-stream-graph: stream dependency graph tool
add_executable(
  milk-stream-graph
  ../overview/tools/milk-stream-graph.c
  ../overview/tools/milk-stream-graph_export.c
  ../overview/tools/milk-stream-graph_tui.c)
target_include_directories(
  milk-stream-graph
  PRIVATE ${CMAKE_SOURCE_DIR}/src/engine/ImageStreamIO
          ${CMAKE_SOURCE_DIR}/src/engine/libprocessinfo
          ${CMAKE_SOURCE_DIR}/src/engine/libfps
          ${CMAKE_SOURCE_DIR}/src/cli/overview
          ${CMAKE_SOURCE_DIR}/src/cli/overview/ansi
          ${CMAKE_SOURCE_DIR}/src/cli/overview/tools)
target_link_libraries(milk-stream-graph PRIVATE milkctrl_model)
install(TARGETS milk-stream-graph DESTINATION bin)

# milk-stream-info: detailed stream info with connections
add_executable(
  milk-stream-info
  ../overview/tools/milk-stream-info.c
  ../overview/tools/milk-stream-info_print.c)
target_include_directories(
  milk-stream-info
  PRIVATE ${CMAKE_SOURCE_DIR}/src/engine/ImageStreamIO
          ${CMAKE_SOURCE_DIR}/src/engine/libprocessinfo
          ${CMAKE_SOURCE_DIR}/src/engine/libfps
          ${CMAKE_SOURCE_DIR}/src/cli/overview
          ${CMAKE_SOURCE_DIR}/src/cli/overview/ansi
          ${CMAKE_SOURCE_DIR}/src/cli/overview/tools)
target_link_libraries(milk-stream-info PRIVATE milkctrl_model)
install(TARGETS milk-stream-info DESTINATION bin)

# milk-procinfo-info: detailed processinfo with connections
add_executable(
  milk-procinfo-info
  ../overview/tools/milk-procinfo-info.c)
target_include_directories(
  milk-procinfo-info
  PRIVATE ${CMAKE_SOURCE_DIR}/src/engine/ImageStreamIO
          ${CMAKE_SOURCE_DIR}/src/engine/libprocessinfo
          ${CMAKE_SOURCE_DIR}/src/engine/libfps
          ${CMAKE_SOURCE_DIR}/src/cli/overview
          ${CMAKE_SOURCE_DIR}/src/cli/overview/ansi
          ${CMAKE_SOURCE_DIR}/src/cli/overview/tools)
target_link_libraries(milk-procinfo-info PRIVATE milkctrl_model)
install(TARGETS milk-procinfo-info DESTINATION bin)

# milk-fuzzy-match: bigram fuzzy string matcher
add_executable(milk-fuzzy-match milk-fuzzy-match.c)
target_include_directories(milk-fuzzy-match
                           PRIVATE ${CMAKE_SOURCE_DIR}/src/engine/libfps)
install(TARGETS milk-fuzzy-match DESTINATION bin)

# TESTING

include(CTest)

set(failRegex "ERROR" "FAIL")

add_test(NAME milklistim COMMAND milk-exec "mem.listim")
set_property(TEST milklistim PROPERTY LABELS "CLI")
set_tests_properties(milklistim PROPERTIES TIMEOUT 1)
set_property(TEST milklistim PROPERTY PASS_REGULAR_EXPRESSION "0 image")

include(${CMAKE_CURRENT_SOURCE_DIR}/../overview/tests/milkctrl_tests.cmake)
