cmake_minimum_required(VERSION 3.8)
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
  # Some systems only have the versioned .so.0 symlink
  # (no -dev package). Temporarily extend suffixes.
  set(_saved_suffixes ${CMAKE_FIND_LIBRARY_SUFFIXES})
  list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES
    ".so.0" ".so.0.22" ".so.0.24")
  find_library(TREESITTER_LIB NAMES tree-sitter)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_saved_suffixes})
  if(TREESITTER_LIB)
    message(STATUS
      "Found tree-sitter: ${TREESITTER_LIB}")
    set(TREESITTER_FOUND TRUE)
  else()
    message(STATUS
      "tree-sitter library not found; "
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
    CLIcore/CLIcore_UI_bookmark.c
    CLIcore/CLIcore_UI_builtins.c
    CLIcore/CLIcore_UI_builtins_fifo.c
    CLIcore/CLIcore_UI_completion.c
    CLIcore/CLIcore_UI_completion_tab.c
    CLIcore/CLIcore_UI_highlight.c
    CLIcore/CLIcore_UI_hintarea.c
    CLIcore/CLIcore_UI_history.c
    CLIcore/CLIcore_UI_history_display.c
    CLIcore/CLIcore_UI_history_cmds.c
    CLIcore/CLIcore_UI_prompt.c
    treesitter/cli_treesitter.c)

# Add vendored parser when tree-sitter is available
if(TREESITTER_FOUND)
  list(APPEND CORE_FILES
    treesitter/milkcli_parser.c)
endif()
    
    link_directories(${NCURSES_LIBRARY_DIRS})
    
        # main
    
        add_library(CLIcore
    
                    SHARED
    
        ${CORE_FILES})

# Add milk-help executable
add_executable(milk-help milk-help.c)
target_link_libraries(milk-help PUBLIC CLIcore)
install(TARGETS milk-help DESTINATION bin)

# Add milk-cli-help executable
add_executable(milk-cli-help milk-cli-help.c)
target_link_libraries(milk-cli-help PUBLIC CLIcore)
install(TARGETS milk-cli-help DESTINATION bin)

# Add milk-cli-help-synchro executable
add_executable(milk-cli-help-synchro milk-cli-help-synchro.c)
target_link_libraries(milk-cli-help-synchro PUBLIC CLIcore)
install(TARGETS milk-cli-help-synchro DESTINATION bin)

if(${CFITSIO_FOUND})
  link_directories(${CFITSIO_LIBRARY_DIRS})
  target_compile_definitions(CLIcore PUBLIC USE_CFITSIO=1)
  target_include_directories(CLIcore PUBLIC ${CFITSIO_INCLUDE_DIRS})
endif()


if (OPENMP_C_FOUND)
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
    target_compile_definitions(CLIcore
        PUBLIC USE_TREESITTER)
    target_include_directories(CLIcore
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/treesitter)
    target_link_libraries(CLIcore
        PUBLIC ${TREESITTER_LIB})
endif()

target_link_libraries(CLIcore
                      PUBLIC m
                              milkscript
                              $<$<BOOL:${USE_READLINE}>:readline>
                              ${CFITSIO_LIBRARIES}
                              dl
                              rt

                              ${CMAKE_THREAD_LIBS_INIT}
                              $<$<BOOL:${USE_NCURSES}>:${NCURSES_LIBRARIES}>
                              milkcommon milkCOREMODarith $<$<BOOL:${USE_CFITSIO}>:milkCOREMODiofits> milkCOREMODmemory milkCOREMODtools milkprocessinfo milkfps milkfpsseq milkdata)


target_include_directories(CLIcore
                           PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
                                  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli>
                                  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/libmilkscript>
                                  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../streamCTRL>
                                  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/CLIcore>
                                  $<INSTALL_INTERFACE:include/${SRCNAME}>
                                  $<$<BOOL:${USE_NCURSES}>:${NCURSES_INCLUDE_DIR}>)
target_compile_options(CLIcore PUBLIC)

target_link_directories(CLIcore
                        PUBLIC $<$<BOOL:${USE_NCURSES}>:${NCURSES_LIBRARY_DIRS}>)

target_link_libraries(CLIcore
                      PUBLIC ${CMAKE_THREAD_LIBS_INIT}
                             $<$<BOOL:${USE_NCURSES}>:${NCURSES_LIBRARIES}> )

if(${python_module})
  add_subdirectory(python_module)
endif()

install(TARGETS CLIcore EXPORT milkTargets DESTINATION lib)


install(FILES CLIcore.h
              ../libmilkscript/cli_calc_parser.h
              ../libmilkscript/cli_calc_tokenizer.h
              cmdsettings.h
              ../../engine/libmilkcommon/milkDebugTools.h
              ../libmilkscript/standalone_dependencies.h
              ../../engine/libprocessinfo/timeutils.h
              DESTINATION include/${SRCNAME})

install(FILES ../libmilkscript/CLIcore_UI_execute.h
              ../libmilkscript/CLIcore_checkargs.h
              ../libmilkscript/CLIcore_datainit.h
              ../libmilkscript/CLIcore_help.h
              ../libmilkscript/CLIcore_memory.h
              ../libmilkscript/CLIcore_modules.h
              ../libmilkscript/CLIcore_setSHMdir.h
              ../libmilkscript/CLIcore_script.h
              ../libmilkscript/CLIcore_signals.h
              ../libmilkscript/CLIcore_utils.h
              DESTINATION include/${SRCNAME}/CLIcore)



add_executable(milk-streamCTRL 
    ../streamCTRL/milk-streamCTRL.c
    ../streamCTRL/streamCTRL_TUI.c
    ../streamCTRL/streamCTRL_TUI_render.c
    ../streamCTRL/streamCTRL_TUI_render_help.c
    ../streamCTRL/streamCTRL_TUI_render_header.c
    ../streamCTRL/streamCTRL_TUI_render_stream_rows.c
    ../streamCTRL/streamCTRL_TUI_render_footer.c
    ../streamCTRL/streamCTRL_TUI_input.c
    ../streamCTRL/streamCTRL_find_streams.c
    ../streamCTRL/streamCTRL_print_inode.c
    ../streamCTRL/streamCTRL_print_procpid.c
    ../streamCTRL/streamCTRL_print_trace.c
    ../streamCTRL/streamCTRL_scan.c
    ../streamCTRL/streamCTRL_utilfuncs.c
    ../streamCTRL/streamCTRL_TUIcompat.c
)

target_link_libraries(milk-streamCTRL PUBLIC ImageStreamIO milkprocessinfo m rt pthread)
target_include_directories(milk-streamCTRL PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)
install(TARGETS milk-streamCTRL DESTINATION bin)

# milk-fpsexec-mmon retired — use milk-streamCTRL instead

add_executable(milk-termview termview/milk-termview.c termview/termview.c)
target_link_libraries(milk-termview PUBLIC ImageStreamIO m)
target_include_directories(milk-termview PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)

install(TARGETS milk-termview DESTINATION bin)

# milkCTRL: unified system dashboard TUI
add_executable(milkCTRL
    ../overview/milkCTRL.c
    ../overview/overview_data.c
    ../overview/overview_data_sys.c
    ../overview/overview_data_cache.c
    ../overview/overview_data_scan.c
    ../overview/overview_data_graph.c
    ../overview/overview_data_sort.c
    ../overview/overview_scan.c
    ../overview/overview_layout.c
    ../overview/overview_cmdlog.c
    ../overview/overview_render.c
    ../overview/overview_render_detail.c
    ../overview/overview_render_utils.c
    ../overview/overview_render_relate.c
    ../overview/overview_render_streams.c
    ../overview/overview_render_procs.c
    ../overview/overview_render_fps.c
    ../overview/overview_render_graph.c
    ../overview/overview_render_status.c
    ../overview/overview_render_help.c
    ../overview/overview_render_cmdlog.c
    ../overview/overview_input.c
    ../overview/overview_ctrl.c
    ../overview/overview_fps_edit.c
    ../overview/overview_render_fps_params.c
    ../overview/stream_graph.c
)
target_include_directories(milkCTRL PRIVATE
    ${CMAKE_SOURCE_DIR}/src/engine/ImageStreamIO
    ${CMAKE_SOURCE_DIR}/src/engine/libprocessinfo
    ${CMAKE_SOURCE_DIR}/src/engine/libfps
    ${CMAKE_SOURCE_DIR}/src/cli/overview
)
target_link_libraries(milkCTRL PUBLIC
    ImageStreamIO milkprocessinfo milkfps m rt pthread)
install(TARGETS milkCTRL DESTINATION bin)
install(PROGRAMS ../overview/milk-setup-caps DESTINATION bin)

# milk-stream-graph: stream dependency graph tool
add_executable(milk-stream-graph
    ../overview/milk-stream-graph.c
    ../overview/stream_graph.c
    ../overview/overview_data.c
    ../overview/overview_data_sys.c
    ../overview/overview_data_cache.c
    ../overview/overview_data_scan.c
    ../overview/overview_data_graph.c
    ../overview/overview_data_sort.c
    ../overview/overview_scan.c
    ../overview/overview_layout.c
)
target_include_directories(milk-stream-graph PRIVATE
    ${CMAKE_SOURCE_DIR}/src/engine/ImageStreamIO
    ${CMAKE_SOURCE_DIR}/src/engine/libprocessinfo
    ${CMAKE_SOURCE_DIR}/src/engine/libfps
    ${CMAKE_SOURCE_DIR}/src/cli/overview
)
target_link_libraries(milk-stream-graph PUBLIC
    ImageStreamIO milkprocessinfo milkfps m rt pthread)
install(TARGETS milk-stream-graph DESTINATION bin)

# milk-stream-info: detailed stream info with connections
add_executable(milk-stream-info
    ../overview/milk-stream-info.c
    ../overview/overview_data.c
    ../overview/overview_data_sys.c
    ../overview/overview_data_cache.c
    ../overview/overview_data_scan.c
    ../overview/overview_data_graph.c
    ../overview/overview_data_sort.c
)
target_include_directories(milk-stream-info PRIVATE
    ${CMAKE_SOURCE_DIR}/src/engine/ImageStreamIO
    ${CMAKE_SOURCE_DIR}/src/engine/libprocessinfo
    ${CMAKE_SOURCE_DIR}/src/engine/libfps
    ${CMAKE_SOURCE_DIR}/src/cli/overview
)
target_link_libraries(milk-stream-info PUBLIC
    ImageStreamIO milkprocessinfo milkfps m rt pthread)
install(TARGETS milk-stream-info DESTINATION bin)

# milk-procinfo-info: detailed processinfo with connections
add_executable(milk-procinfo-info
    ../overview/milk-procinfo-info.c
    ../overview/overview_data.c
    ../overview/overview_data_sys.c
    ../overview/overview_data_cache.c
    ../overview/overview_data_scan.c
    ../overview/overview_data_graph.c
    ../overview/overview_data_sort.c
)
target_include_directories(milk-procinfo-info PRIVATE
    ${CMAKE_SOURCE_DIR}/src/engine/ImageStreamIO
    ${CMAKE_SOURCE_DIR}/src/engine/libprocessinfo
    ${CMAKE_SOURCE_DIR}/src/engine/libfps
    ${CMAKE_SOURCE_DIR}/src/cli/overview
)
target_link_libraries(milk-procinfo-info PUBLIC
    ImageStreamIO milkprocessinfo milkfps m rt pthread)
install(TARGETS milk-procinfo-info DESTINATION bin)

# milk-fuzzy-match: bigram fuzzy string matcher
add_executable(milk-fuzzy-match
    milk-fuzzy-match.c
)
target_include_directories(milk-fuzzy-match PRIVATE
    ${CMAKE_SOURCE_DIR}/src/engine/libfps
)
install(TARGETS milk-fuzzy-match DESTINATION bin)

# TESTING

include(CTest)

set (failRegex "ERROR" "FAIL")

add_test (NAME milklistim COMMAND milk-exec "mem.listim")
set_property (TEST milklistim PROPERTY LABELS "CLI")
set_tests_properties(milklistim PROPERTIES TIMEOUT 1)
set_property (TEST milklistim PROPERTY PASS_REGULAR_EXPRESSION "0 image")
