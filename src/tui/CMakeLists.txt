# Engine-tier terminal UIs. No CLIcore linkage.

# ==========================================
# milk-fpsCTRL executable
# ==========================================

set(SOURCEFILES_FPSCTRL
    fpsCTRL/milk-fpsCTRL.c
    fpsCTRL/fpsCTRL_FPSdisplay.c
    fpsCTRL/fpsCTRL_TUI.c
    fpsCTRL/fpsCTRL_TUIcompat.c
    fpsCTRL/fpsCTRL_TUI_process_user_key.c
    fpsCTRL/level0node_summary.c
    fpsCTRL/print_nodeinfo.c
    fpsCTRL/fpsCTRL_units.c
    fpsCTRL/fpsCTRL_globals.c)

add_executable(milk-fpsCTRL ${SOURCEFILES_FPSCTRL})
target_include_directories(
  milk-fpsCTRL PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                       ${CMAKE_CURRENT_SOURCE_DIR}/fpsCTRL)
target_link_libraries(
  milk-fpsCTRL PUBLIC milkcomputeunit milkfps milkprocessinfo
                      ImageStreamIO pthread m)
install(TARGETS milk-fpsCTRL DESTINATION bin)

# ==========================================
# milk-procCTRL executables
# ==========================================

add_executable(
  milk-procCTRL
  procCTRL/milk-procCTRL.c
  procCTRL/procCTRL_GetCPUloads.c
  procCTRL/procCTRL_GetNumberCPUs.c
  procCTRL/procCTRL_PIDcollectSystemInfo.c
  procCTRL/procCTRL_processinfo_scan.c
  procCTRL/procCTRL_TUI.c
  procCTRL/procCTRL_TUI_render.c
  procCTRL/procCTRL_TUI_input.c
  procCTRL/procCTRL_TUI_sort.c
  procCTRL/procCTRL_TUIcompat.c)
target_link_libraries(milk-procCTRL PUBLIC milkprocessinfo ImageStreamIO
                                           pthread m)
target_include_directories(
  milk-procCTRL
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/procCTRL
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)

add_executable(milk-procCTRL-scan procCTRL/milk-procCTRL-scan.c
                                  procCTRL/procCTRL_PIDcollectSystemInfo.c)
target_link_libraries(milk-procCTRL-scan PUBLIC milkprocessinfo ImageStreamIO
                                                pthread m)
target_include_directories(
  milk-procCTRL-scan
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/procCTRL
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)

install(TARGETS milk-procCTRL DESTINATION bin)
install(TARGETS milk-procCTRL-scan DESTINATION bin)
