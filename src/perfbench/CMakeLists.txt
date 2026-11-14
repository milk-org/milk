# =======================================
# perfbench — Performance Benchmark Framework
# =======================================
#
# Provides: milk-perfbench       C benchmark harness (main) milk-perfbench.sh
# Bash harness (kept as ref) milk-perfbench-readprocinfo  C utility (reads
# processinfo SHM)
#

set(PERFBENCH_SCRIPTS scripts/milk-perfbench)

# ──────────────────────────────────────────
# C harness: milk-perfbench (main program)
# ──────────────────────────────────────────

add_executable(
  milk-perfbench milk-perfbench.c perfbench_setup.c perfbench_perf.c
                 perfbench_sysmon.c perfbench_report.c)

target_include_directories(
  milk-perfbench
  PRIVATE ${PROJECT_SOURCE_DIR}/src/engine/libprocessinfo
          ${PROJECT_SOURCE_DIR}/src/engine/ImageStreamIO
          ${PROJECT_SOURCE_DIR}/src/engine/libfps)

target_link_libraries(milk-perfbench milkprocessinfo ImageStreamIO)

install(TARGETS milk-perfbench DESTINATION bin)

# ──────────────────────────────────────────
# C utility: reads processinfo SHM → JSON
# ──────────────────────────────────────────

add_executable(milk-perfbench-readprocinfo milk-perfbench-readprocinfo.c)

target_include_directories(
  milk-perfbench-readprocinfo
  PRIVATE ${PROJECT_SOURCE_DIR}/src/engine/libprocessinfo
          ${PROJECT_SOURCE_DIR}/src/engine/ImageStreamIO
          ${PROJECT_SOURCE_DIR}/src/engine/libfps)

target_link_libraries(milk-perfbench-readprocinfo milkprocessinfo ImageStreamIO)

install(TARGETS milk-perfbench-readprocinfo DESTINATION bin)

# ──────────────────────────────────────────
# C utility: create dummy SHM streams
# ──────────────────────────────────────────

add_executable(milk-perfbench-mkstream perfbench-mkstream.c)

target_include_directories(
  milk-perfbench-mkstream PRIVATE ${PROJECT_SOURCE_DIR}/src/engine/ImageStreamIO
                                  ${PROJECT_SOURCE_DIR}/src/engine/libfps)

target_link_libraries(milk-perfbench-mkstream ImageStreamIO)

install(TARGETS milk-perfbench-mkstream DESTINATION bin)

# ──────────────────────────────────────────
# Install scripts (bash fallback)
# ──────────────────────────────────────────

# Installed alongside the C binary as .sh
install(
  PROGRAMS ${PERFBENCH_SCRIPTS}
  DESTINATION bin
  RENAME milk-perfbench.sh)
