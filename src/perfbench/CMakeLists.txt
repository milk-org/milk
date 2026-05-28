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

# ──────────────────────────────────────────
# CTest benchmarks (label: perf)
# ──────────────────────────────────────────

include(CTest)

# Smoke test: run the template compute unit through the harness. Uses a wrapper
# script that creates the cam01 stream required by clitest.
set(TESTNAME "perf-clitest")
add_test(NAME "${TESTNAME}"
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/test-perf-clitest.sh 100
                 ${CMAKE_BINARY_DIR}/perfresults)
set_property(TEST "${TESTNAME}" PROPERTY LABELS "perf")
set_property(TEST "${TESTNAME}" PROPERTY TIMEOUT 30)
set_property(
  TEST "${TESTNAME}"
  PROPERTY
    ENVIRONMENT
    "PATH=${CMAKE_BINARY_DIR}/src/perfbench:${CMAKE_BINARY_DIR}/src/milk_module_example:${CMAKE_BINARY_DIR}/src/engine/libfps:${PROJECT_SOURCE_DIR}/scripts:$ENV{PATH}"
)
