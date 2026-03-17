# =======================================
# perfbench — Performance Benchmark Framework
# =======================================
#
# Provides:
#   milk-perfbench              Shell harness script
#   milk-perfbench-readprocinfo C utility (reads
#                               processinfo SHM)
#

set(PERFBENCH_SCRIPTS
    scripts/milk-perfbench
)

# ──────────────────────────────────────────
# C utility: reads processinfo SHM → JSON
# ──────────────────────────────────────────

add_executable(milk-perfbench-readprocinfo
    milk-perfbench-readprocinfo.c)

target_include_directories(
    milk-perfbench-readprocinfo PRIVATE
    ${PROJECT_SOURCE_DIR}/src/engine/libprocessinfo
    ${PROJECT_SOURCE_DIR}/src/engine/ImageStreamIO)

target_link_libraries(
    milk-perfbench-readprocinfo
    milkprocessinfo ImageStreamIO)

install(TARGETS milk-perfbench-readprocinfo
    DESTINATION bin)

# ──────────────────────────────────────────
# Install scripts
# ──────────────────────────────────────────

install(PROGRAMS ${PERFBENCH_SCRIPTS}
    DESTINATION bin)

# ──────────────────────────────────────────
# CTest benchmarks (label: perf)
# ──────────────────────────────────────────

include(CTest)

# Smoke test: run the template compute unit
# through the harness with 1 iteration
set(TESTNAME "perf-clitest")
add_test(NAME "${TESTNAME}"
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/milk-perfbench
        milk-fpsexec-clitest 1
        -o ${CMAKE_BINARY_DIR}/perfresults)
set_property(TEST "${TESTNAME}"
    PROPERTY LABELS "perf")
set_property(TEST "${TESTNAME}"
    PROPERTY TIMEOUT 30)
set_property(TEST "${TESTNAME}"
    PROPERTY ENVIRONMENT
    "PATH=${CMAKE_BINARY_DIR}/src/perfbench:${CMAKE_BINARY_DIR}/src/milk_module_example:$ENV{PATH}")
