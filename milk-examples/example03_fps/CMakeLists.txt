cmake_minimum_required(VERSION 3.10)
project(example03_fps)

option(USE_NCURSES "Enable ncurses and TUI support" OFF)

set(MILK_ROOT "../..")
set(MILK_INSTALL "${MILK_ROOT}/_build/_install")

include_directories(${MILK_ROOT}/src)
include_directories(${MILK_ROOT}/src/engine/ImageStreamIO)
include_directories(${MILK_ROOT}/src/engine/libfps)
include_directories(${MILK_ROOT}/src/engine/libcomputeunit)
include_directories(${MILK_ROOT}/src/engine/libprocessinfo)
include_directories(${MILK_ROOT}/src/engine/libmilkdata)
include_directories(${MILK_ROOT}/src/engine/libmilkcommon)
include_directories(${MILK_ROOT}/src/cli/CLIcore)
include_directories(${MILK_ROOT}/src/cli/libmilkscript)

link_directories(${MILK_INSTALL}/lib)

add_executable(milk-fpsexec-example-03-writer writer.c)

target_link_libraries(milk-fpsexec-example-03-writer milkfps milkcomputeunit
                      milkprocessinfo ImageStreamIO m rt)

add_executable(milk-fpsexec-example-03-processor processor.c)

target_link_libraries(milk-fpsexec-example-03-processor milkfps milkcomputeunit
                      milkprocessinfo ImageStreamIO m rt)

# Module loaded by milk-cli: the module file exports its MILK_RUNNABLE list
# (MILK_MODULE_RUNNABLES), the CLI core registers the commands after dlopen().
add_library(example03fps MODULE example03fps_module.c writer_module.c writer.c
                                processor_module.c processor.c)

set_target_properties(example03fps PROPERTIES PREFIX "")

# Excludes the standalone main() of writer.c / processor.c from the module
target_compile_definitions(example03fps PRIVATE EXAMPLE03_MODULE_BUILD)

target_link_libraries(
  example03fps
  milkfps
  milkcomputeunit
  milkprocessinfo
  ImageStreamIO
  m
  rt)
