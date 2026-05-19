cmake_minimum_required(VERSION 3.10)
project(example03_fps)

option(USE_NCURSES "Enable ncurses and TUI support" OFF)

set(MILK_ROOT "../..")

include_directories(${MILK_ROOT}/src/ImageStreamIO)
include_directories(${MILK_ROOT}/src)
include_directories(${MILK_ROOT}/src/libprocessinfo)
include_directories(${MILK_ROOT}/src/libfps)

include_directories(${MILK_ROOT}/src/libmilkTUI)
include_directories(${MILK_ROOT}/src/CommandLineInterface)
include_directories(${MILK_ROOT}/src/CommandLineInterface/streamCTRL)
include_directories(${MILK_ROOT}/src/CommandLineInterface/CLIcore)

link_directories(${MILK_ROOT}/_build/src/ImageStreamIO)
link_directories(${MILK_ROOT}/_build/src/libprocessinfo)
link_directories(${MILK_ROOT}/_build/src/libfps)

link_directories(${MILK_ROOT}/_build/src/CommandLineInterface)

# The libraries are targets in the parent project if added via add_subdirectory,
# but here they might be built separately. 
# We should prefer using the actual build artifact locations.

add_executable(milk-fpsexec-example-03-writer writer.c)



target_link_libraries(milk-fpsexec-example-03-writer milkfps milkprocessinfo ImageStreamIO m rt)



add_executable(milk-fpsexec-example-03-processor processor.c)

target_link_libraries(milk-fpsexec-example-03-processor milkfps milkprocessinfo ImageStreamIO m rt)



# Build as a single shared object module for milk-CLI



add_library(example03fps MODULE 



    example03fps_module.c 



    writer_module.c 



    writer.c 



    processor_module.c 



    processor.c



)



set_target_properties(example03fps PROPERTIES PREFIX "")



target_compile_definitions(example03fps PRIVATE MILK_MODULE)



target_link_libraries(example03fps milkfps milkprocessinfo ImageStreamIO CLIcore m rt)








