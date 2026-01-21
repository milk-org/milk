cmake_minimum_required(VERSION 3.10)
project(example03_fps)

set(MILK_ROOT "../..")

include_directories(${MILK_ROOT}/src/ImageStreamIO)
include_directories(${MILK_ROOT}/src)
include_directories(${MILK_ROOT}/src/libprocessinfo)
include_directories(${MILK_ROOT}/src/libfps)
include_directories(${MILK_ROOT}/src/libmilkTUI)

link_directories(${MILK_ROOT}/build/src/ImageStreamIO)
link_directories(${MILK_ROOT}/build/src/libprocessinfo)
link_directories(${MILK_ROOT}/build/src/libfps)
link_directories(${MILK_ROOT}/build/src/CommandLineInterface)

add_executable(milk-example-03-writer writer.c)
target_link_libraries(milk-example-03-writer ImageStreamIO m rt)

add_executable(milk-example-03-processor processor.c)
target_link_libraries(milk-example-03-processor milkfps milkprocessinfo ImageStreamIO CLIcore m rt)
