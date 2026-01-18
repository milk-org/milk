cmake_minimum_required(VERSION 3.10)
project(example02_processinfo)

set(MILK_ROOT "../..")

include_directories(${MILK_ROOT}/src/ImageStreamIO)
include_directories(${MILK_ROOT}/src)
include_directories(${MILK_ROOT}/src/libprocessinfo)

link_directories(${MILK_ROOT}/_build/src/ImageStreamIO)
link_directories(${MILK_ROOT}/_build/src/libprocessinfo)

add_executable(writer writer.c)
target_link_libraries(writer ImageStreamIO m rt)

add_executable(processor processor.c)
target_link_libraries(processor processinfo ImageStreamIO m rt)
