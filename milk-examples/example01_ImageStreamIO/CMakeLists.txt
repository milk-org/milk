cmake_minimum_required(VERSION 3.10)
project(example01_ImageStreamIO)

set(MILK_ROOT "../..")

include_directories(${MILK_ROOT}/src/ImageStreamIO)

# Find ImageStreamIO library in the build directory
link_directories(${MILK_ROOT}/_build/src/ImageStreamIO)

add_executable(milk-example-01-writer writer.c)
target_link_libraries(milk-example-01-writer ImageStreamIO m rt)

add_executable(milk-example-01-processor processor.c)
target_link_libraries(milk-example-01-processor ImageStreamIO m rt)
