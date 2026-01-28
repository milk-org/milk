cmake_minimum_required(VERSION 3.10)
project(example03_fps)

set(MILK_ROOT "../..")

include_directories(${MILK_ROOT}/src/ImageStreamIO)
include_directories(${MILK_ROOT}/src)
include_directories(${MILK_ROOT}/src/libprocessinfo)
include_directories(${MILK_ROOT}/src/libfps)
include_directories(${MILK_ROOT}/src/libmilkTUI)
include_directories(${MILK_ROOT}/src/CommandLineInterface)
include_directories(${MILK_ROOT}/src/CommandLineInterface/streamCTRL)
include_directories(${MILK_ROOT}/src/CommandLineInterface/CLIcore)

link_directories(${MILK_ROOT}/build/src/ImageStreamIO)
link_directories(${MILK_ROOT}/build/src/libprocessinfo)
link_directories(${MILK_ROOT}/build/src/libfps)
link_directories(${MILK_ROOT}/build/src/CommandLineInterface)

add_executable(milk-example-03-writer writer.c)

target_link_libraries(milk-example-03-writer milkfps milkprocessinfo ImageStreamIO CLIcore m rt)



add_executable(milk-example-03-processor processor.c)

target_link_libraries(milk-example-03-processor milkfps milkprocessinfo ImageStreamIO CLIcore m rt)



# Build as a shared object module for milk-CLI



add_library(processor03 MODULE processor_module.c processor.c)



set_target_properties(processor03 PROPERTIES PREFIX "")



target_compile_definitions(processor03 PRIVATE MILK_MODULE)



target_link_libraries(processor03 milkfps milkprocessinfo ImageStreamIO CLIcore m rt)







add_library(writer03 MODULE writer_module.c writer.c)



set_target_properties(writer03 PROPERTIES PREFIX "")



target_compile_definitions(writer03 PRIVATE MILK_MODULE)



target_link_libraries(writer03 milkfps milkprocessinfo ImageStreamIO CLIcore m rt)




