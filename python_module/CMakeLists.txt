cmake_minimum_required(VERSION 3.6)

# Global option #
#

set(CMAKE_CXX_STANDARD 14)

project(CacaoProcessTools LANGUAGES CXX)

if(NOT TARGET pybind11)
  execute_process(COMMAND bash -c "python -m pybind11 --includes"
                  OUTPUT_VARIABLE pybind11_inc)
  execute_process(COMMAND bash -c "python3-config --extension-suffix"
                  OUTPUT_VARIABLE PYTHON_MODULE_EXTENSION)
  string(REPLACE "-I" "" pybind11_inc ${pybind11_inc})
  string(REPLACE " " ";" pybind11_inc ${pybind11_inc})
  string(REGEX REPLACE "\n$" "" pybind11_inc "${pybind11_inc}")
  string(REGEX REPLACE "\n$" "" PYTHON_MODULE_EXTENSION "${PYTHON_MODULE_EXTENSION}")

  add_library(CacaoProcessTools MODULE CacaoProcessTools.cpp)

  target_compile_features(CacaoProcessTools PRIVATE cxx_std_14)
  target_include_directories(CacaoProcessTools PRIVATE "${pybind11_inc}")
  target_compile_options(CacaoProcessTools PRIVATE "-Wno-deprecated-declarations")
  set_target_properties(CacaoProcessTools
          PROPERTIES PREFIX
              "${PYTHON_MODULE_PREFIX}"
              SUFFIX
              "${PYTHON_MODULE_EXTENSION}")

else()
  pybind11_add_module(CacaoProcessTools CacaoProcessTools.cpp)
endif()

target_link_libraries(CacaoProcessTools PRIVATE CLIcore ImageStreamIO)

install(TARGETS CacaoProcessTools
        EXPORT CacaoProcessToolsConfig
        ARCHIVE DESTINATION python
        LIBRARY DESTINATION python)
