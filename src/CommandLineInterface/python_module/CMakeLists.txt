cmake_minimum_required(VERSION 3.6)

# Global option #
#################

set (CMAKE_CXX_STANDARD 14)

project(CacaoProcessTools LANGUAGES CXX)

hunter_add_package(pybind11)
find_package(pybind11 CONFIG REQUIRED)

add_library(CacaoProcessTools MODULE
CacaoProcessTools.cpp
)

# add_definitions(${CLIcore_DEFINITIONS})

# target_compile_options(CacaoProcessTools PRIVATE -fopenmp)

target_link_libraries(CacaoProcessTools PRIVATE pybind11::module CLIcore)

set_target_properties(CacaoProcessTools PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                            SUFFIX "${PYTHON_MODULE_EXTENSION}")
