cmake_minimum_required(VERSION 3.16)
project(fpsvalkey C)

set(CMAKE_C_STANDARD 11)
add_definitions(-D_GNU_SOURCE)

# ==========================================
# Find dependencies
# ==========================================

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# libvalkey
# If not system-installed, set PKG_CONFIG_PATH:
#   PKG_CONFIG_PATH=/path/to/libvalkey/build cmake ..
pkg_check_modules(VALKEY REQUIRED valkey)

# ==========================================
# Locate milk source tree and build tree
# ==========================================

# MILK_SRC: root of the milk source tree
# Default: two levels up from this CMakeLists.txt
if(NOT DEFINED MILK_SRC)
    get_filename_component(MILK_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
endif()

# MILK_BUILD: milk build directory
# Default: MILK_SRC/_build
if(NOT DEFINED MILK_BUILD)
    set(MILK_BUILD "${MILK_SRC}/_build")
endif()

message(STATUS "MILK_SRC:   ${MILK_SRC}")
message(STATUS "MILK_BUILD: ${MILK_BUILD}")

if(NOT EXISTS "${MILK_SRC}/src/libfps/fps.h")
    message(FATAL_ERROR
        "Cannot find milk source tree. "
        "Set -DMILK_SRC=/path/to/milk")
endif()

if(NOT EXISTS "${MILK_BUILD}")
    message(FATAL_ERROR
        "Cannot find milk build dir. "
        "Set -DMILK_BUILD=/path/to/milk/_build")
endif()

# ==========================================
# milk-fps-valkey executable
# ==========================================

add_executable(milk-fps-valkey
    milk-fps-valkey.c
    fps_valkey.c
)

# Use milk source tree for includes
# (installed headers are incomplete)
target_include_directories(milk-fps-valkey PRIVATE
    ${VALKEY_INCLUDE_DIRS}
    ${MILK_SRC}/src
    ${MILK_SRC}/src/libfps
    ${MILK_SRC}/src/CommandLineInterface
    ${MILK_SRC}/src/CommandLineInterface/CLIcore
    ${MILK_SRC}/src/ImageStreamIO
    ${MILK_SRC}/src/libprocessinfo
    ${MILK_BUILD}
)

# link against built milk libraries
target_link_directories(milk-fps-valkey PRIVATE
    ${VALKEY_LIBRARY_DIRS}
    ${MILK_BUILD}/src/libfps
    ${MILK_BUILD}/src/libprocessinfo
    ${MILK_BUILD}/src/ImageStreamIO
)

target_link_libraries(milk-fps-valkey PRIVATE
    ${VALKEY_LIBRARIES}
    milkfps
    milkprocessinfo
    ImageStreamIO
    Threads::Threads
    m
    rt
)

# ==========================================
# Install
# ==========================================

install(TARGETS milk-fps-valkey DESTINATION bin)
