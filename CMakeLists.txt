cmake_minimum_required(VERSION 3.17)

project(milk LANGUAGES C)

# Examples : cmake .. -DUSE_CUDA=ON -DINSTALLMAKEDEFAULT=ON cmake ..
# -DUSE_CUDA=ON -DUSE_MAGMA=ON cmake .. -DCMAKE_BUILD_TYPE=Debug

# =======================================
# BUILD PROFILE OPTIONS
# =======================================
#
# USE_CLI (default ON): Build the interactive CLI (milk-cli), including CLIcore,
# readline, bison/flex parser, and TUI. When OFF, only core libraries and
# standalone fpsexec executables are built. This removes the need for readline,
# ncurses, bison, and flex.
#
# USE_NCURSES (default ON, requires USE_CLI): Enable ncurses TUI (milk-fpsCTRL,
# streamCTRL).
#
# USE_READLINE (default ON, requires USE_CLI): Enable readline for interactive
# CLI input.
#
# USE_CFITSIO (default ON): Enable FITS file I/O (COREMOD_iofits). Requires
# cfitsio. Disable for a POSIX-only core build.
#

# Build tiers (see docs/install/build_tiers.md):
#
# Engine-only (POSIX deps only): cmake .. -DUSE_COREMODS=OFF -DUSE_CLI=OFF
#
# Core (engine + coremods, no CLI/plugins): cmake .. -DUSE_CLI=OFF
#
# Core without cfitsio: cmake .. -DUSE_CLI=OFF -DUSE_CFITSIO=OFF
#
# Full build (default): cmake ..
#
# Full build with CUDA: cmake .. -DUSE_CUDA=ON
#
# Debug build:     -DCMAKE_BUILD_TYPE=Debug Coverage build:
# -DCMAKE_BUILD_TYPE=Coverage

option(USE_COREMODS "Build core modules (COREMOD_*)" ON)
option(USE_CFITSIO "Build FITS I/O (needs cfitsio)" ON)
option(USE_CLI "Build interactive CLI (milk-cli)" ON)
option(USE_SEQ "Build sequencer module (requires CLI)" ON)
option(USE_NCURSES "Enable ncurses and TUI support" ON)
option(USE_READLINE "Enable readline support" ON)
option(USE_TREESITTER "Enable tree-sitter syntax highlighting (CLI)" ON)

option(USE_STATIC_LTO "Static-link standalone exes for full LTO" OFF)

# Enforce dependency chain: USE_CLI requires USE_COREMODS
if(USE_CLI AND NOT USE_COREMODS)
  message(STATUS "USE_CLI requires USE_COREMODS; enabling USE_COREMODS")
  set(USE_COREMODS
    ON
    CACHE BOOL "" FORCE)
endif()

# USE_SEQ requires USE_CLI since it relies on CLIcore for commands
if(USE_SEQ AND NOT USE_CLI)
  message(STATUS "USE_SEQ requires USE_CLI; disabling USE_SEQ")
  set(USE_SEQ
    OFF
    CACHE BOOL "" FORCE)
endif()

# =======================================
# CMAKE GENERAL CONFIG
# =======================================
include(CheckLanguage)

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

enable_language(C)

# =======================================
# PACKAGE GENERAL CONFIG
# =======================================
set(VERSION_MAJOR 1)
set(VERSION_MINOR 03)
set(VERSION_PATCH 00)
set(MILK_VERSION_OPTION "")
set(PROJECT_VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
set(PROJECT_DESCRIPTION "Modular Image processing tooLKit")

set(SOURCEDIR "${PROJECT_SOURCE_DIR}")

option(INSTALLMAKEDEFAULT "Make this installation the system default" OFF)

# turn VERBOSE ON set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_LINK_DEPENDS_NO_SHARED true)
set(MILK_SUFFIX "milk-${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
string(FIND "${CMAKE_INSTALL_PREFIX}" "${MILK_SUFFIX}" _suffix_pos REVERSE)
if(_suffix_pos EQUAL -1)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/${MILK_SUFFIX}")
endif()
set(INSTALLDIR "${CMAKE_INSTALL_PREFIX}")

# Configure a header file to pass some of the CMake settings to the source code
configure_file("${PROJECT_SOURCE_DIR}/src/milk_config.h.in"
  "${PROJECT_SOURCE_DIR}/src/milk_config.h")

configure_file("${PROJECT_SOURCE_DIR}/src/config.h.in"
  "${PROJECT_SOURCE_DIR}/src/config.h")

# select most recent version available
set(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)
set(CMAKE_FIND_PACKAGE_SORT_DIRECTION DEC)

# =======================================
# LIBRARIES
# =======================================

# options
option(USE_CUDA "Use CUDA" OFF)
option(USE_MAGMA "Use MAGMA" OFF)

# BLAS + LAPACKE Options
option(USE_MKL "Use MKL" OFF)
option(USE_OPENBLAS "Use OpenBLAS" OFF)
option(USE_LAPACKE "Use LAPACKE" OFF)

if(USE_CUDA)
  # --> Either HAVE_CUDA or USE_CUDA=OFF
  include(cmake/have_cuda.cmake)
endif()

if(USE_CUDA AND USE_MAGMA)
  # --> Either HAVE_MAGMA or USE_MAGMA=OFF
  include(cmake/have_magma.cmake)
endif()

if(USE_CFITSIO)
  # --> Either HAVE_CFITSIO or USE_CFITSIO=OFF
  include(cmake/have_cfitsio.cmake)
endif()

if(USE_MKL
  OR USE_OPENBLAS
  OR USE_LAPACKE)
  # --> Processes all 3 and sets HAVE_BLAS, HAVE_MKL, HAVE_OPENBLAS and
  # HAVE_LAPACKE
  include(cmake/have_blas_lapacke.cmake)
endif()

# apply_*_to_target() helpers are now defined; load the extension scanner.
include(cmake/milk_extensions.cmake)

# =======================================
# MODULES INCLUDED
# =======================================
# modules are added to two lists: libsrcdir : source code directory libname   :
# shared object name

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  # Update submodules as needed
  option(GIT_SUBMODULE "Check submodules during build" ON)
  if(GIT_SUBMODULE)
    message(STATUS "Submodule update")
    execute_process(
      COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(
        FATAL_ERROR
        "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules"
      )
    endif()
  endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/src/engine/ImageStreamIO/CMakeLists.txt")
  message(
    FATAL_ERROR
    "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again."
  )
endif()

if(USE_CLI)
  list(APPEND libsrcdir src/cli/libmilkscript)
  list(APPEND libname milkscript)

  list(APPEND libsrcdir src/cli/CLIcore)
  list(APPEND libname CLIcore)
else()
  # Without CLI, modules use CLIcore_standalone.h stubs via the MILK_NO_CLI
  # compile definition.
  add_definitions(-DMILK_NO_CLI)
  # Provide CLI header paths so modules can find CLIcore.h (which redirects to
  # standalone stubs)
  include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cli
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cli/CLIcore
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cli/CLIcore/CLIcore
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cli/streamCTRL
    ${CMAKE_CURRENT_SOURCE_DIR}/src/engine/libprocessinfo/procCTRL
    ${CMAKE_CURRENT_SOURCE_DIR}/src/engine/libfps
    ${CMAKE_CURRENT_SOURCE_DIR}/src/engine/libfps/fpsCTRL
    ${CMAKE_CURRENT_SOURCE_DIR}/src/engine/libprocessinfo
    ${CMAKE_CURRENT_SOURCE_DIR}/src/engine/libmilkdata
    ${CMAKE_CURRENT_SOURCE_DIR}/src/engine/libmilkcommon
    ${CMAKE_CURRENT_SOURCE_DIR}/src/engine/ImageStreamIO
    ${CMAKE_CURRENT_SOURCE_DIR}/src/coremods
    ${CMAKE_CURRENT_SOURCE_DIR}/src/coremods/COREMOD_memory
    ${CMAKE_CURRENT_SOURCE_DIR}/src/coremods/COREMOD_iofits)
  install(FILES src/cli/CLIcore/CLIcore.h src/cli/CLIcore/CLIcore_standalone.h
    src/cli/CLIcore/cmdsettings.h
    DESTINATION include/CommandLineInterface)
  # timeutils.h, milkDebugTools.h, and IMGID.h have moved to their respective
  # engine modules (libprocessinfo, libmilkcommon, libfps) which install them
  # unconditionally via their own CMakeLists.txt targets.
  install(FILES src/cli/CLIcore/CLIcore/CLIcore_checkargs.h
    DESTINATION include/CommandLineInterface/CLIcore)
  # Create an INTERFACE library target named CLIcore so that existing
  # target_link_libraries(... CLIcore) calls in plugin CMakeLists work without
  # modification. The INTERFACE target provides header paths only;
  # CLIcore_standalone.h stubs handle all symbols.
  add_library(CLIcore INTERFACE)
  target_include_directories(
    CLIcore
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/cli>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/cli/streamCTRL>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/engine/libprocessinfo/procCTRL>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/engine/libfps>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/engine/libfps/fpsCTRL>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/engine/libprocessinfo>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/engine/libmilkdata>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/engine/ImageStreamIO>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/coremods/COREMOD_memory>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/coremods/COREMOD_iofits>
    $<INSTALL_INTERFACE:include>)
  install(TARGETS CLIcore EXPORT milkTargets)
endif()

list(APPEND libsrcdir src/engine/ImageStreamIO)
list(APPEND libname ImageStreamIO m)

if(USE_COREMODS)
  list(APPEND modulelist COREMOD_arith COREMOD_memory COREMOD_tools)
  if(USE_CFITSIO)
    list(APPEND modulelist COREMOD_iofits)
  endif()

  foreach(mname IN LISTS modulelist)
    string(REPLACE "_" "" mname1 "${mname}")
    list(APPEND libsrcdir src/coremods/${mname})
    list(APPEND libname milk${mname1})
    message("======= adding module  ${mname}")
  endforeach()
else()
  message(STATUS "USE_COREMODS=OFF: skipping COREMOD_*")
endif()

if(USE_SEQ)
  list(APPEND libsrcdir src/sequencer)
  list(APPEND libname milksequencer)
  message("======= adding module  sequencer")
endif()

# =======================================
# BUILD FLAGS
# =======================================

# Can add -fanalyzer if gcc10+ (warning: this will take a while to run)
set(CMAKE_C_FLAGS_DEBUG "-O0 -g -Wall -Wextra -std=gnu17 -Wno-sign-compare")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra -std=c++17 -Wno-sign-compare")

set(CMAKE_C_FLAGS_RELEASE "-Ofast -DNDEBUG -std=gnu17 -funroll-loops")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG -std=c++17 -funroll-loops")

set(CMAKE_C_FLAGS_COVERAGE "-O0 -g --coverage -fprofile-arcs -ftest-coverage")
set(CMAKE_CXX_FLAGS_COVERAGE "-O0 -g --coverage -fprofile-arcs -ftest-coverage")

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# ======== PGO ========
include(cmake/pgo.cmake) # -DUSE_PGO=(""|GENERATE|USE)

# =======================================
# STATIC LTO FOR STANDALONE EXECUTABLES
# =======================================
# When USE_STATIC_LTO is ON, standalone fpsexec executables link static archives
# (.a) of all milk libraries so that the LTO linker has full cross-module
# visibility. Usage: cmake .. -DUSE_STATIC_LTO=ON
if(USE_STATIC_LTO)
  message(
    STATUS "STATIC_LTO: standalones will link static archives for full LTO")
  add_link_options(-flto=auto)
endif()

# =======================================
# SANITIZERS (development opt-in)
# =======================================
# Usage: cmake .. -DUSE_SANITIZERS=ON Enables AddressSanitizer and
# UndefinedBehaviorSanitizer for catching memory errors and undefined behavior.
# Not for production — significant overhead.
option(USE_SANITIZERS "Enable ASan + UBSan (development only)" OFF)
if(USE_SANITIZERS)
  message(STATUS "SANITIZERS: ASan + UBSan enabled")
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

# =======================================
# COVERAGE BUILD — global propagation
# =======================================
if(CMAKE_BUILD_TYPE STREQUAL "Coverage")
  add_compile_definitions(COVERAGE_BUILD=1)
  add_link_options(--coverage)
  link_libraries(gcov)
endif()

# =======================================
# MISC OPTIONS
# =======================================
# Catch printf-family format string bugs
add_compile_options(
  $<$<COMPILE_LANGUAGE:C>:-Wformat=2>
  $<$<COMPILE_LANGUAGE:C>:-Wformat-overflow>
  $<$<COMPILE_LANGUAGE:C>:-Wformat-truncation>)
# Some optims for faster compilation
add_compile_options($<$<COMPILE_LANGUAGE:C>:-march=native>)
add_compile_options($<$<COMPILE_LANGUAGE:C>:-pipe>)

# =======================================
# COMPILE OPTIONS
# =======================================

# Vectorization diagnostics (development aid)
option(VEC_REPORT "Print GCC vectorization report" OFF)
if(VEC_REPORT)
  add_compile_options(-fopt-info-vec-missed)
endif()

# link-time optimizer
option(USE_LTO "Enable Link-Time Optimization (-flto=auto)" ON)
if(USE_LTO)
  add_compile_options($<$<COMPILE_LANGUAGE:C>:-flto=auto>)
endif()
# NOTE: -fwhole-program is applied only to standalone executables (via
# milk_standalone.cmake) where main() is the sole entry point.  Applying it
# globally would cause GCC to discard exported symbols from shared libraries.
# NOTE: -fstack-usage is incompatible with link-time optimizer
# add_compile_options(-fstack-usage)

message("COMPILE_OPTIONS:  ${COMPILE_OPTIONS}")

message("====================================================")
message("VERSION  ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
message("====================================================")
message("CMAKE_THREAD_LIBS_INIT  :  ${CMAKE_THREAD_LIBS_INIT}")
message("PROJECT_SOURCE_DIR       = ${PROJECT_SOURCE_DIR}")
message("PROJECT_BINARY_DIR       = ${PROJECT_BINARY_DIR}")

message("CMAKE_INSTALL_PREFIX     = ${CMAKE_INSTALL_PREFIX}")

message("USE_CUDA                 = ${USE_CUDA}")
message("USE_MAGMA                = ${USE_MAGMA}")

# TODO I can't find a trace of where EXTRAMODULES is used?
message("EXTRAMODULES             = ${EXTRAMODULES}")

# =======================================
# TESTING
# =======================================

include(CTest)

# =======================================
# STANDALONE EXECUTABLE HELPERS
# =======================================
# Functions: add_milk_standalone, add_cacao_standalone,
# add_cacao_standalone_plugins See cmake/milk_standalone.cmake for docs.

include(${PROJECT_SOURCE_DIR}/cmake/milk_standalone.cmake)

# =======================================
# INSTALL PATHS
# =======================================

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Expose architectural tiers globally so that legacy #include
# =====================================================================
# 1. INCLUDES
# =====================================================================

# Global includes removed in favor of target_include_directories Exception:
# src/engine is needed globally so that all targets can resolve #include
# "ImageStreamIO/ImageStreamIO.h"
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/engine)

# =======================================
# SOURCE DIRECTORIES
# =======================================
# Core modules (libsrcdir) and plugins (libextradir) are discovered and added as
# subdirectories. Discover plugins (only when coremods are enabled)
if(USE_COREMODS)
  execute_process(
    COMMAND
    bash -c
    "find -L ${PROJECT_SOURCE_DIR}/plugins -mindepth 1 -maxdepth 2 -type d | grep -v .git | grep -v .vscode"
    OUTPUT_VARIABLE libextradir)
  string(REGEX REPLACE "\n" "; " libextradir "${libextradir}")
endif()

foreach(lsrcdir IN LISTS libsrcdir libextradir)
  string(REGEX REPLACE " " "" lsrcdir "${lsrcdir}")
  if(NOT "${lsrcdir}" STREQUAL "")
    if(IS_ABSOLUTE "${lsrcdir}")
      set(abs_lsrcdir "${lsrcdir}")
    else()
      set(abs_lsrcdir "${CMAKE_CURRENT_SOURCE_DIR}/${lsrcdir}")
    endif()
    if(EXISTS "${abs_lsrcdir}/CMakeLists.txt")
      message("ADDING SUBDIR = ${lsrcdir}")
      add_subdirectory(${lsrcdir})
    endif()
  endif()
endforeach()

# example module and isio-tools depend on coremods
if(USE_COREMODS)
  add_subdirectory(src/milk_module_example)
  add_subdirectory(src/isio-tools)
  add_subdirectory(src/perfbench)
endif()

# add libfps (standalone fps library exploration)
add_subdirectory(src/engine/libmilkcommon)
add_subdirectory(src/engine/libfps)
add_subdirectory(src/engine/libfpsseq)

# add libmilkdata (core data layer, no CLI dependency)
add_subdirectory(src/engine/libmilkdata)

# add libprocessinfo
add_subdirectory(src/engine/libprocessinfo)

# Package definitions (needed by both CLI and standalone)
add_definitions(
  -DPACKAGE_NAME="milk" -DCONFIGDIR="/config" -DINSTALLDIR="${INSTALLDIR}"
  -DSOURCEDIR="${SOURCEDIR}" -DABSSRCTOPDIR="${SOURCEDIR}"
  -DPACKAGE_BUGREPORT="https://github.com/milk-org/milk/issues")

set(PROGRAM_PERMISSIONS_DEFAULT
  OWNER_WRITE
  OWNER_READ
  OWNER_EXECUTE
  GROUP_READ
  GROUP_EXECUTE
  WORLD_READ
  WORLD_EXECUTE)

message("====================================================")
message("CMAKE_BUILD_TYPE        : ${CMAKE_BUILD_TYPE}")
message("CMAKE_C_FLAGS_DEBUG           : ${CMAKE_C_FLAGS_DEBUG}")
message("CMAKE_C_FLAGS_RELEASE         : ${CMAKE_C_FLAGS_RELEASE}")
message("CMAKE_C_FLAGS_COVERAGE        : ${CMAKE_C_FLAGS_COVERAGE}")
get_cmake_property(_all_vars VARIABLES)
set(_use_line "USE_*  :")
set(_have_line "HAVE_* :")
foreach(_v IN LISTS _all_vars)
  if(_v MATCHES "^USE_")
    string(APPEND _use_line "  ${_v}=${${_v}}")
  elseif(_v MATCHES "^HAVE_")
    string(APPEND _have_line "  ${_v}=${${_v}}")
  endif()
endforeach()
unset(_all_vars)
message("${_use_line}")
message("${_have_line}")
message("====================================================")


if(USE_CLI)
  # main CLI executable (only when USE_CLI is ON)
  add_executable(milk-cli src/CLImain.c)
  target_link_libraries(milk-cli PUBLIC ${libname})
  install(
    TARGETS milk-cli
    DESTINATION bin
    PERMISSIONS ${PROGRAM_PERMISSIONS_DEFAULT})
else()

  message("====================================================")
  message("USE_CLI=OFF: Skipping milk-cli, CLIcore, TUI")
  message("  Only standalone fpsexec programs will be built")
  message("====================================================")

endif(USE_CLI)

# =======================================
# MAKE DEFAULT
# =======================================

set(CMAKE_INSTALL_PREFIX_LINK "/usr/local/${PROJECT_NAME}")
if(INSTALLMAKEDEFAULT)
  install(
    CODE "
    EXECUTE_PROCESS(COMMAND ln -snf ${CMAKE_INSTALL_PREFIX} ${CMAKE_INSTALL_PREFIX_LINK})
    ")
endif(INSTALLMAKEDEFAULT)

# =======================================
# CONFIG.CMAKE
# =======================================

set(INCLUDE_INSTALL_DIR
  ${CMAKE_INSTALL_PREFIX}/include
  CACHE PATH "Location of header files")
set(LIBRARY_INSTALL_DIR
  ${CMAKE_INSTALL_PREFIX}/lib
  CACHE PATH "Location of libraries")

# Export the targets to a file for use in the Config.cmake file
install(
  EXPORT ${PROJECT_NAME}Targets
  FILE "${PROJECT_NAME}Targets.cmake"
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION lib/cmake)

# Set alias for collective milk::milk_all target as well
add_library(${PROJECT_NAME}_all INTERFACE)
target_link_libraries(${PROJECT_NAME}_all INTERFACE ${libname})
target_include_directories(
  ${PROJECT_NAME}_all INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)
install(TARGETS ${PROJECT_NAME}_all EXPORT milkTargets)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION lib/cmake
  PATH_VARS INCLUDE_INSTALL_DIR LIBRARY_INSTALL_DIR)

# Install the Config.cmake and ConfigVersion.cmake files
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION lib/cmake)

# =======================================
# PKG-CONFIG
# =======================================

foreach(lname IN LISTS libname)
  string(APPEND LINKSTRING "-l${lname} ")
endforeach()

message("LINKSTRING: ${LINKSTRING}")

set(INSTALL_PKGCONFIG_DIR
  "lib/pkgconfig"
  CACHE PATH "Installation directory for pkgconfig (.pc) files")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/milk.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/milk.pc @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/milk.pc
  DESTINATION "${INSTALL_PKGCONFIG_DIR}")

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/milk_config.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/config.h DESTINATION include)

if(build_python_module)
  add_subdirectory(python_module)
endif()

if(USE_CLI)

  install(TARGETS CLIcore DESTINATION bin)
  install(
    PROGRAMS scripts/milk-cli-all
    scripts/milk-argparse
    scripts/milk-check
    scripts/milk-completion.sh
    scripts/milk-commands
    scripts/milk-cubeslice2shm
    scripts/milk-exec
    scripts/milk-fpslist-addentry
    scripts/milk-cr2tofits
    scripts/milk-FITS2shm
    scripts/milk-fitsheader
    scripts/milk-fpsmkcmd
    scripts/milk-fpsinit
    scripts/milk-logshim
    scripts/milk-logshimkill
    scripts/milk-logshimoff
    scripts/milk-logshimon
    scripts/milk-logshimstat
    scripts/merge3DfitsTelemetry
    scripts/milk-images-merge
    scripts/milk-scriptexample
    scripts/milk-script-advanced
    scripts/milk-script-std-config
    scripts/milk-streamlink
    scripts/milk-shm2FITSloop
    scripts/tmuxkillall
    scripts/tmuxsessionname
    scripts/waitforfile
    DESTINATION bin)

endif(USE_CLI)

install(FILES docs/cli/help.txt DESTINATION doc)

# milk-cli script payloads (.milk files) – installed as data, not executables
install(FILES scripts/makecircleofdisks.milk DESTINATION share/milk/scripts)

# Guard: verify no standalone links CLIcore unexpectedly
add_test(NAME standalone-dep-check
  COMMAND ${PROJECT_SOURCE_DIR}/scripts/milk-check-standalone-deps
  ${CMAKE_CURRENT_BINARY_DIR})

install(PROGRAMS scripts/milk-check-standalone-deps DESTINATION bin)
