cmake_minimum_required(VERSION 3.14)

project(milk LANGUAGES C)


# Examples :
# cmake .. -DUSE_CUDA=ON -DINSTALLMAKEDEFAULT=ON
# cmake .. -DUSE_CUDA=ON -DUSE_MAGMA=ON
# cmake .. -DCMAKE_BUILD_TYPE=Debug

option(build_python_module "Compile Python Wrappers" OFF)

include(CheckLanguage)

# =======================================
# GPU ACCELERATION
# =======================================

# options
option(USE_CUDA  "Use CUDA        library" OFF)
option(USE_MAGMA "Use MAGMA       library" OFF)

if(USE_CUDA)
#find_package(CUDA 9.0 REQUIRED)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else()
  #message(FATAL_ERROR "No CUDA support")
  message(WARNING "No CUDA support found")
  set(CMAKE_CUDA_COMPILER "/usr/bin/nvcc")
  message(" setting CMAKE_CUDA_COMPILER to ${CMAKE_CUDA_COMPILER}")
endif()
endif(USE_CUDA)




find_package(PkgConfig REQUIRED)
#find_package(MKL CONFIG)

pkg_check_modules(OPENBLAS openblas)
pkg_check_modules(MKL mkl-sdl)
#pkg_check_modules(MKL mkl-static-ilp64-gomp)

enable_language(C)


# Version number
set ( VERSION_MAJOR 1 )
set ( VERSION_MINOR 03 )
set ( VERSION_PATCH 00 )
set ( MILK_VERSION_OPTION "" )
set ( PROJECT_VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}" )
set ( PROJECT_DESCRIPTION "Modular Image processing tooLKit" )




option ( INSTALLMAKEDEFAULT "Make this installation the system default" OFF )

# turn VERBOSE ON
#set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/milk-${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")


# Configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/src/milk_config.h.in"
  "${PROJECT_SOURCE_DIR}/src/milk_config.h"
  )

configure_file (
  "${PROJECT_SOURCE_DIR}/src/config.h.in"
  "${PROJECT_SOURCE_DIR}/src/config.h"
  )


# select most recent version available
SET(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)
SET(CMAKE_FIND_PACKAGE_SORT_DIRECTION DEC)

#
# main libraries
# linked to executable by default
#


# =======================================
# MODULES INCLUDED
# =======================================
# modules are added to two lists:
# libsrcdir : source code directory
# libname   : shared object name

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/src/ImageStreamIO/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

list(APPEND libsrcdir src/CommandLineInterface)
list(APPEND libname CLIcore)

list(APPEND libsrcdir src/ImageStreamIO)
list(APPEND libname ImageStreamIO m)


list(APPEND modulelist COREMOD_arith COREMOD_iofits COREMOD_memory COREMOD_tools)

foreach(mname IN LISTS modulelist)
string(REPLACE "_" "" mname1 "${mname}")
list(APPEND libsrcdir src/${mname})
list(APPEND libname milk${mname1})
message("======= adding module  ${mname}")
endforeach()

# =======================================
# BUILD TYPE
# =======================================

# Set Release build flags
# cmake -DCMAKE_BUILD_TYPE=Debug
# option: add -fanalyzer if gcc10+ (warning: this will take a while to run)
set(CMAKE_C_FLAGS_DEBUG     "-O0 -g -Wall -Wextra")

# Set Release build flags
# cmake -DCMAKE_BUILD_TYPE=Release
set(CMAKE_C_FLAGS_RELEASE     "-Ofast -DNDEBUG")


# Set a default build type if none was specified
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}")
endif()

message("CMAKE_BUILD_TYPE : ${CMAKE_BUILD_TYPE}")



# =======================================
# COMPILE OPTIONS
# =======================================

add_compile_options($<$<COMPILE_LANGUAGE:C>:-march=native>)

# link-time optimizer
add_compile_options($<$<COMPILE_LANGUAGE:C>:-flto=auto>)
add_compile_options($<$<COMPILE_LANGUAGE:C>:-fwhole-program>)
#add_compile_options(-fwhole-program)
# NOTE: -fstack-usage is incompatible with link-time optimizer
#add_compile_options(-fstack-usage)


# A common flag is -pipe. This flag has no effect on the generated code,
# but it makes the compilation process faster. It tells the compiler
# to use pipes instead of temporary files during the different stages
# of compilation, which uses more memory. On systems with low memory,
# GCC might get killed. In those cases do not use this flag.
add_compile_options($<$<COMPILE_LANGUAGE:C>:-pipe>)

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

message("EXTRAMODULES             = ${EXTRAMODULES}")






set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)




# ADD SOURCE DIRECTORIES
# recursively add libraries to be compiled
execute_process(COMMAND bash -c "find ${PROJECT_SOURCE_DIR}/plugins -mindepth 2 -maxdepth 2 -type d| grep -v .git| grep -v .vscode"
                OUTPUT_VARIABLE libextradir)
string(REGEX REPLACE "\n" "; " libextradir "${libextradir}")

foreach(lsrcdir IN LISTS libsrcdir libextradir)
string(REGEX REPLACE " " "" lsrcdir "${lsrcdir}")
if (NOT ${lsrcdir} EQUAL "")
message("ADDING SUBDIR = ${lsrcdir}")
add_subdirectory ( ${lsrcdir} )
endif()
endforeach()

# add example module
add_subdirectory(src/milk_module_example)

# add isio-tools
add_subdirectory(src/isio-tools)


# main
add_executable(milk src/CLImain.c)


# note: set -SOURCEDIR=\\\"${CMAKE_INSTALL_PREFIX}\\\" for autoload
set(BUILD_FLAGS "-DPACKAGE_NAME=\\\"milk\\\" -DCONFIGDIR=\\\"${PROJECT_SOURCE_DIR}/config\\\" -DINSTALLDIR=\\\"${CMAKE_INSTALL_PREFIX}\\\" -DSOURCEDIR=\\\"${PROJECT_SOURCE_DIR}\\\" -DABSSRCTOPDIR=\\\"${PROJECT_SOURCE_DIR}\\\" -DPACKAGE_BUGREPORT=\\\"https://github.com/milk-org/milk/issues\\\"" )


target_link_libraries (milk PUBLIC
${libname}
)


if(USE_CUDA)
target_link_libraries (milk PUBLIC ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
set(BUILD_FLAGS "${BUILD_FLAGS} -DHAVE_CUDA" )
endif(USE_CUDA)




set_target_properties(milk PROPERTIES COMPILE_FLAGS "${BUILD_FLAGS}" )

message("====================================================")
get_target_property(milkcompopts milk COMPILE_OPTIONS)

message("BUILD_FLAGS             :  ${BUILD_FLAGS}")
message("COMPILE_FLAGS           :  ${COMPILE_FLAGS}")
message("COMPILE_OPTIONS         :  ${milkcompopts}")
message("CMAKE_EXE_LINKER_FLAGS  :  ${CMAKE_EXE_LINKER_FLAGS}")

message("CMAKE_C_FLAGS           :  ${CMAKE_C_FLAGS}")
message("CMAKE_CXX_FLAGS         :  ${CMAKE_CXX_FLAGS}")

message("CMAKE_C_FLAGS_DEBUG           : ${CMAKE_C_FLAGS_DEBUG}")
message("CMAKE_C_FLAGS_RELEASE         : ${CMAKE_C_FLAGS_RELEASE}")
message("CMAKE_C_FLAGS_RELWITHDEBINFO  : ${CMAKE_C_FLAGS_RELWITHDEBINFO}")
message("CMAKE_C_FLAGS_MINSIZEREL      : ${CMAKE_C_FLAGS_MINSIZEREL}")

message("CMAKE_CURRENT_SOURCE_DIR      : ${CMAKE_CURRENT_SOURCE_DIR}")
message("CMAKE_CURRENT_BINARY_DIR      : ${CMAKE_CURRENT_BINARY_DIR}")
message("INSTALL_PKGCONFIG_DIR         : ${INSTALL_PKGCONFIG_DIR}")
message("====================================================")


set(PROGRAM_PERMISSIONS_DEFAULT
    OWNER_WRITE OWNER_READ OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE)


install(TARGETS milk DESTINATION bin PERMISSIONS ${PROGRAM_PERMISSIONS_DEFAULT} SETUID)



# =======================================
# MAKE DEFAULT
# =======================================

set(CMAKE_INSTALL_PREFIX_LINK "/usr/local/${PROJECT_NAME}")
if(INSTALLMAKEDEFAULT)
    install(CODE "
    EXECUTE_PROCESS(COMMAND ln -snf ${CMAKE_INSTALL_PREFIX} ${CMAKE_INSTALL_PREFIX_LINK})
    ")
endif(INSTALLMAKEDEFAULT)



# =======================================
# PKG-CONFIG
# =======================================

foreach(lname IN LISTS libname)
string(APPEND LINKSTRING "-l${lname} ")
endforeach()

message("LINKSTRING: ${LINKSTRING}")

set(INSTALL_PKGCONFIG_DIR "lib/pkgconfig"
    CACHE PATH "Installation directory for pkgconfig (.pc) files")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/milk.pc.in
               ${CMAKE_CURRENT_BINARY_DIR}/milk.pc @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/milk.pc
        DESTINATION "${INSTALL_PKGCONFIG_DIR}")


install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/milk_config.h
              ${CMAKE_CURRENT_SOURCE_DIR}/src/config.h
        DESTINATION include
        )


if(build_python_module)
add_subdirectory (python_module)
endif()

install(TARGETS CLIcore DESTINATION bin)
install(PROGRAMS
			scripts/milk-all
			scripts/milk-argparse
			scripts/milk-check
			scripts/milk-cubeslice2shm
			scripts/milk-exec
			scripts/milk-streamCTRL
			scripts/milk-procCTRL
			scripts/milk-fpsCTRL
			scripts/milk-cr2tofits
			scripts/milk-FITS2shm
			scripts/milk-fpsmkcmd
			scripts/milk-fpsinit
			scripts/milk-logshim
			scripts/milk-logshimkill
			scripts/milk-logshimoff
			scripts/milk-logshimon
			scripts/milk-logshimstat
			scripts/milk-makecsetandrt
			scripts/merge3DfitsTelemetry
      scripts/milk-images-merge
			scripts/milk-scriptexample
			scripts/milk-script-std-config
			scripts/milk-streamlink
      scripts/milk-stream-scan
			scripts/milk-shm2FITS
			scripts/milk-shm2FITSloop
			scripts/milk-shmim-rm
			scripts/milk-shmim-zero
			scripts/tmuxkillall
			scripts/tmuxsessionname
			scripts/waitforfile
      DESTINATION bin)


# ====================================
# TESTING
# ====================================

include(CTest)

