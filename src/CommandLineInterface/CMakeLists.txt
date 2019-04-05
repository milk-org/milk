cmake_minimum_required (VERSION 3.8)

option(python_build "Compile Python Wrappers" OFF)
option(use_hunter "Use hunter package manager" OFF)

if(use_hunter)
  include("cmake/HunterGate.cmake")
  huntergate(URL
             "https://github.com/ruslo/hunter/archive/v0.23.126.tar.gz"
             SHA1
             "ccaf37e820cb54d6a5c02993c7f389b9861fc38e")
endif()

project (CLIcore C)
 
if(${python_build})
    set(CORE_FILES standalone_dependencies.c)
else()
    include_directories ("${PROJECT_SOURCE_DIR}/src")
    include_directories ("${PROJECT_SOURCE_DIR}/..")

    find_package(BISON)
    find_package(FLEX)
    #find_package(Curses REQUIRED) # already performed in main build dir
    find_package (Threads REQUIRED)
    find_package(PkgConfig REQUIRED)
    find_package(GSL QUIET REQUIRED)
    pkg_check_modules(FFTW REQUIRED fftw3)
    pkg_check_modules(FFTWF REQUIRED fftw3f)

    BISON_TARGET(MilkBison calc_bison.y ${PROJECT_SOURCE_DIR}/calc_bison.c)
    FLEX_TARGET(MilkFlex calc_flex.l ${PROJECT_SOURCE_DIR}/calc_flex.c)
    ADD_FLEX_BISON_DEPENDENCY(MilkFlex MilkBison)
    set(CORE_FILES 
        CLIcore.c 
        CLIcore.h 
        processtools.c
        streamCTRL.c
	function_parameters.c
        calc.h
        ${BISON_MilkBison_OUTPUTS}
        ${FLEX_MilkFlex_OUTPUTS})        
endif()


find_package(PkgConfig REQUIRED)
pkg_check_modules(NCURSES REQUIRED ncurses)
pkg_check_modules(HWLOC hwloc)

link_directories( ${NCURSES_LIBRARY_DIRS} ${HWLOC_LIBRARY_DIRS} )

# main
add_library(CLIcore SHARED
processtools.c
streamCTRL.c
${CORE_FILES}
)

if(${python_build})
    set(USE_CUDA ON)
    if(NOT TARGET ImageStreamIO)
        add_subdirectory(../../tplib/ImageStreamIO ImageStreamIO)
    endif()
    target_compile_definitions(CLIcore PRIVATE STANDALONE)
    target_include_directories(CLIcore PUBLIC ${CUDA_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../ImageStreamIO)
    target_link_libraries (CLIcore PUBLIC ImageStreamIO)
else()
    target_include_directories(CLIcore PUBLIC
    ${PROJECT_SOURCE_DIR}/..
    ${PROJECT_SOURCE_DIR}/src
    ${GSL_INCLUDE_DIRS}
    ${FFTW_INCLUDE_DIRS}
    ${FFTWF_INCLUDE_DIRS}
    )

    target_compile_options(CLIcore PUBLIC 
    ${FFTW_CFLAGS_OTHER}
    ${FFTWF_CFLAGS_OTHER}
    )

    target_link_libraries (CLIcore PUBLIC
    m
    readline
    cfitsio
    dl
    rt
    ${GSL_LIBRARIES} 
    ${FFTW_LIBRARIES}
    ${FFTWF_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${CURSES_LIBRARIES}
    )
endif()

if(${HWLOC_FOUND})
    target_compile_definitions(CLIcore PRIVATE USE_HWLOC)
endif()

target_include_directories(CLIcore PUBLIC
${CMAKE_CURRENT_SOURCE_DIR}
${HWLOC_INCLUDE_DIR}
${NCURSES_INCLUDE_DIR}
)

target_compile_options(CLIcore PUBLIC )

target_link_libraries (CLIcore PUBLIC
${CMAKE_THREAD_LIBS_INIT}
${HWLOC_LIBRARIES}
${NCURSES_LIBRARIES}
)

if(${python_build})
    add_subdirectory(python_module)
endif()

install(TARGETS CLIcore DESTINATION bin)
install(TARGETS CLIcore DESTINATION lib)
install(FILES CLIcore.h processtools.h DESTINATION include)
