cmake_minimum_required(VERSION 3.8)

# Add milk-help executable
add_executable(milk-help milk-help.c)
install(TARGETS milk-help DESTINATION bin)

set(SRCNAME "CommandLineInterface")


option(python_module "Compile Python Wrappers" OFF)

project(CLIcore C)

find_package(PkgConfig REQUIRED)
if(USE_GSL)
    find_package(GSL QUIET REQUIRED)
endif()
pkg_check_modules(FFTW REQUIRED fftw3)
pkg_check_modules(FFTWF REQUIRED fftw3f)
if(USE_NCURSES)
    pkg_check_modules(NCURSES REQUIRED ncurses)
endif()
if(USE_READLINE)
    pkg_check_modules(READLINE REQUIRED readline)
endif()
find_package(OpenMP)

pkg_check_modules(HWLOC hwloc)

pkg_check_modules(CFITSIO cfitsio)

find_package(BISON)
find_package(FLEX)
# find_package(Curses REQUIRED) # already performed in main build dir
find_package(Threads REQUIRED)
bison_target(MilkBison calc_bison.y ${PROJECT_SOURCE_DIR}/calc_bison.c)
flex_target(MilkFlex calc_flex.l ${PROJECT_SOURCE_DIR}/calc_flex.c)
add_flex_bison_dependency(MilkFlex MilkBison)
set(CORE_FILES
    CLIcore.c
    CLIcore/CLIcore_utils.c
    CLIcore/CLIcore_UI.c
    CLIcore/CLIcore_checkargs.c
    CLIcore/CLIcore_datainit.c
    CLIcore/CLIcore_help.c
    CLIcore/CLIcore_memory.c
    CLIcore/CLIcore_modules.c
    CLIcore/CLIcore_setSHMdir.c
    CLIcore/CLIcore_signals.c
    ${BISON_MilkBison_OUTPUTS}
                ${FLEX_MilkFlex_OUTPUTS})
    
    link_directories(${NCURSES_LIBRARY_DIRS} ${HWLOC_LIBRARY_DIRS})
    
        # main
    
        add_library(CLIcore
    
                    SHARED
    
                    streamCTRL/streamCTRL_TUI.c
            streamCTRL/streamCTRL_find_streams.c
            streamCTRL/streamCTRL_print_inode.c
            streamCTRL/streamCTRL_print_procpid.c
            streamCTRL/streamCTRL_print_trace.c
            streamCTRL/streamCTRL_scan.c
            streamCTRL/streamCTRL_utilfuncs.c
            ${CORE_FILES})

if(${CFITSIO_FOUND})
  link_directories(${CFITSIO_LIBRARY_DIRS})
  target_compile_definitions(CLIcore PUBLIC USE_CFITSIO=1)
  target_include_directories(CLIcore PUBLIC ${CFITSIO_INCLUDE_DIRS})
endif()

target_include_directories(CLIcore
                PUBLIC ${GSL_INCLUDE_DIRS}
                      ${FFTW_INCLUDE_DIRS}
                      ${FFTWF_INCLUDE_DIRS})

target_compile_options(CLIcore
                        PUBLIC ${FFTW_CFLAGS_OTHER} ${FFTWF_CFLAGS_OTHER})
if (OPENMP_C_FOUND)
  message("Found OpenMP")
  target_link_libraries(CLIcore PUBLIC OpenMP::OpenMP_C)
endif()

if(USE_READLINE)
    target_compile_definitions(CLIcore PUBLIC USE_READLINE)
endif()

if(USE_NCURSES)
    target_compile_definitions(CLIcore PUBLIC USE_NCURSES)
endif()

target_link_libraries(CLIcore
                      PUBLIC m
                              $<$<BOOL:${USE_READLINE}>:readline>
                              ${CFITSIO_LIBRARIES}
                              dl
                              rt
                              ${GSL_LIBRARIES}
                              ${FFTW_LIBRARIES}
                              ${FFTWF_LIBRARIES}
                              ${CMAKE_THREAD_LIBS_INIT}
                              $<$<BOOL:${USE_NCURSES}>:${NCURSES_LIBRARIES}>
                              $<$<BOOL:${USE_NCURSES}>:milkTUI>
                              $<$<BOOL:${USE_NCURSES}>:milkprocessinfoTUI>
                              $<$<BOOL:${USE_NCURSES}>:milkfpsTUI>
                              milkCOREMODarith milkCOREMODiofits milkCOREMODmemory milkCOREMODtools milkprocessinfo milkfps)

if(${HWLOC_FOUND})
  target_compile_definitions(CLIcore PUBLIC USE_HWLOC)
endif()
target_include_directories(CLIcore
                           PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/streamCTRL>
                                  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/CLIcore>
                                  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/procCTRL>
                                  $<INSTALL_INTERFACE:include/${SRCNAME}>
                                  ${FFTW_INCLUDE_DIRS} ${FFTWF_INCLUDE_DIRS}
                                  ${HWLOC_INCLUDE_DIR} $<$<BOOL:${USE_NCURSES}>:${NCURSES_INCLUDE_DIR}>)
target_compile_options(CLIcore PUBLIC)

target_link_directories(CLIcore
                        PUBLIC $<$<BOOL:${USE_NCURSES}>:${NCURSES_LIBRARY_DIRS}> ${HWLOC_LIBRARY_DIRS})

target_link_libraries(CLIcore
                      PUBLIC ${CMAKE_THREAD_LIBS_INIT} ${HWLOC_LIBRARIES}
                             $<$<BOOL:${USE_NCURSES}>:${NCURSES_LIBRARIES}> )

if(${python_module})
  add_subdirectory(python_module)
endif()

install(TARGETS CLIcore EXPORT milkTargets DESTINATION lib)


install(FILES CLIcore.h
              IMGID.h
              ../libmilkTUI/TUItools.h
              calc.h
              calc_bison.h
              cmdsettings.h
              milkDebugTools.h
              standalone_dependencies.h
              ../libprocessinfo/timeutils.h
              DESTINATION include/${SRCNAME})

install(FILES CLIcore/CLIcore_UI.h
              CLIcore/CLIcore_checkargs.h
              CLIcore/CLIcore_datainit.h
              CLIcore/CLIcore_help.h
              CLIcore/CLIcore_memory.h
              CLIcore/CLIcore_modules.h
              CLIcore/CLIcore_setSHMdir.h
              CLIcore/CLIcore_signals.h
              CLIcore/CLIcore_utils.h
              DESTINATION include/${SRCNAME}/CLIcore)

install(FILES ../libprocessinfo/procCTRL/procCTRL_GetCPUloads.h
              ../libprocessinfo/procCTRL/procCTRL_GetNumberCPUs.h
              ../libprocessinfo/procCTRL/procCTRL_PIDcollectSystemInfo.h
              ../libprocessinfo/procCTRL/procCTRL_TUI.h
              ../libprocessinfo/procCTRL/procCTRL_processinfo_scan.h
              DESTINATION include/${SRCNAME}/procCTRL)

add_executable(milk-streamCTRL streamCTRL/milk-streamCTRL.c)
target_link_libraries(milk-streamCTRL PUBLIC CLIcore ImageStreamIO)

install(TARGETS milk-streamCTRL DESTINATION bin)

add_executable(milk-termview termview/milk-termview.c termview/termview.c)
target_link_libraries(milk-termview PUBLIC CLIcore ImageStreamIO)

install(TARGETS milk-termview DESTINATION bin)

# TESTING

include(CTest)

set (failRegex "ERROR" "FAIL")

add_test (NAME milklistim COMMAND milk-exec "listim")
set_property (TEST milklistim PROPERTY LABELS "CLI")
set_tests_properties(milklistim PROPERTIES TIMEOUT 1)
set_property (TEST milklistim PROPERTY PASS_REGULAR_EXPRESSION "0 image")
