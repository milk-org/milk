cmake_minimum_required(VERSION 3.8)


set(SRCNAME "CommandLineInterface")


option(python_module "Compile Python Wrappers" OFF)

project(CLIcore C)

find_package(PkgConfig REQUIRED)
find_package(GSL QUIET REQUIRED)
pkg_check_modules(FFTW REQUIRED fftw3)
pkg_check_modules(FFTWF REQUIRED fftw3f)
pkg_check_modules(NCURSES REQUIRED ncurses)

pkg_check_modules(HWLOC hwloc)



if(${python_module})
  set(CORE_FILES standalone_dependencies.c)
else()
  include_directories("${PROJECT_SOURCE_DIR}/src")
  include_directories("${PROJECT_SOURCE_DIR}/..")

  find_package(BISON)
  find_package(FLEX)
  # find_package(Curses REQUIRED) # already performed in main build dir
  find_package(Threads REQUIRED)
  bison_target(MilkBison calc_bison.y ${PROJECT_SOURCE_DIR}/calc_bison.c)
  flex_target(MilkFlex calc_flex.l ${PROJECT_SOURCE_DIR}/calc_flex.c)
  add_flex_bison_dependency(MilkFlex MilkBison)
  set(CORE_FILES
      CLIcore.c
      CLIcore_UI.c
      ${BISON_MilkBison_OUTPUTS}
      ${FLEX_MilkFlex_OUTPUTS})
endif()

link_directories(${NCURSES_LIBRARY_DIRS} ${HWLOC_LIBRARY_DIRS})

# main
add_library(CLIcore
            SHARED
            processtools.c
            streamCTRL.c
            function_parameters.c
            ${CORE_FILES})

if(${python_module})
  set(USE_CUDA ON)
  if(NOT TARGET ImageStreamIO)
    if(IS_DIRECTORY ${PROJECT_SOURCE_DIR}/../ImageStreamIO)
      add_subdirectory( ${PROJECT_SOURCE_DIR}/../ImageStreamIO ImageStreamIO)
    else()
      add_subdirectory($ENV{OCTOPUS_ROOT}/tplib/ImageStreamIO ImageStreamIO)
    endif()
  endif()
  target_compile_definitions(CLIcore PUBLIC STANDALONE)
  target_include_directories(
    CLIcore
    PUBLIC ${CUDA_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../ImageStreamIO)
  target_link_libraries(CLIcore PUBLIC ImageStreamIO)
else()
  target_include_directories(CLIcore
                             PUBLIC ${PROJECT_SOURCE_DIR}/..
                                    ${PROJECT_SOURCE_DIR}/src
                                    ${GSL_INCLUDE_DIRS}
                                    ${FFTW_INCLUDE_DIRS}
                                    ${FFTWF_INCLUDE_DIRS})

  target_compile_options(CLIcore
                         PUBLIC ${FFTW_CFLAGS_OTHER} ${FFTWF_CFLAGS_OTHER})

  target_link_libraries(CLIcore
                        PUBLIC m
                               readline
                               cfitsio
                               dl
                               rt
                               ${GSL_LIBRARIES}
                               ${FFTW_LIBRARIES}
                               ${FFTWF_LIBRARIES}
                               ${CMAKE_THREAD_LIBS_INIT}
                               ${CURSES_LIBRARIES})
endif()

if(${HWLOC_FOUND})
  target_compile_definitions(CLIcore PRIVATE USE_HWLOC)
endif()

target_include_directories(CLIcore
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                  ${HWLOC_INCLUDE_DIR} ${NCURSES_INCLUDE_DIR})

target_compile_options(CLIcore PUBLIC)

target_link_libraries(CLIcore
                      PUBLIC ${CMAKE_THREAD_LIBS_INIT} ${HWLOC_LIBRARIES}
                             ${NCURSES_LIBRARIES})

if(${python_module})
  add_subdirectory(python_module)
endif()

install(TARGETS CLIcore DESTINATION bin)
install(PROGRAMS
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
			scripts/merge3DfitsTelemetry
			scripts/milk-streamlink
			scripts/milk-shm2FITS
			scripts/milk-shm2FITSloop
			scripts/tmuxkillall
			scripts/tmuxsessionname
			scripts/waitforfile
        DESTINATION bin)

install(TARGETS CLIcore DESTINATION lib)

install(FILES CLIcore.h
              CLIcore_UI.h
              processtools.h  
              streamCTRL.h  
              function_parameters.h 
              standalone_dependencies.h 
        DESTINATION include/${SRCNAME})
