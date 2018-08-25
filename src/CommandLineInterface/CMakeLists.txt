cmake_minimum_required (VERSION 3.1)
project (CLIcore C)

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

# main
add_library(CLIcore SHARED
CLIcore.c 
CLIcore.h 
processtools.c
calc.h
${BISON_MilkBison_OUTPUTS}
${FLEX_MilkFlex_OUTPUTS})

target_include_directories(CLIcore PUBLIC
${PROJECT_SOURCE_DIR}/..
${PROJECT_SOURCE_DIR}/src
${GSL_INCLUDE_DIRS}
${FFTW_INCLUDE_DIRS}
${FFTWF_INCLUDE_DIRS}
${CURSES_INCLUDE_DIR}
)

target_compile_options(CLIcore PUBLIC 
${FFTW_CFLAGS_OTHER}
${FFTWF_CFLAGS_OTHER}
)

#set_target_properties(CLIcore PROPERTIES COMPILE_FLAGS "-DPACKAGE_NAME=\\\"milk\\\" -DPACKAGE_VERSION=\\\"0.0.0\\\" -DCONFIGDIR=\\\"${PROJECT_SOURCE_DIR}/config\\\" -DSOURCEDIR=\\\"${PROJECT_SOURCE_DIR}\\\" -DABSSRCTOPDIR=\\\"${PROJECT_SOURCE_DIR}\\\" -DPACKAGE_BUGREPORT=\\\"me@me.com\\\"" )

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

install(TARGETS CLIcore DESTINATION bin)
install(TARGETS CLIcore DESTINATION lib)
install(FILES CLIcore.h DESTINATION include)
