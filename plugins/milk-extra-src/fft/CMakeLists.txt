# library name
set(LIBNAME "milkfft")
set(SRCNAME "fft")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED fftw3)
pkg_check_modules(FFTWF REQUIRED fftw3f)

set(SOURCEFILES
    ${SRCNAME}.c
    DFT.c
    dofft.c
    dofft_1d.c
    dofft_2d.c
    fftcorrelation.c
    ffttranslate.c
    fftzoom.c
    fft_autocorrelation.c
    fft_structure_function.c
    init_fftwplan.c
    permut.c
    pup2foc.c
    testfftspeed.c
    wisdom.c)

set(INCLUDEFILES
    ${SRCNAME}.h
    DFT.h
    dofft.h
    dofft_internal.h
    fftcorrelation.h
    ffttranslate.h
    fftzoom.h
    fft_autocorrelation.h
    fft_structure_function.h
    init_fftwplan.h
    permut.h
    pup2foc.h
    testfftspeed.h
    wisdom.h)

# DEFAULT SETTINGS Do not change unless needed
# =====================================================================

project(lib_${LIBNAME}_project)

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/..")

# Library can be compiled from multiple source files Convention: the main souce
# file is named <libname>.c
#
add_library(${LIBNAME} SHARED ${SOURCEFILES})

target_link_libraries(${LIBNAME} PUBLIC ${FFTW_LIBRARIES} ${FFTWF_LIBRARIES}
                                        CLIcore)

set_target_properties(
  ${LIBNAME} PROPERTIES COMPILE_FLAGS
                        "-DFFTCONFIGDIR=\\\"${PROJECT_SOURCE_DIR}/config\\\"")

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})

# Compute-only variant: no CLI code, no CLIcore dependency
set(LIBNAME_COMPUTE "${LIBNAME}_compute")
add_library(${LIBNAME_COMPUTE} SHARED ${SOURCEFILES})
target_compile_definitions(${LIBNAME_COMPUTE} PRIVATE MILK_NO_CLI)
target_compile_definitions(${LIBNAME_COMPUTE}
                           PRIVATE FFTCONFIGDIR="${PROJECT_SOURCE_DIR}/config")
if(${CFITSIO_FOUND})
  target_compile_definitions(${LIBNAME_COMPUTE} PUBLIC USE_CFITSIO=1)
  target_include_directories(
    ${LIBNAME_COMPUTE}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           ${CFITSIO_INCLUDE_DIRS})
  target_link_directories(${LIBNAME_COMPUTE} PUBLIC ${CFITSIO_LIBRARY_DIRS})
endif()
target_include_directories(
  ${LIBNAME_COMPUTE}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
         $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<INSTALL_INTERFACE:include/${SRCNAME}>)
target_link_libraries(
  ${LIBNAME_COMPUTE}
  PUBLIC ${FFTW_LIBRARIES} ${FFTWF_LIBRARIES} m ${CFITSIO_LIBRARIES}
         milkCOREMODmemory_compute milkCOREMODiofits_compute)
install(
  TARGETS ${LIBNAME_COMPUTE}
  EXPORT milkTargets
  DESTINATION lib)

add_milk_standalone(fft-dofft dofft.c)
target_link_libraries(milk-fpsexec-fft-dofft PUBLIC ${LIBNAME_COMPUTE})
add_milk_standalone(fft-pup2foc pup2foc.c)
target_link_libraries(milk-fpsexec-fft-pup2foc PUBLIC ${LIBNAME_COMPUTE})
