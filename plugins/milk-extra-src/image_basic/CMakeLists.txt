# library name
set(LIBNAME "milkimagebasic")
set(SRCNAME "image_basic")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

set(SOURCEFILES
    ${SRCNAME}.c
    cubecollapse.c
    extrapolate_nearestpixel.c
    im3Dto2D.c
    image_add.c
    imcontract.c
    imexpand.c
    imgetcircsym.c
    imgetcircasym.c
    imresize.c
    imrotate.c
    imstretch.c
    imswapaxis2D.c
    indexmap.c
    loadfitsimgcube.c
    measure_transl.c
    naninf2zero.c
    streamfeed.c
    streamrecord.c
    tableto2Dim.c)

set(INCLUDEFILES
    ${SRCNAME}.h
    cubecollapse.h
    extrapolate_nearestpixel.h
    im3Dto2D.h
    image_add.h
    imcontract.h
    imexpand.h
    imgetcircsym.h
    imgetcircasym.h
    imresize.h
    imrotate.h
    imstretch.h
    imswapaxis2D.h
    indexmap.h
    loadfitsimgcube.h
    measure_transl.h
    naninf2zero.h
    streamfeed.h
    streamrecord.h
    tableto2Dim.h)

# DEFAULT SETTINGS Do not change unless needed
# =====================================================================

project(lib_${LIBNAME}_project)

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/..")

# Library can be compiled from multiple source files Convention: the main souce
# file is named <libname>.c
#
add_library(${LIBNAME} SHARED ${SOURCEFILES})
target_link_libraries(${LIBNAME} PRIVATE CLIcore)

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})

# Compute-only variant: no CLI code, no CLIcore dependency
set(LIBNAME_COMPUTE "${LIBNAME}_compute")
add_library(${LIBNAME_COMPUTE} SHARED ${SOURCEFILES})
target_compile_definitions(${LIBNAME_COMPUTE} PRIVATE MILK_NO_CLI)
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
  ${LIBNAME_COMPUTE} PUBLIC m ${CFITSIO_LIBRARIES} milkCOREMODmemory_compute
                            milkCOREMODiofits_compute)
install(
  TARGETS ${LIBNAME_COMPUTE}
  EXPORT milkTargets
  DESTINATION lib)

add_milk_standalone(imgbasic-cubecollapse cubecollapse.c)
add_milk_standalone(imgbasic-rotateim imrotate.c)
add_milk_standalone(imgbasic-resizeim imresize.c)
