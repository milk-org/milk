# library name
set(LIBNAME "milkimagefilter")
set(SRCNAME "image_filter")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

set(SOURCEFILES
	${SRCNAME}.c
	cubepercentile.c
	fconvolve.c
	fit1D.c
	fit2DcosKernel.c
	fit2Dcossin.c
	gaussfilter.c
	im2Dfilter_1pixbblurr.c
	medianfilter.c
	percentile_interpolation.c
)

set(INCLUDEFILES
	${SRCNAME}.h
	cubepercentile.h
	fconvolve.h
	fit1D.h
	fit2DcosKernel.h
	fit2Dcossin.h
	im2Dfilter_1pixbblurr.h
	gaussfilter.h
	medianfilter.h
	percentile_interpolation.h
)


# DEFAULT SETTINGS
# Do not change unless needed
# =====================================================================

project(lib_${LIBNAME}_project)

include_directories ("${PROJECT_SOURCE_DIR}/src")
include_directories ("${PROJECT_SOURCE_DIR}/..")


# Library can be compiled from multiple source files
# Convention: the main souce file is named <libname>.c
#
add_library(${LIBNAME} SHARED ${SOURCEFILES})
target_link_libraries(${LIBNAME} PRIVATE CLIcore)

install(TARGETS ${LIBNAME} EXPORT milkTargets DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})

# Compute-only variant: no CLI code, no CLIcore dependency
set(LIBNAME_COMPUTE "${LIBNAME}_compute")
add_library(${LIBNAME_COMPUTE} SHARED ${SOURCEFILES})
target_compile_definitions(${LIBNAME_COMPUTE} PRIVATE MILK_NO_CLI)
if(${CFITSIO_FOUND})
  target_compile_definitions(${LIBNAME_COMPUTE} PUBLIC USE_CFITSIO=1)
  target_include_directories(${LIBNAME_COMPUTE} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    ${CFITSIO_INCLUDE_DIRS})
  target_link_directories(${LIBNAME_COMPUTE} PUBLIC ${CFITSIO_LIBRARY_DIRS})
endif()
target_include_directories(${LIBNAME_COMPUTE} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/${SRCNAME}>)
target_link_libraries(${LIBNAME_COMPUTE} PUBLIC
    m ${CFITSIO_LIBRARIES}
    milkCOREMODmemory_compute milkCOREMODiofits_compute)
install(TARGETS ${LIBNAME_COMPUTE} EXPORT milkTargets DESTINATION lib)

add_milk_standalone(imgfilt-gaussfilt gaussfilter.c)

add_milk_standalone(imgfilt-im2Dfilt1pxbb im2Dfilter_1pixbblurr.c)
