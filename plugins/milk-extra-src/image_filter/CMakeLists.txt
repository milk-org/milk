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

install(TARGETS ${LIBNAME} DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
