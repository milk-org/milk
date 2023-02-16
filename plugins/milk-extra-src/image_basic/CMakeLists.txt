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
	tableto2Dim.c
)

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
	tableto2Dim.h
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
