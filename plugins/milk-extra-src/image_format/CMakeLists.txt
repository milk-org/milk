# library name
set(LIBNAME "milkimageformat")
set(SRCNAME "image_format")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

set(SOURCEFILES
	${SRCNAME}.c
	combineHDR.c
	CR2toFITS.c
	CR2tomov.c
	extract_RGGBchan.c
	FITS_to_ushortintbin_lock.c
	FITS_to_floatbin_lock.c
	FITStorgbFITSsimple.c
	imtoASCII.c
	loadCR2toFITSRGB.c
	read_binary32f.c
	readPGM.c
	writeBMP.c
	extract_utr.c
	stream_temporal_stats.c
)

set(INCLUDEFILES
	${SRCNAME}.h
	combineHDR.h
	CR2toFITS.h
	CR2tomov.h
	extract_RGGBchan.h
	FITS_to_ushortintbin_lock.h
	FITS_to_floatbin_lock.h
	FITStorgbFITSsimple.h
	imtoASCII.h
	loadCR2toFITSRGB.h
	read_binary32f.h
	readPGM.h
	writeBMP.h
	extract_utr.h
	stream_temporal_stats.h
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


# PKGCONFIG
# =====================================================================
set(INSTALL_PKGCONFIG_DIR "lib/pkgconfig"
    CACHE PATH "Installation directory for pkgconfig (.pc) files")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${LIBNAME}.pc.in
               ${CMAKE_CURRENT_BINARY_DIR}/${LIBNAME}.pc @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${LIBNAME}.pc
        DESTINATION "${INSTALL_PKGCONFIG_DIR}")
