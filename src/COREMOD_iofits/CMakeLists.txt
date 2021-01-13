# library name
set(LIBNAME "milkCOREMODiofits")
set(SRCNAME "COREMOD_iofits")


set(SOURCEFILES
	${SRCNAME}.c
	breakcube.c
	check_fitsio_status.c
	data_type_code.c
	file_exists.c
	images2cube.c
	is_fits_file.c
	loadfits.c
	loadmemstream.c
	read_keyword.c
	savefits.c
)

set(INCLUDEFILES
	${SRCNAME}.h
	COREMOD_iofits_common.h
	breakcube.h
	check_fitsio_status.h
	data_type_code.h
	file_exists.h
	images2cube.h
	is_fits_file.h
	loadfits.h
	loadmemstream.h
	read_keyword.h
	savefits.h
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

install(TARGETS ${LIBNAME} DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
