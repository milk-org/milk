# library name
set(LIBNAME "milkCOREMODmemory")
set(SRCNAME "COREMOD_memory")

set(SOURCEFILES
	${SRCNAME}.c
	image_ID.c
	image_keyword.c
	compute_nb_image.c
	compute_image_memory.c
	list_image.c
	variable_ID.c
	create_image.c
	delete_image.c
	image_copy.c
	image_complex.c
	stream_sem.c
	read_shmim.c
	stream_TCP.c
	logshmim.c
)

set(INCLUDEFILES
	${SRCNAME}.h
	image_ID.h
	image_keyword.h
	compute_nb_image.h
	compute_image_memory.h
	list_image.h
	variable_ID.h	
	create_image.h	
	delete_image.h
	image_copy.h
	image_complex.h
	stream_sem.h
	read_shmim.h
	stream_TCP.h
	logshmim.h
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


