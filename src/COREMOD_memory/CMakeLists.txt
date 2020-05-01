# library name
set(LIBNAME "milkCOREMODmemory")
set(SRCNAME "COREMOD_memory")

set(SOURCEFILES
	${SRCNAME}.c
	image_ID.c
	image_keyword.c
	image_set_counters.c
	compute_nb_image.c
	compute_nb_variable.c
	compute_image_memory.c
	list_image.c
	variable_ID.c
	create_image.c
	delete_image.c
	delete_variable.c
	image_copy.c
	image_complex.c
	stream_sem.c
	read_shmim.c
	stream_TCP.c
	stream_poke.c
	stream_diff.c
	stream_paste.c
	stream_halfimdiff.c
	stream_ave.c
	stream_updateloop.c
	stream_delay.c
	stream_pixmapdecode.c
	logshmim.c
	saveall.c
	clearall.c
)

set(INCLUDEFILES
	${SRCNAME}.h
	image_ID.h
	image_keyword.h
	image_set_counters.h
	compute_nb_image.h
	compute_nb_variable.h
	compute_image_memory.h
	list_image.h
	variable_ID.h	
	create_image.h	
	delete_image.h
	delete_variable.h
	image_copy.h
	image_complex.h
	stream_sem.h
	read_shmim.h
	stream_TCP.h
	stream_poke.h
	stream_diff.h
	stream_paste.h
	stream_halfimdiff.h
	stream_ave.h
	stream_updateloop.h
	stream_delay.h
	stream_pixmapdecode.h
	logshmim.h
	saveall.h
	clearall.h
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


