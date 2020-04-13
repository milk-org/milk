# library name
set(LIBNAME "milkmemory")


set(SOURCEFILES
	COREMOD_memory.c)

set(INCLUDEFILES
	COREMOD_memory.h)


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
install(FILES ${INCLUDEFILES} DESTINATION include)

