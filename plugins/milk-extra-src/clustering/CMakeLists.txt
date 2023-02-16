set(LIBNAME "milkclustering")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

# list source files (.c) other than modulename.c
set(SOURCEFILES
	cubecluster.c
	CFmeminit.c
	CFtree_rebuild.c
	addvector_to_CF.c
	compute_imdistance_double.c
	condense.c
	create_new_leaf.c
	ctree_init.c
	ctree_memallocate.c
	ctree_memfree.c
	droptree.c
	get_availableCFindex.c
	leaf_addentry.c
	leafnode_attachleaf.c
	mindiffscan.c
	node_attachnode.c
	printCFtree.c
	split_CF_node.c
	update_level.c
	)

# list include files (.h) that should be installed on system
set(INCLUDEFILES
	cubecluster.h
	)

# list scripts that should be installed on system
set(SCRIPTFILES
)

set(LINKLIBS
	CLIcore
)


# DEFAULT SETTINGS
# Do not change unless needed
# =====================================================================

# SRCNAME is current directory (last subdir)
set(NAME0 "${CMAKE_CURRENT_SOURCE_DIR}")
string(REPLACE "/" " " NAME1 ${NAME0})
string(REGEX MATCH "[a-zA-Z0-9_]+$" SRCNAME "${NAME0}")
message(" SRCNAME = ${SRCNAME}")

project(lib_${LIBNAME}_project)

# Library can be compiled from multiple source files
# Convention: the main souce file is named <libname>.c
#
add_library(${LIBNAME} SHARED ${SRCNAME}.c ${SOURCEFILES})

target_include_directories(${LIBNAME} PUBLIC ${PROJECT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${LIBNAME} PRIVATE ${LINKLIBS})

install(TARGETS ${LIBNAME} DESTINATION lib)
install(FILES ${SRCNAME}.h ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTFILES} DESTINATION bin)
