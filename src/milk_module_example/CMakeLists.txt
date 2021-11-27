set(LIBNAME "milkmoduleexample")

# list source files (.c) other than modulename.c
set(SOURCEFILES
	examplefunc1.c
	examplefunc2_FPS.c
	examplefunc3_updatestreamloop.c
	examplefunc4_streamprocess.c
)

# list include files (.h) that should be installed on system
set(INCLUDEFILES
)

# list scripts that should be installed on system
set(SCRIPTFILES
	scripts/milk-tutorial
	scripts/milk-streampoke
	scripts/milk-semtestspeed
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

# LIBNAME constructed by removing spaces
#string(REPLACE "_" "" LIBNAME "${SRCNAME}")
#message(" LIBNAME = ${LIBNAME}")



project(lib_${LIBNAME}_project)

# Library can be compiled from multiple source files
# Convention: the main souce file is named <libname>.c
#
add_library(${LIBNAME} SHARED ${SRCNAME}.c ${SOURCEFILES})

# note: set to PUBLIC if other modules link to this one
target_include_directories(${LIBNAME} PRIVATE ${PROJECT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${LIBNAME} PRIVATE ${LINKLIBS})

install(TARGETS ${LIBNAME} DESTINATION lib)
install(FILES ${SRCNAME}.h ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTFILES} DESTINATION bin)
