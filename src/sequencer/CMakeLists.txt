# library name
set(LIBNAME "milksequencer")
set(SRCNAME "sequencer")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

set(SOURCEFILES
	${SRCNAME}.c
	seq_cli.c
)

set(INCLUDEFILES
	${SRCNAME}.h
	seq_cli.h
)

# DEFAULT SETTINGS
# Do not change unless needed
# =====================================================================

project(lib_${LIBNAME}_project)

add_library(${LIBNAME} SHARED ${SOURCEFILES})

target_include_directories(${LIBNAME} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
											$<INSTALL_INTERFACE:include>)
target_link_libraries(${LIBNAME} PUBLIC m milkfps milkfpsseq CLIcore)

install(TARGETS ${LIBNAME} EXPORT milkTargets DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
