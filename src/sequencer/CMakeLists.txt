# library name
set(LIBNAME "milksequencer")
set(SRCNAME "sequencer")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

set(SOURCEFILES ${SRCNAME}.c seq_cli.c)

set(INCLUDEFILES ${SRCNAME}.h seq_cli.h)

# DEFAULT SETTINGS Do not change unless needed
# =====================================================================

project(lib_${LIBNAME}_project)

add_library(${LIBNAME} SHARED ${SOURCEFILES})

target_include_directories(
  ${LIBNAME} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
                    $<INSTALL_INTERFACE:include>)
target_link_libraries(${LIBNAME} PUBLIC m milkfps milkcomputeunit milkfpsseq CLIcore)

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})

# Add CTest checks to verify that the sequencer module's CLI commands are
# registered. These tests are only enabled when BUILD_TESTING is ON.
if(BUILD_TESTING)
  add_test(NAME sequencer_cmd_help_seq_list COMMAND milk-exec "-T"
                                                    "@help seq.list")
endif()
