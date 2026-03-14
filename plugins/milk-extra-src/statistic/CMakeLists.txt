# library name
set(LIBNAME "milkstatistic")
set(SRCNAME "statistic")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

set(SOURCEFILES
	${SRCNAME}.c)

set(INCLUDEFILES
	${SRCNAME}.h)


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

install(TARGETS ${LIBNAME} EXPORT milkTargets DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})

# Compute-only variant: no CLI code, no CLIcore dependency
set(LIBNAME_COMPUTE "${LIBNAME}_compute")
add_library(${LIBNAME_COMPUTE} SHARED ${SOURCEFILES})
target_compile_definitions(${LIBNAME_COMPUTE} PRIVATE MILK_NO_CLI)
if(${CFITSIO_FOUND})
  target_compile_definitions(${LIBNAME_COMPUTE} PUBLIC USE_CFITSIO=1)
  target_include_directories(${LIBNAME_COMPUTE} PUBLIC ${CFITSIO_INCLUDE_DIRS})
  target_link_directories(${LIBNAME_COMPUTE} PUBLIC ${CFITSIO_LIBRARY_DIRS})
endif()
target_include_directories(${LIBNAME_COMPUTE} PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/${SRCNAME}>)
target_link_libraries(${LIBNAME_COMPUTE} PUBLIC
    m ${CFITSIO_LIBRARIES}
    milkCOREMODmemory_compute milkCOREMODiofits_compute)
install(TARGETS ${LIBNAME_COMPUTE} EXPORT milkTargets DESTINATION lib)
