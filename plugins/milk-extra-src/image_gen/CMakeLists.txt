# library name
set(LIBNAME "milkimagegen")
# set(SRCNAME "image_gen")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

set(SOURCEFILES
	mkrandomim.c
	voronoi.c
)


set(INCLUDEFILES
	mkrandomim.h
	voronoi.h
)


set(LINKLIBS
	CLIcore
	milkstatistic
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

target_include_directories(${LIBNAME} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
											$<INSTALL_INTERFACE:include>
											$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
											$<INSTALL_INTERFACE:include/${SRCNAME}>)

target_link_libraries(${LIBNAME} PRIVATE ${LINKLIBS})

install(TARGETS ${LIBNAME} EXPORT milkTargets DESTINATION lib)
install(FILES ${SRCNAME}.h ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})

# Compute-only variant: no CLI code, no CLIcore dependency
set(LIBNAME_COMPUTE "${LIBNAME}_compute")
add_library(${LIBNAME_COMPUTE} SHARED ${SRCNAME}.c ${SOURCEFILES})
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
    milkCOREMODmemory_compute milkCOREMODiofits_compute milkstatistic)
install(TARGETS ${LIBNAME_COMPUTE} EXPORT milkTargets DESTINATION lib)

add_milk_standalone(imggen-mkrandom mkrandomim.c)
target_link_libraries(milk-fpsexec-imggen-mkrandom PUBLIC milkstatistic)
add_milk_standalone(imggen-voronoi voronoi.c)
