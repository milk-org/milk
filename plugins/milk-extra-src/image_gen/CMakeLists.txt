# library name
set(LIBNAME "milkimagegen")
# set(SRCNAME "image_gen")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

set(SOURCEFILES
	mkdisk.c
	mkpolygon.c
	mkspdisk.c
	mkrandomim.c
	voronoi.c
	image_gen_basic_shapes.c
	image_gen_polygons.c
	image_gen_profiles.c
	image_gen_complex.c
)


set(INCLUDEFILES
	mkdisk.h
	mkpolygon.h
	mkspdisk.h
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

target_include_directories(${LIBNAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/${SRCNAME}>
)

target_link_libraries(${LIBNAME} PRIVATE ${LINKLIBS})

install(TARGETS ${LIBNAME} EXPORT milkTargets DESTINATION lib)
install(FILES ${SRCNAME}.h ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})

# Compute-only variant: no CLI code, no CLIcore dependency
set(LIBNAME_COMPUTE "${LIBNAME}_compute")
add_library(${LIBNAME_COMPUTE} SHARED ${SRCNAME}.c ${SOURCEFILES})
target_compile_definitions(${LIBNAME_COMPUTE} PRIVATE MILK_NO_CLI)
if(USE_CFITSIO AND CFITSIO_FOUND)
  target_compile_definitions(${LIBNAME_COMPUTE} PUBLIC USE_CFITSIO=1)
  target_include_directories(${LIBNAME_COMPUTE} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore> ${CFITSIO_INCLUDE_DIRS})
  target_link_directories(${LIBNAME_COMPUTE} PUBLIC ${CFITSIO_LIBRARY_DIRS})
endif()
target_include_directories(${LIBNAME_COMPUTE} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/${SRCNAME}>)
if(USE_CFITSIO AND CFITSIO_FOUND)
  target_link_libraries(${LIBNAME_COMPUTE} PUBLIC
      m ${CFITSIO_LIBRARIES}
      milkCOREMODmemory_compute milkCOREMODiofits_compute milkstatistic_compute)
else()
  target_link_libraries(${LIBNAME_COMPUTE} PUBLIC
      m milkCOREMODmemory_compute milkstatistic_compute)
endif()
install(TARGETS ${LIBNAME_COMPUTE} EXPORT milkTargets DESTINATION lib)

# Static archive for USE_STATIC_LTO builds (standalone can't link .so)
if(USE_STATIC_LTO)
  set(LIBNAME_COMPUTE_STATIC "${LIBNAME_COMPUTE}_static")
  add_library(${LIBNAME_COMPUTE_STATIC} STATIC ${SRCNAME}.c ${SOURCEFILES})
  set_target_properties(${LIBNAME_COMPUTE_STATIC}
      PROPERTIES OUTPUT_NAME ${LIBNAME_COMPUTE})
  target_compile_definitions(${LIBNAME_COMPUTE_STATIC} PRIVATE MILK_NO_CLI)
  target_include_directories(${LIBNAME_COMPUTE_STATIC} PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<INSTALL_INTERFACE:include>)
  if(USE_CFITSIO AND CFITSIO_FOUND)
    target_compile_definitions(${LIBNAME_COMPUTE_STATIC}
        PUBLIC USE_CFITSIO=1)
    target_include_directories(${LIBNAME_COMPUTE_STATIC}
        PUBLIC ${CFITSIO_INCLUDE_DIRS})
    target_link_libraries(${LIBNAME_COMPUTE_STATIC} PUBLIC
        m ${CFITSIO_LIBRARIES}
        milkCOREMODmemory_compute_static milkCOREMODiofits_compute_static milkstatistic_compute_static)
  else()
    target_link_libraries(${LIBNAME_COMPUTE_STATIC} PUBLIC
        m milkCOREMODmemory_compute_static milkstatistic_compute_static)
  endif()
endif()

add_milk_standalone(imggen-mkrandom mkrandomim.c)
if(USE_STATIC_LTO)
  target_link_libraries(milk-fpsexec-imggen-mkrandom PUBLIC
      ${LIBNAME_COMPUTE_STATIC} milkstatistic_compute_static)
else()
  target_link_libraries(milk-fpsexec-imggen-mkrandom PUBLIC
      ${LIBNAME} milkstatistic_compute)
endif()
add_milk_standalone(imggen-voronoi voronoi.c)
add_milk_standalone(imggen-mkdisk mkdisk.c)
target_link_libraries(milk-fpsexec-imggen-mkdisk PUBLIC
    ${LIBNAME_COMPUTE})
add_milk_standalone(imggen-mkspdisk mkspdisk.c)
target_link_libraries(milk-fpsexec-imggen-mkspdisk PUBLIC
    ${LIBNAME_COMPUTE})
add_milk_standalone(imggen-mkpolygon mkpolygon.c)
target_link_libraries(milk-fpsexec-imggen-mkpolygon PUBLIC
    ${LIBNAME_COMPUTE})
