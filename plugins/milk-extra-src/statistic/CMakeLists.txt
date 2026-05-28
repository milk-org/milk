# library name
set(LIBNAME "milkstatistic")
set(SRCNAME "statistic")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

set(SOURCEFILES ${SRCNAME}.c)

set(INCLUDEFILES ${SRCNAME}.h)

# DEFAULT SETTINGS Do not change unless needed
# =====================================================================

project(lib_${LIBNAME}_project)

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/..")

# Library can be compiled from multiple source files Convention: the main souce
# file is named <libname>.c
#
add_library(${LIBNAME} SHARED ${SOURCEFILES})
target_link_libraries(${LIBNAME} PRIVATE CLIcore)

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})

# Compute-only variant: no CLI code, no CLIcore dependency
set(LIBNAME_COMPUTE "${LIBNAME}_compute")
add_library(${LIBNAME_COMPUTE} SHARED ${SOURCEFILES})
target_compile_definitions(${LIBNAME_COMPUTE} PRIVATE MILK_NO_CLI)
if(USE_CFITSIO AND CFITSIO_FOUND)
  target_compile_definitions(${LIBNAME_COMPUTE} PUBLIC USE_CFITSIO=1)
  target_include_directories(
    ${LIBNAME_COMPUTE}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           ${CFITSIO_INCLUDE_DIRS})
  target_link_directories(${LIBNAME_COMPUTE} PUBLIC ${CFITSIO_LIBRARY_DIRS})
endif()
target_include_directories(
  ${LIBNAME_COMPUTE}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
         $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<INSTALL_INTERFACE:include/${SRCNAME}>)
if(USE_CFITSIO AND CFITSIO_FOUND)
  target_link_libraries(
    ${LIBNAME_COMPUTE} PUBLIC m ${CFITSIO_LIBRARIES} milkCOREMODmemory_compute
                              milkCOREMODiofits_compute)
else()
  target_link_libraries(${LIBNAME_COMPUTE} PUBLIC m milkCOREMODmemory_compute)
endif()
install(
  TARGETS ${LIBNAME_COMPUTE}
  EXPORT milkTargets
  DESTINATION lib)

# Static archive for USE_STATIC_LTO builds
if(USE_STATIC_LTO)
  set(LIBNAME_COMPUTE_STATIC "${LIBNAME_COMPUTE}_static")
  add_library(${LIBNAME_COMPUTE_STATIC} STATIC ${SOURCEFILES})
  set_target_properties(${LIBNAME_COMPUTE_STATIC} PROPERTIES OUTPUT_NAME
                                                             ${LIBNAME_COMPUTE})
  target_compile_definitions(${LIBNAME_COMPUTE_STATIC} PRIVATE MILK_NO_CLI)
  target_include_directories(
    ${LIBNAME_COMPUTE_STATIC}
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
           $<INSTALL_INTERFACE:include>)
  if(USE_CFITSIO AND CFITSIO_FOUND)
    target_link_libraries(
      ${LIBNAME_COMPUTE_STATIC}
      PUBLIC m ${CFITSIO_LIBRARIES} milkCOREMODmemory_compute_static
             milkCOREMODiofits_compute_static)
  else()
    target_link_libraries(${LIBNAME_COMPUTE_STATIC}
                          PUBLIC m milkCOREMODmemory_compute_static)
  endif()
endif()
