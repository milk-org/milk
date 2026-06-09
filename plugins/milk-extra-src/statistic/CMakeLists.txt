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

# Static archive for USE_STATIC_LTO builds
if(USE_STATIC_LTO)
  set(LIBNAME_STATIC "${LIBNAME}_static")
  add_library(${LIBNAME_STATIC} STATIC ${SOURCEFILES})
  set_target_properties(${LIBNAME_STATIC} PROPERTIES OUTPUT_NAME ${LIBNAME})
  target_include_directories(
    ${LIBNAME_STATIC}
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
           $<INSTALL_INTERFACE:include>)
  if(USE_CFITSIO AND CFITSIO_FOUND)
    target_link_libraries(
      ${LIBNAME_STATIC} PUBLIC m ${CFITSIO_LIBRARIES} milkCOREMODmemory_static
                               milkCOREMODiofits_static)
  else()
    target_link_libraries(${LIBNAME_STATIC} PUBLIC m milkCOREMODmemory_static)
  endif()
  install(
    TARGETS ${LIBNAME_STATIC}
    EXPORT milkTargets
    DESTINATION lib)
endif()
