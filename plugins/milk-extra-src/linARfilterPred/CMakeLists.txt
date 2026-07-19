# library name
set(LIBNAME "milklinARfilterPred")
set(SRCNAME "linARfilterPred")

message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

file(GLOB SOURCEFILES "*.c")
file(GLOB INCLUDEFILES "*.h")

file(GLOB SCRIPTS "scripts/*")

# DEFAULT SETTINGS Do not change unless needed
# =====================================================================

project(lib_${LIBNAME}_project)

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/..")

add_library(${LIBNAME} SHARED ${SOURCEFILES})
target_include_directories(
  ${LIBNAME} PRIVATE $<TARGET_PROPERTY:CLIcore,INTERFACE_INCLUDE_DIRECTORIES>)

message("---- module linalgebra ------------------")

milk_apply_extensions(${LIBNAME})

# Additional cross-module dependency
target_link_libraries(${LIBNAME} PUBLIC milklinalgebra)

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTS} DESTINATION bin)

# List standalones
# FIXME can do better with a MILK_CMAKE tag in the file.
add_milk_standalone(linpred-applyPF applyPF.c)
add_milk_standalone(linpred-buildlinPF build_linPF.c)
add_milk_standalone(linAR-pfloadascii pfloadascii.c)
add_milk_standalone(linAR-mselblock mselblock.c)
add_milk_standalone(linAR-imrepshiftx imrepshiftx.c)
add_milk_standalone(linAR-mkARpfilt mkARpfilt.c)
add_milk_standalone(linAR-applyARpfilt applyARpfilt.c)
add_milk_standalone(linAR-mscangain mscangain.c)
add_milk_standalone(linAR-PFMupdate linARPFMupdate.c)
add_milk_standalone(linAR-applyRT linARapplyRT.c)

# Link module library into every standalone in this directory
# FIXME can do better than this in the future I'm sure !
get_property(_stdalones DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
             PROPERTY BUILDSYSTEM_TARGETS)
foreach(_t IN LISTS _stdalones)
  get_target_property(_type ${_t} TYPE)
  if(_type STREQUAL "EXECUTABLE")
    target_link_libraries(${_t} PUBLIC ${LIBNAME})
  endif()
endforeach()
