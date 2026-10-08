set(LIBNAME "milklinARfilterPred")
set(SRCNAME "linARfilterPred")

message("Module CMakeLists.txt ---- SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

file(GLOB SOURCEFILES "*.c" "*.cpp")
file(GLOB INCLUDEFILES "*.h" "*.hpp")
file(GLOB SCRIPTS "scripts/*")

project(lib_${LIBNAME}_project)

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/..")

add_library(${LIBNAME} SHARED ${SOURCEFILES})
target_include_directories(
  ${LIBNAME} PRIVATE $<TARGET_PROPERTY:CLIcore,INTERFACE_INCLUDE_DIRECTORIES>)
milk_apply_extensions(${LIBNAME})

# CUSTOM ===========
target_link_libraries(${LIBNAME} PUBLIC milklinalgebra)

milk_add_static_lto(${LIBNAME} LINK_LIBS milklinalgebra_static)
# ==================

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTS} DESTINATION bin)

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

get_property(_stdalones DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  PROPERTY BUILDSYSTEM_TARGETS)
foreach(_t IN LISTS _stdalones)
  get_target_property(_type ${_t} TYPE)
  if(_type STREQUAL "EXECUTABLE")
    milk_link_module(${_t} ${LIBNAME})
  endif()
endforeach()
