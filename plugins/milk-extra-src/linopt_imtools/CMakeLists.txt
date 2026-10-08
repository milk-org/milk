set(LIBNAME "milklinoptimtools")
set(SRCNAME "linopt_imtools")

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
milk_add_static_lto(${LIBNAME})

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTS} DESTINATION bin)

add_milk_standalone(linopt-SVDdecomp compute_SVDdecomp.c)
add_milk_standalone(linopt-SVDpInverse compute_SVDpseudoInverse.c)
add_milk_standalone(linopt-imgconstruct image_construct.c)
add_milk_standalone(linopt-imgFitModes image_fitModes.c)
add_milk_standalone(linopt-img2vec image_to_vec.c)
add_milk_standalone(linopt-imcXprod imcube_crossproduct.c)
add_milk_standalone(linopt-lin1Dfit lin1Dfit.c)
add_milk_standalone(linopt-linRM linRM_from_inout.c)
add_milk_standalone(linopt-mkCPAmodes makeCPAmodes.c)
add_milk_standalone(linopt-mkCosRadModes makeCosRadModes.c)
add_milk_standalone(linopt-mask2pixtab mask_to_pixtable.c)
add_milk_standalone(linopt-vec2Dimg vec_to_2Dimage.c)

get_property(_stdalones DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  PROPERTY BUILDSYSTEM_TARGETS)
foreach(_t IN LISTS _stdalones)
  get_target_property(_type ${_t} TYPE)
  if(_type STREQUAL "EXECUTABLE")
    milk_link_module(${_t} ${LIBNAME})
  endif()
endforeach()
