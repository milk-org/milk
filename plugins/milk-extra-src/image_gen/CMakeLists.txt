set(LIBNAME "milkimagegen")
set(SRCNAME "image_gen")

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
target_link_libraries(${LIBNAME} PRIVATE milkstatistic)

milk_add_static_lto(${LIBNAME} LINK_LIBS milkstatistic_static)
# ==================

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTS} DESTINATION bin)

add_milk_standalone(imggen-mkrandom mkrandomim.c)
add_milk_standalone(imggen-voronoi voronoi.c)
add_milk_standalone(imggen-mkdisk mkdisk.c)
add_milk_standalone(imggen-mkspdisk mkspdisk.c)
add_milk_standalone(imggen-mkpolygon mkpolygon.c)
add_milk_standalone(imggen-mkgauss imgen_mkgauss.c)
add_milk_standalone(imggen-mkfiberclpoverlap imgen_mkfiberclpoverlap.c)
add_milk_standalone(imggen-mkslopexy imgen_mkslopexy.c)
add_milk_standalone(imggen-mkdist imgen_mkdist.c)
add_milk_standalone(imggen-mkhexsegpup imgen_mkhexsegpup.c)
add_milk_standalone(imggen-segs2wfmodes imgen_segs2wfmodes.c)
add_milk_standalone(imggen-mkrect imgen_mkrect.c)
add_milk_standalone(imggen-mkline imgen_mkline.c)
add_milk_standalone(imggen-mklincoord imgen_mklincoord.c)
add_milk_standalone(imggen-mkgridpix imgen_mkgridpix.c)
add_milk_standalone(imggen-mkrndim imgen_mkrndim.c)
add_milk_standalone(imggen-mkrndgim imgen_mkrndgim.c)
add_milk_standalone(imggen-im2coord imgen_im2coord.c)

get_property(_stdalones DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  PROPERTY BUILDSYSTEM_TARGETS)
foreach(_t IN LISTS _stdalones)
  get_target_property(_type ${_t} TYPE)
  if(_type STREQUAL "EXECUTABLE")
    milk_link_module(${_t} ${LIBNAME})
  endif()
endforeach()

# CUSTOM ===========
if(NOT USE_STATIC_LTO)
  target_link_libraries(milk-fpsexec-imggen-mkrandom PUBLIC milkstatistic)
endif()
# ==================
