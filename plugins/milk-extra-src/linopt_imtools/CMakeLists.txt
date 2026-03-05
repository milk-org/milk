# library name
set(LIBNAME "milklinoptimtools")
set(SRCNAME "linopt_imtools")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

set(SOURCEFILES
	compute_SVDdecomp.c
	compute_SVDpseudoInverse.c
	image_construct.c
	image_fitModes.c
	image_to_vec.c
	imcube_crossproduct.c
	lin1Dfit.c
	linRM_from_inout.c
	makeCosRadModes.c
	makeCPAmodes.c
	mask_to_pixtable.c
	vec_to_2Dimage.c
)

set(INCLUDEFILES
	compute_SVDdecomp.h
	compute_SVDpseudoInverse.h
	image_construct.h
	image_fitModes.h
	image_to_vec.h
	imcube_crossproduct.h
	lin1Dfit.h
	linRM_from_inout.h
	makeCosRadModes.h
	makeCPAmodes.h
	mask_to_pixtable.h
	vec_to_2Dimage.h
)


set(LINKLIBS
	CLIcore
)



# DEFAULT SETTINGS
# Do not change unless needed
# =====================================================================

project(lib_${LIBNAME}_project)

include_directories ("${PROJECT_SOURCE_DIR}/src")
include_directories ("${PROJECT_SOURCE_DIR}/..")


# Library can be compiled from multiple source files
# Convention: the main souce file is named <libname>.c
#
add_library(${LIBNAME} SHARED ${SRCNAME}.c ${SOURCEFILES})
target_include_directories(${LIBNAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
											$<INSTALL_INTERFACE:include>)
target_link_libraries(${LIBNAME} PRIVATE ${LINKLIBS})


install(TARGETS ${LIBNAME} EXPORT milkTargets DESTINATION lib)
install(FILES ${SRCNAME}.h ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
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
