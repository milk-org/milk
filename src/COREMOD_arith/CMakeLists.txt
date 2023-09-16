# library name
set(LIBNAME "milkCOREMODarith")
set(SRCNAME "COREMOD_arith")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

set(SOURCEFILES
	${SRCNAME}.c
	image_crop.c
	image_cropmask.c
	image_merge3D.c
	image_set_1Dpixrange.c
	image_set_2Dpix.c
	image_set_col.c
	image_set_row.c
	image_setzero.c
	image_stats.c
	image_total.c
	image_dxdy.c
	imfunctions.c
	mathfuncs.c
	image_arith__im__im.c
	image_arith__im_im__im.c
	image_arith__Cim_Cim__Cim.c
	image_arith__im_f__im.c
	image_arith__im_f_f__im.c
	execute_arith.c
)

set(INCLUDEFILES
	${SRCNAME}.h
	image_crop.h
	image_cropmask.h
	image_merge3D.h
	image_set_1Dpixrange.h
	image_set_2Dpix.h
	image_set_col.h
	image_set_row.h
	image_setzero.h
	image_stats.h
	image_total.h
	image_dxdy.h
	imfunctions.h
	mathfuncs.h
	image_arith__im__im.h
	image_arith__im_im__im.h
	image_arith__Cim_Cim__Cim.h
	image_arith__im_f__im.h
	image_arith__im_f_f__im.h
	execute_arith.h
)

set(SCRIPTS
    scripts/milk-fpslistadd-WFSCROPMASK
	scripts/milk-fpslistadd-SCICROPMASK
)




# TESTING

include(CTest)

# test module is loaded
set(TESTNAME "CLIfunc-COREMOD_arith-module")
add_test (NAME "${TESTNAME}" COMMAND milk-exec "-T" "m?")
set_property (TEST "${TESTNAME}" PROPERTY LABELS "CLImodule")
set_property (TEST "${TESTNAME}" PROPERTY TIMEOUT 1)
set_property (TEST "${TESTNAME}" PROPERTY PASS_REGULAR_EXPRESSION "COREMOD_arith")


# test that commands are registered

list(APPEND commandlist "extractim" "extract3Dim" "setpix" "setpix1Drange" "setrow" "setcol" "imzero" "imtrunc" "cropmask")

foreach(CLIcmdname IN LISTS commandlist)

set(TESTCLIFUNC "${CLIcmdname}")
set(TESTNAME "CLIfunc-COREMOD_arith-${TESTCLIFUNC}")
add_test (NAME "${TESTNAME}" COMMAND milk-exec "-T" "cmd? ${CLIcmdname}")
set_property (TEST "${TESTNAME}" PROPERTY LABELS "CLIfunc")
set_property (TEST "${TESTNAME}" PROPERTY TIMEOUT 1)
set_property (TEST "${TESTNAME}" PROPERTY PASS_REGULAR_EXPRESSION "src:")

endforeach()




# DEFAULT SETTINGS
# Do not change unless needed
# =====================================================================

project(lib_${LIBNAME}_project)

# Library can be compiled from multiple source files
# Convention: the main souce file is named <libname>.c
#
add_library(${LIBNAME} SHARED ${SOURCEFILES})

find_package(PkgConfig REQUIRED)
pkg_check_modules(CFITSIO cfitsio)
if(${CFITSIO_FOUND})
  link_directories(${CFITSIO_LIBRARY_DIRS})
  target_compile_definitions(${LIBNAME} PUBLIC USE_CFITSIO=1)
  target_include_directories(${LIBNAME} PUBLIC ${CFITSIO_INCLUDE_DIRS})
  target_link_directories(${LIBNAME} PUBLIC ${CFITSIO_LIBRARY_DIRS})
endif()

target_include_directories(${LIBNAME} PUBLIC ${PROJECT_SOURCE_DIR}/..)
target_link_libraries(${LIBNAME} PUBLIC m ${CFITSIO_LIBRARIES})

install(TARGETS ${LIBNAME} DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})

install(PROGRAMS ${SCRIPTS} DESTINATION bin)
