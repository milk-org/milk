# library name
set(LIBNAME "milkCOREMODiofits")

# list source files (.c) other than modulename.c
set(SOURCEFILES
	breakcube.c
	check_fitsio_status.c
	data_type_code.c
	file_exists.c
	images2cube.c
	is_fits_file.c
	loadfits.c
	loadmemstream.c
	read_keyword.c
	savefits.c
)

# list include files (.h) that should be installed on system
set(INCLUDEFILES
	COREMOD_iofits_common.h
	breakcube.h
	check_fitsio_status.h
	data_type_code.h
	file_exists.h
	images2cube.h
	is_fits_file.h
	loadfits.h
	loadmemstream.h
	read_keyword.h
	savefits.h
)

# list scripts that should be installed on system
set(SCRIPTFILES
	)



# DEFAULT SETTINGS
# Do not change unless needed
# =====================================================================

# SRCNAME is current directory (last subdir)
set(NAME0 "${CMAKE_CURRENT_SOURCE_DIR}")
string(REPLACE "/" " " NAME1 ${NAME0})
string(REGEX MATCH "[a-zA-Z0-9_]+$" SRCNAME "${NAME0}")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

project(lib_${LIBNAME}_project)

add_library(${LIBNAME} SHARED ${SRCNAME}.c ${SOURCEFILES})

find_package(PkgConfig REQUIRED)
pkg_check_modules(CFITSIO cfitsio)
if(${CFITSIO_FOUND})
message("Found cfitsio")
  link_directories(${CFITSIO_LIBRARY_DIRS})
  target_compile_definitions(${LIBNAME} PUBLIC USE_CFITSIO=1)
  target_include_directories(${LIBNAME} PUBLIC ${CFITSIO_INCLUDE_DIRS})
  target_link_directories(${LIBNAME} PUBLIC ${CFITSIO_LIBRARY_DIRS})
endif()

target_include_directories(${LIBNAME} PRIVATE ${PROJECT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${LIBNAME} PRIVATE m ${CFITSIO_LIBRARIES})

message(" CFITSIO_LIBRARIES : ${CFITSIO_LIBRARIES}")

install(TARGETS ${LIBNAME} DESTINATION lib)
install(FILES ${SRCNAME}.h ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTFILES} DESTINATION bin)
