# library name
set(LIBNAME "milkfft")
set(SRCNAME "fft")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED fftw3)
pkg_check_modules(FFTWF REQUIRED fftw3f)

set(SOURCEFILES
	${SRCNAME}.c
	DFT.c
	dofft.c
	fftcorrelation.c
	ffttranslate.c
	fftzoom.c
	fft_autocorrelation.c
	fft_structure_function.c
	init_fftwplan.c
	permut.c
	pupfft.c
	testfftspeed.c
	wisdom.c
)

set(INCLUDEFILES
	${SRCNAME}.h
	DFT.h
	dofft.h
	fftcorrelation.h
	ffttranslate.h
	fftzoom.h
	fft_autocorrelation.h
	fft_structure_function.h
	init_fftwplan.h
	permut.h
	pupfft.h
	testfftspeed.h
	wisdom.h
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
add_library(${LIBNAME} SHARED ${SOURCEFILES})

target_link_libraries(${LIBNAME} PUBLIC ${FFTW_LIBRARIES} ${FFTWF_LIBRARIES} CLIcore)

set_target_properties(${LIBNAME} PROPERTIES COMPILE_FLAGS "-DFFTCONFIGDIR=\\\"${PROJECT_SOURCE_DIR}/config\\\"")

install(TARGETS ${LIBNAME} DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
