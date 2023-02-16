# library name
set(LIBNAME "milklinARfilterPred")
set(SRCNAME "linARfilterPred")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")




set(SOURCEFILES
	${SRCNAME}.c
	applyPF.c
	build_linPF.c
)

set(INCLUDEFILES
	${SRCNAME}.h
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


if(USE_CUDA)
find_package(CUDA 9.0 REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
message("---- CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS}")
message("---- CUDA_LIBRARIES    = ${CUDA_LIBRARIES}" )
target_include_directories(${LIBNAME} PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(${LIBNAME} PRIVATE ${CUDA_LIBRARIES})
target_compile_options(${LIBNAME} PUBLIC -DHAVE_CUDA)
endif(USE_CUDA)


if(USE_MAGMA)
target_include_directories(${LIBNAME} PUBLIC ${MAGMA_INCLUDE_DIRS})
set_target_properties(${LIBNAME} PROPERTIES COMPILE_FLAGS "-DHAVE_CUDA -DHAVE_MAGMA")
endif(USE_MAGMA)

target_link_libraries(${LIBNAME} PRIVATE CLIcore)



install(TARGETS ${LIBNAME} DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
