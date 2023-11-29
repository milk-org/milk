# library name
set(LIBNAME "milklinalgebra")
set(SRCNAME "linalgebra")


set(SOURCEFILES
	${SRCNAME}.c
	basis_rotate_match.c
	cublas_Coeff2Map_Loop.c
	linalgebrainit.c
	cublas_linalgebratest.c
	cublas_linalgebra_MVMextractModesLoop.c
	GPU_loop_MultMat_free.c
	GPU_loop_MultMat_execute.c
	GPU_loop_MultMat_setup.c
	GPU_SVD_computeControlMatrix.c
	GPUloadCmat.c
	GramSchmidt.c
	magma_compute_SVDpseudoInverse_SVD.c
	magma_compute_SVDpseudoInverse.c
	magma_MatMatMult_testPseudoInverse.c
	modalremap.c
	MVM_CPU.c
	PCAmatch.c
	cublas_PCA.c
	printGPUMATMULTCONF.c
	MVMextractModes.c
	SGEMM.c
	SingularValueDecomp.c
	SingularValueDecomp_mkM.c
	SingularValueDecomp_mkU.c
)

set(INCLUDEFILES
	${SRCNAME}.h
	basis_rotate_match.h
	cublas_Coeff2Map_Loop.h
	linalgebra_types.h
	linalgebrainit.h
	cublas_linalgebratest.h
	cublas_linalgebra_MVMextractModesLoop.h
	GPU_loop_MultMat_free.h
	GPU_loop_MultMat_execute.h
	GPU_loop_MultMat_setup.h
	GPU_SVD_computeControlMatrix.h
	GPUloadCmat.h
	GramSchmidt.h
	magma_compute_SVDpseudoInverse_SVD.h
	magma_compute_SVDpseudoInverse.h
	magma_MatMatMult_testPseudoInverse.h
	modalremap.h
	MVM_CPU.h
	PCAmatch.h
	cublas_PCA.h
	printGPUMATMULTCONF.h
	MVMextractModes.h
	SGEMM.h
	SingularValueDecomp.h
	SingularValueDecomp_mkM.h
	SingularValueDecomp_mkU.h
)

set(SCRIPTS
    scripts/milk-modes-project
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



message("---- module linalgebra ------------------")

find_package(PkgConfig REQUIRED)

if(USE_CUDA)
find_package(CUDA 9.0 REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
message("---- CUDA_INCLUDE_DIRS   = ${CUDA_INCLUDE_DIRS}")
message("---- CUDA_LIBRARIES      = ${CUDA_LIBRARIES} ${CUDA_cusolver_LIBRARY}" )
message("---- CUDA_CUBLAS_LIBRARIES    = ${CUDA_CUBLAS_LIBRARIES}" )
target_include_directories(${LIBNAME} PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(${LIBNAME} PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY})
target_compile_options(${LIBNAME} PUBLIC -DHAVE_CUDA)
endif(USE_CUDA)



if(USE_MAGMA)
find_package(PkgConfig REQUIRED)
pkg_check_modules(MAGMA REQUIRED magma)
message("---- MAGMA_LIBRARY_DIRS =  ${MAGMA_LIBRARY_DIRS}")
message("---- MAGMA_LIBRARIES    =  ${MAGMA_LIBRARIES}" )
message("---- MAGMA_CFLAGS_OTHER =  ${MAGMA_CFLAGS_OTHER}" )
target_include_directories(${LIBNAME} PUBLIC ${MAGMA_INCLUDE_DIRS})
target_link_directories(${LIBNAME} PUBLIC ${MAGMA_LIBRARY_DIRS})
target_link_libraries(${LIBNAME} PRIVATE ${MAGMA_LIBRARIES})
target_compile_options(${LIBNAME} PRIVATE ${MAGMA_CFLAGS_OTHER})
target_compile_options(${LIBNAME} PUBLIC -DHAVE_CUDA -DHAVE_MAGMA -DUSE_MAGMA)
#target_compile_options(${LIBNAME} PUBLIC -DHAVE_CUDA -DHAVE_MAGMA -DUSE_MAGMA)
endif(USE_MAGMA)



#find_package(PkgConfig REQUIRED)
#pkg_check_modules(OPENBLAS openblas)

# OpenBLAS detected in upsteam CMakeList with pkg_check_modules(OPENBLAS openblas)
if(OPENBLAS_FOUND)
message("---- OPENBLAS_LIBRARY_DIRS =  ${OPENBLAS_LIBRARY_DIRS}")
message("---- OPENBLAS_LIBRARIES    =  ${OPENBLAS_LIBRARIES}" )
message("---- OPENBLAS_CFLAGS_OTHER =  ${OPENBLAS_CFLAGS_OTHER}" )
target_include_directories(${LIBNAME} PUBLIC ${OPENBLAS_INCLUDE_DIRS})
target_link_directories(${LIBNAME} PUBLIC ${OPENBLAS_LIBRARY_DIRS})
target_link_libraries(${LIBNAME} PUBLIC ${OPENBLAS_LIBRARIES})
target_compile_options(${LIBNAME} PUBLIC -DHAVE_OPENBLAS ${OPENBLAS_CFLAGS_OTHER})
endif()


# MKL detected in upstream CMakeList with pkg_check_modules(MKL mkl-sdl)
if(MKL_FOUND)
message("---- MKL_LIBRARY_DIRS =  ${MKL_LIBRARY_DIRS}")
message("---- MKL_LIBRARIES    =  ${MKL_LIBRARIES}" )
message("---- MKL_CFLAGS_OTHER =  ${MKL_CFLAGS_OTHER}" )
target_include_directories(${LIBNAME} PUBLIC ${MKL_INCLUDE_DIRS})
target_link_directories(${LIBNAME} PUBLIC ${MKL_LIBRARY_DIRS})
target_link_libraries(${LIBNAME} PUBLIC ${MKL_LIBRARIES})
target_compile_options(${LIBNAME} PUBLIC -DHAVE_MKL ${MKL_CFLAGS_OTHER})
endif()

#If not using MKL, check if OpenBLAS has lapacke symbols.  Include lapacke if not.
if(NOT MKL_FOUND)

    check_library_exists(openblas LAPACKE_sgemlq "" DONT_NEED_LAPACKE)

    if(DONT_NEED_LAPACKE)
       message("-- MKL NOT FOUND")
       message("   LINKING OpenBLAS")
    else()
       message("-- MKL NOT FOUND")
       message("   LINKING LAPACKE")
       list(APPEND LINKLIBS lapacke)    
    endif()

endif()

target_link_libraries(${LIBNAME} PRIVATE CLIcore)

message("---------------------------------------")


install(TARGETS ${LIBNAME} DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTS} DESTINATION bin)
