# library name
set(LIBNAME "milklinalgebra")
set(SRCNAME "linalgebra")


set(SOURCEFILES
	${SRCNAME}.c
	basis_rotate_match.c
	cublas_Coeff2Map_Loop.c
	linalgebrainit.c
	cublas_linalgebratest.c
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
	Qexpand.c
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
	Qexpand.h
	SGEMM.h
	SingularValueDecomp.h
	SingularValueDecomp_mkM.h
	SingularValueDecomp_mkU.h
)

set(SCRIPTS
	scripts/milk-linmap-calib
    scripts/milk-modes-project
	scripts/milk-modes-project-U
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
  find_package(CUDAToolkit 9.0 REQUIRED)
  message(STATUS "Found CUDA ${CUDAToolkit_VERSION} at ${CUDAToolkit_LIBRARY_DIR}")
  message("---- CUDA_INCLUDE_DIRS = ${CUDAToolkit_INCLUDE_DIRS}")
  target_include_directories(${LIBNAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore> ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(${LIBNAME} PRIVATE CUDA::cublas CUDA::cusolver)
  target_compile_options(${LIBNAME} PUBLIC -DHAVE_CUDA)

  if(USE_MAGMA)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(MAGMA REQUIRED magma)
    message("---- MAGMA_LIBRARY_DIRS =  ${MAGMA_LIBRARY_DIRS}")
    message("---- MAGMA_LIBRARIES    =  ${MAGMA_LIBRARIES}" )
    message("---- MAGMA_CFLAGS_OTHER =  ${MAGMA_CFLAGS_OTHER}" )
    target_include_directories(${LIBNAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore> ${MAGMA_INCLUDE_DIRS})
    target_link_directories(${LIBNAME} PUBLIC ${MAGMA_LIBRARY_DIRS})
    target_link_libraries(${LIBNAME} PRIVATE ${MAGMA_LIBRARIES})
    target_compile_options(${LIBNAME} PRIVATE ${MAGMA_CFLAGS_OTHER})
    target_compile_options(${LIBNAME} PUBLIC -DHAVE_CUDA -DHAVE_MAGMA -DUSE_MAGMA)
    #target_compile_options(${LIBNAME} PUBLIC -DHAVE_CUDA -DHAVE_MAGMA -DUSE_MAGMA)
  endif(USE_MAGMA)
endif(USE_CUDA)



#find_package(PkgConfig REQUIRED)
#pkg_check_modules(OPENBLAS openblas)

# OpenBLAS detected in upsteam CMakeList with pkg_check_modules(OPENBLAS openblas)
if(OPENBLAS_FOUND)
message("---- OPENBLAS_LIBRARY_DIRS =  ${OPENBLAS_LIBRARY_DIRS}")
message("---- OPENBLAS_LIBRARIES    =  ${OPENBLAS_LIBRARIES}" )
message("---- OPENBLAS_CFLAGS_OTHER =  ${OPENBLAS_CFLAGS_OTHER}" )
target_include_directories(${LIBNAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore> ${OPENBLAS_INCLUDE_DIRS})
target_link_directories(${LIBNAME} PUBLIC ${OPENBLAS_LIBRARY_DIRS})
target_link_libraries(${LIBNAME} PUBLIC ${OPENBLAS_LIBRARIES})
target_compile_options(${LIBNAME} PUBLIC -DHAVE_OPENBLAS ${OPENBLAS_CFLAGS_OTHER})
endif()


# MKL detected in upstream CMakeList with pkg_check_modules(MKL mkl-sdl)
if(MKL_FOUND)
message("---- MKL_LIBRARY_DIRS =  ${MKL_LIBRARY_DIRS}")
message("---- MKL_LIBRARIES    =  ${MKL_LIBRARIES}" )
message("---- MKL_CFLAGS_OTHER =  ${MKL_CFLAGS_OTHER}" )
target_include_directories(${LIBNAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore> ${MKL_INCLUDE_DIRS})
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
       target_link_libraries(${LIBNAME} PRIVATE lapacke)
    endif()

endif()

target_link_libraries(${LIBNAME} PRIVATE CLIcore)

message("---------------------------------------")


install(TARGETS ${LIBNAME} EXPORT milkTargets DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTS} DESTINATION bin)

# Compute-only variant: no CLI code, no CLIcore dependency
set(LIBNAME_COMPUTE "${LIBNAME}_compute")
add_library(${LIBNAME_COMPUTE} SHARED ${SOURCEFILES})
target_compile_definitions(${LIBNAME_COMPUTE}
    PRIVATE MILK_NO_CLI)
if(USE_CFITSIO AND CFITSIO_FOUND)
  target_compile_definitions(${LIBNAME_COMPUTE}
      PUBLIC USE_CFITSIO=1)
  target_include_directories(${LIBNAME_COMPUTE}
      PUBLIC ${CFITSIO_INCLUDE_DIRS})
  target_link_directories(${LIBNAME_COMPUTE}
      PUBLIC ${CFITSIO_LIBRARY_DIRS})
endif()
target_include_directories(${LIBNAME_COMPUTE} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/coremods>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/${SRCNAME}>)
target_link_libraries(${LIBNAME_COMPUTE} PUBLIC
    m milkfps)
if(USE_CUDA)
  target_include_directories(${LIBNAME_COMPUTE}
      PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(${LIBNAME_COMPUTE}
      PRIVATE CUDA::cublas CUDA::cusolver)
  target_compile_options(${LIBNAME_COMPUTE}
      PUBLIC -DHAVE_CUDA)
  if(USE_MAGMA)
    target_include_directories(${LIBNAME_COMPUTE}
        PUBLIC ${MAGMA_INCLUDE_DIRS})
    target_link_directories(${LIBNAME_COMPUTE}
        PUBLIC ${MAGMA_LIBRARY_DIRS})
    target_link_libraries(${LIBNAME_COMPUTE}
        PRIVATE ${MAGMA_LIBRARIES})
    target_compile_options(${LIBNAME_COMPUTE}
        PRIVATE ${MAGMA_CFLAGS_OTHER})
    target_compile_options(${LIBNAME_COMPUTE}
        PUBLIC -DHAVE_CUDA -DHAVE_MAGMA
               -DUSE_MAGMA)
  endif()
endif()
if(OPENBLAS_FOUND)
  target_include_directories(${LIBNAME_COMPUTE}
      PUBLIC ${OPENBLAS_INCLUDE_DIRS})
  target_link_directories(${LIBNAME_COMPUTE}
      PUBLIC ${OPENBLAS_LIBRARY_DIRS})
  target_link_libraries(${LIBNAME_COMPUTE}
      PUBLIC ${OPENBLAS_LIBRARIES})
  target_compile_options(${LIBNAME_COMPUTE}
      PUBLIC -DHAVE_OPENBLAS
             ${OPENBLAS_CFLAGS_OTHER})
endif()
if(MKL_FOUND)
  target_include_directories(${LIBNAME_COMPUTE}
      PUBLIC ${MKL_INCLUDE_DIRS})
  target_link_directories(${LIBNAME_COMPUTE}
      PUBLIC ${MKL_LIBRARY_DIRS})
  target_link_libraries(${LIBNAME_COMPUTE}
      PUBLIC ${MKL_LIBRARIES})
  target_compile_options(${LIBNAME_COMPUTE}
      PUBLIC -DHAVE_MKL ${MKL_CFLAGS_OTHER})
endif()
if(NOT MKL_FOUND AND NOT DONT_NEED_LAPACKE)
  target_link_libraries(${LIBNAME_COMPUTE}
      PRIVATE lapacke)
endif()
install(TARGETS ${LIBNAME_COMPUTE} EXPORT milkTargets
    DESTINATION lib)

add_milk_standalone(linalg-gramschmidt GramSchmidt.c)
add_milk_standalone(linalg-MVMextract MVMextractModes.c)
if(USE_CUDA)
    target_link_libraries(milk-fpsexec-linalg-MVMextract PRIVATE CUDA::cudart CUDA::cublas)
endif()
add_milk_standalone(linalg-PCAmatch PCAmatch.c)
target_link_libraries(milk-fpsexec-linalg-PCAmatch
    PUBLIC ${LIBNAME_COMPUTE})
add_milk_standalone(linalg-Qexpand Qexpand.c)
add_milk_standalone(linalg-SGEMM SGEMM.c)
target_link_libraries(milk-fpsexec-linalg-SGEMM PUBLIC ${LIBNAME_COMPUTE})
if(USE_CUDA)
    target_link_libraries(milk-fpsexec-linalg-SGEMM PRIVATE CUDA::cudart CUDA::cublas)
endif()

add_milk_standalone(linalg-SVD SingularValueDecomp.c)
target_link_libraries(milk-fpsexec-linalg-SVD
    PUBLIC ${LIBNAME_COMPUTE})
if(OPENBLAS_FOUND)
target_link_libraries(milk-fpsexec-linalg-SVD
    PUBLIC ${OPENBLAS_LIBRARIES} lapacke)
target_link_directories(milk-fpsexec-linalg-SVD
    PUBLIC ${OPENBLAS_LIBRARY_DIRS})
endif()
add_milk_standalone(linalg-SVDmkM SingularValueDecomp_mkM.c)
target_link_libraries(milk-fpsexec-linalg-SVDmkM
    PUBLIC ${LIBNAME_COMPUTE})
add_milk_standalone(linalg-SVDmkU SingularValueDecomp_mkU.c)
target_link_libraries(milk-fpsexec-linalg-SVDmkU
    PUBLIC ${LIBNAME_COMPUTE})
add_milk_standalone(linalg-basisrotmatch basis_rotate_match.c)
target_link_libraries(milk-fpsexec-linalg-basisrotmatch
    PUBLIC ${LIBNAME_COMPUTE})
add_milk_standalone(linalg-cublasPCA cublas_PCA.c)
if(USE_CUDA)
    target_link_libraries(milk-fpsexec-linalg-cublasPCA PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
endif()

target_link_libraries(milk-fpsexec-linalg-cublasPCA
    PUBLIC ${LIBNAME_COMPUTE})
add_milk_standalone(linalg-modalremap modalremap.c)
target_link_libraries(milk-fpsexec-linalg-modalremap
    PUBLIC ${LIBNAME_COMPUTE})
