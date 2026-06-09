# library name
set(LIBNAME "milklinARfilterPred")
set(SRCNAME "linARfilterPred")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

set(SOURCEFILES
    ${SRCNAME}.c
    linARfilterPred_data.c
    linARfilterPred_build.c
    linARfilterPred_apply.c
    linARfilterPred_pf.c
    applyPF.c
    build_linPF.c
    pfloadascii.c
    mselblock.c
    imrepshiftx.c
    mkARpfilt.c
    applyARpfilt.c
    mscangain.c
    linARPFMupdate.c
    linARapplyRT.c)

set(INCLUDEFILES ${SRCNAME}.h)

# DEFAULT SETTINGS Do not change unless needed
# =====================================================================

project(lib_${LIBNAME}_project)

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/..")

# Library can be compiled from multiple source files Convention: the main souce
# file is named <libname>.c
#
add_library(${LIBNAME} SHARED ${SOURCEFILES})

if(USE_CUDA)
  find_package(CUDAToolkit 9.0 REQUIRED)
  target_include_directories(
    ${LIBNAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           ${CUDAToolkit_INCLUDE_DIRS})
  target_compile_options(${LIBNAME} PUBLIC -DHAVE_CUDA)
endif(USE_CUDA)

# OpenBLAS detected in upsteam CMakeList with pkg_check_modules(OPENBLAS
# openblas)
if(OPENBLAS_FOUND)
  message("---- OPENBLAS_LIBRARY_DIRS =  ${OPENBLAS_LIBRARY_DIRS}")
  message("---- OPENBLAS_LIBRARIES    =  ${OPENBLAS_LIBRARIES}")
  message("---- OPENBLAS_CFLAGS_OTHER =  ${OPENBLAS_CFLAGS_OTHER}")
  target_include_directories(
    ${LIBNAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           ${OPENBLAS_INCLUDE_DIRS})
  target_link_directories(${LIBNAME} PUBLIC ${OPENBLAS_LIBRARY_DIRS})
  target_link_libraries(${LIBNAME} PUBLIC ${OPENBLAS_LIBRARIES})
  target_compile_options(${LIBNAME} PUBLIC -DHAVE_OPENBLAS
                                           ${OPENBLAS_CFLAGS_OTHER})
endif()

# MKL detected in upstream CMakeList with pkg_check_modules(MKL mkl-sdl)
if(MKL_FOUND)
  message("---- MKL_LIBRARY_DIRS =  ${MKL_LIBRARY_DIRS}")
  message("---- MKL_LIBRARIES    =  ${MKL_LIBRARIES}")
  message("---- MKL_CFLAGS_OTHER =  ${MKL_CFLAGS_OTHER}")
  target_include_directories(
    ${LIBNAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           ${MKL_INCLUDE_DIRS})
  target_link_directories(${LIBNAME} PUBLIC ${MKL_LIBRARY_DIRS})
  target_link_libraries(${LIBNAME} PUBLIC ${MKL_LIBRARIES})
  target_compile_options(${LIBNAME} PUBLIC -DHAVE_MKL ${MKL_CFLAGS_OTHER})
endif()

if(USE_MAGMA)
  target_include_directories(
    ${LIBNAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           ${MAGMA_INCLUDE_DIRS})
  set_target_properties(${LIBNAME} PROPERTIES COMPILE_FLAGS
                                              "-DHAVE_CUDA -DHAVE_MAGMA")
endif(USE_MAGMA)

target_link_libraries(${LIBNAME} PRIVATE CLIcore)

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})

add_milk_standalone(linpred-applyPF applyPF.c)
target_link_libraries(milk-fpsexec-linpred-applyPF PUBLIC ${LIBNAME}
                                                          milklinalgebra)

add_milk_standalone(linpred-buildlinPF build_linPF.c)
target_link_libraries(milk-fpsexec-linpred-buildlinPF PUBLIC ${LIBNAME}
                                                             milklinalgebra)

add_milk_standalone(linAR-pfloadascii pfloadascii.c)
target_link_libraries(milk-fpsexec-linAR-pfloadascii PUBLIC ${LIBNAME})

add_milk_standalone(linAR-mselblock mselblock.c)
target_link_libraries(milk-fpsexec-linAR-mselblock PUBLIC ${LIBNAME})

add_milk_standalone(linAR-imrepshiftx imrepshiftx.c)
target_link_libraries(milk-fpsexec-linAR-imrepshiftx PUBLIC ${LIBNAME})

add_milk_standalone(linAR-mkARpfilt mkARpfilt.c)
target_link_libraries(milk-fpsexec-linAR-mkARpfilt PUBLIC ${LIBNAME}
                                                          milklinalgebra)

add_milk_standalone(linAR-applyARpfilt applyARpfilt.c)
target_link_libraries(milk-fpsexec-linAR-applyARpfilt PUBLIC ${LIBNAME})

add_milk_standalone(linAR-mscangain mscangain.c)
target_link_libraries(milk-fpsexec-linAR-mscangain PUBLIC ${LIBNAME})

add_milk_standalone(linAR-PFMupdate linARPFMupdate.c)
target_link_libraries(milk-fpsexec-linAR-PFMupdate PUBLIC ${LIBNAME})

add_milk_standalone(linAR-applyRT linARapplyRT.c)
target_link_libraries(milk-fpsexec-linAR-applyRT PUBLIC ${LIBNAME})
