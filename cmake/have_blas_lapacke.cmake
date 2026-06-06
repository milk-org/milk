# ~~~
# have_blas_lapacke.cmake
#
# Scans for BLAS & LAPACKE capability
#
# Inputs (will be set to OFF if unable):
#     USE_MKL       ON/OFF
#     USE_OPENBLAS  ON/OFF
#     USE_LAPACKE   ON/OFF
#
# Provides    HAVE_BLAS     (unset / TRUE) <- OpenBLAS or MKL
#             HAVE_MKL      (unset / TRUE) <- MKL
#             HAVE_OPENBLAS (unset / TRUE) <- OpenBLAS
#             HAVE_LAPACKE  (unset / TRUE) <- OpenBLAS or MKL or Lapacke standalone
# ~~~

if(USE_MKL)
  pkg_check_modules(MKL IMPORTED_TARGET GLOBAL mkl-sdl)
endif()
if(USE_OPENBLAS)
  pkg_check_modules(OPENBLAS IMPORTED_TARGET GLOBAL openblas)
endif()
if(USE_LAPACKE)
  pkg_check_modules(LAPACKE IMPORTED_TARGET GLOBAL lapacke)
endif()

if(MKL_FOUND)
  set(HAVE_MKL TRUE)
  set(HAVE_BLAS TRUE)
  set(HAVE_LAPACKE TRUE)
  add_compile_definitions(HAVE_MKL HAVE_BLAS HAVE_LAPACKE)
  include_directories(${MKL_INCLUDE_DIRS})
  message(
    "Computation libraries: using MKL [-DHAVE_MKL -DHAVE_BLAS -DHAVE_LAPACKE]")
  return()
else()
  set(USE_MKL OFF)
endif()

if(OPENBLAS_FOUND)
  set(HAVE_OPENBLAS TRUE)
  set(HAVE_BLAS TRUE)
  set(HAVE_LAPACKE TRUE)
  add_compile_definitions(HAVE_OPENBLAS HAVE_BLAS HAVE_LAPACKE)
  include_directories(${OPENBLAS_INCLUDE_DIRS})
  message(
    "Computation libraries: using openblas [-DHAVE_OPENBLAS -DHAVE_BLAS -DHAVE_LAPACKE]"
  )
  return()
else()
  set(USE_OPENBLAS OFF)
endif()

if(LAPACKE_FOUND)
  set(HAVE_LAPACKE TRUE)
  add_compile_definitions(HAVE_LAPACKE)
  include_directories(${LAPACKE_INCLUDE_DIRS})
  message("Computation libraries: using standalone lapacke [-DHAVE_LAPACKE]")
else()
  set(USE_LAPACKE OFF)
endif()

function(apply_blas_to_target target)
  if(MKL_FOUND)
    target_link_libraries(${target} PRIVATE PkgConfig::MKL)
  elseif(OPENBLAS_FOUND)
    target_link_libraries(${target} PRIVATE PkgConfig::OPENBLAS)
  else()
    message(WARNING "apply_blas_to_target: no BLAS found, skipping ${target}")
  endif()
endfunction()

function(apply_lapacke_to_target target)
  if(MKL_FOUND)
    target_link_libraries(${target} PRIVATE PkgConfig::MKL)
  elseif(OPENBLAS_FOUND)
    target_link_libraries(${target} PRIVATE PkgConfig::OPENBLAS)
  elseif(LAPACKE_FOUND)
    target_link_libraries(${target} PRIVATE PkgConfig::LAPACKE)
  else()
    message(
      WARNING "apply_lapacke_to_target: no LAPACKE found, skipping ${target}")
  endif()
endfunction()
