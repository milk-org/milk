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

# Functions to apply to targets

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
    if(LAPACKE_FOUND) # separate liblapacke on this distro
      target_link_libraries(${target} PRIVATE PkgConfig::LAPACKE)
    endif()
  elseif(LAPACKE_FOUND)
    target_link_libraries(${target} PRIVATE PkgConfig::LAPACKE)
  else()
    message(
      WARNING "apply_lapacke_to_target: no LAPACKE found, skipping ${target}")
  endif()
endfunction()

# Parse the options and scan for libraries -- the following contains "return()"s.
# Keep function definitions hereabove.

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
else()
  set(USE_MKL OFF)
endif()

if(OPENBLAS_FOUND AND NOT HAVE_BLAS)
  set(HAVE_OPENBLAS TRUE)
  set(HAVE_BLAS TRUE)

  add_compile_definitions(HAVE_OPENBLAS HAVE_BLAS)
  include_directories(${OPENBLAS_INCLUDE_DIRS})

  # Check whether this OpenBLAS build bundles the LAPACKE C interface
  include(CheckSymbolExists)
  set(CMAKE_REQUIRED_LIBRARIES PkgConfig::OPENBLAS)
  set(CMAKE_REQUIRED_INCLUDES ${OPENBLAS_INCLUDE_DIRS})
  check_symbol_exists(LAPACKE_ssytrd "lapacke.h" OPENBLAS_HAS_LAPACKE)
  unset(CMAKE_REQUIRED_LIBRARIES)
  unset(CMAKE_REQUIRED_INCLUDES)

  if(OPENBLAS_HAS_LAPACKE)
    set(HAVE_LAPACKE TRUE)
    add_compile_definitions(HAVE_LAPACKE)
    message("Computation libraries: openblas (with LAPACKE) "
            "[-DHAVE_OPENBLAS -DHAVE_BLAS -DHAVE_LAPACKE]")
  else()
    message("Computation libraries: openblas (no LAPACKE) "
            "[-DHAVE_OPENBLAS -DHAVE_BLAS]")
  endif()
else()
  set(USE_OPENBLAS OFF)
endif()

if(LAPACKE_FOUND AND NOT HAVE_LAPACKE)
  set(HAVE_LAPACKE TRUE)
  add_compile_definitions(HAVE_LAPACKE)
  include_directories(${LAPACKE_INCLUDE_DIRS})
  message("Computation libraries: using standalone lapacke [-DHAVE_LAPACKE]")
else()
  set(USE_LAPACKE OFF)
endif()
