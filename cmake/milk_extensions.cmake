# ─────────────────────────────────────────────────────────────────────────────
# milk_extensions.cmake — Auto-wire build extensions from source REQUEST_
# defines
# ─────────────────────────────────────────────────────────────────────────────
#
# milk_apply_extensions(target)
#
# Scans all C/C++ source files belonging to <target> for #define REQUEST_<EXT>
# directives and calls the corresponding apply_*_to_target() function when
# found.  Also applies to <target>_compute and <target>_compute_static variants
# if those targets exist.
#
# Valid directives (place near the top of a .c / .cpp file):
#
# #define REQUEST_BLAS #define REQUEST_LAPACKE #define REQUEST_CUDA #define
# REQUEST_MAGMA #define REQUEST_CFITSIO
#
# The apply_*_to_target() functions are defined in the have_*.cmake files and
# are only available when the corresponding USE_<EXT> option is ON.  If the
# extension is disabled the REQUEST_ directive is silently ignored.
# ─────────────────────────────────────────────────────────────────────────────

function(milk_apply_extensions target)
  if(NOT TARGET ${target})
    message(WARNING "milk_apply_extensions: target '${target}' not found")
    return()
  endif()

  # SOURCE_DIR gives the directory where add_library/add_executable was called,
  # needed to resolve relative source paths.
  get_target_property(_srcdir ${target} SOURCE_DIR)
  get_target_property(_sources ${target} SOURCES)

  set(_want_blas FALSE)
  set(_want_lapacke FALSE)
  set(_want_cuda FALSE)
  set(_want_magma FALSE)
  set(_want_cfitsio FALSE)

  foreach(_src IN LISTS _sources)
    if(NOT IS_ABSOLUTE "${_src}")
      set(_src "${_srcdir}/${_src}")
    endif()
    if(NOT EXISTS "${_src}")
      continue()
    endif()

    file(STRINGS "${_src}" _req_lines REGEX "// MILK_COMPILE_REQUEST_")
    foreach(_line IN LISTS _req_lines)
      if(_line MATCHES "// MILK_COMPILE_REQUEST_BLAS")
        set(_want_blas TRUE)
      endif()
      if(_line MATCHES "// MILK_COMPILE_REQUEST_LAPACKE")
        set(_want_lapacke TRUE)
      endif()
      if(_line MATCHES "// MILK_COMPILE_REQUEST_CUDA")
        set(_want_cuda TRUE)
      endif()
      if(_line MATCHES "// MILK_COMPILE_REQUEST_MAGMA")
        set(_want_magma TRUE)
      endif()
      if(_line MATCHES "// MILK_COMPILE_REQUEST_CFITSIO")
        set(_want_cfitsio TRUE)
      endif()
    endforeach() # foreach(_line IN LISTS _req_lines)
  endforeach() # foreach(_src IN LISTS _sources)

  # Apply collected extensions to the target. Guarded by if(COMMAND ...) so
  # missing extensions (USE_X=OFF) are silent.
  if(_want_blas AND COMMAND apply_blas_to_target)
    apply_blas_to_target(${target})
  endif()
  if(_want_lapacke AND COMMAND apply_lapacke_to_target)
    apply_lapacke_to_target(${target})
  endif()
  if(_want_cuda AND COMMAND apply_cuda_to_target)
    apply_cuda_to_target(${target})
  endif()
  if(_want_magma AND COMMAND apply_magma_to_target)
    apply_magma_to_target(${target})
  endif()
  if(_want_cfitsio AND COMMAND apply_cfitsio_to_target)
    apply_cfitsio_to_target(${target})
  endif()
endfunction()
