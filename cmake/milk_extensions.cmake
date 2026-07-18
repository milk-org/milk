# ─────────────────────────────────────────────────────────────────────────────
# milk_extensions.cmake — Auto-wire build extensions from source directives
# ─────────────────────────────────────────────────────────────────────────────
#
# milk_apply_extensions(target)
#
# Scans all C/C++ source files belonging to <target> for two families of comment
# directives and wires the corresponding build extensions.
#
# ── REQUEST directives ─────────────────────────────────────
#
# MILK_CMAKE_REQUEST_BLAS
# MILK_CMAKE_REQUEST_LAPACKE
# MILK_CMAKE_REQUEST_CUDA
# MILK_CMAKE_REQUEST_MAGMA
# MILK_CMAKE_REQUEST_CFITSIO
#
# If any source in the target carries the directive, the corresponding
# apply_*_to_target() function is called for the whole target.
# When the extension is unavailable (USE_<EXT>=OFF) the directive is silently ignored.
#
# ── MANDATE directives ─────────────────────────────────────────
#
# MILK_CMAKE_MANDATE_BLAS
# MILK_CMAKE_MANDATE_LAPACKE
# MILK_CMAKE_MANDATE_CUDA
# MILK_CMAKE_MANDATE_MAGMA
# MILK_CMAKE_MANDATE_CFITSIO
#
# The source file declares a hard requirement on an extension.
# - Extension AVAILABLE  (HAVE_<EXT> TRUE):  the directive implies REQUEST
# -- apply_*_to_target() is called for the whole target.
# - Extension UNAVAILABLE (HAVE_<EXT> falsy): the source file is excluded from compilation
# (HEADER_FILE_ONLY) and a CMake WARNING is emitted.
# Any single unmet mandate triggers exclusion regardless of other mandates.
#
# The apply_*_to_target() functions are defined in the have_*.cmake files.
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

  string(ASCII 27 _esc)

  set(_want_blas FALSE)
  set(_want_lapacke FALSE)
  set(_want_cuda FALSE)
  set(_want_magma FALSE)
  set(_want_cfitsio FALSE)

  set(_valid_src_count 0)
  set(_excluded_src_count 0)

  foreach(_src IN LISTS _sources)
    if(NOT IS_ABSOLUTE "${_src}")
      set(_src "${_srcdir}/${_src}")
    endif()
    if(NOT EXISTS "${_src}")
      continue()
    endif()
    math(EXPR _valid_src_count "${_valid_src_count} + 1")

    file(STRINGS "${_src}" _req_lines REGEX "MILK_CMAKE_REQUEST_")
    foreach(_line IN LISTS _req_lines)
      if(_line MATCHES "MILK_CMAKE_REQUEST_BLAS")
        set(_want_blas TRUE)
      endif()
      if(_line MATCHES "MILK_CMAKE_REQUEST_LAPACKE")
        set(_want_lapacke TRUE)
      endif()
      if(_line MATCHES "MILK_CMAKE_REQUEST_CUDA")
        set(_want_cuda TRUE)
      endif()
      if(_line MATCHES "MILK_CMAKE_REQUEST_MAGMA")
        set(_want_magma TRUE)
      endif()
      if(_line MATCHES "MILK_CMAKE_REQUEST_CFITSIO")
        set(_want_cfitsio TRUE)
      endif()
    endforeach() # foreach(_line IN LISTS _req_lines)

    # Per-file MANDATE_ scan.  Any single unmet mandate excludes this source
    # from compilation.  Passing mandates imply REQUEST (auto-link).
    file(STRINGS "${_src}" _mnd_lines REGEX "MILK_CMAKE_MANDATE_")
    set(_src_excluded FALSE)
    foreach(_line IN LISTS _mnd_lines)
      if(_line MATCHES "MILK_CMAKE_MANDATE_BLAS")
        if(HAVE_BLAS)
          set(_want_blas TRUE)
        else()
          set(_src_excluded TRUE)
        endif()
      endif()
      if(_line MATCHES "MILK_CMAKE_MANDATE_LAPACKE")
        if(HAVE_LAPACKE)
          set(_want_lapacke TRUE)
        else()
          set(_src_excluded TRUE)
        endif()
      endif()
      if(_line MATCHES "MILK_CMAKE_MANDATE_CUDA")
        if(HAVE_CUDA)
          set(_want_cuda TRUE)
        else()
          set(_src_excluded TRUE)
        endif()
      endif()
      if(_line MATCHES "MILK_CMAKE_MANDATE_MAGMA")
        if(HAVE_MAGMA)
          set(_want_magma TRUE)
        else()
          set(_src_excluded TRUE)
        endif()
      endif()
      if(_line MATCHES "MILK_CMAKE_MANDATE_CFITSIO")
        if(HAVE_CFITSIO)
          set(_want_cfitsio TRUE)
        else()
          set(_src_excluded TRUE)
        endif()
      endif()
    endforeach() # foreach(_line IN LISTS _mnd_lines)
    if(_src_excluded)
      get_filename_component(_src_name "${_src}" NAME)
      message(
        STATUS
        "${_esc}[33m[SKIP] ${target}: ${_src_name} (unmet MANDATE_*)${_esc}[0m"
      )
      set_source_files_properties("${_src}" PROPERTIES HEADER_FILE_ONLY TRUE)
      math(EXPR _excluded_src_count "${_excluded_src_count} + 1")
    endif()
  endforeach() # foreach(_src IN LISTS _sources)

  # Check standalone targets -- if no more source files (ie all files missing a
  # required MANDATE_) then the target is moot and should be disabled
  get_target_property(_tgt_type ${target} TYPE)
  if(_tgt_type STREQUAL "EXECUTABLE"
     AND _valid_src_count GREATER 0
     AND _excluded_src_count GREATER_EQUAL _valid_src_count)
    message(STATUS "milk_apply_extensions: [${target}] all sources excluded "
                    "due to unmet MILK_CMAKE_MANDATE_* — target disabled "
                    "(EXCLUDE_FROM_ALL)")
    set_target_properties(${target} PROPERTIES EXCLUDE_FROM_ALL TRUE)
    return()
  endif()

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
