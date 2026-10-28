# ═══════════════════════════════════════════════════════════
# milk_standalone.cmake — fpsexec link sets and per-executable helpers
# ═══════════════════════════════════════════════════════════
#
# Link sets and PGO / LTO / build-tag helpers shared by every fpsexec. The
# executables themselves are created by milk_add_fpsexec_stubs()
# (cmake/milk_runnable.cmake); they do not link CLIcore.

# Common link set for all standalone executables
set(_MILK_STANDALONE_LIBS
    milkfps
    milkcomputeunit
    milkdata
    milkprocessinfo
    ImageStreamIO
    milkCOREMODmemory
    milkCOREMODtools
    milkCOREMODarith
    m
    rt
    pthread
    -Wl,--allow-shlib-undefined)
if(USE_CFITSIO AND CFITSIO_FOUND)
  list(APPEND _MILK_STANDALONE_LIBS milkCOREMODiofits ${CFITSIO_LIBRARIES})
endif()

# Static link set for full LTO optimization Used when USE_STATIC_LTO=ON.  Static
# archives give the LTO linker full cross-module visibility.
#
# --start-group / --end-group resolves circular references between the archives
# (e.g. COREMOD functions calling FPS functions and vice versa).
if(USE_STATIC_LTO)
  # Core static libs always required
  set(_MILK_STANDALONE_STATIC_LIBS
      -Wl,--start-group
      milkCOREMODmemory_static
      milkCOREMODtools_static
      milkCOREMODarith_static
      milkcomputeunit_static
      milkfps_static
      milkdata_static
      milkprocessinfo_static
      -Wl,--end-group
      m
      rt
      pthread
      -Wl,--allow-shlib-undefined)
  # CFITSIO-dependent libs only when CFITSIO is enabled
  if(USE_CFITSIO AND CFITSIO_FOUND)
    list(INSERT _MILK_STANDALONE_STATIC_LIBS 1 milkCOREMODiofits_static
         ImageStreamIO_static ${CFITSIO_LIBRARIES})
  else()
    # Link ImageStreamIO static archive when CFITSIO is not used
    list(APPEND _MILK_STANDALONE_STATIC_LIBS ImageStreamIO_static)
  endif()
endif()

# Link set of every fpsexec: static archives in USE_STATIC_LTO builds, shared libs otherwise
if(USE_STATIC_LTO)
  set(_MILK_FPSEXEC_LIBS ${_MILK_STANDALONE_STATIC_LIBS})
else()
  set(_MILK_FPSEXEC_LIBS ${_MILK_STANDALONE_LIBS})
endif()

# ── milk_pgo_target ─────────────────────────────
#
# Applies per-executable PGO profile directory. Each standalone gets its own
# subdirectory under PGO_DIR so profiles are fully isolated.
#
# Called internally by milk_configure_fpsexec().
#
function(milk_pgo_target EXE_NAME)
  if(USE_PGO STREQUAL "GENERATE")
    target_compile_options(${EXE_NAME}
                           PRIVATE -fprofile-generate=${PGO_DIR}/${EXE_NAME})
    target_link_options(${EXE_NAME} PRIVATE
                        -fprofile-generate=${PGO_DIR}/${EXE_NAME})
  elseif(USE_PGO STREQUAL "USE")
    target_compile_options(
      ${EXE_NAME} PRIVATE -fprofile-use=${PGO_DIR}/${EXE_NAME}
                          -fprofile-correction)
    target_link_options(${EXE_NAME} PRIVATE
                        -fprofile-use=${PGO_DIR}/${EXE_NAME})
  endif()
endfunction()

# ── milk_build_tag_target ──────────────────────
#
# Injects compile-time MILK_BUILD_* defines so that every standalone binary
# embeds a detectable tag string.  milk-perfbench uses `strings` to extract this
# tag and report whether the binary was built with PGO or LTO.
#
# Defines set: MILK_BUILD_PGO_GENERATE  — pass-1 (instrument) MILK_BUILD_PGO_USE
# — pass-2 (use profiles) MILK_BUILD_LTO           — any LTO variant
# MILK_BUILD_STATIC        — compiled as static LTO MILK_BUILD_OPT           —
# -O2/-O3 detected
#
# Called internally by milk_configure_fpsexec().
#
function(milk_build_tag_target EXE_NAME)
  # PGO state from the USE_PGO option
  if(USE_PGO STREQUAL "GENERATE")
    target_compile_definitions(${EXE_NAME} PRIVATE MILK_BUILD_PGO_GENERATE)
  elseif(USE_PGO STREQUAL "USE")
    target_compile_definitions(${EXE_NAME} PRIVATE MILK_BUILD_PGO_USE)
  endif()

  # LTO from USE_STATIC_LTO option
  if(USE_STATIC_LTO)
    target_compile_definitions(${EXE_NAME} PRIVATE MILK_BUILD_LTO
                                                   MILK_BUILD_STATIC)
  endif()

  # Detect manual -flto flag in CMAKE_C_FLAGS
  if(CMAKE_C_FLAGS MATCHES "-flto")
    target_compile_definitions(${EXE_NAME} PRIVATE MILK_BUILD_LTO)
  endif()

  # Detect manual PGO flags when USE_PGO not set
  if(CMAKE_C_FLAGS MATCHES "-fprofile-generate")
    target_compile_definitions(${EXE_NAME} PRIVATE MILK_BUILD_PGO_GENERATE)
  endif()
  if(CMAKE_C_FLAGS MATCHES "-fprofile-use")
    target_compile_definitions(${EXE_NAME} PRIVATE MILK_BUILD_PGO_USE)
  endif()

  # Optimisation tier
  if(CMAKE_C_FLAGS MATCHES "-O3" OR CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(${EXE_NAME} PRIVATE MILK_BUILD_OPT)
  endif()

  # Inject the binary name for the sentinel string
  target_compile_definitions(${EXE_NAME}
                             PRIVATE MILK_BUILD_BINNAME="${EXE_NAME}")
endfunction()

# ── milk_lto_target ─────────────────────────────
#
# Applies LTO-specific link options to a standalone executable.  -flto=auto is
# passed to the linker so it processes LTO IR in the static archives.
#
# Only applied when USE_STATIC_LTO is ON. Called internally by
# milk_configure_fpsexec().
#
function(milk_lto_target EXE_NAME)
  if(USE_STATIC_LTO)
    target_link_options(${EXE_NAME} PRIVATE -flto=auto)
  endif()
endfunction()

# ── milk_configure_fpsexec ──────────────────────
#
# Common setup of an fpsexec executable: include paths, framework libs, PGO/LTO,
# build tag, install (unless EXCLUDE_FROM_ALL).
#
# milk_configure_fpsexec(<exe> PREFIX milk|cacao [MODULE <lib>])
#
# MODULE links the module library (its _static archive in USE_STATIC_LTO builds).
#
function(milk_configure_fpsexec EXE_NAME)
  cmake_parse_arguments(ARG "" "PREFIX;MODULE" "" ${ARGN})

  # CMAKE_SOURCE_DIR (not PROJECT_SOURCE_DIR): each plugin subdirectory calls
  # its own project(), which would otherwise re-root this path locally.
  target_include_directories(${EXE_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src)
  if(ARG_PREFIX STREQUAL "cacao")
    target_include_directories(
      ${EXE_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/plugins/milk-extra-src)
  endif()
  target_link_libraries(${EXE_NAME} PUBLIC ${_MILK_FPSEXEC_LIBS})
  if(ARG_MODULE)
    milk_link_module(${EXE_NAME} ${ARG_MODULE})
  endif()
  milk_pgo_target(${EXE_NAME})
  milk_lto_target(${EXE_NAME})
  milk_build_tag_target(${EXE_NAME})
  get_target_property(_excl ${EXE_NAME} EXCLUDE_FROM_ALL)
  if(NOT _excl)
    install(TARGETS ${EXE_NAME} DESTINATION bin)
  endif()
endfunction()
