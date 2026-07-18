# ═══════════════════════════════════════════════════════════
# MilkStandalone.cmake — Standalone executable helpers
# ═══════════════════════════════════════════════════════════
#
# Provides three CMake functions for creating fpsexec standalone executables:
#
# add_milk_standalone(name src.c) add_cacao_standalone(name src.c)
# add_cacao_standalone_plugins(name src.c [plugin...])
#
# These functions are used in module CMakeLists.txt files. They compile the
# source with -DFPS_STANDALONE and link against COREMOD libraries (no CLIcore
# linkage).
#
# The module's shared library (${LIBNAME}) is NOT linked by default. If a
# standalone needs module-lib symbols, add an explicit target_link_libraries()
# after the call.
#

# Compile fps_standalone_data.c once as an OBJECT library so all standalone
# executables share the same .o rather than recompiling it per target.
add_library(fps_standalone_data_obj OBJECT
            "${PROJECT_SOURCE_DIR}/src/engine/libfps/fps_standalone_data.c")
target_compile_definitions(
  fps_standalone_data_obj
  PRIVATE FPS_STANDALONE $<$<BOOL:${USE_STATIC_LTO}>:FPS_STANDALONE_SKIP_STUBS>)
target_include_directories(fps_standalone_data_obj
                           PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(USE_CLI)
  target_include_directories(
    fps_standalone_data_obj
    PRIVATE # ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/src/cli
            ${PROJECT_SOURCE_DIR}/src/cli/CLIcore
            ${PROJECT_SOURCE_DIR}/src/cli/libmilkscript
            ${PROJECT_SOURCE_DIR}/src/coremods
            ${PROJECT_SOURCE_DIR}/src/engine/libfps
            ${PROJECT_SOURCE_DIR}/src/engine/libprocessinfo
            ${PROJECT_SOURCE_DIR}/src/engine/libmilkcommon)
endif()

# Common link set for all standalone executables
set(_MILK_STANDALONE_LIBS
    milkfps
    milkfpsStandalone
    milkfpsseq
    milkdata
    milkprocessinfo
    ImageStreamIO
    milkCOREMODmemory
    milkCOREMODtools
    milkCOREMODarith
    m
    rt
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
      milkfpsStandalone_static
      milkfpsseq_static
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

# ── milk_scan_standalone ────────────────────────
# TODO
# Take a list of source files
# grep for FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
# or FPS_MAIN_STANDALONE_V2_CONFIG()
# figures out the target name
# call add_milk_standalone


# ── milk_pgo_target ─────────────────────────────
#
# Applies per-executable PGO profile directory. Each standalone gets its own
# subdirectory under PGO_DIR so profiles are fully isolated.
#
# Called internally by add_*_standalone().
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
# Called internally by add_*_standalone().
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
# add_*_standalone().
#
function(milk_lto_target EXE_NAME)
  if(USE_STATIC_LTO)
    target_link_options(${EXE_NAME} PRIVATE -flto=auto)
  endif()
endfunction()

# ── add_milk_standalone ─────────────────────────
#
# Creates a milk-fpsexec-<name> standalone binary.
#
# Usage: add_milk_standalone(myfunction myfunction.c)
#
function(add_milk_standalone FUNC_NAME SRC_FILE)
  set(EXE_NAME "milk-fpsexec-${FUNC_NAME}")
  add_executable(${EXE_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_FILE}"
                             $<TARGET_OBJECTS:fps_standalone_data_obj>)
  target_compile_definitions(${EXE_NAME} PRIVATE FPS_STANDALONE MILK_NO_CLI)
  target_include_directories(${EXE_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src)
  if(USE_STATIC_LTO)
    target_link_libraries(${EXE_NAME} PUBLIC ${_MILK_STANDALONE_STATIC_LIBS})
  else()
    target_link_libraries(${EXE_NAME} PUBLIC ${_MILK_STANDALONE_LIBS})
  endif()
  milk_apply_extensions(${EXE_NAME})
  milk_pgo_target(${EXE_NAME})
  milk_lto_target(${EXE_NAME})
  milk_build_tag_target(${EXE_NAME})
  get_target_property(_excl ${EXE_NAME} EXCLUDE_FROM_ALL)
  if(NOT _excl)
    install(TARGETS ${EXE_NAME} DESTINATION bin)
  endif()
endfunction()

# ── add_cacao_standalone ────────────────────────
#
# Creates a cacao-fpsexec-<name> standalone binary. Includes plugin header paths
# (milk-extra-src/).
#
# Usage: add_cacao_standalone(myfunction myfunction.c)
#
function(add_cacao_standalone FUNC_NAME SRC_FILE)
  set(EXE_NAME "cacao-fpsexec-${FUNC_NAME}")
  add_executable(${EXE_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_FILE}"
                             $<TARGET_OBJECTS:fps_standalone_data_obj>)
  target_compile_definitions(${EXE_NAME} PRIVATE FPS_STANDALONE MILK_NO_CLI)
  target_include_directories(
    ${EXE_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src
                        ${PROJECT_SOURCE_DIR}/plugins/milk-extra-src)
  if(USE_STATIC_LTO)
    target_link_libraries(${EXE_NAME} PUBLIC ${_MILK_STANDALONE_STATIC_LIBS})
  else()
    target_link_libraries(${EXE_NAME} PUBLIC ${_MILK_STANDALONE_LIBS})
  endif()
  milk_apply_extensions(
    ${EXE_NAME}) # May set EXCLUDE_FROM_ALL on a standalone target that misses
                 # its MILK_CMAKE_MANDATE_X
  milk_pgo_target(${EXE_NAME})
  milk_lto_target(${EXE_NAME})
  milk_build_tag_target(${EXE_NAME})
  get_target_property(_excl ${EXE_NAME} EXCLUDE_FROM_ALL)
  if(NOT _excl)
    install(TARGETS ${EXE_NAME} DESTINATION bin)
  endif()
endfunction()

# ── add_cacao_standalone_plugins ────────────────
#
# Like add_cacao_standalone(), but additionally links plugin libraries.
#
# Usage: add_cacao_standalone_plugins(name src.c) → links ALL 4 plugin libs
#
# add_cacao_standalone_plugins(name src.c fft imagegen) → links ONLY the listed
# plugin libs
#
# Valid plugin names: fft, imagegen, imagefilter, imagebasic
#
function(add_cacao_standalone_plugins FUNC_NAME SRC_FILE)
  add_cacao_standalone(${FUNC_NAME} ${SRC_FILE})
  set(EXE_NAME "cacao-fpsexec-${FUNC_NAME}")

  set(_all_plugins fft imagegen imagefilter imagebasic)
  set(_requested ${ARGN})
  if(NOT _requested)
    set(_requested ${_all_plugins})
  endif()

  foreach(_p IN LISTS _requested)
    if(_p STREQUAL "fft")
      target_link_libraries(${EXE_NAME} PUBLIC milkfft)
    elseif(_p STREQUAL "imagegen")
      target_link_libraries(${EXE_NAME} PUBLIC milkimagegen)
    elseif(_p STREQUAL "imagefilter")
      target_link_libraries(${EXE_NAME} PUBLIC milkimagefilter)
    elseif(_p STREQUAL "imagebasic")
      target_link_libraries(${EXE_NAME} PUBLIC milkimagebasic)
    else()
      message(WARNING "Unknown plugin '${_p}' in "
                      "add_cacao_standalone_plugins()")
    endif()
  endforeach()
endfunction()
