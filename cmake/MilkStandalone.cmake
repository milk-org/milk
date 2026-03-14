# ═══════════════════════════════════════════════════════════
#  MilkStandalone.cmake — Standalone executable helpers
# ═══════════════════════════════════════════════════════════
#
# Provides three CMake functions for creating fpsexec
# standalone executables:
#
#   add_milk_standalone(name src.c)
#   add_cacao_standalone(name src.c)
#   add_cacao_standalone_plugins(name src.c [plugin...])
#
# These functions are used in module CMakeLists.txt files.
# They compile the source with -DFPS_STANDALONE and link
# against _compute variants of COREMOD libraries (no
# CLIcore dependency).
#
# The module's shared library (${LIBNAME}) is NOT linked
# by default. If a standalone needs module-lib symbols,
# add an explicit target_link_libraries() after the call.
#

# Shared source file for FPS standalone data storage
set(FPS_STANDALONE_DATA_SRC
    "${PROJECT_SOURCE_DIR}/src/engine/libfps/fps_standalone_data.c")

# Common link set for all standalone executables
set(_MILK_STANDALONE_LIBS
    milkfps
    milkfpsStandalone
    milkdata
    milkprocessinfo
    ImageStreamIO
    milkCOREMODmemory_compute
    milkCOREMODtools_compute
    milkCOREMODarith_compute
    milkCOREMODiofits_compute
    ${CFITSIO_LIBRARIES}
    m rt
    -Wl,--allow-shlib-undefined
)

# Static link set for full LTO optimization
# Used when USE_STATIC_LTO=ON.  Static archives
# give the LTO linker full cross-module visibility.
#
# --start-group / --end-group resolves circular
# references between the archives (e.g. COREMOD
# functions calling FPS functions and vice versa).
if(USE_STATIC_LTO)
  set(_MILK_STANDALONE_STATIC_LIBS
      -Wl,--start-group
      milkCOREMODmemory_compute_static
      milkCOREMODtools_compute_static
      milkCOREMODarith_compute_static
      milkCOREMODiofits_compute_static
      milkfpsStandalone_static
      milkfps_static
      milkdata_static
      milkprocessinfo_static
      ImageStreamIO_static
      -Wl,--end-group
      ${CFITSIO_LIBRARIES}
      m rt pthread
      -Wl,--allow-shlib-undefined
  )
endif()


# ── milk_pgo_target ─────────────────────────────
#
# Applies per-executable PGO profile directory.
# Each standalone gets its own subdirectory under
# PGO_DIR so profiles are fully isolated.
#
# Called internally by add_*_standalone().
#
function(milk_pgo_target EXE_NAME)
    if(USE_PGO STREQUAL "GENERATE")
        target_compile_options(${EXE_NAME} PRIVATE
            -fprofile-generate=${PGO_DIR}/${EXE_NAME})
        target_link_options(${EXE_NAME} PRIVATE
            -fprofile-generate=${PGO_DIR}/${EXE_NAME})
    elseif(USE_PGO STREQUAL "USE")
        target_compile_options(${EXE_NAME} PRIVATE
            -fprofile-use=${PGO_DIR}/${EXE_NAME}
            -fprofile-correction)
        target_link_options(${EXE_NAME} PRIVATE
            -fprofile-use=${PGO_DIR}/${EXE_NAME})
    endif()
endfunction()


# ── milk_lto_target ─────────────────────────────
#
# Applies LTO-specific link options to a standalone
# executable.  -flto=auto is passed to the linker
# so it processes LTO IR in the static archives.
#
# Only applied when USE_STATIC_LTO is ON.
# Called internally by add_*_standalone().
#
function(milk_lto_target EXE_NAME)
    if(USE_STATIC_LTO)
        target_link_options(${EXE_NAME} PRIVATE
            -flto=auto)
    endif()
endfunction()


# ── add_milk_standalone ─────────────────────────
#
# Creates a milk-fpsexec-<name> standalone binary.
#
# Usage:
#   add_milk_standalone(myfunction myfunction.c)
#
function(add_milk_standalone FUNC_NAME SRC_FILE)
    set(EXE_NAME "milk-fpsexec-${FUNC_NAME}")
    add_executable(${EXE_NAME}
        "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_FILE}"
        "${FPS_STANDALONE_DATA_SRC}")
    target_compile_definitions(${EXE_NAME}
        PRIVATE FPS_STANDALONE)
    if(USE_STATIC_LTO)
        # fps_standalone_data.c provides stub symbols
        # that clash with real implementations in the
        # static archives.  Compiling it with
        # MILK_NO_CLI skips those stubs.
        set_source_files_properties(
            "${FPS_STANDALONE_DATA_SRC}"
            TARGET_DIRECTORY ${EXE_NAME}
            PROPERTIES COMPILE_DEFINITIONS
                "FPS_STANDALONE_SKIP_STUBS")
    endif()
    target_include_directories(${EXE_NAME}
        PRIVATE ${PROJECT_SOURCE_DIR}/src)
    if(USE_STATIC_LTO)
        target_link_libraries(${EXE_NAME}
            PUBLIC ${_MILK_STANDALONE_STATIC_LIBS})
    else()
        target_link_libraries(${EXE_NAME}
            PUBLIC ${_MILK_STANDALONE_LIBS})
    endif()
    milk_pgo_target(${EXE_NAME})
    milk_lto_target(${EXE_NAME})
    install(TARGETS ${EXE_NAME} DESTINATION bin)
endfunction()


# ── add_cacao_standalone ────────────────────────
#
# Creates a cacao-fpsexec-<name> standalone binary.
# Includes plugin header paths (milk-extra-src/).
#
# Usage:
#   add_cacao_standalone(myfunction myfunction.c)
#
function(add_cacao_standalone FUNC_NAME SRC_FILE)
    set(EXE_NAME "cacao-fpsexec-${FUNC_NAME}")
    add_executable(${EXE_NAME}
        "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_FILE}"
        "${FPS_STANDALONE_DATA_SRC}")
    target_compile_definitions(${EXE_NAME}
        PRIVATE FPS_STANDALONE)
    if(USE_STATIC_LTO)
        set_source_files_properties(
            "${FPS_STANDALONE_DATA_SRC}"
            TARGET_DIRECTORY ${EXE_NAME}
            PROPERTIES COMPILE_DEFINITIONS
                "FPS_STANDALONE_SKIP_STUBS")
    endif()
    target_include_directories(${EXE_NAME}
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/plugins/milk-extra-src)
    if(USE_STATIC_LTO)
        target_link_libraries(${EXE_NAME}
            PUBLIC ${_MILK_STANDALONE_STATIC_LIBS})
    else()
        target_link_libraries(${EXE_NAME}
            PUBLIC ${_MILK_STANDALONE_LIBS})
    endif()
    milk_pgo_target(${EXE_NAME})
    milk_lto_target(${EXE_NAME})
    install(TARGETS ${EXE_NAME} DESTINATION bin)
endfunction()


# ── add_cacao_standalone_plugins ────────────────
#
# Like add_cacao_standalone(), but additionally
# links plugin _compute libraries.
#
# Usage:
#   add_cacao_standalone_plugins(name src.c)
#     → links ALL 4 plugin _compute libs
#
#   add_cacao_standalone_plugins(name src.c fft imagegen)
#     → links ONLY the listed plugin _compute libs
#
# Valid plugin names:
#   fft, imagegen, imagefilter, imagebasic
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
            target_link_libraries(${EXE_NAME}
                PUBLIC milkfft_compute)
        elseif(_p STREQUAL "imagegen")
            target_link_libraries(${EXE_NAME}
                PUBLIC milkimagegen_compute)
        elseif(_p STREQUAL "imagefilter")
            target_link_libraries(${EXE_NAME}
                PUBLIC milkimagefilter_compute)
        elseif(_p STREQUAL "imagebasic")
            target_link_libraries(${EXE_NAME}
                PUBLIC milkimagebasic_compute)
        else()
            message(WARNING
                "Unknown plugin '${_p}' in "
                "add_cacao_standalone_plugins()")
        endif()
    endforeach()
endfunction()
