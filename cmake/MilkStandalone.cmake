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
    "${PROJECT_SOURCE_DIR}/src/libfps/fps_standalone_data.c")

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
    target_include_directories(${EXE_NAME}
        PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(${EXE_NAME}
        PUBLIC ${_MILK_STANDALONE_LIBS})
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
    target_include_directories(${EXE_NAME}
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/plugins/milk-extra-src)
    target_link_libraries(${EXE_NAME}
        PUBLIC ${_MILK_STANDALONE_LIBS})
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
