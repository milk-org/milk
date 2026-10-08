# ─────────────────────────────────────────────────────────────────────────────
# milk_static_lto.cmake — Static archive of a module for USE_STATIC_LTO builds
# ─────────────────────────────────────────────────────────────────────────────

# milk_link_module(<target> <LIBNAME>)
#
# Links <LIBNAME> into <target>; in USE_STATIC_LTO builds, links the
# <LIBNAME>_static archive instead so the executable owns a single copy of the
# framework state.
#
function(milk_link_module target LIBNAME)
    #if(USE_STATIC_LTO AND TARGET ${LIBNAME}_static)
    if(USE_STATIC_LTO)
        target_link_libraries(${target} PUBLIC ${LIBNAME}_static)
    else()
        target_link_libraries(${target} PUBLIC ${LIBNAME})
    endif()
endfunction()

# milk_add_static_lto(<LIBNAME>
#                     [LINK_LIBS <libs...>]
#                     [CFITSIO_LINK_LIBS <libs...>])
#
# Builds <LIBNAME>_static from the sources of <LIBNAME> (call after
# add_library). No-op unless USE_STATIC_LTO is ON. The archive is neither
# installed nor exported (see docs/notes_for_future.md).
#
# LINK_LIBS         static dependencies, always linked.
# CFITSIO_LINK_LIBS static dependencies linked only when HAVE_CFITSIO is set.
#
function(milk_add_static_lto LIBNAME)
    if(NOT USE_STATIC_LTO)
        return()
    endif()
    cmake_parse_arguments(ARG "" "" "LINK_LIBS;CFITSIO_LINK_LIBS" ${ARGN})

    set(_static ${LIBNAME}_static)
    get_target_property(_sources ${LIBNAME} SOURCES)
    add_library(${_static} STATIC ${_sources})
    set_target_properties(${_static} PROPERTIES OUTPUT_NAME ${LIBNAME})
    target_compile_definitions(${_static} PRIVATE MILK_NO_CLI)
    target_include_directories(
        ${_static}
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<TARGET_PROPERTY:milkfps_static,INTERFACE_INCLUDE_DIRECTORIES>
        $<INSTALL_INTERFACE:include>)
    target_link_libraries(${_static} PUBLIC m milkfps_static milkcommon
        ${ARG_LINK_LIBS})
    # Unresolved symbols are looked up in archives that follow this one.
    set(_core milkCOREMODmemory_static milkCOREMODtools_static
        milkCOREMODarith_static)
    if(HAVE_CFITSIO)
        list(APPEND _core milkCOREMODiofits_static)
    endif()
    list(REMOVE_ITEM _core ${_static})
    target_link_libraries(${_static} PUBLIC ${_core})
    if(HAVE_CFITSIO)
        target_link_libraries(${_static} PUBLIC ${CFITSIO_LIBRARIES}
            ${ARG_CFITSIO_LINK_LIBS})
    endif()
    milk_apply_extensions(${_static})
endfunction()
