set(LIBNAME "milkimageformat")
set(SRCNAME "image_format")

message("Module CMakeLists.txt ---- SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

file(GLOB SOURCEFILES "*.c" "*.cpp")
file(GLOB INCLUDEFILES "*.h" "*.hpp")
file(GLOB SCRIPTS "scripts/*")

project(lib_${LIBNAME}_project)

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/..")

milk_add_runnable_module(${LIBNAME} SOURCES ${SOURCEFILES})
milk_add_fpsexec_stubs(${LIBNAME})

# CUSTOM ===========
set(INSTALL_PKGCONFIG_DIR
    "lib/pkgconfig"
    CACHE PATH "Installation directory for pkgconfig (.pc) files")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${LIBNAME}.pc.in
               ${CMAKE_CURRENT_BINARY_DIR}/${LIBNAME}.pc @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${LIBNAME}.pc
        DESTINATION "${INSTALL_PKGCONFIG_DIR}")
# ==================

install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTS} DESTINATION bin)

# CUSTOM ===========
milk_link_module(milk-fpsexec-imgfmt-combineHDR milkimagefilter)
# ==================
