set(LIBNAME "milkimageformat")
set(SRCNAME "image_format")

message("Module CMakeLists.txt ---- SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

file(GLOB SOURCEFILES "*.c" "*.cpp")
file(GLOB INCLUDEFILES "*.h" "*.hpp")
file(GLOB SCRIPTS "scripts/*")

project(lib_${LIBNAME}_project)

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/..")

add_library(${LIBNAME} SHARED ${SOURCEFILES})
target_include_directories(
  ${LIBNAME} PRIVATE $<TARGET_PROPERTY:CLIcore,INTERFACE_INCLUDE_DIRECTORIES>)
milk_apply_extensions(${LIBNAME})
milk_add_static_lto(${LIBNAME})

# CUSTOM ===========
set(INSTALL_PKGCONFIG_DIR
    "lib/pkgconfig"
    CACHE PATH "Installation directory for pkgconfig (.pc) files")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${LIBNAME}.pc.in
               ${CMAKE_CURRENT_BINARY_DIR}/${LIBNAME}.pc @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${LIBNAME}.pc
        DESTINATION "${INSTALL_PKGCONFIG_DIR}")
# ==================

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTS} DESTINATION bin)

add_milk_standalone(imgfmt-combineHDR combineHDR.c)
add_milk_standalone(imgfmt-extractRGGB extract_RGGBchan.c)
add_milk_standalone(imgfmt-extractutr extract_utr.c)
add_milk_standalone(imgfmt-strmtempstat stream_temporal_stats.c)
add_milk_standalone(imgfmt-writeBMP writeBMP.c)

get_property(_stdalones DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  PROPERTY BUILDSYSTEM_TARGETS)
foreach(_t IN LISTS _stdalones)
  get_target_property(_type ${_t} TYPE)
  if(_type STREQUAL "EXECUTABLE")
    milk_link_module(${_t} ${LIBNAME})
  endif()
endforeach()

# CUSTOM ===========
milk_link_module(milk-fpsexec-imgfmt-combineHDR milkimagefilter)
# ==================
