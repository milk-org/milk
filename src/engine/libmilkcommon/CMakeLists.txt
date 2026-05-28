set(LIBNAME "milkcommon")

add_library(${LIBNAME} SHARED milk_rt.c)

target_include_directories(
  ${LIBNAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libmilkdata>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/ImageStreamIO>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libprocessinfo>
         $<INSTALL_INTERFACE:include>)

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES milkDebugTools.h milk_compiler.h pixel_dispatch.h milk_types.h
              multiselect_parse.h milk_rt.h DESTINATION include/libmilkcommon)
