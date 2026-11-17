set(LIBNAME "milkcommon")

set(SOURCEFILES milk_compiler.c milk_rt.c milk_path.c)
set(INCLUDEDIRS
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${MILK_SOURCE_DIR}/src/engine/libmilkdata>
    $<BUILD_INTERFACE:${MILK_SOURCE_DIR}/src/engine/ImageStreamIO>
    $<BUILD_INTERFACE:${MILK_SOURCE_DIR}/src/engine/libprocessinfo>
    $<INSTALL_INTERFACE:include>)

add_library(${LIBNAME} SHARED ${SOURCEFILES})
set_target_properties(${LIBNAME} PROPERTIES UNITY_BUILD ON)
target_include_directories(${LIBNAME} PUBLIC ${INCLUDEDIRS})

if(USE_STATIC_LTO)
  add_library(${LIBNAME}_static STATIC ${SOURCEFILES})
  set_target_properties(${LIBNAME}_static PROPERTIES OUTPUT_NAME ${LIBNAME} UNITY_BUILD ON)
  target_include_directories(${LIBNAME}_static PUBLIC ${INCLUDEDIRS})
endif()

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES milkDebugTools.h milk_compiler.h pixel_dispatch.h milk_types.h
              multiselect_parse.h milk_rt.h milk_path.h DESTINATION include/libmilkcommon)
