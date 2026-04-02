set(LIBNAME "milkdata")

set(SOURCEFILES
    milkdata.c
)

set(INCLUDEFILES
   milkdata.h
   milkdata_macros.h
)

add_library(${LIBNAME} SHARED ${SOURCEFILES})

target_include_directories(${LIBNAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(${LIBNAME} PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libprocessinfo>
    # fps.h -> fps_cli_binding.h -> CLIcore/CLIcore_checkargs.h
    # TODO: remove when CLICMDARGDEF moves to engine tier
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
)

target_link_libraries(${LIBNAME}
    PUBLIC ImageStreamIO milkcommon m rt
)

install(TARGETS ${LIBNAME}
    EXPORT milkTargets DESTINATION lib)
install(FILES milkdata.h milkdata_macros.h milkdata_clicmd.h
    DESTINATION include/libmilkdata)

# Static archive for standalone LTO builds
if(USE_STATIC_LTO)
  add_library(${LIBNAME}_static STATIC ${SOURCEFILES})
  set_target_properties(${LIBNAME}_static
      PROPERTIES OUTPUT_NAME ${LIBNAME})
  target_include_directories(${LIBNAME}_static PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<INSTALL_INTERFACE:include>)
  target_include_directories(${LIBNAME}_static PRIVATE
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libprocessinfo>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>)
endif()
