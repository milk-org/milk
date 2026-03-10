set(LIBNAME "milkdata")

set(SOURCEFILES
    milkdata.c
)

set(INCLUDEFILES
   milkdata.h
   milkdata_macros.h
   milkDebugTools.h
)

add_library(${LIBNAME} SHARED ${SOURCEFILES})

target_include_directories(${LIBNAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(${LIBNAME} PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libprocessinfo>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine>
)

target_link_libraries(${LIBNAME}
    PUBLIC ImageStreamIO m rt
)

install(TARGETS ${LIBNAME}
    EXPORT milkTargets DESTINATION lib)
install(FILES milkdata.h milkdata_macros.h
    DESTINATION include/libmilkdata)
