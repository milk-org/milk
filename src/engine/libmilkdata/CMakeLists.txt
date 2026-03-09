set(LIBNAME "milkdata")

set(SOURCEFILES
    milkdata.c
)

add_library(${LIBNAME} SHARED ${SOURCEFILES})

target_include_directories(${LIBNAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/CommandLineInterface>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(${LIBNAME}
    PUBLIC ImageStreamIO milkfps milkprocessinfo m rt
)

install(TARGETS ${LIBNAME}
    EXPORT milkTargets DESTINATION lib)
install(FILES milkdata.h milkdata_macros.h
    DESTINATION include/libmilkdata)
