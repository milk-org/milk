set(LIBNAME "milkTUI")

option(USE_NCURSES "Enable ncurses support" ON)

if(USE_NCURSES)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(NCURSES REQUIRED ncurses)
endif()

set(SOURCEFILES
    TUItools.c
)

add_library(${LIBNAME} SHARED ${SOURCEFILES})

if(USE_NCURSES)
    target_compile_definitions(${LIBNAME} PUBLIC USE_NCURSES)
endif()

target_include_directories(${LIBNAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

if(USE_NCURSES)
    target_link_libraries(${LIBNAME} PUBLIC ImageStreamIO ${NCURSES_LIBRARIES} m)
else()
    target_link_libraries(${LIBNAME} PUBLIC ImageStreamIO m)
endif()

install(TARGETS ${LIBNAME} EXPORT milkTargets DESTINATION lib)
install(FILES TUItools.h DESTINATION include/libmilkTUI)
