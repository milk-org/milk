set(LIBNAME "milkTUI")

find_package(PkgConfig REQUIRED)
pkg_check_modules(NCURSES REQUIRED ncurses)

set(SOURCEFILES
    TUItools.c
    timeutils.c
    quicksort.c
)

add_library(${LIBNAME} SHARED ${SOURCEFILES})

target_include_directories(${LIBNAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(${LIBNAME} PUBLIC ImageStreamIO ${NCURSES_LIBRARIES} m)

install(TARGETS ${LIBNAME} EXPORT milkTargets DESTINATION lib)
install(FILES TUItools.h timeutils.h quicksort.h DESTINATION include/libmilkTUI)
