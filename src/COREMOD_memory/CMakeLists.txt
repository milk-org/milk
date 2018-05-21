project(lib_COREMOD_memory_project)

include_directories ("${PROJECT_SOURCE_DIR}/src")
include_directories ("${PROJECT_SOURCE_DIR}/..")


add_library(COREMOD_memory SHARED COREMOD_memory.c)

#SET(CURSES_USE_NCURSES TRUE)
#find_package(Curses REQUIRED)
#include_directories(${CURSES_INCLUDE_DIR})
#target_link_libraries(COREMOD_memory PUBLIC ${CURSES_LIBRARIES})

install(TARGETS COREMOD_memory DESTINATION lib)
install(FILES COREMOD_memory.h DESTINATION include)

