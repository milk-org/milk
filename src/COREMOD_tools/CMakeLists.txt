project(lib_COREMOD_tools_project)

include_directories ("${PROJECT_SOURCE_DIR}/src")
include_directories ("${PROJECT_SOURCE_DIR}/..")
add_library(COREMOD_tools SHARED COREMOD_tools.c)

install(TARGETS COREMOD_tools DESTINATION lib)
install(FILES COREMOD_tools.h DESTINATION include)

