project(lib_COREMOD_memory_project)

include_directories ("${PROJECT_SOURCE_DIR}/src")
include_directories ("${PROJECT_SOURCE_DIR}/..")
add_library(COREMOD_memory SHARED COREMOD_memory.c)

install(TARGETS COREMOD_memory DESTINATION lib)
install(FILES COREMOD_memory.h DESTINATION include)

