project(lib_COREMOD_iofits_project)

include_directories ("${PROJECT_SOURCE_DIR}/src")
include_directories ("${PROJECT_SOURCE_DIR}/..")
add_library(COREMOD_iofits SHARED COREMOD_iofits.c)

install(TARGETS COREMOD_iofits DESTINATION lib)
install(FILES COREMOD_iofits.h DESTINATION include)

