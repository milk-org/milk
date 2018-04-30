project(lib_COREMOD_arith_project)

include_directories ("${PROJECT_SOURCE_DIR}/src")
include_directories ("${PROJECT_SOURCE_DIR}/..")
add_library(COREMOD_arith SHARED COREMOD_arith.c)

install(TARGETS COREMOD_arith DESTINATION lib)
install(FILES COREMOD_arith.h DESTINATION include)

