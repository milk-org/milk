set(LIBNAME "milklinalgebra")
set(SRCNAME "linalgebra")

message("Module CMakeLists.txt ---- SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

file(GLOB SOURCEFILES "*.c" "*.cpp")
file(GLOB INCLUDEFILES "*.h" "*.hpp")
file(GLOB SCRIPTS "scripts/*")

project(lib_${LIBNAME}_project)

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/..")

# Sources are compiled once (-DMILK_NO_CLI) into the shared library; mandates and requests
# are resolved per file by milk_apply_extensions().
milk_add_runnable_module(${LIBNAME} SOURCES ${SOURCEFILES})

# One milk-fpsexec-<id> per EXPORT_FPSEXEC line (see cmake/milk_runnable.cmake).
milk_add_fpsexec_stubs(${LIBNAME})

install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTS} DESTINATION bin)
