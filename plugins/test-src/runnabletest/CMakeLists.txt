set(LIBNAME "milkrunnabletest")
set(SRCNAME "runnabletest")

message("Module CMakeLists.txt ---- SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

file(GLOB SOURCEFILES "*.c" "*.cpp")
file(GLOB INCLUDEFILES "*.h" "*.hpp")
file(GLOB SCRIPTS "scripts/*")

project(lib_${LIBNAME}_project)

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/..")

# Sources are compiled once (-DMILK_NO_CLI) into the shared library, which the CLI loads and the
# generated fpsexec stubs link. Requests/mandates are resolved by milk_apply_extensions().
milk_add_runnable_module(${LIBNAME} SOURCES ${SOURCEFILES})

# One milk-fpsexec-<id> per EXPORT_FPSEXEC(milk, "<id>", unit) line in the sources
milk_add_fpsexec_stubs(${LIBNAME})

install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTS} DESTINATION bin)
