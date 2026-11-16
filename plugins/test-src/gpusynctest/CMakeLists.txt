set(LIBNAME "milkgpusynctest")
set(SRCNAME "gpusynctest")

message("Module CMakeLists.txt ---- SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

# *.cu: CUDA kernels, dropped (MILK_CMAKE_MANDATE_CUDA) when CUDA is off
file(GLOB SOURCEFILES "*.c" "*.cu")
file(GLOB INCLUDEFILES "*.h")

project(lib_${LIBNAME}_project)

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/..")

milk_add_runnable_module(${LIBNAME} SOURCES ${SOURCEFILES})
milk_add_fpsexec_stubs(${LIBNAME})

install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
