# library name
set(LIBNAME "milkfft")
set(SRCNAME "fft")

message("Module CMakeLists.txt ---- SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

# CUSTOM ===========
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED fftw3)
pkg_check_modules(FFTWF REQUIRED fftw3f)
# ==================

file(GLOB SOURCEFILES "*.c")
file(GLOB INCLUDEFILES "*.h")
file(GLOB SCRIPTS "scripts/*")

project(lib_${LIBNAME}_project)

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/..")

milk_add_runnable_module(${LIBNAME} SOURCES ${SOURCEFILES}
                         LINK_LIBS ${FFTW_LIBRARIES} ${FFTWF_LIBRARIES})
milk_add_fpsexec_stubs(${LIBNAME})

# CUSTOM ===========
# Applied to the objects: they feed both the shared and the static library.
set_target_properties(
  ${LIBNAME}_objs PROPERTIES COMPILE_FLAGS
  "-DFFTCONFIGDIR=\\\"${PROJECT_SOURCE_DIR}/config\\\"")
target_link_libraries(${LIBNAME} PUBLIC ${FFTW_LIBRARIES} ${FFTWF_LIBRARIES})
# ==================

install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
