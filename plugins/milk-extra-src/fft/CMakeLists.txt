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

add_library(${LIBNAME} SHARED ${SOURCEFILES})
target_include_directories(
  ${LIBNAME} PRIVATE $<TARGET_PROPERTY:CLIcore,INTERFACE_INCLUDE_DIRECTORIES>)
milk_apply_extensions(${LIBNAME})

# CUSTOM ===========
set_target_properties(
  ${LIBNAME} PROPERTIES COMPILE_FLAGS
  "-DFFTCONFIGDIR=\\\"${PROJECT_SOURCE_DIR}/config\\\"")
target_link_libraries(${LIBNAME} PUBLIC ${FFTW_LIBRARIES} ${FFTWF_LIBRARIES})
# ==================

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})

add_milk_standalone(fft-dofft dofft.c)
add_milk_standalone(fft-pup2foc pup2foc.c)

get_property(_stdalones DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  PROPERTY BUILDSYSTEM_TARGETS)
foreach(_t IN LISTS _stdalones)
  get_target_property(_type ${_t} TYPE)
  if(_type STREQUAL "EXECUTABLE")
    target_link_libraries(${_t} PUBLIC ${LIBNAME})
  endif()
endforeach()
