# library name
set(LIBNAME "milklinalgebra")
set(SRCNAME "linalgebra")

message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

# This is only good so far as we're taking everything
# And no mixing of CPP files
file(GLOB SOURCEFILES "*.c" "*.cpp")
file(GLOB INCLUDEFILES "*.h" "*.hpp")

file(GLOB SCRIPTS "scripts/*")

# DEFAULT SETTINGS Do not change unless needed
# =====================================================================

project(lib_${LIBNAME}_project)

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/..")

add_library(${LIBNAME} SHARED ${SOURCEFILES})
target_include_directories(
  ${LIBNAME} PRIVATE $<TARGET_PROPERTY:CLIcore,INTERFACE_INCLUDE_DIRECTORIES>)

message("---- module linalgebra ------------------")

milk_apply_extensions(${LIBNAME})

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTS} DESTINATION bin)

# List standalones
# FIXME can do better with a MILK_CMAKE tag in the file.
add_milk_standalone(linalg-gramschmidt GramSchmidt.c)
add_milk_standalone(linalg-MVMextract MVMextractModes.cpp)
add_milk_standalone(linalg-PCAmatch PCAmatch.c)
add_milk_standalone(linalg-Qexpand Qexpand.c)
add_milk_standalone(linalg-SGEMM SGEMM.c)
add_milk_standalone(linalg-SVD SingularValueDecomp.c)
add_milk_standalone(linalg-SVDmkM SingularValueDecomp_mkM.c)
add_milk_standalone(linalg-SVDmkU SingularValueDecomp_mkU.c)
add_milk_standalone(linalg-basisrotmatch basis_rotate_match.c)
add_milk_standalone(linalg-cublasPCA cublas_PCA.c)
add_milk_standalone(linalg-modalremap modalremap.c)

# Link module library into every standalone in this directory
# FIXME can do better than this in the future I'm sure !
get_property(_stdalones DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
             PROPERTY BUILDSYSTEM_TARGETS)
foreach(_t IN LISTS _stdalones)
  get_target_property(_type ${_t} TYPE)
  if(_type STREQUAL "EXECUTABLE")
    target_link_libraries(${_t} PUBLIC ${LIBNAME})
  endif()
endforeach()
