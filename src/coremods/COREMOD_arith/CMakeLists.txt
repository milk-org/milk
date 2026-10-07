set(LIBNAME "milkCOREMODarith")
set(SRCNAME "COREMOD_arith")

message("Module CMakeLists.txt ---- SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

file(GLOB SOURCEFILES "*.c" "*.cpp")
file(GLOB INCLUDEFILES "*.h" "*.hpp")
file(GLOB SCRIPTS "scripts/*")

project(lib_${LIBNAME}_project)

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/..")

add_library(${LIBNAME} SHARED ${SOURCEFILES})
# CLIcore does not exist at this tier: reuse the include dirs of libfps
target_include_directories(
  ${LIBNAME} PRIVATE $<TARGET_PROPERTY:milkfps,INCLUDE_DIRECTORIES>)
milk_apply_extensions(${LIBNAME})

# CUSTOM ===========
find_package(PkgConfig REQUIRED)
if(USE_CFITSIO)
  pkg_check_modules(CFITSIO cfitsio)
  if(${CFITSIO_FOUND})
    link_directories(${CFITSIO_LIBRARY_DIRS})
    target_compile_definitions(${LIBNAME} PUBLIC USE_CFITSIO=1)
    target_include_directories(${LIBNAME} PUBLIC ${CFITSIO_INCLUDE_DIRS})
    target_link_directories(${LIBNAME} PUBLIC ${CFITSIO_LIBRARY_DIRS})
  endif()
endif()

target_include_directories(
  ${LIBNAME} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
                    $<INSTALL_INTERFACE:include>)
target_link_libraries(
  ${LIBNAME} PUBLIC m ${CFITSIO_LIBRARIES} milkfps
                    $<$<BOOL:${USE_CFITSIO}>:milkCOREMODiofits>)

milk_add_static_lto(${LIBNAME} CFITSIO_LINK_LIBS milkCOREMODiofits_static)
# ==================

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTS} DESTINATION bin)

add_milk_standalone(arith-crop2D image_crop2D.c)
add_milk_standalone(arith-multicrop2D image_multicrop2D.c)
add_milk_standalone(arith-immerge image_merge3D.c)
add_milk_standalone(arith-setrow image_set_row.c)
add_milk_standalone(arith-setcol image_set_col.c)
add_milk_standalone(arith-setpix image_set_2Dpix.c)
add_milk_standalone(arith-imsetzero image_setzero.c)
add_milk_standalone(arith-setpix1Drange image_set_1Dpixrange.c)
add_milk_standalone(arith-set3Daxes image_set_3Daxes.c)
add_milk_standalone(arith-normslice image_norm.c)
add_milk_standalone(arith-cropmask image_cropmask.c)
add_milk_standalone(arith-unfold image_unfold.c)
add_milk_standalone(arith-pixremap image_pixremap.c)
add_milk_standalone(arith-slicenormalize image_slicenormalize.c)
add_milk_standalone(arith-pixunmap image_pixunmap.c)
add_milk_standalone(arith-vecmult image_vecmult.c)
add_milk_standalone(arith-imtrunc image_arith__im_f_f__im.c)

get_property(_stdalones DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  PROPERTY BUILDSYSTEM_TARGETS)
foreach(_t IN LISTS _stdalones)
  get_target_property(_type ${_t} TYPE)
  if(_type STREQUAL "EXECUTABLE")
    target_link_libraries(${_t} PUBLIC ${LIBNAME})
  endif()
endforeach()

# CUSTOM ===========
include(CTest)

set(TESTNAME "CLIfunc-COREMOD_arith-module")
add_test(NAME "${TESTNAME}" COMMAND milk-exec "-T" "m?")
set_property(TEST "${TESTNAME}" PROPERTY LABELS "CLImodule")
set_property(TEST "${TESTNAME}" PROPERTY TIMEOUT 1)
set_property(TEST "${TESTNAME}" PROPERTY PASS_REGULAR_EXPRESSION
                                         "COREMOD_arith")

list(
  APPEND
  commandlist
  "extractim"
  "extract3Dim"
  "setpix"
  "setpix1Drange"
  "setrow"
  "setcol"
  "imzero"
  "imtrunc"
  "cropmask")

foreach(CLIcmdname IN LISTS commandlist)

  set(TESTCLIFUNC "${CLIcmdname}")
  set(TESTNAME "CLIfunc-COREMOD_arith-${TESTCLIFUNC}")
  add_test(NAME "${TESTNAME}" COMMAND milk-exec "-T" "cmd? arith.${CLIcmdname}")
  set_property(TEST "${TESTNAME}" PROPERTY LABELS "CLIfunc")
  set_property(TEST "${TESTNAME}" PROPERTY TIMEOUT 1)
  set_property(TEST "${TESTNAME}" PROPERTY PASS_REGULAR_EXPRESSION "src:")

endforeach()
# ==================
