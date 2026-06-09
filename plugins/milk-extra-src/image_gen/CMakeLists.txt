# library name
set(LIBNAME "milkimagegen")
# set(SRCNAME "image_gen")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

set(SOURCEFILES
    mkdisk.c
    mkpolygon.c
    mkspdisk.c
    mkrandomim.c
    voronoi.c
    imgen_mkgauss.c
    imgen_mkfiberclpoverlap.c
    imgen_mkslopexy.c
    imgen_mkdist.c
    imgen_mkhexsegpup.c
    imgen_segs2wfmodes.c
    imgen_mkrect.c
    imgen_mkline.c
    imgen_mklincoord.c
    imgen_mkgridpix.c
    imgen_mkrndim.c
    imgen_mkrndgim.c
    imgen_im2coord.c
    image_gen_basic_shapes.c
    image_gen_polygons.c
    image_gen_profiles.c
    image_gen_complex.c)

set(INCLUDEFILES mkdisk.h mkpolygon.h mkspdisk.h mkrandomim.h voronoi.h)

set(LINKLIBS CLIcore milkstatistic)

# DEFAULT SETTINGS Do not change unless needed
# =====================================================================

# SRCNAME is current directory (last subdir)
set(NAME0 "${CMAKE_CURRENT_SOURCE_DIR}")
string(REPLACE "/" " " NAME1 ${NAME0})
string(REGEX MATCH "[a-zA-Z0-9_]+$" SRCNAME "${NAME0}")
message(" SRCNAME = ${SRCNAME}")

project(lib_${LIBNAME}_project)

# Library can be compiled from multiple source files Convention: the main souce
# file is named <libname>.c
#
add_library(${LIBNAME} SHARED ${SRCNAME}.c ${SOURCEFILES})

target_include_directories(
  ${LIBNAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<INSTALL_INTERFACE:include>
         $<INSTALL_INTERFACE:include/${SRCNAME}>)

target_link_libraries(${LIBNAME} PRIVATE ${LINKLIBS})

install(
  TARGETS ${LIBNAME}
  EXPORT milkTargets
  DESTINATION lib)
install(FILES ${SRCNAME}.h ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})

# Static archive for USE_STATIC_LTO builds
if(USE_STATIC_LTO)
  set(LIBNAME_STATIC "${LIBNAME}_static")
  add_library(${LIBNAME_STATIC} STATIC ${SRCNAME}.c ${SOURCEFILES})
  set_target_properties(${LIBNAME_STATIC} PROPERTIES OUTPUT_NAME ${LIBNAME})
  target_include_directories(
    ${LIBNAME_STATIC}
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
           $<INSTALL_INTERFACE:include>)
  if(USE_CFITSIO AND CFITSIO_FOUND)
    target_link_libraries(${LIBNAME_STATIC} PUBLIC m ${CFITSIO_LIBRARIES}
                                                   milkstatistic_static)
  else()
    target_link_libraries(${LIBNAME_STATIC} PUBLIC m milkstatistic_static)
  endif()
  install(
    TARGETS ${LIBNAME_STATIC}
    EXPORT milkTargets
    DESTINATION lib)
endif()

add_milk_standalone(imggen-mkrandom mkrandomim.c)
target_link_libraries(milk-fpsexec-imggen-mkrandom PUBLIC ${LIBNAME}
                                                          milkstatistic)
add_milk_standalone(
  imggen-voronoi
  voronoi.c
  imgen_mkgauss.c
  imgen_mkfiberclpoverlap.c
  imgen_mkslopexy.c
  imgen_mkdist.c
  imgen_mkhexsegpup.c
  imgen_segs2wfmodes.c
  imgen_mkrect.c
  imgen_mkline.c
  imgen_mklincoord.c
  imgen_mkgridpix.c
  imgen_mkrndim.c
  imgen_mkrndgim.c
  imgen_im2coord.c)
add_milk_standalone(imggen-mkdisk mkdisk.c)
target_link_libraries(milk-fpsexec-imggen-mkdisk PUBLIC ${LIBNAME})
add_milk_standalone(imggen-mkspdisk mkspdisk.c)
target_link_libraries(milk-fpsexec-imggen-mkspdisk PUBLIC ${LIBNAME})
add_milk_standalone(imggen-mkpolygon mkpolygon.c)
target_link_libraries(milk-fpsexec-imggen-mkpolygon PUBLIC ${LIBNAME})

add_milk_standalone(imggen-mkgauss imgen_mkgauss.c)
target_link_libraries(milk-fpsexec-imggen-mkgauss PUBLIC ${LIBNAME})
add_milk_standalone(imggen-mkfiberclpoverlap imgen_mkfiberclpoverlap.c)
target_link_libraries(milk-fpsexec-imggen-mkfiberclpoverlap PUBLIC ${LIBNAME})
add_milk_standalone(imggen-mkslopexy imgen_mkslopexy.c)
target_link_libraries(milk-fpsexec-imggen-mkslopexy PUBLIC ${LIBNAME})
add_milk_standalone(imggen-mkdist imgen_mkdist.c)
target_link_libraries(milk-fpsexec-imggen-mkdist PUBLIC ${LIBNAME})
add_milk_standalone(imggen-mkhexsegpup imgen_mkhexsegpup.c)
target_link_libraries(milk-fpsexec-imggen-mkhexsegpup PUBLIC ${LIBNAME})
add_milk_standalone(imggen-segs2wfmodes imgen_segs2wfmodes.c)
target_link_libraries(milk-fpsexec-imggen-segs2wfmodes PUBLIC ${LIBNAME})
add_milk_standalone(imggen-mkrect imgen_mkrect.c)
target_link_libraries(milk-fpsexec-imggen-mkrect PUBLIC ${LIBNAME})
add_milk_standalone(imggen-mkline imgen_mkline.c)
target_link_libraries(milk-fpsexec-imggen-mkline PUBLIC ${LIBNAME})
add_milk_standalone(imggen-mklincoord imgen_mklincoord.c)
target_link_libraries(milk-fpsexec-imggen-mklincoord PUBLIC ${LIBNAME})
add_milk_standalone(imggen-mkgridpix imgen_mkgridpix.c)
target_link_libraries(milk-fpsexec-imggen-mkgridpix PUBLIC ${LIBNAME})
add_milk_standalone(imggen-mkrndim imgen_mkrndim.c)
target_link_libraries(milk-fpsexec-imggen-mkrndim PUBLIC ${LIBNAME})
add_milk_standalone(imggen-mkrndgim imgen_mkrndgim.c)
target_link_libraries(milk-fpsexec-imggen-mkrndgim PUBLIC ${LIBNAME})
add_milk_standalone(imggen-im2coord imgen_im2coord.c)
target_link_libraries(milk-fpsexec-imggen-im2coord PUBLIC ${LIBNAME})
