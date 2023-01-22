# library name
set(LIBNAME "milkCOREMODmemory")
set(SRCNAME "COREMOD_memory")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

set(SOURCEFILES
    ${SRCNAME}.c
    clearall.c
    compute_image_memory.c
    compute_nb_image.c
    compute_nb_variable.c
    create_image.c
    create_variable.c
    delete_image.c
    delete_sharedmem_image.c
    delete_variable.c
    fps_create.c
    fps_ID.c
    fps_list.c
    image_checksize.c
    image_complex.c
    image_copy.c
    image_copy_shm.c
    image_ID.c
    image_keyword.c
    image_keyword_addD.c
    image_keyword_addL.c
    image_keyword_addS.c
    image_keyword_list.c
    image_make2D.c
    image_make3D.c
    image_mk_complex_from_amph.c
    image_mk_complex_from_reim.c
    image_mk_amph_from_complex.c
    image_mk_reim_from_complex.c
    image_set_counters.c
    list_image.c
    list_variable.c
    logshmim.c
    read_shmim.c
    read_shmim_size.c
    read_shmimall.c
    saveall.c
    shmimlog.c
    shmimlogcmd.c
    shmim_purge.c
    shmim_setowner.c
    stream_ave.c
    stream_copy.c
    stream_merge.c
    stream_delay.c
    stream_diff.c
    stream_halfimdiff.c
    stream_monitorlimits.c
    stream_paste.c
    stream_pixmapdecode.c
    stream_poke.c
    stream_sem.c
    stream_TCP.c
    stream_UDP.c
    stream_updateloop.c
    variable_ID.c
   )

set(INCLUDEFILES
    ${SRCNAME}.h
    clearall.h
    compute_image_memory.h
    compute_nb_image.h
    compute_nb_variable.h
    create_image.h
    create_variable.h
    delete_image.h
    delete_sharedmem_image.h
    delete_variable.h
    fps_create.h
    fps_ID.h
    fps_list.h
    image_checksize.h
    image_complex.h
    image_copy.h
    image_copy_shm.h
    image_ID.h
    image_keyword.h
    image_keyword_addD.h
    image_keyword_addL.h
    image_keyword_addS.h
    image_keyword_list.h
    image_make2D.h
    image_make3D.h
    image_mk_complex_from_amph.h
    image_mk_complex_from_reim.h
    image_mk_amph_from_complex.h
    image_mk_reim_from_complex.h
    image_set_counters.h
    list_image.h
    list_variable.h
    logshmim.h
    shmimlog_types.h
    read_shmim.h
    read_shmim_size.h
    read_shmimall.h
    saveall.h
    shmimlog.h
    shmimlogcmd.h
    shmim_purge.h
    shmim_setowner.h
    stream_ave.h
    stream_copy.h
    stream_delay.h
    stream_merge.h
    stream_diff.h
    stream_halfimdiff.h
    stream_monitorlimits.h
    stream_paste.h
    stream_pixmapdecode.h
    stream_poke.h
    stream_sem.h
    stream_TCP.h
    stream_UDP.h
    stream_updateloop.h
    variable_ID.h
   )

# list scripts that should be installed on system
set(SCRIPTFILES
  scripts/milk-rmshmim
  scripts/milk-shmimave
	scripts/milk-shmimpurge
  scripts/milk-shmimcopy-semtrig
	scripts/milk-shmimpoke
  scripts/milk-shmimpoke-semtrig
)



# DEFAULT SETTINGS
# Do not change unless needed
# =====================================================================

project(lib_${LIBNAME}_project)

# Library can be compiled from multiple source files
# Convention: the main souce file is named <libname>.c
#
add_library(${LIBNAME} SHARED ${SOURCEFILES})

find_package(PkgConfig REQUIRED)
pkg_check_modules(CFITSIO cfitsio)
if(${CFITSIO_FOUND})
  link_directories(${CFITSIO_LIBRARY_DIRS})
  target_compile_definitions(${LIBNAME} PUBLIC USE_CFITSIO=1)
  target_include_directories(${LIBNAME} PUBLIC ${CFITSIO_INCLUDE_DIRS})
  target_link_directories(${LIBNAME} PUBLIC ${CFITSIO_LIBRARY_DIRS})
endif()

target_include_directories(${LIBNAME} PUBLIC ${PROJECT_SOURCE_DIR}/..)
target_link_libraries(${LIBNAME} PUBLIC m ${CFITSIO_LIBRARIES})

install(TARGETS ${LIBNAME} DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTFILES} DESTINATION bin)
