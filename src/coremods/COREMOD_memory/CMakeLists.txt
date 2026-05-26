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
    fps_ID.c
    fps_list.c
    im3D_to_stream2D.c
    image_checksize.c
    image_complex.c
    image_copy.c
    image_copy_shm.c
    image_ID.c
    image_keyword.c
    image_keyword_add.c
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
    logshmim_save.c
    read_shmim.c
    read_shmim_size.c
    read_shmimall.c
    saveall.c
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
    stream_TCP_receive.c
    stream_UDP.c
    stream_UDP_receive.c
    stream_updateloop.c
    stream_slice.c
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
    fps_ID.h
    fps_list.h
    im3D_to_stream2D.h
    image_checksize.h
    image_complex.h
    image_copy.h
    image_copy_shm.h
    image_ID.h
    imageID.h
    image_keyword.h
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
    stream_slice.h
    variable_ID.h
   )

# list scripts that should be installed on system
set(SCRIPTFILES
  scripts/milk-nettransmit
  scripts/milk-shmimave
  scripts/milk-shmimcopy-semtrig
  scripts/milk-shmimpoke
  scripts/milk-shmimpoke-semtrig
  scripts/milk-semloopspeed
  scripts/milk-streamFITSlog
)

# TESTING

include(CTest)

# test module is loaded
set(TESTNAME "CLIfunc-COREMOD_memory-module")
add_test (NAME "${TESTNAME}" COMMAND milk-exec "-T" "m?")
set_property (TEST "${TESTNAME}" PROPERTY LABELS "CLImodule")
set_property (TEST "${TESTNAME}" PROPERTY TIMEOUT 1)
set_property (TEST "${TESTNAME}" PROPERTY PASS_REGULAR_EXPRESSION "COREMOD_memory")

# test that commands are registered

list(APPEND commandlist "mk2Dim" "mk3Dim" "listim" "rmall")

foreach(CLIcmdname IN LISTS commandlist)

set(TESTCLIFUNC "${CLIcmdname}")
set(TESTNAME "CLIfunc-COREMOD_memory-${TESTCLIFUNC}")
add_test (NAME "${TESTNAME}" COMMAND milk-exec "-T" "cmd? mem.${CLIcmdname}")
set_property (TEST "${TESTNAME}" PROPERTY LABELS "CLIfunc")
set_property (TEST "${TESTNAME}" PROPERTY TIMEOUT 1)
set_property (TEST "${TESTNAME}" PROPERTY PASS_REGULAR_EXPRESSION "src:")

endforeach()

# Semaphore latency - speed test

set(TESTNAME "milksemloopspeed")
add_test (NAME "${TESTNAME}" COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/milk-semloopspeed "-D" "123456")
set_property (TEST "${TESTNAME}" PROPERTY LABELS "perf")
set_property (TEST "${TESTNAME}" PROPERTY TIMEOUT 5)
set_property (TEST "${TESTNAME}" PROPERTY PASS_REGULAR_EXPRESSION "cnt0 = 123457")

# DEFAULT SETTINGS
# Do not change unless needed
# =====================================================================

project(lib_${LIBNAME}_project)

# Library can be compiled from multiple source files
# Convention: the main souce file is named <libname>.c
#
add_library(${LIBNAME} SHARED ${SOURCEFILES})

find_package(PkgConfig REQUIRED)
if(USE_CFITSIO)
  pkg_check_modules(CFITSIO cfitsio)
  if(${CFITSIO_FOUND})
    link_directories(${CFITSIO_LIBRARY_DIRS})
    target_compile_definitions(${LIBNAME} PUBLIC USE_CFITSIO=1)
    target_include_directories(${LIBNAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore> ${CFITSIO_INCLUDE_DIRS})
    target_link_directories(${LIBNAME} PUBLIC ${CFITSIO_LIBRARY_DIRS})
  endif()
endif()

target_include_directories(${LIBNAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore> $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
                                              $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli>
                                              $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/streamCTRL>
											  $<INSTALL_INTERFACE:include>)
if(USE_CFITSIO AND CFITSIO_LIBRARIES)
  target_link_libraries(${LIBNAME} PUBLIC m ${CFITSIO_LIBRARIES} milkfps)
else()
  target_link_libraries(${LIBNAME} PUBLIC m milkfps)
endif()

install(TARGETS ${LIBNAME} EXPORT milkTargets DESTINATION lib)
install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})
install(PROGRAMS ${SCRIPTFILES} DESTINATION bin)

# Compute-only library for standalone builds (no CLI code)
set(LIBNAME_COMPUTE "${LIBNAME}_compute")
add_library(${LIBNAME_COMPUTE} SHARED ${SOURCEFILES})
target_compile_definitions(${LIBNAME_COMPUTE} PRIVATE MILK_NO_CLI)
if(USE_CFITSIO AND CFITSIO_FOUND)
  target_compile_definitions(${LIBNAME_COMPUTE} PUBLIC USE_CFITSIO=1)
  target_include_directories(${LIBNAME_COMPUTE} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore> ${CFITSIO_INCLUDE_DIRS})
  target_link_directories(${LIBNAME_COMPUTE} PUBLIC ${CFITSIO_LIBRARY_DIRS})
endif()
target_include_directories(${LIBNAME_COMPUTE} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>)
if(USE_CFITSIO AND CFITSIO_LIBRARIES)
  target_link_libraries(${LIBNAME_COMPUTE} PUBLIC
      m ${CFITSIO_LIBRARIES} milkfps)
else()
  target_link_libraries(${LIBNAME_COMPUTE} PUBLIC m milkfps)
endif()
install(TARGETS ${LIBNAME_COMPUTE} EXPORT milkTargets
    DESTINATION lib)

# Static archive for standalone LTO builds
if(USE_STATIC_LTO)
  set(LIBNAME_COMPUTE_STATIC "${LIBNAME_COMPUTE}_static")
  add_library(${LIBNAME_COMPUTE_STATIC} STATIC
      ${SOURCEFILES})
  set_target_properties(${LIBNAME_COMPUTE_STATIC}
      PROPERTIES OUTPUT_NAME ${LIBNAME_COMPUTE})
  target_compile_definitions(${LIBNAME_COMPUTE_STATIC}
      PRIVATE MILK_NO_CLI)
  if(USE_CFITSIO AND CFITSIO_FOUND)
    target_compile_definitions(${LIBNAME_COMPUTE_STATIC}
        PUBLIC USE_CFITSIO=1)
    target_include_directories(${LIBNAME_COMPUTE_STATIC}
        PUBLIC ${CFITSIO_INCLUDE_DIRS})
    target_link_directories(${LIBNAME_COMPUTE_STATIC}
        PUBLIC ${CFITSIO_LIBRARY_DIRS})
  endif()
  target_include_directories(${LIBNAME_COMPUTE_STATIC}
      PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/CLIcore/CLIcore>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/streamCTRL>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps/fpsCTRL>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libprocessinfo>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libmilkdata>
      $<INSTALL_INTERFACE:include>)
  if(USE_CFITSIO AND CFITSIO_LIBRARIES)
    target_link_libraries(${LIBNAME_COMPUTE_STATIC} PUBLIC
        m ${CFITSIO_LIBRARIES} milkfps_static milkcommon)
  else()
    target_link_libraries(${LIBNAME_COMPUTE_STATIC} PUBLIC m milkfps_static milkcommon)
  endif()
endif()

add_milk_standalone(mem-streamave stream_ave.c)
add_milk_standalone(mem-streampoke stream_poke.c)
add_milk_standalone(mem-streamcopy stream_copy.c)
add_milk_standalone(imkwadd image_keyword_add.c)

# Standalone executable
add_executable(milk-fpsexec-cnt2push cnt2push_main.c)
target_include_directories(milk-fpsexec-cnt2push PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>)
target_link_libraries(milk-fpsexec-cnt2push ImageStreamIO m)
install(TARGETS milk-fpsexec-cnt2push DESTINATION bin)

add_executable(milk-stream-list milk-stream-list.c)
target_include_directories(milk-stream-list PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)
target_link_libraries(milk-stream-list ImageStreamIO)
install(TARGETS milk-stream-list DESTINATION bin)

add_executable(milk-stream-create milk-stream-create.c)
target_include_directories(milk-stream-create PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)
target_link_libraries(milk-stream-create ImageStreamIO)
install(TARGETS milk-stream-create DESTINATION bin)

add_executable(milk-stream-rm milk-stream-rm.c)
target_include_directories(milk-stream-rm PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cli/libmilkscript>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)
target_link_libraries(milk-stream-rm ImageStreamIO)
install(TARGETS milk-stream-rm DESTINATION bin)

add_executable(milk-streamCTRL-cli milk-streamCTRL-cli.c)
target_include_directories(milk-streamCTRL-cli PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)
target_link_libraries(milk-streamCTRL-cli ImageStreamIO)
install(TARGETS milk-streamCTRL-cli DESTINATION bin)

# ==========================================
# milk-makecsetandrt
# Replaces bash script milk-makecsetandrt
# ==========================================
add_executable(milk-makecsetandrt milk-makecsetandrt.c)
target_include_directories(milk-makecsetandrt PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)
target_link_libraries(milk-makecsetandrt PRIVATE milkcommon)
install(TARGETS milk-makecsetandrt DESTINATION bin)

# ==========================================
# milk-shmimpurge
# Replaces bash script milk-shmimpurge
# ==========================================
add_executable(milk-shmimpurge milk-shmimpurge.c)
target_include_directories(milk-shmimpurge PRIVATE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)
target_link_libraries(milk-shmimpurge PRIVATE ImageStreamIO)
install(TARGETS milk-shmimpurge DESTINATION bin)

# ==========================================
# milk-logshim-ctrl
# Unified replacement for the 5 milk-logshim* scripts:
#   milk-logshim, milk-logshimkill, milk-logshimon,
#   milk-logshimoff, milk-logshimstat
# ==========================================
add_executable(milk-logshim-ctrl milk-logshim-ctrl.c)
target_include_directories(milk-logshim-ctrl PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>)
install(TARGETS milk-logshim-ctrl DESTINATION bin)

# ==========================================
# milk-stream-link
# C replacement for the milk-streamlink bash script.
# Reads conf/streamlink.*.name.txt, creates SHM symlink,
# writes imsize.txt via ImageStreamIO.
# ==========================================
add_executable(milk-stream-link milk-stream-link.c)
target_include_directories(milk-stream-link PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/ImageStreamIO>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>)
target_link_libraries(milk-stream-link PRIVATE ImageStreamIO)
install(TARGETS milk-stream-link DESTINATION bin)
