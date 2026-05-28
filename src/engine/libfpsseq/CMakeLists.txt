set(FPSSEQ_SOURCES fpsseq_shm.c fpsseq_scheduler.c fpsseq_fifo.c
                   fpsseq_cmdexec.c fpsseq_script.c)

add_library(milkfpsseq SHARED ${FPSSEQ_SOURCES})
target_include_directories(
  milkfpsseq PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                    $<INSTALL_INTERFACE:include/libfpsseq>)
target_include_directories(
  milkfpsseq
  PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/engine/libfps/cmd>)
target_link_libraries(milkfpsseq PUBLIC milkfps milkprocessinfo ImageStreamIO
                                        rt)
install(
  TARGETS milkfpsseq
  EXPORT milkTargets
  DESTINATION lib)
install(FILES fpsseq.h fpsseq_types.h DESTINATION include/libfpsseq)

if(USE_STATIC_LTO)
  add_library(milkfpsseq_static STATIC ${FPSSEQ_SOURCES})
  set_target_properties(milkfpsseq_static PROPERTIES OUTPUT_NAME milkfpsseq)
  target_include_directories(
    milkfpsseq_static PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                             $<INSTALL_INTERFACE:include>)
  target_link_libraries(
    milkfpsseq_static PUBLIC milkfps_static milkprocessinfo_static
                             ImageStreamIO_static)
endif()

# milk-seq standalone executable
add_executable(milk-seq milk-seq.c)
target_include_directories(milk-seq
                           PRIVATE ${CMAKE_SOURCE_DIR}/src/engine/libfps)
target_link_libraries(
  milk-seq
  PUBLIC milkfpsseq
         milkfps
         milkfpsStandalone
         milkdata
         milkprocessinfo
         ImageStreamIO
         pthread
         m
         rt)
install(TARGETS milk-seq DESTINATION bin)

# milk-seq-help standalone documentation
add_executable(milk-seq-help milk-seq-help.c)
install(TARGETS milk-seq-help DESTINATION bin)
