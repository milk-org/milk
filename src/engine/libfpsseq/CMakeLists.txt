set(FPSSEQ_SOURCES
    fpsseq_shm.c
    fpsseq_scheduler.c
    fpsseq_fifo.c
    fpsseq_cmdexec.c
    fpsseq_script.c
)

add_library(milkfpsseq SHARED ${FPSSEQ_SOURCES})
target_include_directories(milkfpsseq PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/libfpsseq>
)
target_link_libraries(milkfpsseq PUBLIC
    milkfps milkprocessinfo ImageStreamIO rt)
install(TARGETS milkfpsseq
    EXPORT milkTargets DESTINATION lib)
install(FILES fpsseq.h fpsseq_types.h
    DESTINATION include/libfpsseq)

# milk-seq standalone executable
add_executable(milk-seq milk-seq.c)
target_link_libraries(milk-seq PUBLIC
    milkfpsseq milkfps milkfpsStandalone
    milkdata milkprocessinfo ImageStreamIO
    pthread m rt)
install(TARGETS milk-seq DESTINATION bin)
