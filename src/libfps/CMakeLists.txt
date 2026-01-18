set(LIBNAME "fps")
set(SRCNAME "libfps")

set(SOURCEFILES
    fps_add_entry.c
    fps_checkparameter.c
    fps_connect.c
    fps_connectExternalFPS.c
    fps_disconnect.c
    fps_GetFileName.c
    fps_GetParamIndex.c
    fps_GetTypeString.c
    fps_paramvalue.c
    fps_printlist.c
    fps_PrintParameterInfo.c
    fps_printparameter_valuestring.c
    fps_save2disk.c
    fps_shmdirname.c
    fps_WriteParameterToDisk.c
    fps_loadstream.c
    fps_outlog.c
    fps_globals.c
    timeutils.c
)

add_library(${LIBNAME} SHARED ${SOURCEFILES})

target_include_directories(${LIBNAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(${LIBNAME} PUBLIC milkCOREMODiofits ImageStreamIO)

install(TARGETS ${LIBNAME} EXPORT milkTargets DESTINATION lib)
install(FILES fps.h fps_internal.h fps_globals.h timeutils.h DESTINATION include/libfps)
