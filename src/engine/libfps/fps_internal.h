/**
 * @file fps_internal.h
 * @brief Fps internal module
 */

#ifndef FPS_INTERNAL_H
#define FPS_INTERNAL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "milkDebugTools.h"
#include "ImageStreamIO/ImageStruct.h"

errno_t function_parameter_struct_create(int NBparamMAX, const char *name);
errno_t function_parameter_struct_realloc(FPS *fps, int NBparamMAX_new);

/* Forward declarations for cross-file libfps functions */
imageID COREMOD_IOFITS_LoadMemStream(
    const char *sname,
    uint64_t   *streamflag,
    uint32_t   *imLOC);

int file_exists(const char *filename);
int is_fits_file(const char *filename);
int functionparameter_ConnectExternalFPS(
    FPS *fpsentry,
    int  pindex,
    FPS *fpstest);

#endif // FPS_INTERNAL_H
