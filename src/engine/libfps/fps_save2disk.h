/**
 * @file    fps_save2disk.h
 * @brief   Save FPS content to disk
 */

#ifndef FPS_SAVE2DISK_H
#define FPS_SAVE2DISK_H

#include "fps.h"

int functionparameter_SaveParam2disk(
    FPS        *fpsentry,
    const char *paramname);

int functionparameter_SaveFPS2disk_dir(
    FPS  *fpsentry,
    char *dirname);

int functionparameter_SaveFPS2disk(FPS *fpsentry);

errno_t functionparameter_write_archivescript(FPS *fps);

errno_t fps_write_RUNoutput_image(
    FPS        *fps,
    const char *imagename,
    const char *outname);

FILE *fps_write_RUNoutput_file(FPS *fps,
                               const char                *filename,
                               const char                *extension);

errno_t fps_datadir_to_confdir(FPS *fps);

#endif
