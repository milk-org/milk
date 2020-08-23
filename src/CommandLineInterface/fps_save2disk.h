/**
 * @file    fps_save2disk.h
 * @brief   Save FPS content to disk
 */

#ifndef FPS_SAVE2DISK_H
#define FPS_SAVE2DISK_H
#endif

#include "function_parameters.h"

int functionparameter_SaveParam2disk(
    FUNCTION_PARAMETER_STRUCT *fpsentry,
    const char *paramname
);


int functionparameter_SaveFPS2disk_dir(
    FUNCTION_PARAMETER_STRUCT *fpsentry,
    char *dirname
);


int functionparameter_SaveFPS2disk(
    FUNCTION_PARAMETER_STRUCT *fpsentry
);


errno_t	functionparameter_write_archivescript(
    FUNCTION_PARAMETER_STRUCT *fps
);


errno_t fps_write_RUNoutput_image(
	FUNCTION_PARAMETER_STRUCT *fps,
	const char *imagename,
	const char *outname
);

FILE *fps_write_RUNoutput_file(
	FUNCTION_PARAMETER_STRUCT *fps,
	const char *filename,
	const char *extension
);

errno_t fps_datadir_to_confdir(
	FUNCTION_PARAMETER_STRUCT *fps
);
