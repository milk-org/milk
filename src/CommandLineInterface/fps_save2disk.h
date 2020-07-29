/**
 * @file    fps_save2disk.h
 * @brief   Save FPS content to disk
 */


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
    FUNCTION_PARAMETER_STRUCT *fps,
    char *archdirname
);
