#ifndef IMAGE_BASIC_IMROTATE_H
#define IMAGE_BASIC_IMROTATE_H

#include "fps.h"
#include "fps_cli_binding.h"
#include "processinfo.h"

errno_t imrotate_addCLIcmd();

imageID basic_rotate(const char *__restrict ID_name,
                     const char *__restrict IDout_name,
                     float angle);

imageID basic_rotate90(const char *__restrict ID_name,
                       const char *__restrict ID_out_name);

imageID basic_rotate_int(const char *__restrict ID_name,
                         const char *__restrict ID_out_name,
                         long nbstep);

imageID basic_rotate2(const char *__restrict ID_name_in,
                      const char *__restrict ID_name_out,
                      float angle);

#define IMROTATE_PARAMS(X) \
    X( \
        ".in_name", \
        &imrotate_inimname, \
        FPTYPE_STREAMNAME, \
        1, \
        FPFLAG_DEFAULT_INPUT, \
        "input image" \
    ) \
    X( \
        ".out_name", \
        &imrotate_outimname, \
        FPTYPE_STREAMNAME, \
        1, \
        FPFLAG_DEFAULT_INPUT, \
        "output image" \
    ) \
    X( \
        ".angle", \
        &imrotate_angle, \
        FPTYPE_FLOAT32, \
        1, \
        FPFLAG_DEFAULT_INPUT, \
        "rotate angle" \
    )

extern char  *imrotate_inimname;
extern char  *imrotate_outimname;
extern float *imrotate_angle;

#define IMROTATE_HELPTEXT \
    "rotateim: rotate 2D image\n" \
    "========================\n"

#endif