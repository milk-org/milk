#ifndef IMAGE_BASIC_IMROTATE_H
#define IMAGE_BASIC_IMROTATE_H

#include "fps.h"
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
    X(    FPTYPE_STREAMNAME, char*,  ".in_name",  "input image",   "im1", "im1", &imrotate_inimname,  (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT)) \
    X(    FPTYPE_STREAMNAME, char*,  ".out_name", "output image",  "out1", "out1", &imrotate_outimname,  (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT)) \
    X(FPTYPE_FLOAT32,    float,  ".angle",    "rotate angle",  "0.0", 0.0,   &imrotate_angle, (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT))

extern char  *imrotate_inimname;
extern char  *imrotate_outimname;
extern float *imrotate_angle;

#define IMROTATE_HELPTEXT \
    "rotateim: rotate 2D image\n" \
    "========================\n"

#endif