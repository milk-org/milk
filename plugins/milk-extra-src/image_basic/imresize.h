#ifndef IMAGE_BASIC_IMRESIZE_H
#define IMAGE_BASIC_IMRESIZE_H

#include "fps.h"
#include "fps_cli_binding.h"
#include "processinfo.h"

errno_t imresize_addCLIcmd();

long basic_resizeim(const char *imname_in,
                    const char *imname_out,
                    long        xsizeout,
                    long        ysizeout);

#define IMRESIZE_PARAMS(X) \
    X( \
        ".in_name", \
        &imresize_inimname, \
        FPTYPE_STREAMNAME, \
        1, \
        FPFLAG_DEFAULT_INPUT, \
        "input image" \
    ) \
    X( \
        ".out_name", \
        &imresize_outimname, \
        FPTYPE_STREAMNAME, \
        1, \
        FPFLAG_DEFAULT_INPUT, \
        "output image" \
    ) \
    X( \
        ".xsize", \
        &imresize_xsize, \
        FPTYPE_INT64, \
        1, \
        FPFLAG_DEFAULT_INPUT, \
        "new x size" \
    ) \
    X( \
        ".ysize", \
        &imresize_ysize, \
        FPTYPE_INT64, \
        1, \
        FPFLAG_DEFAULT_INPUT, \
        "new y size" \
    )

extern char *imresize_inimname;
extern char *imresize_outimname;
extern long *imresize_xsize;
extern long *imresize_ysize;

#define IMRESIZE_HELPTEXT \
    "resizeim: resize 2D image\n" \
    "========================\n"

#endif