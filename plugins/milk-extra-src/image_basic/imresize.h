#ifndef IMAGE_BASIC_IMRESIZE_H
#define IMAGE_BASIC_IMRESIZE_H

#include "fps.h"
#include "processinfo.h"

errno_t imresize_addCLIcmd();

long basic_resizeim(const char *imname_in,
                    const char *imname_out,
                    long        xsizeout,
                    long        ysizeout);

#define IMRESIZE_PARAMS(X) \
    X(CLIARG_IMG,     FPTYPE_STREAMNAME, char*,  ".in_name",  "input image",   "im1", "im1", &imresize_inimname, (void*)val,  CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_STR,     FPTYPE_STREAMNAME, char*,  ".out_name", "output image",  "out1", "out1", &imresize_outimname, (void*)val,  CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_INT64,   FPTYPE_INT64,      long,   ".xsize",    "new x size",    "128", 128,   &imresize_xsize,    (void*)&val, CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_INT64,   FPTYPE_INT64,      long,   ".ysize",    "new y size",    "128", 128,   &imresize_ysize,    (void*)&val, CLIARG_VISIBLE_DEFAULT)

extern char *imresize_inimname;
extern char *imresize_outimname;
extern long *imresize_xsize;
extern long *imresize_ysize;

#define IMRESIZE_HELPTEXT \
    "resizeim: resize 2D image\n" \
    "========================\n"

#endif