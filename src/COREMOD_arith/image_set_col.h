/**
 * @file image_set_col.h
 * @brief Set image column pixel values.
 */

#ifndef COREMOD_ARITH_IMAGE_SET_COL_H
#define COREMOD_ARITH_IMAGE_SET_COL_H

#include "fps.h"
#include "processinfo.h"

extern char     *setcol_inimname;
extern float    *setcol_pixval;
extern uint32_t *setcol_colindex;

void image_set_col_compute(
    FUNCTION_PARAMETER_STRUCT *fps,
    PROCESSINFO               *processinfo,
    IMAGE                     *inimg
);
errno_t image_set_col(
    IMGID    inimg,
    double   value,
    uint32_t colindex
);

errno_t CLIADDCMD_COREMOD_arith__imset_col();

#define SETCOL_PARAMS(X) \
    X(".imname", &setcol_inimname,          \
      FPTYPE_STREAMNAME, 1,                 \
      (FPFLAG_DEFAULT_INPUT                 \
       | FPFLAG_CLI_INPUT),                 \
      "input image")                        \
    X(".pixval", &setcol_pixval,            \
      FPTYPE_FLOAT32, 1,                    \
      (FPFLAG_DEFAULT_INPUT                 \
       | FPFLAG_CLI_INPUT),                 \
      "pixel value")                        \
    X(".col", &setcol_colindex,             \
      FPTYPE_UINT32, 1,                     \
      (FPFLAG_DEFAULT_INPUT                 \
       | FPFLAG_CLI_INPUT),                 \
      "column index")

#define SETCOL_HELPTEXT \
    "setcol: set image column pixel values\n" \
    "======================================\n" \
    "Sets all pixels in a specified column " \
    "of an image stream to a given value.\n"

#endif