/**
 * @file image_set_2Dpix.h
 * @brief Set a single pixel value in a 2D image.
 */

#ifndef COREMOD_ARITH_IMAGE_SET_2DPIX_H
#define COREMOD_ARITH_IMAGE_SET_2DPIX_H

#include "fps.h"
#include "processinfo.h"

extern char     *setpix_inimname;
extern float    *setpix_pixval;
extern uint32_t *setpix_colindex;
extern uint32_t *setpix_rowindex;

void image_set_2Dpix_compute(
    FUNCTION_PARAMETER_STRUCT *fps,
    PROCESSINFO               *processinfo,
    IMAGE                     *inimg
);
errno_t image_set_2Dpix(
    IMGID inimg, double value,
    uint32_t colindex, uint32_t rowindex
);

errno_t CLIADDCMD_COREMOD_arith__imset_2Dpix();

#define SETPIX_PARAMS(X) \
    X(".imname", &setpix_inimname,          \
      FPTYPE_STREAMNAME, 1,                 \
      (FPFLAG_DEFAULT_INPUT                 \
       | FPFLAG_CLI_INPUT),                 \
      "input image")                        \
    X(".pixval", &setpix_pixval,            \
      FPTYPE_FLOAT32, 1,                    \
      (FPFLAG_DEFAULT_INPUT                 \
       | FPFLAG_CLI_INPUT),                 \
      "pixel value")                        \
    X(".col", &setpix_colindex,             \
      FPTYPE_UINT32, 1,                     \
      (FPFLAG_DEFAULT_INPUT                 \
       | FPFLAG_CLI_INPUT),                 \
      "col index")                          \
    X(".row", &setpix_rowindex,             \
      FPTYPE_UINT32, 1,                     \
      (FPFLAG_DEFAULT_INPUT                 \
       | FPFLAG_CLI_INPUT),                 \
      "row index")

#define SETPIX_HELPTEXT \
    "setpix: set image pixel value\n" \
    "Assigns a value to a single pixel.\n"

#endif