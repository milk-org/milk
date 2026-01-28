/**
 * @file image_set_col.h
 * @brief Header for the image set column function.
 */

#ifndef COREMOD_ARITH_IMAGE_SET_COL_H
#define COREMOD_ARITH_IMAGE_SET_COL_H

#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

/* ================================================================== */
/* GLOBAL PARAMETERS (SHARED)                                         */
/* ================================================================== */

extern char     *setcol_inimname;
extern float    *setcol_pixval;
extern uint32_t *setcol_colindex;

/* ================================================================== */
/* SHARED FUNCTIONS                                                   */
/* ================================================================== */

void image_set_col_compute(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *processinfo, IMAGE *inimg);
errno_t image_set_col(IMGID inimg, double value, uint32_t colindex);

errno_t CLIADDCMD_COREMOD_arith__imset_col();

/* ================================================================== */
/* PARAMETER DEFINITION (X-MACRO)                                     */
/* ================================================================== */

#define SETCOL_PARAMS(X) \
    X(CLIARG_IMG,     FPTYPE_STREAMNAME, char*,    ".imname", "input image",  "im1", "im1", &setcol_inimname, (void*)val,  CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_FLOAT32, FPTYPE_FLOAT32,    float,    ".pixval", "pixel value",  "3.2", 3.2f,  &setcol_pixval,   (void*)&val, CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_UINT32,  FPTYPE_UINT32,     uint32_t, ".col",    "column index", "100", 100,   &setcol_colindex, (void*)&val, CLIARG_VISIBLE_DEFAULT)

#define SETCOL_HELPTEXT \
    "setcol: set image column pixels values\n" \
    "======================================\n" \
    "Sets all pixels in a specified column of an image stream to a given value.\n\n" \
    "Parameters:\n" \
    "  .imname : Input stream name\n" \
    "  .pixval : Value to set pixels to\n" \
    "  .col    : Index of the column to modify\n"

#endif