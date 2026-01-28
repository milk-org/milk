/**
 * @file image_set_row.h
 * @brief Header for the image set row function.
 */

#ifndef COREMOD_ARITH_IMAGE_SET_ROW_H
#define COREMOD_ARITH_IMAGE_SET_ROW_H

#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

/* ================================================================== */
/* GLOBAL PARAMETERS (SHARED)                                         */
/* ================================================================== */

extern char     *setrow_inimname;
extern float    *setrow_pixval;
extern uint32_t *setrow_rowindex;

/* ================================================================== */
/* SHARED FUNCTIONS                                                   */
/* ================================================================== */

void image_set_row_compute(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *processinfo, IMAGE *inimg);
errno_t image_set_row(IMGID inimg, double value, uint32_t rowindex);

errno_t CLIADDCMD_COREMOD_arith__imset_row();

/* ================================================================== */
/* PARAMETER DEFINITION (X-MACRO)                                     */
/* ================================================================== */

#define SETROW_PARAMS(X) \
    X(CLIARG_IMG,     FPTYPE_STREAMNAME, char*,    ".imname", "input image",  "im1", "im1", &setrow_inimname, (void*)val,  CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_FLOAT32, FPTYPE_FLOAT32,    float,    ".pixval", "pixel value",  "3.2", 3.2f,  &setrow_pixval,   (void*)&val, CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_UINT32,  FPTYPE_UINT32,     uint32_t, ".row",    "row index",    "100", 100,   &setrow_rowindex, (void*)&val, CLIARG_VISIBLE_DEFAULT)

#define SETROW_HELPTEXT \
    "setrow: set image row pixels values\n" \
    "===================================\n" \
    "Sets all pixels in a specified row of an image stream to a given value.\n\n" \
    "Parameters:\n" \
    "  .imname : Input stream name\n" \
    "  .pixval : Value to set pixels to\n" \
    "  .row    : Index of the row to modify\n"

#endif