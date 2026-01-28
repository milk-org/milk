/**
 * @file image_set_2Dpix.h
 * @brief Header for the image set pixel function.
 */

#ifndef COREMOD_ARITH_IMAGE_SET_2DPIX_H
#define COREMOD_ARITH_IMAGE_SET_2DPIX_H

#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

/* ================================================================== */
/* GLOBAL PARAMETERS (SHARED)                                         */
/* ================================================================== */

extern char     *setpix_inimname;
extern float    *setpix_pixval;
extern uint32_t *setpix_colindex;
extern uint32_t *setpix_rowindex;

/* ================================================================== */
/* SHARED FUNCTIONS                                                   */
/* ================================================================== */

void image_set_2Dpix_compute(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *processinfo, IMAGE *inimg);
errno_t image_set_2Dpix(IMGID inimg, double value, uint32_t colindex, uint32_t rowindex);

errno_t CLIADDCMD_COREMOD_arith__imset_2Dpix();

/* ================================================================== */
/* PARAMETER DEFINITION (X-MACRO)                                     */
/* ================================================================== */

#define SETPIX_PARAMS(X) \
    X(CLIARG_IMG,     FPTYPE_STREAMNAME, char*,    ".imname", "input image",  "im1", "im1", &setpix_inimname, (void*)val,  CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_FLOAT32, FPTYPE_FLOAT32,    float,    ".pixval", "pixel value",  "3.2", 3.2f,  &setpix_pixval,   (void*)&val, CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_UINT32,  FPTYPE_UINT32,     uint32_t, ".col",    "col index",    "100", 100,   &setpix_colindex, (void*)&val, CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_UINT32,  FPTYPE_UINT32,     uint32_t, ".row",    "row index",    "100", 100,   &setpix_rowindex, (void*)&val, CLIARG_VISIBLE_DEFAULT)

#define SETPIX_HELPTEXT \
    "setpix: set image pixel value\n" \
    "=============================\n" \
    "Sets a specific pixel (column, row) in an image stream to a given value.\n\n" \
    "Parameters:\n" \
    "  .imname : Input stream name\n" \
    "  .pixval : Value to set pixel to\n" \
    "  .col    : Column index\n" \
    "  .row    : Row index\n"

#endif