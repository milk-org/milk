/**
 * @file image_set_row.h
 * @brief Header for the image set row function.
 */

#ifndef COREMOD_ARITH_IMAGE_SET_ROW_H
#define COREMOD_ARITH_IMAGE_SET_ROW_H

#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

/* ============================================ */
/* GLOBAL PARAMETERS (SHARED)                   */
/* ============================================ */

extern char     *setrow_inimname;
extern float    *setrow_pixval;
extern uint32_t *setrow_rowindex;

/* ============================================ */
/* SHARED FUNCTIONS                             */
/* ============================================ */

void image_set_row_compute(
    FUNCTION_PARAMETER_STRUCT *fps,
    PROCESSINFO               *processinfo,
    IMAGE                     *inimg
);

errno_t image_set_row(
    IMGID    inimg,
    double   value,
    uint32_t rowindex
);


errno_t CLIADDCMD_COREMOD_arith__imset_row();

/* ============================================ */
/* PARAMETER DEFINITION (X-MACRO)               */
/* ============================================ */

/**
 * V2 format: X(keyword, ptr, type,
 *              is_primary, fpflag, descr)
 */
#define SETROW_PARAMS(X) \
    X(".imname", &setrow_inimname,          \
      FPTYPE_STREAMNAME, 1,                 \
      (FPFLAG_DEFAULT_INPUT                 \
       | FPFLAG_CLI_INPUT),                 \
      "input image")                        \
    X(".pixval", &setrow_pixval,            \
      FPTYPE_FLOAT32, 1,                    \
      (FPFLAG_DEFAULT_INPUT                 \
       | FPFLAG_CLI_INPUT),                 \
      "pixel value")                        \
    X(".row", &setrow_rowindex,             \
      FPTYPE_UINT32, 1,                     \
      (FPFLAG_DEFAULT_INPUT                 \
       | FPFLAG_CLI_INPUT),                 \
      "row index")

#define SETROW_HELPTEXT \
    "setrow: set image row pixel values\n" \
    "===================================\n" \
    "Sets all pixels in a specified row " \
    "of an image stream to a given " \
    "value.\n\n" \
    "Parameters:\n" \
    "  .imname : Input stream name\n" \
    "  .pixval : Value to set pixels to\n" \
    "  .row    : Index of the row\n"

#endif