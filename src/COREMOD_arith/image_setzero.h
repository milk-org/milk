/**
 * @file image_setzero.h
 * @brief Header for the image setzero function.
 */

#ifndef COREMOD_ARITH_IMAGE_SETZERO_H
#define COREMOD_ARITH_IMAGE_SETZERO_H

#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

/* ================================================================== */
/* GLOBAL PARAMETERS (SHARED)                                         */
/* ================================================================== */

extern char *imsetzero_inimname;

/* ================================================================== */
/* SHARED FUNCTIONS                                                   */
/* ================================================================== */

void image_setzero_compute(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *processinfo, IMAGE *inimg);
errno_t image_setzero_IMGID(IMGID *inimg);
errno_t image_setzero(IMGID inimg);

errno_t CLIADDCMD_COREMOD_arith__imsetzero();

/* ================================================================== */
/* PARAMETER DEFINITION (X-MACRO)                                     */
/* ================================================================== */

#define IMSETZERO_PARAMS(X) \
    X(FPTYPE_STREAMNAME, char*, ".imname", "input image", "im1", "im1", &imsetzero_inimname, (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT))

#define IMSETZERO_HELPTEXT \
    "imzero: set all image pixels to zero\n" \
    "====================================\n" \
    "Sets all elements of the specified image stream to zero.\n\n" \
    "Parameters:\n" \
    "  .imname : Input stream name\n"

#endif