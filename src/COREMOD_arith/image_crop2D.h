/**
 * @file image_crop2D.h
 * @brief Header for the 2D crop function.
 */

#ifndef COREMOD_ARITH_CROP2D_H
#define COREMOD_ARITH_CROP2D_H

#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

/* ================================================================== */
/* GLOBAL PARAMETERS (SHARED)                                         */
/* ================================================================== */

extern char     *cropinsname;
extern char     *outsname;
extern uint32_t *cropxstart;
extern uint32_t *cropxsize;
extern uint32_t *cropystart;
extern uint32_t *cropysize;

/* ================================================================== */
/* SHARED FUNCTIONS                                                   */
/* ================================================================== */

void image_crop2D_compute(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *processinfo, IMAGE *input_image, IMAGE *output_image);
errno_t image_crop2D_validate();

errno_t CLIADDCMD_COREMODE_arith__crop2D();

/* ================================================================== */
/* PARAMETER DEFINITION (X-MACRO)                                     */
/* ================================================================== */

#define CROP2D_PARAMS(X) \
    X(CLIARG_IMG,    FPTYPE_STREAMNAME, char*,    ".insname",    "Input stream name",  "inim",  "inim",  &cropinsname, (void*)val,  CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_STR,    FPTYPE_STREAMNAME, char*,    ".outsname",   "Output stream name", "outim", "outim", &outsname,    (void*)val,  CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_UINT32, FPTYPE_UINT32,     uint32_t, ".cropxstart", "crop x coord start", "30",    30,      &cropxstart,  (void*)&val, CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_UINT32, FPTYPE_UINT32,     uint32_t, ".cropxsize",  "crop x coord size",  "32",    32,      &cropxsize,   (void*)&val, CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_UINT32, FPTYPE_UINT32,     uint32_t, ".cropystart", "crop y coord start", "20",    20,      &cropystart,  (void*)&val, CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_UINT32, FPTYPE_UINT32,     uint32_t, ".cropysize",  "crop y coord size",  "32",    32,      &cropysize,   (void*)&val, CLIARG_VISIBLE_DEFAULT)

#define CROP2D_HELPTEXT \
    "Crop 2D: extract sub-region from image stream\n" \
    "============================================\n" \
    "Extracts a rectangular region from an input stream and writes it to an\n" \
    "output stream.\n\n" \
    "Parameters:\n" \
    "  .insname    : Input stream name\n" \
    "  .outsname   : Output stream name\n" \
    "  .cropxstart : Start coordinate in X\n" \
    "  .cropxsize  : Size in X\n" \
    "  .cropystart : Start coordinate in Y\n" \
    "  .cropysize  : Size in Y\n"

#endif