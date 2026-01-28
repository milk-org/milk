/**
 * @file image_set_1Dpixrange.h
 * @brief Header for the image set 1D pixel range function.
 */

#ifndef COREMOD_ARITH_IMAGE_SET_1DPIXRANGE_H
#define COREMOD_ARITH_IMAGE_SET_1DPIXRANGE_H

#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

/* ==================================================================
 * GLOBAL PARAMETERS (SHARED)                                         
 * ==================================================================
 */

extern char     *setpix1d_inimname;
extern float    *setpix1d_pixval;
extern uint32_t *setpix1d_minindex;
extern uint32_t *setpix1d_maxindex;

/* ==================================================================
 * SHARED FUNCTIONS                                                   
 * ==================================================================
 */

void image_set_1Dpixrange_compute(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *processinfo, IMAGE *inimg);
errno_t image_set_1Dpixrange(IMGID inimg, double value, uint32_t minindex, uint32_t maxindex);

errno_t CLIADDCMD_COREMOD_arith__imset_1Dpixrange();

/* ==================================================================
 * PARAMETER DEFINITION (X-MACRO)                                     
 * ==================================================================
 */

#define SETPIX1D_PARAMS(X) \
    X(CLIARG_IMG,     FPTYPE_STREAMNAME, char*,    ".imname", "input image",  "im1", "im1", &setpix1d_inimname, (void*)val,  CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_FLOAT32, FPTYPE_FLOAT32,    float,    ".pixval", "pixel value",  "3.2", 3.2f,  &setpix1d_pixval,   (void*)&val, CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_UINT32,  FPTYPE_UINT32,     uint32_t, ".mini",   "min index",    "10",  10,    &setpix1d_minindex, (void*)&val, CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_UINT32,  FPTYPE_UINT32,     uint32_t, ".maxi",   "max index",    "50",  50,    &setpix1d_maxindex, (void*)&val, CLIARG_VISIBLE_DEFAULT)

#define SETPIX1D_HELPTEXT \
    "setpix1Drange: set image pixel value over range\n" \
    "===============================================\n" \
    "Sets pixels in a specified 1D index range to a given value.\n\n" \
    "Parameters:\n" \
    "  .imname : Input stream name\n" \
    "  .pixval : Value to set pixels to\n" \
    "  .mini   : Minimum index (inclusive)\n" \
    "  .maxi   : Maximum index (exclusive)\n"

#endif
