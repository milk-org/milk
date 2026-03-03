/**
 * @file image_set_1Dpixrange.h
 * @brief Set pixels in a 1D index range.
 */

#ifndef COREMOD_ARITH_IMAGE_SET_1DPIXRANGE_H
#define COREMOD_ARITH_IMAGE_SET_1DPIXRANGE_H

#include "fps.h"
#include "processinfo.h"

extern char     *setpix1d_inimname;
extern float    *setpix1d_pixval;
extern uint32_t *setpix1d_minindex;
extern uint32_t *setpix1d_maxindex;

void image_set_1Dpixrange_compute(
    FUNCTION_PARAMETER_STRUCT *fps,
    PROCESSINFO *processinfo, IMAGE *inimg
);
errno_t image_set_1Dpixrange(
    IMGID inimg, double value,
    uint32_t minindex, uint32_t maxindex
);

errno_t
CLIADDCMD_COREMOD_arith__imset_1Dpixrange();

#define SETPIX1D_PARAMS(X) \
    X(".imname", &setpix1d_inimname,        \
      FPTYPE_STREAMNAME, 1,                 \
      (FPFLAG_DEFAULT_INPUT                 \
       | FPFLAG_CLI_INPUT),                 \
      "input image")                        \
    X(".pixval", &setpix1d_pixval,          \
      FPTYPE_FLOAT32, 1,                    \
      (FPFLAG_DEFAULT_INPUT                 \
       | FPFLAG_CLI_INPUT),                 \
      "pixel value")                        \
    X(".mini", &setpix1d_minindex,          \
      FPTYPE_UINT32, 1,                     \
      (FPFLAG_DEFAULT_INPUT                 \
       | FPFLAG_CLI_INPUT),                 \
      "min index")                          \
    X(".maxi", &setpix1d_maxindex,          \
      FPTYPE_UINT32, 1,                     \
      (FPFLAG_DEFAULT_INPUT                 \
       | FPFLAG_CLI_INPUT),                 \
      "max index")

#define SETPIX1D_HELPTEXT \
    "setpix1Drange: set pixel range\n" \
    "Sets pixels in a 1D index range.\n"

#endif
