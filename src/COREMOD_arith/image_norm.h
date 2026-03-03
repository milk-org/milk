#ifndef COREMOD_ARITH_IMAGE_NORM_H
#define COREMOD_ARITH_IMAGE_NORM_H

#include "fps.h"
#include "processinfo.h"

extern char     *norm_inimname;
extern char     *norm_outimname;
extern uint32_t *norm_sliceaxis;

void image_norm_compute(
    FUNCTION_PARAMETER_STRUCT *fps,
    PROCESSINFO *processinfo,
    IMAGE *inimg, IMAGE *outimg
);

#ifndef FPS_STANDALONE
errno_t image_slicenorm_IMGID(
    IMGID *inimg, IMGID *outimg,
    uint8_t sliceaxis
);
errno_t image_slicenorm(
    const char *inname,
    const char *outname,
    uint8_t sliceaxis
);
#endif

errno_t
CLIADDCMD_COREMOD_arith__image_normslice();

#define NORMSLICE_PARAMS(X) \
    X(".in0name", &norm_inimname,            \
      FPTYPE_STREAMNAME, 1,                  \
      (FPFLAG_DEFAULT_INPUT                  \
       | FPFLAG_CLI_INPUT),                  \
      "input image 0")                       \
    X(".outname", &norm_outimname,           \
      FPTYPE_STREAMNAME, 1,                  \
      (FPFLAG_DEFAULT_INPUT                  \
       | FPFLAG_CLI_INPUT),                  \
      "output image")                        \
    X(".axis", &norm_sliceaxis,              \
      FPTYPE_UINT32, 1,                      \
      (FPFLAG_DEFAULT_INPUT                  \
       | FPFLAG_CLI_INPUT),                  \
      "norm axis")

#define NORMSLICE_HELPTEXT \
    "normslice: image norm by slice\n"

#endif
