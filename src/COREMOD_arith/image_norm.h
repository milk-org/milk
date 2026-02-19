#ifndef COREMOD_ARITH_IMAGE_NORM_H
#define COREMOD_ARITH_IMAGE_NORM_H

#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

extern char     *norm_inimname;
extern char     *norm_outimname;
extern uint32_t *norm_sliceaxis;

void image_norm_compute(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *processinfo, IMAGE *inimg, IMAGE *outimg);
errno_t image_slicenorm_IMGID(IMGID *inimg, IMGID *outimg, uint8_t sliceaxis);
errno_t image_slicenorm(const char *inname, const char *outname, uint8_t sliceaxis);

errno_t CLIADDCMD_COREMOD_arith__image_normslice();

#define NORMSLICE_PARAMS(X) \
    X(FPTYPE_STREAMNAME, char*,    ".in0name", "input image 0",  "im0", "im0", &norm_inimname,  (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT)) \
    X(FPTYPE_STREAMNAME, char*,    ".outname", "output image",   "im0", "im0", &norm_outimname,  (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT)) \
    X(FPTYPE_UINT32,     uint32_t, ".axis",    "norm axis",      "0",   0,     &norm_sliceaxis, (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT))

#define NORMSLICE_HELPTEXT "normslice: image norm by slice"

#endif
