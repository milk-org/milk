#ifndef COREMOD_ARITH_IMAGE_MERGE3D_H
#define COREMOD_ARITH_IMAGE_MERGE3D_H

#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

extern char     *immerge_inimname0;
extern char     *immerge_inimname1;
extern char     *immerge_outimname;
extern uint32_t *immerge_mergeaxis;

void image_merge_compute(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *processinfo, IMAGE *inimg0, IMAGE *inimg1, IMAGE *outimg);
errno_t image_marge(IMGID inimg0, IMGID inimg1, IMGID *outimg, uint8_t mergeaxis);
errno_t CLIADDCMD_COREMOD_arith__image_merge();

#define IMMERGE_PARAMS(X) \
    X(   FPTYPE_STREAMNAME, char*,    ".in0name", "input image 0",  "im0", "im0", &immerge_inimname0,  (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT)) \
    X(   FPTYPE_STREAMNAME, char*,    ".in1name", "input image 1",  "im1", "im1", &immerge_inimname1,  (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT)) \
    X(   FPTYPE_STREAMNAME, char*,    ".outname", "output image",   "im0", "im0", &immerge_outimname,  (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT)) \
    X(FPTYPE_UINT32,     uint32_t, ".axis",    "merge axis",     "0",   0,     &immerge_mergeaxis, (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT))

#define IMMERGE_HELPTEXT "immerge: merge images along axis"

#endif