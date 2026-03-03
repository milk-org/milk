#ifndef COREMOD_ARITH_IMAGE_MERGE3D_H
#define COREMOD_ARITH_IMAGE_MERGE3D_H

#include "fps.h"
#include "processinfo.h"

extern char     *immerge_inimname0;
extern char     *immerge_inimname1;
extern char     *immerge_outimname;
extern uint32_t *immerge_mergeaxis;

void image_merge_compute(
    FUNCTION_PARAMETER_STRUCT *fps,
    PROCESSINFO *processinfo,
    IMAGE *in0, IMAGE *in1, IMAGE *out
);
errno_t image_marge(
    IMGID inimg0, IMGID inimg1,
    IMGID *outimg, uint8_t mergeaxis
);

errno_t CLIADDCMD_COREMOD_arith__image_merge();

#define IMMERGE_PARAMS(X) \
    X(".in0name", &immerge_inimname0,        \
      FPTYPE_STREAMNAME, 1,                  \
      (FPFLAG_DEFAULT_INPUT                  \
       | FPFLAG_CLI_INPUT),                  \
      "input image 0")                       \
    X(".in1name", &immerge_inimname1,        \
      FPTYPE_STREAMNAME, 1,                  \
      (FPFLAG_DEFAULT_INPUT                  \
       | FPFLAG_CLI_INPUT),                  \
      "input image 1")                       \
    X(".outname", &immerge_outimname,        \
      FPTYPE_STREAMNAME, 1,                  \
      (FPFLAG_DEFAULT_INPUT                  \
       | FPFLAG_CLI_INPUT),                  \
      "output image")                        \
    X(".axis", &immerge_mergeaxis,           \
      FPTYPE_UINT32, 1,                      \
      (FPFLAG_DEFAULT_INPUT                  \
       | FPFLAG_CLI_INPUT),                  \
      "merge axis")

#define IMMERGE_HELPTEXT \
    "immerge: merge images along axis\n"

#endif