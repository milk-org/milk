#ifndef COREMOD_ARITH_IMAGE_SET_3DAXES_H
#define COREMOD_ARITH_IMAGE_SET_3DAXES_H

#include "fps.h"
#include "processinfo.h"

extern char     *set3d_inimname;
extern uint32_t *set3d_size0;
extern uint32_t *set3d_size1;
extern uint32_t *set3d_size2;

void image_set_3Daxes_compute(
    FUNCTION_PARAMETER_STRUCT *fps,
    PROCESSINFO *processinfo, IMAGE *inimg
);
errno_t image_set_3Daxes(
    IMGID inimg, uint32_t s0,
    uint32_t s1, uint32_t s2
);

errno_t CLIADDCMD_COREMOD_arith__imset_3Daxes();

#define SET3DAXES_PARAMS(X) \
    X(".imname", &set3d_inimname,            \
      FPTYPE_STREAMNAME, 1,                  \
      (FPFLAG_DEFAULT_INPUT                  \
       | FPFLAG_CLI_INPUT),                  \
      "input image")                         \
    X(".size0", &set3d_size0,                \
      FPTYPE_UINT32, 1,                      \
      (FPFLAG_DEFAULT_INPUT                  \
       | FPFLAG_CLI_INPUT),                  \
      "axis 0 size")                         \
    X(".size1", &set3d_size1,                \
      FPTYPE_UINT32, 1,                      \
      (FPFLAG_DEFAULT_INPUT                  \
       | FPFLAG_CLI_INPUT),                  \
      "axis 1 size")                         \
    X(".size2", &set3d_size2,                \
      FPTYPE_UINT32, 1,                      \
      (FPFLAG_DEFAULT_INPUT                  \
       | FPFLAG_CLI_INPUT),                  \
      "axis 2 size")

#define SET3DAXES_HELPTEXT \
    "set3Daxes: reshape to 3D volume\n" \
    "Sets axis sizes of a 3D image.\n"

#endif