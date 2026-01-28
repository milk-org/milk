/**
 * @file image_set_3Daxes.h
 * @brief Header for the image set 3D axes function.
 */

#ifndef COREMOD_ARITH_IMAGE_SET_3DAXES_H
#define COREMOD_ARITH_IMAGE_SET_3DAXES_H

#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

/* ================================================================== */
/* GLOBAL PARAMETERS (SHARED)                                         */
/* ================================================================== */

extern char     *set3d_inimname;
extern uint32_t *set3d_size0;
extern uint32_t *set3d_size1;
extern uint32_t *set3d_size2;

/* ================================================================== */
/* SHARED FUNCTIONS                                                   */
/* ================================================================== */

void image_set_3Daxes_compute(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *processinfo, IMAGE *inimg);
errno_t image_set_3Daxes(IMGID inimg, uint32_t imsize0, uint32_t imsize1, uint32_t imsize2);

errno_t CLIADDCMD_COREMOD_arith__imset_3Daxes();

/* ================================================================== */
/* PARAMETER DEFINITION (X-MACRO)                                     */
/* ================================================================== */

#define SET3DAXES_PARAMS(X) \
    X(CLIARG_IMG,    FPTYPE_STREAMNAME, char*,    ".imname", "input image", "im1", "im1", &set3d_inimname, (void*)val,  CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_UINT32, FPTYPE_UINT32,     uint32_t, ".size0",  "axis 0 size", "128", 128,   &set3d_size0,    (void*)&val, CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_UINT32, FPTYPE_UINT32,     uint32_t, ".size1",  "axis 1 size", "128", 128,   &set3d_size1,    (void*)&val, CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_UINT32, FPTYPE_UINT32,     uint32_t, ".size2",  "axis 2 size", "128", 128,   &set3d_size2,    (void*)&val, CLIARG_VISIBLE_DEFAULT)

#define SET3DAXES_HELPTEXT \
    "set3Daxes: set 3D image axes size\n" \
    "=================================\n" \
    "Reshapes an image stream into a 3D volume by specifying axis sizes.\n" \
    "The total number of elements must remain constant.\n\n" \
    "Parameters:\n" \
    "  .imname : Input stream name\n" \
    "  .size0  : New size for axis 0\n" \
    "  .size1  : New size for axis 1\n" \
    "  .size2  : New size for axis 2\n"

#endif