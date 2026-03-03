/**
 * @file image_crop2D.h
 * @brief Crop a 2D rectangular region from stream.
 */

#ifndef COREMOD_ARITH_CROP2D_H
#define COREMOD_ARITH_CROP2D_H

#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

extern char     *cropinsname;
extern char     *outsname;
extern uint32_t *cropxstart;
extern uint32_t *cropxsize;
extern uint32_t *cropystart;
extern uint32_t *cropysize;

void image_crop2D_compute(
    FUNCTION_PARAMETER_STRUCT *fps,
    PROCESSINFO *processinfo,
    IMAGE *input_image,
    IMAGE *output_image
);
errno_t image_crop2D_validate();

errno_t CLIADDCMD_COREMODE_arith__crop2D();

#define CROP2D_PARAMS(X) \
    X(".insname", &cropinsname,              \
      FPTYPE_STREAMNAME, 1,                  \
      (FPFLAG_DEFAULT_INPUT                  \
       | FPFLAG_CLI_INPUT),                  \
      "Input stream name")                   \
    X(".outsname", &outsname,                \
      FPTYPE_STREAMNAME, 1,                  \
      (FPFLAG_DEFAULT_INPUT                  \
       | FPFLAG_CLI_INPUT),                  \
      "Output stream name")                  \
    X(".cropxstart", &cropxstart,            \
      FPTYPE_UINT32, 1,                      \
      (FPFLAG_DEFAULT_INPUT                  \
       | FPFLAG_CLI_INPUT),                  \
      "crop x coord start")                  \
    X(".cropxsize", &cropxsize,              \
      FPTYPE_UINT32, 1,                      \
      (FPFLAG_DEFAULT_INPUT                  \
       | FPFLAG_CLI_INPUT),                  \
      "crop x coord size")                   \
    X(".cropystart", &cropystart,            \
      FPTYPE_UINT32, 1,                      \
      (FPFLAG_DEFAULT_INPUT                  \
       | FPFLAG_CLI_INPUT),                  \
      "crop y coord start")                  \
    X(".cropysize", &cropysize,              \
      FPTYPE_UINT32, 1,                      \
      (FPFLAG_DEFAULT_INPUT                  \
       | FPFLAG_CLI_INPUT),                  \
      "crop y coord size")

#define CROP2D_HELPTEXT \
    "Crop 2D: extract sub-region\n" \
    "from image stream.\n"

#endif