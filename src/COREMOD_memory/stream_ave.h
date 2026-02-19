/**
 * @file stream_ave.h
 * @brief Header for the stream average function.
 */

#ifndef COREMOD_MEMORY_STREAM_AVE_H
#define COREMOD_MEMORY_STREAM_AVE_H

#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

/* ================================================================== */
/* GLOBAL PARAMETERS (SHARED)                                         */
/* ================================================================== */

extern char     *streamave_inimname;
extern char     *streamave_outimave;
extern uint32_t *streamave_outimshared;
extern char     *streamave_outimrms;
extern uint64_t *streamave_NBcoadd;
extern uint64_t *streamave_cntindex;
extern uint64_t *streamave_compave;
extern uint64_t *streamave_comprms;

/* ================================================================== */
/* SHARED FUNCTIONS                                                   */
/* ================================================================== */

void stream_ave_compute(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *processinfo, IMAGE *imgin, IMAGE *imgoutave, IMAGE *imgoutrms, double *imdataarray, double *imdataarrayPOW);

errno_t CLIADDCMD_streamaverage();

/* ================================================================== */
/* PARAMETER DEFINITION (X-MACRO)                                     */
/* ================================================================== */

#define STREAMAVE_PARAMS(X) \
    X(   FPTYPE_STREAMNAME, char*,    ".in_name",      "input image",            "im1",  "im1",  &streamave_inimname,  (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT)) \
    X(   FPTYPE_STREAMNAME, char*,    ".outave_name",  "output average image",   "out1", "out1", &streamave_outimave,  (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT)) \
    X(FPTYPE_UINT32,     uint32_t, ".out_shared",   "output shared flag",     "1",    1,      &streamave_outimshared, FPFLAG_DEFAULT_INPUT) \
    X(   FPTYPE_STREAMNAME, char*,    ".outrms_name",  "output RMS image",       "out1", "out1", &streamave_outimrms,  FPFLAG_DEFAULT_INPUT) \
    X(FPTYPE_UINT64,     uint64_t, ".NBcoadd",      "number of coadded",      "100",  100,    &streamave_NBcoadd, (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT)) \
    X(FPTYPE_UINT64,     uint64_t, ".cntindex",     "counter index",          "0",    0,      &streamave_cntindex, FPFLAG_DEFAULT_INPUT) \
    X( FPTYPE_ONOFF,      uint64_t, ".comp.ave",     "compute average",        "1",    1,      &streamave_compave, FPFLAG_DEFAULT_INPUT) \
    X( FPTYPE_ONOFF,      uint64_t, ".comp.rms",     "compute rms",            "0",    0,      &streamave_comprms, FPFLAG_DEFAULT_INPUT)

#define STREAMAVE_HELPTEXT \
    "streamave: average stream of images\n" \
    "==================================\n" \
    "Computes the average and optionally the RMS of a stream of images over\n" \
    "a specified number of frames.\n\n" \
    "Parameters:\n" \
    "  .in_name      : Input stream\n" \
    "  .outave_name  : Average output stream\n" \
    "  .NBcoadd      : Number of frames to average\n"

#endif