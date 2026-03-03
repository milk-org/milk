/**
 * @file stream_ave.h
 * @brief Header for the stream average function.
 */

#ifndef COREMOD_MEMORY_STREAM_AVE_H
#define COREMOD_MEMORY_STREAM_AVE_H

#include "fps.h"
#include "fps_cli_binding.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

/* =========================================
 * GLOBAL PARAMETERS (SHARED)
 * ========================================= */

extern char     *streamave_inimname;
extern char     *streamave_outimave;
extern uint32_t *streamave_outimshared;
extern char     *streamave_outimrms;
extern uint64_t *streamave_NBcoadd;
extern uint64_t *streamave_cntindex;
extern uint64_t *streamave_compave;
extern uint64_t *streamave_comprms;

/* =========================================
 * SHARED FUNCTIONS
 * ========================================= */

void stream_ave_compute(
    FUNCTION_PARAMETER_STRUCT *fps,
    PROCESSINFO               *processinfo,
    IMAGE                     *imgin,
    IMAGE                     *imgoutave,
    IMAGE                     *imgoutrms,
    double                    *imdataarray,
    double                    *imdataarrayPOW
);

errno_t CLIADDCMD_streamaverage();

/* =========================================
 * PARAMETER DEFINITION (V2 6-arg X-MACRO)
 * ========================================= */

#define STREAMAVE_PARAMS(X)                 \
    X(                                      \
        ".in_name",                         \
        &streamave_inimname,                \
        FPTYPE_STREAMNAME,                  \
        1,                                  \
        FPFLAG_DEFAULT_INPUT,               \
        "input image"                       \
    )                                       \
    X(                                      \
        ".outave_name",                     \
        &streamave_outimave,                \
        FPTYPE_STREAMNAME,                  \
        1,                                  \
        FPFLAG_DEFAULT_INPUT,               \
        "output average image"              \
    )                                       \
    X(                                      \
        ".out_shared",                      \
        &streamave_outimshared,             \
        FPTYPE_UINT32,                      \
        0,                                  \
        FPFLAG_DEFAULT_INPUT,               \
        "output shared flag"                \
    )                                       \
    X(                                      \
        ".outrms_name",                     \
        &streamave_outimrms,                \
        FPTYPE_STREAMNAME,                  \
        0,                                  \
        FPFLAG_DEFAULT_INPUT,               \
        "output RMS image"                  \
    )                                       \
    X(                                      \
        ".NBcoadd",                         \
        &streamave_NBcoadd,                 \
        FPTYPE_UINT64,                      \
        1,                                  \
        FPFLAG_DEFAULT_INPUT,               \
        "number of coadded"                 \
    )                                       \
    X(                                      \
        ".cntindex",                        \
        &streamave_cntindex,                \
        FPTYPE_UINT64,                      \
        0,                                  \
        FPFLAG_DEFAULT_INPUT,               \
        "counter index"                     \
    )                                       \
    X(                                      \
        ".comp.ave",                        \
        &streamave_compave,                 \
        FPTYPE_ONOFF,                       \
        0,                                  \
        FPFLAG_DEFAULT_INPUT,               \
        "compute average"                   \
    )                                       \
    X(                                      \
        ".comp.rms",                        \
        &streamave_comprms,                 \
        FPTYPE_ONOFF,                       \
        0,                                  \
        FPFLAG_DEFAULT_INPUT,               \
        "compute rms"                       \
    )

#define STREAMAVE_HELPTEXT                   \
    "streamave: average stream of images\n"  \
    "==================================\n"   \
    "Computes the average and optionally "   \
    "the RMS of a stream of images over\n"   \
    "a specified number of frames.\n\n"      \
    "Parameters:\n"                          \
    "  .in_name      : Input stream\n"       \
    "  .outave_name  : Average output\n"     \
    "  .NBcoadd      : Frames to average\n"

#endif