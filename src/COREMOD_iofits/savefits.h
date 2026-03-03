/**
 * @file savefits.h
 * @brief Header for the FITS save function.
 */

#ifndef COREMOD_IOFITS_SAVEFITS_H
#define COREMOD_IOFITS_SAVEFITS_H

#include "fps.h"
#include "fps_cli_binding.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

/* =========================================
 * GLOBAL PARAMETERS (SHARED)
 * ========================================= */

extern char *savefits_inimname;
extern char *savefits_outfname;
extern int  *savefits_outbitpix;
extern char *savefits_inheader;

/* =========================================
 * SHARED FUNCTIONS
 * ========================================= */

void savefits_compute(
    FUNCTION_PARAMETER_STRUCT *fps,
    PROCESSINFO               *processinfo,
    IMAGE                     *imgin
);

errno_t saveFITS_opt_trunc_IMGID(
    IMGID          *imgin,
    int             truncate,
    const char     *outputFITSname,
    int             outputbitpix,
    const char     *importheaderfile,
    IMAGE_KEYWORD  *kwarray,
    int             kwarraysize,
    const char     *FITSIOext
);

errno_t saveFITS_opt_trunc(
    const char     *inputimname,
    int             truncate,
    const char     *outputFITSname,
    int             outputbitpix,
    const char     *importheaderfile,
    IMAGE_KEYWORD  *kwarray,
    int             kwarraysize,
    const char     *FITSIOext
);

errno_t saveFITS(
    const char     *inputimname,
    const char     *outputFITSname,
    int             outputbitpix,
    const char     *importheaderfile,
    IMAGE_KEYWORD  *kwarray,
    int             kwarraysize
);

errno_t save_fl_fits(
    const char *inputimname,
    const char *outputFITSname
);

errno_t saveall_fits(const char *savedirname);

errno_t save_fits(
    const char *inputimname,
    const char *outputFITSname
);

errno_t CLIADDCMD_COREMOD_iofits__saveFITS();

/* =========================================
 * PARAMETER DEFINITION (V2 6-arg X-MACRO)
 * ========================================= */

#define SAVEFITS_PARAMS(X)                  \
    X(                                      \
        ".in_name",                         \
        &savefits_inimname,                 \
        FPTYPE_STREAMNAME,                  \
        1,                                  \
        FPFLAG_DEFAULT_INPUT,               \
        "input image"                       \
    )                                       \
    X(                                      \
        ".out_fname",                       \
        &savefits_outfname,                 \
        FPTYPE_FILENAME,                    \
        1,                                  \
        FPFLAG_DEFAULT_INPUT,               \
        "output FITS file"                  \
    )                                       \
    X(                                      \
        ".bitpix",                          \
        &savefits_outbitpix,                \
        FPTYPE_INT32,                       \
        0,                                  \
        FPFLAG_DEFAULT_INPUT,               \
        "FITS bitpix"                       \
    )                                       \
    X(                                      \
        ".in_header",                       \
        &savefits_inheader,                 \
        FPTYPE_FILENAME,                    \
        0,                                  \
        FPFLAG_DEFAULT_INPUT,               \
        "header import file"               \
    )

#define SAVEFITS_HELPTEXT                   \
    "saveFITS: save image as FITS\n"        \
    "==========================\n"          \
    "Saves an image stream to a FITS "      \
    "file. Can be run in a loop to log "    \
    "data.\n\n"                             \
    "Parameters:\n"                         \
    "  .in_name   : Input stream name\n"    \
    "  .out_fname : Output filename\n"      \
    "  .bitpix    : FITS bitpix "           \
    "(0 for auto, -32 for float, etc.)\n"

#endif