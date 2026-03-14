/**
 * @file    savefits.h
 * @brief   Save image to FITS file.
 *
 * Exports public save functions used by many
 * modules and the CLIADDCMD registration.
 */

#ifndef COREMOD_IOFITS_SAVEFITS_H
#define COREMOD_IOFITS_SAVEFITS_H

#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO/ImageStreamIO.h"

/* =========================================
 * PUBLIC PARAMETER GLOBALS
 * ========================================= */

extern char *savefits_inimname;
extern char *savefits_outfname;
extern int  *savefits_outbitpix;
extern char *savefits_inheader;

/* =========================================
 * PUBLIC SAVE FUNCTIONS
 * ========================================= */

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

errno_t saveall_fits(
    const char *savedirname
);

errno_t save_fits(
    const char *inputimname,
    const char *outputFITSname
);

errno_t CLIADDCMD_COREMOD_iofits__saveFITS();

#endif