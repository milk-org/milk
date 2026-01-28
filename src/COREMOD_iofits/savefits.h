/**
 * @file savefits.h
 * @brief Header for the FITS save function.
 */

#ifndef COREMOD_IOFITS_SAVEFITS_H
#define COREMOD_IOFITS_SAVEFITS_H

#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

/* ==================================================================
 * GLOBAL PARAMETERS (SHARED)                                         
 * ==================================================================
 */

extern char *savefits_inimname;
extern char *savefits_outfname;
extern int  *savefits_outbitpix;
extern char *savefits_inheader;

/* ==================================================================
 * SHARED FUNCTIONS                                                   
 * ==================================================================
 */

void savefits_compute(FUNCTION_PARAMETER_STRUCT *fps, PROCESSINFO *processinfo, IMAGE *imgin);
errno_t saveFITS_opt_trunc_IMGID(IMGID *imgin, int truncate, const char *outputFITSname, int outputbitpix, const char *importheaderfile, IMAGE_KEYWORD *kwarray, int kwarraysize, const char *FITSIOext);
errno_t saveFITS_opt_trunc(const char *inputimname, int truncate, const char *outputFITSname, int outputbitpix, const char *importheaderfile, IMAGE_KEYWORD *kwarray, int kwarraysize, const char *FITSIOext);
errno_t saveFITS(const char *inputimname, const char *outputFITSname, int outputbitpix, const char *importheaderfile, IMAGE_KEYWORD *kwarray, int kwarraysize);
errno_t save_fl_fits(const char *inputimname, const char *outputFITSname);
errno_t saveall_fits(const char *savedirname);
errno_t save_fits(const char *inputimname, const char *outputFITSname);

errno_t CLIADDCMD_COREMOD_iofits__saveFITS();

/* ==================================================================
 * PARAMETER DEFINITION (X-MACRO)                                     
 * ==================================================================
 */

#define SAVEFITS_PARAMS(X) \
    X(CLIARG_IMG,   FPTYPE_STREAMNAME, char*, ".in_name",   "input image",       "im1",      "im1",      &savefits_inimname,  (void*)val,  CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_STR,   FPTYPE_FILENAME,   char*, ".out_fname", "output FITS file",  "out.fits", "out.fits", &savefits_outfname,  (void*)val,  CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_INT32, FPTYPE_INT32,      int,   ".bitpix",    "FITS bitpix",       "0",        0,          &savefits_outbitpix, (void*)&val, CLIARG_HIDDEN_DEFAULT) \
    X(CLIARG_STR,   FPTYPE_FILENAME,   char*, ".in_header", "header import file", "",         "",         &savefits_inheader,  (void*)val,  CLIARG_HIDDEN_DEFAULT)

#define SAVEFITS_HELPTEXT \
    "saveFITS: save image as FITS\n" \
    "==========================\n" \
    "Saves an image stream to a FITS file. Can be run in a loop to log data.\n\n" \
    "Parameters:\n" \
    "  .in_name   : Input stream name\n" \
    "  .out_fname : Output filename\n" \
    "  .bitpix    : FITS bitpix (0 for auto, -32 for float, etc.)\n"

#endif