/**
 * @file    savefits.c
 * @brief   Save image to FITS file
 *
 * V2 FPS framework migration.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

#include "CLIcore.h"
#include "savefits.h"
#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO/ImageStreamIO.h"
#include "COREMOD_iofits_common.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "check_fitsio_status.h"
#include "file_exists.h"
#include "is_fits_file.h"

extern COREMOD_IOFITS_DATA COREMOD_iofits_data;


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info =
{
    .fps_name    = "savefits",
    .cmdkey      = "saveFITS",
    .description = "save image as FITS",
    .description_long =
    "Save a shared memory image stream to a FITS file on disk. Preserves keywords, data type, and multi-dimensional structure."
};


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".in_name", &savefits_inimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input image") \
    X(".out_fname", &savefits_outfname, \
      FPTYPE_FILENAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output FITS file") \
    X(".bitpix", &savefits_outbitpix, \
      FPTYPE_INT32, 0, \
      FPFLAG_DEFAULT_INPUT, \
      "FITS bitpix") \
    X(".in_header", &savefits_inheader, \
      FPTYPE_FILENAME, 0, \
      FPFLAG_DEFAULT_INPUT, \
      "header import file")

char savefits_inimname[FUNCTION_PARAMETER_STRMAXLEN]  = "";
char savefits_outfname[FUNCTION_PARAMETER_STRMAXLEN]  = "";
int32_t savefits_outbitpix = 0;
char savefits_inheader[FUNCTION_PARAMETER_STRMAXLEN]  = "";


/* =========================================
 * Core FITS write — no CLIcore dependency
 * ========================================= */

errno_t saveFITS_opt_trunc_IMGID(
    IMGID         *imgin,
    int           truncate,
    const char    *outputFITSname,
    int           outputbitpix,
    const char    *importheaderfile __attribute__((unused)),
    IMAGE_KEYWORD *kwarray __attribute__((unused)),
    int           kwarraysize __attribute__((unused)),
    const char    *FITSIOext)
{
    COREMOD_iofits_data.FITSIO_status = 0;
    pthread_t self_id = pthread_self();

    char fnametmp[STRINGMAXLEN_FILENAME];
    char fnametmpext[STRINGMAXLEN_FILENAME];
    WRITE_FILENAME(fnametmp, "%s.%d.%ld.tmp",
                   outputFITSname,
                   (int) getpid(),
                   (long) self_id);
    WRITE_FILENAME(fnametmpext, "%s%s",
                   fnametmp, FITSIOext);

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
    resolveIMGID(imgin, ERRMODE_WARN,
                 dcimg, dcnimg);
#endif
    if(imgin->ID == -1)
    {
        return RETURN_SUCCESS;
    }

    int bitpix = (outputbitpix != 0)
                 ? outputbitpix
                 : ImageStreamIO_FITSIObitpix(
                     imgin->md->datatype);
    if(bitpix == -1)
    {
        bitpix = FLOAT_IMG;
    }

    fitsfile *fptr;
    fits_create_file(&fptr, fnametmpext,
                     &COREMOD_iofits_data.FITSIO_status);
    if(check_FITSIO_status(__FILE__, __func__,
                           __LINE__, 1) != 0)
    {
        return RETURN_FAILURE;
    }

    int naxis = imgin->md->naxis;
    long naxesl[3];
    long nelements = 1;
    for(int i = 0; i < naxis; i++)
    {
        naxesl[i] = (long) imgin->md->size[i];
        if(truncate >= 0 && i == naxis - 1)
        {
            naxesl[naxis - 1] = truncate;
        }
        nelements *= naxesl[i];
    }

    fits_create_img(fptr, bitpix, naxis,
                    naxesl,
                    &COREMOD_iofits_data.FITSIO_status);
    if(check_FITSIO_status(__FILE__, __func__,
                           __LINE__, 1) != 0)
    {
        remove(fnametmp);
        return RETURN_FAILURE;
    }

    fits_write_img(fptr,
                   ImageStreamIO_FITSIOdatatype(
                       imgin->md->datatype),
                   1, nelements,
                   imgin->im->array.raw,
                   &COREMOD_iofits_data.FITSIO_status);
    fits_close_file(fptr,
                    &COREMOD_iofits_data.FITSIO_status);
    rename(fnametmp, outputFITSname);
    return RETURN_SUCCESS;
}


#ifndef FPS_STANDALONE
/**
 * @brief Save a FITS file with optional truncation.
 *
 * Writes an image to FITS with optional precision
 * reduction for smaller file sizes.
 */
errno_t saveFITS_opt_trunc(
    const char    *inputimname,
    int           truncate,
    const char    *outputFITSname,
    int           outputbitpix,
    const char    *importheaderfile,
    IMAGE_KEYWORD *kwarray,
    int           kwarraysize,
    const char    *FITSIOext)
{
    IMGID id = imgid_make_from_name(inputimname);
    return saveFITS_opt_trunc_IMGID(
               &id,          truncate, outputFITSname,
               outputbitpix, importheaderfile,
               kwarray,      kwarraysize, FITSIOext);
}

/**
 * @brief Save a float image to FITS.
 */
errno_t save_fl_fits(
    const char *inputimname,
    const char *outputFITSname)
{
    return saveFITS_opt_trunc(
               inputimname, -1, outputFITSname,
               -32,             NULL, NULL, 0, "");
}

/**
 * @brief Save an image to a FITS file.
 *
 * Standard save with automatic type detection.
 */
errno_t saveFITS(
    const char    *inputimname,
    const char    *outputFITSname,
    int           outputbitpix,
    const char    *importheaderfile,
    IMAGE_KEYWORD *kwarray,
    int           kwarraysize)
{
    return saveFITS_opt_trunc(
               inputimname, -1, outputFITSname,
               outputbitpix, importheaderfile,
               kwarray, kwarraysize, "");
}

errno_t saveall_fits(const char *savedirname)
{
    for(int i = 0; i < dcnimg; i++)
    {
        if(dcimg[i].used == 1)
        {
            char fname[STRINGMAXLEN_FILENAME];
            WRITE_FILENAME(fname, "%s/%s.fits",
                           savedirname,
                           dcimg[i].name);
            saveFITS(dcimg[i].name,
                     fname, 0, NULL, NULL, 0);
        }
    }
    return RETURN_SUCCESS;
}

errno_t save_fits(
    const char *inputimname,
    const char *outputFITSname)
{
    return saveFITS(inputimname,
                    outputFITSname, 0, NULL, NULL, 0);
}
#endif


/* ================================================================
 * 4.  FPS COMPUTE KERNEL
 * ============================================================= */

static MILK_HOT errno_t fpsexec(IMGID *imgin)
{
    if(savefits_outfname[0] == '\0')
    {
        return RETURN_FAILURE;
    }

    return saveFITS_opt_trunc_IMGID(
               imgin, -1, savefits_outfname,
               savefits_outbitpix,
               savefits_inheader[0] ? savefits_inheader : NULL,
               NULL, 0, "");
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

static FPS_CLI_BINDING my_bindings[] =
{
    FPS_PARAMS(FPS_X_BINDING)
};

static const int __attribute__((unused)) nb_bindings =
    sizeof(my_bindings)
    / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] =
{
    FPS_PARAMS(FPS_X_FARG)
};

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata =
{
#else
static CLICMDDATA CLIcmddata =
{
#endif
    "",
    "",
    CLICMD_FIELDS_DEFAULTS
};

FPS_CMDSETTINGS_INIT(dft, CLIcmddata, FPS_app_info)


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    IMGID in =
        imgid_make_from_name(
            savefits_inimname);
    resolveIMGID(
        &in,   ERRMODE_ABORT,
        dcimg, dcnimg);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    fpsexec(&in);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
               &FPS_app_info, farg, &CLIcmddata,
               my_bindings, nb_bindings,
               compute_function);
}

errno_t CLIADDCMD_COREMOD_iofits__saveFITS()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    FPS_app_info,
    FPS_PARAMS,
    compute_function)
#endif
