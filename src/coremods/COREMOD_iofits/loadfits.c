/**
 * @file    loadfits.c
 * @brief   load FITS format files
 */

// MILK_CMAKE_MANDATE_CFITSIO

#include <stdlib.h>

#include "CLIcore.h"
#include "fps.h"
#include "loadfits.h"

#include "COREMOD_iofits_common.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "check_fitsio_status.h"
#include "data_type_code.h"

#include "COREMOD_memory/image_keyword.h"
extern COREMOD_IOFITS_DATA COREMOD_iofits_data;

// ==========================================
// FPS V2
// ==========================================

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "loadfits",
    .cmdkey           = "loadfits",
    .description      = "load FITS format file",
    .description_long = "Load a FITS file from disk into a shared memory image stream. Supports 2D "
                        "and 3D images with automatic data type detection."
};

// CLI function arguments and parameters
static char    infilename[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char    outimname[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static int64_t FITSIOerrmode                            = 2;

#define FPS_PARAMS(X)                                                                            \
    X(".infname", &infilename, FPTYPE_FILENAME, 1, FPFLAG_DEFAULT_INPUT, "input file")           \
    X(".outimname", &outimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "output image name") \
    X(".errmode", &FITSIOerrmode, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT,                         \
      "FITSIO errors mode (0:ignore) (1:warning) (2:error) (3:exit)")


FPS_V2_SECTION5(FPS_PARAMS)


/// errmode values :
/// LOADFITS_ERRMODE_IGNORE  (0) print warning, do not show error messages, continue
/// LOADFITS_ERRMODE_WARNING (1) print error, continue
/// LOADFITS_ERRMODE_ERROR   (2) return error
/// LOADFITS_ERRMODE_EXIT    (3) exit program at error

/**
 * @brief Load a FITS file and return an IMGID.
 *
 * Reads a FITS file from disk into the image array
 * and returns the IMGID handle.
 */
errno_t load_fits_IMGID(const char *__restrict file_name, IMGID *imgout, int errmode)
{
    DEBUG_TRACE_FSTART();

    fitsfile *fptr = NULL; /* pointer to the FITS file; defined in fitsio.h */
    int       nulval, anynul;
    long      bitpixl = 0;

    uint32_t naxes[3];

    double         bscale;
    double         bzero;
    unsigned char *barray = NULL;
    long          *larray = NULL;

    nulval = 0;
    anynul = 0;
    bscale = 1;
    bzero  = 0;

    naxes[0] = 0;
    naxes[1] = 0;
    naxes[2] = 0;

    DEBUG_TRACEPOINT("FARG \"%s\" %s %d", file_name, imgout->name, errmode);

    {
        // Open fitsio file pointer
        // tyr 3 consecutive times and then give up if not successful
        int fileOK = 0;
        int NBtry  = 3;
        for (int tr = 0; tr < NBtry; tr++)
        {
            if (fileOK == 0)
            {
                int status = 0;
                fits_open_file(&fptr, file_name, READONLY, &status);

                if (status != 0)
                {
                    if (errmode > 0)
                    {
                        printf("attempt # %d failed\n", tr);
                    }

                    //void fits_get_errstatus(int status, char *err_text)
                    if (status != 0)
                    {
                        if (errmode > 1)
                        {
                            if (tr == NBtry - 1)
                            {
                                FITSIO_CHECK_ERROR(status, errmode,
                                                   "can't load %s "
                                                   "(tried %d times)",
                                                   file_name, NBtry);
                            }
                        }
                        if (tr != NBtry - 1) // don't wait on last try
                        {
                            usleep(10000);
                        }
                    }

                    imgout->ID = -1;
                }
                else
                {
                    fileOK = 1;
                }
            }
        }

        //printf("fileOK = %d\n", fileOK);

        if (fileOK == 0)
        {
            // if image not loaded, set output identifier to -1
            imgout->ID = -1;

            if (errmode == 0)
            {
                DEBUG_TRACE_FEXIT();
                return RETURN_SUCCESS;
            }

            if (errmode == 1)
            {
                PRINT_WARNING("Image \"%s\" could not be loaded from file \"%s\"", imgout->name,
                              file_name);
                DEBUG_TRACE_FEXIT();
                return RETURN_SUCCESS;
            }

            if (errmode == 2)
            {
                FUNC_RETURN_FAILURE("Image \"%s\" could not be loaded from file \"%s\"",
                                    imgout->name, file_name);
            }

            if (errmode == 3)
            {
                abort();
            }
            DEBUG_TRACE_FEXIT();
            return -1;
        }
    }

    DEBUG_TRACEPOINT("File %s open", file_name);

    long fpixel = 1;
    long naxis  = 0;
    long nelements;

    // Keywords
    int nbFITSkeys = 0;

    {
        int status = 0;
        fits_get_hdrspace(fptr, &nbFITSkeys, NULL, &status);
        FITSIO_CHECK_ERROR(status, errmode, "fits_get_hdrspace error on %s", file_name);
    }

    {
        int status = 0;
        fits_read_key(fptr, TLONG, "NAXIS", &naxis, NULL, &status);
        FITSIO_CHECK_ERROR(status, errmode, "File %s has no NAXIS", file_name);
        DEBUG_TRACEPOINT("naxis = %ld", naxis);
    }

    for (long i = 0; i < naxis; i++)
    {
        char keyword[STRINGMAXLEN_FITSKEYWORDNAME];
        WRITE_FITSKEYWNAME(keyword, "NAXIS%ld", i + 1);

        {
            int status = 0;
            fits_read_key(fptr, TLONG, keyword, &naxes[i], NULL, &status);
            FITSIO_CHECK_ERROR(status, errmode, "File %s has no NAXIS%ld", file_name, i);
        }
    }

    {
        int status = 0;
        fits_read_key(fptr, TLONG, "BITPIX", &bitpixl, NULL, &status);
        FITSIO_CHECK_ERROR(status, errmode, "File %s has no BITPIX", file_name);
    }

    int bitpix = (int) bitpixl;
    {
        int status = 0;
        fits_read_key(fptr, TDOUBLE, "BSCALE", &bscale, NULL, &status);
        if (status != 0)
        {
            bscale = 1.0;
        }
    }

    {
        int status = 0;
        fits_read_key(fptr, TDOUBLE, "BZERO", &bzero, NULL, &status);
        if (status != 0)
        {
            bzero = 0.0;
        }
    }

    {
        int status = 0;
        fits_set_bscale(fptr, bscale, bzero, &status);
        FITSIO_CHECK_ERROR(status, errmode, "bscale set error");
    }

    nelements = 1;
    for (long i = 0; i < naxis; i++)
    {
        nelements *= naxes[i];
    }

    /* bitpix = -32  TFLOAT */
    if (bitpix == -32)
    {
        imgout->mdt->naxis = naxis;
        for (long i = 0; i < naxis; i++)
        {
            imgout->mdt->size[i] = naxes[i];
        }
        imgout->mdt->datatype = _DATATYPE_FLOAT;
        imgout->mdt->shared   = dcshareddft;
        imgout->mdt->NBkw     = NB_KEYWNODE_MAX;
        imgout->im            = (IMAGE *) calloc(1, sizeof(IMAGE));
        imgid_mkimage(imgout);

        {
            int status = 0;
            fits_read_img(fptr, data_type_code(bitpix), fpixel, nelements, &nulval,
                          imgout->im->array.F, &anynul, &status);
            FITSIO_CHECK_ERROR(status, errmode, "fits_read_img bitpix=%d", bitpix);
        }
    }

    /* bitpix = -64  TDOUBLE */
    if (bitpix == -64)
    {
        imgout->mdt->naxis = naxis;
        for (long i = 0; i < naxis; i++)
        {
            imgout->mdt->size[i] = naxes[i];
        }
        imgout->mdt->datatype = _DATATYPE_DOUBLE;
        imgout->mdt->shared   = dcshareddft;
        imgout->mdt->NBkw     = NB_KEYWNODE_MAX;
        imgout->im            = (IMAGE *) calloc(1, sizeof(IMAGE));
        imgid_mkimage(imgout);

        {
            int status = 0;
            fits_read_img(fptr, data_type_code(bitpix), fpixel, nelements, &nulval,
                          imgout->im->array.D, &anynul, &status);
            FITSIO_CHECK_ERROR(status, errmode, "fits_read_img bitpix=%d", bitpix);
        }
    }

    /* bitpix = 16   TSHORT */
    if (bitpix == 16)
    {
        imgout->mdt->naxis = naxis;
        for (long i = 0; i < naxis; i++)
        {
            imgout->mdt->size[i] = naxes[i];
        }
        imgout->mdt->datatype = _DATATYPE_UINT16;
        imgout->mdt->shared   = dcshareddft;
        imgout->mdt->NBkw     = NB_KEYWNODE_MAX;
        imgout->im            = (IMAGE *) calloc(1, sizeof(IMAGE));
        imgid_mkimage(imgout);

        {
            int status = 0;
            fits_read_img(fptr, 20, fpixel, nelements, &nulval, imgout->im->array.UI16, &anynul,
                          &status);
            FITSIO_CHECK_ERROR(status, errmode, "fits_read_img bitpix=%d", bitpix);
        }
    }

    /* bitpix = 32   TLONG */
    if (bitpix == 32)
    {
        imgout->mdt->naxis = naxis;
        for (long i = 0; i < naxis; i++)
        {
            imgout->mdt->size[i] = naxes[i];
        }
        imgout->mdt->datatype = _DATATYPE_INT32;
        imgout->mdt->shared   = dcshareddft;
        imgout->mdt->NBkw     = NB_KEYWNODE_MAX;
        imgout->im            = (IMAGE *) calloc(1, sizeof(IMAGE));
        imgid_mkimage(imgout);

        larray = (long *) malloc(sizeof(long) * nelements);
        if (larray == NULL)
        {
            PRINT_ERROR("malloc error");
            return RETURN_FAILURE;
        }
        {
            int status = 0;
            fits_read_img(fptr, data_type_code(bitpix), fpixel, nelements, &nulval, larray, &anynul,
                          &status);
            FITSIO_CHECK_ERROR(status, errmode, "fits_read_img bitpix=%d", bitpix);
        }

        bzero = 0.0;
        for (uint_fast64_t ii = 0; ii < (uint_fast64_t) nelements; ii++)
        {
            imgout->im->array.SI32[ii] = larray[ii] * bscale + bzero;
        }
        free(larray);
        larray = NULL;
    }

    /* bitpix = 64   TLONG  */
    if (bitpix == 64)
    {
        imgout->mdt->naxis = naxis;
        for (long i = 0; i < naxis; i++)
        {
            imgout->mdt->size[i] = naxes[i];
        }
        imgout->mdt->datatype = _DATATYPE_INT64;
        imgout->mdt->shared   = dcshareddft;
        imgout->mdt->NBkw     = NB_KEYWNODE_MAX;
        imgout->im            = (IMAGE *) calloc(1, sizeof(IMAGE));
        imgid_mkimage(imgout);

        larray = (long *) malloc(sizeof(long) * nelements);
        if (larray == NULL)
        {
            PRINT_ERROR("malloc error");
            abort();
        }

        {
            int status = 0;
            fits_read_img(fptr, data_type_code(bitpix), fpixel, nelements, &nulval, larray, &anynul,
                          &status);
            FITSIO_CHECK_ERROR(status, errmode, "fits_read_img bitpix=%d", bitpix);
        }

        bzero = 0.0;
        for (uint_fast64_t ii = 0; ii < (uint_fast64_t) nelements; ii++)
        {
            imgout->im->array.SI64[ii] = larray[ii] * bscale + bzero;
        }
        free(larray);
        larray = NULL;
    }

    /* bitpix = 8   TBYTE */
    if (bitpix == 8)
    {
        imgout->mdt->naxis = naxis;
        for (long i = 0; i < naxis; i++)
        {
            imgout->mdt->size[i] = naxes[i];
        }
        imgout->mdt->datatype = _DATATYPE_FLOAT;
        imgout->mdt->shared   = dcshareddft;
        imgout->mdt->NBkw     = NB_KEYWNODE_MAX;
        imgout->im            = (IMAGE *) calloc(1, sizeof(IMAGE));
        imgid_mkimage(imgout);

        barray = (unsigned char *) malloc(sizeof(unsigned char) * naxes[1] * naxes[0]);
        if (barray == NULL)
        {
            PRINT_ERROR("malloc error");
            return RETURN_FAILURE;
        }

        {
            int status = 0;
            fits_read_img(fptr, data_type_code(bitpix), fpixel, nelements, &nulval, barray, &anynul,
                          &status);
            FITSIO_CHECK_ERROR(status, errmode, "fits_read_img bitpix=%d", bitpix);
        }

        for (uint_fast64_t ii = 0; ii < (uint_fast64_t) nelements; ii++)
        {
            imgout->im->array.F[ii] = (1.0 * barray[ii] * bscale + bzero);
        }
        free(barray);
        barray = NULL;
    }

    // keywords to ignore
    char *keywordignore[] = { "BITPIX", "NAXIS",  "SIMPLE", "EXTEND", "COMMENT", "DATE", "NAXIS1",
                              "NAXIS2", "NAXIS3", "NAXIS4", "BSCALE", "BZERO",   0 };
    //printf("%d FITS keywords detected\n", nbFITSkeys);
    for (int kwnum = 0; kwnum < nbFITSkeys; kwnum++)
    {
        char keyname[9];
        char kwvaluestr[21];
        char kwcomment[81];
        {
            int status = 0;
            fits_read_keyn(fptr, kwnum + 1, keyname, kwvaluestr, kwcomment, &status);
        }

        //printf("FITS KEYW %3d  %8s %20s / %s\n", kwnum, keyname, kwvaluestr, kwcomment);

        int kwignore = 0;
        int ki       = 0;
        while (keywordignore[ki])
        {
            if (strcmp(keywordignore[ki], keyname) == 0)
            {
                //printf("%3d IGNORING %s\n", kwnum, keyname);
                kwignore = 1;
                break;
            }
            ki++;
        }

        if ((kwignore == 0) && (strlen(kwvaluestr) > 0))
        {
            int kwtypeOK = 0;

            // is this a long ?
            char *tailstr;
            long  kwlongval = strtol(kwvaluestr, &tailstr, 10);
            if (strlen(tailstr) == 0)
            {
                kwtypeOK = 1;
                image_keyword_addL(*imgout, keyname, kwlongval, kwcomment);
            }

            if (kwtypeOK == 0)
            {
                // is this a float ?
                double kwdoubleval = strtold(kwvaluestr, &tailstr);
                if (strlen(tailstr) == 0)
                {
                    kwtypeOK = 1;
                    image_keyword_addD(*imgout, keyname, kwdoubleval, kwcomment);
                }

                if (kwtypeOK == 0)
                {
                    // default to string
                    // remove leading and trailing '
                    kwvaluestr[strlen(kwvaluestr) - 1] = '\0';
                    char *kwvaluestr1;
                    kwvaluestr1 = kwvaluestr + 1;
                    image_keyword_addS(*imgout, keyname, kwvaluestr1, kwcomment);
                }
            }
        }
    }

    {
        int status = 0;
        fits_close_file(fptr, &status);
        FITSIO_CHECK_ERROR(status, errmode, "fits_close_file error in image %s", file_name);
    }

    DEBUG_TRACEPOINT("FOUT IDout %ld", imgout->ID);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

/**
 * @brief Load a FITS file into the image array.
 *
 * Legacy interface returning an imageID integer.
 */
errno_t load_fits(const char *__restrict file_name,
                  const char *__restrict ID_name,
                  int      errmode,
                  imageID *IDout)
{
    IMGID imgout = imgid_make_from_name(ID_name);

    errno_t retval = load_fits_IMGID(file_name, &imgout, errmode);

    if (IDout != NULL)
    {
        *IDout = imgout.ID;
    }

    return retval;
}

static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START IMGID imgout = imgid_make_from_name(outimname);
    FUNC_CHECK_RETURN(load_fits_IMGID(infilename, &imgout, FITSIOerrmode));

    INSERT_STD_PROCINFO_COMPUTEFUNC_END DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

// Register function in CLI
errno_t CLIADDCMD_COREMOD_iofits__loadfits()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);

    int cmdi               = RegisterCLIcmd(CLIcmddata, CLIfunction);
    CLIcmddata.cmdsettings = &data.cmd[cmdi].cmdsettings;

    return RETURN_SUCCESS;
}
#endif

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif
