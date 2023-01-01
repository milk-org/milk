/**
 * @file    loadfits.c
 * @brief   load FITS format files
 */

#include <stdlib.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_iofits_common.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "check_fitsio_status.h"
#include "data_type_code.h"

#include "COREMOD_memory/image_keyword_addD.h"
#include "COREMOD_memory/image_keyword_addL.h"
#include "COREMOD_memory/image_keyword_addS.h"

extern COREMOD_IOFITS_DATA COREMOD_iofits_data;

// CLI function arguments and parameters
static char *infilename;
static char *outimname;
static long *FITSIOerrmode;

// CLI function arguments and parameters
static CLICMDARGDEF farg[] =
{
    {
        CLIARG_STR,
        ".infname",
        "input file",
        "imfname",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &infilename,
        NULL
    },
    {
        CLIARG_STR_NOT_IMG,
        ".outimname",
        "output image name",
        "outimname",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimname,
        NULL
    },
    {
        CLIARG_INT64,
        ".errmode",
        "FITSIO errors mode \n(0:ignore) (1:warning) (2:error) (3:exit)",
        "1",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &FITSIOerrmode,
        NULL
    }
};

// CLI function initialization data
static CLICMDDATA CLIcmddata =
{
    "loadfits", "load FITS format file", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{

    printf(
        "Load FITS file from filesystem\n"
        "Uses fitsio library, supports extended fitsio file syntax\n"
        "File name should be in double quotes unless free of special chars\n"
        "Examples:\n"
        "   loadfits \"im1.fits\" im\n");

    return RETURN_SUCCESS;
}

/// errmode values :
/// LOADFITS_ERRMODE_IGNORE  (0) print warning, do not show error messages, continue
/// LOADFITS_ERRMODE_WARNING (1) print error, continue
/// LOADFITS_ERRMODE_ERROR   (2) return error
/// LOADFITS_ERRMODE_EXIT    (3) exit program at error

errno_t load_fits(const char *restrict file_name,
                  const char *restrict ID_name,
                  int      errmode,
                  imageID *IDout)
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
    //    unsigned short *sarray = NULL;
    //    long      NDR = 1; /* non-destructive reads */

    imageID ID;

    nulval = 0;
    anynul = 0;
    bscale = 1;
    bzero  = 0;

    naxes[0] = 0;
    naxes[1] = 0;
    naxes[2] = 0;

    DEBUG_TRACEPOINT("FARG \"%s\" %s %d", file_name, ID_name, errmode);

    {
        // Open fitsio file pointer
        // tyr 3 consecutive times and then give up if not successful
        int fileOK = 0;
        int NBtry  = 3;
        for(int tr = 0; tr < NBtry; tr++)
        {
            if(fileOK == 0)
            {
                int status = 0;
                fits_open_file(&fptr, file_name, READONLY, &status);

                if(status != 0)
                {
                    if(errmode > 0)
                    {

                        printf("attempt # %d failed\n", tr);
                    }

                    //void fits_get_errstatus(int status, char *err_text)
                    if(status != 0)
                    {
                        if(errmode > 1)
                        {
                            if(tr == NBtry - 1)
                            {
                                FITSIO_CHECK_ERROR(status,
                                                   errmode,
                                                   "can't load %s "
                                                   "(tried %d times)",
                                                   file_name,
                                                   NBtry);
                            }
                        }
                        if(tr != NBtry - 1)  // don't wait on last try
                        {
                            usleep(10000);
                        }
                    }

                    ID = -1;
                }
                else
                {
                    fileOK = 1;
                }
            }
        }
        printf("fileOK = %d\n", fileOK);

        if(fileOK == 0)
        {
            // if image not loaded, set output identifier to -1
            if(IDout != NULL)
            {
                *IDout = -1;
            }

            if(errmode == 0)
            {
                DEBUG_TRACE_FEXIT();
                return RETURN_SUCCESS;
            }

            if(errmode == 1)
            {
                PRINT_WARNING(
                    "Image \"%s\" could not be loaded from file \"%s\"",
                    ID_name,
                    file_name);
                DEBUG_TRACE_FEXIT();
                return RETURN_SUCCESS;
            }

            if(errmode == 2)
            {
                FUNC_RETURN_FAILURE(
                    "Image \"%s\" could not be loaded from file \"%s\"",
                    ID_name,
                    file_name);
            }

            if(errmode == 3)
            {
                abort();
            }
            DEBUG_TRACE_FEXIT();
            return -1;
        }
    }

    DEBUG_TRACEPOINT("File %s open", file_name);

    char keyword[STRINGMAXLEN_FITSKEYWORDNAME];
    long fpixel = 1;
    char comment[STRINGMAXLEN_FITSKEYWCOMMENT];
    long nelements;
    long naxis = 0;

    // Keywords
    int nbFITSkeys = 0;

    {
        int status = 0;
        fits_get_hdrspace(fptr, &nbFITSkeys, NULL, &status);
        FITSIO_CHECK_ERROR(status,
                           errmode,
                           "fits_get_hdrspace error on %s",
                           file_name);
    }

    {
        int status = 0;
        fits_read_key(fptr, TLONG, "NAXIS", &naxis, comment, &status);
        FITSIO_CHECK_ERROR(status, errmode, "File %s has no NAXIS", file_name);
    }
    printf("naxis = %ld\n", naxis);
    DEBUG_TRACEPOINT("naxis = %ld", naxis);

    for(long i = 0; i < naxis; i++)
    {
        WRITE_FITSKEYWNAME(keyword, "NAXIS%ld", i + 1);

        {
            int status = 0;
            fits_read_key(fptr, TLONG, keyword, &naxes[i], comment, &status);
            FITSIO_CHECK_ERROR(status,
                               errmode,
                               "File %s has no NAXIS%ld",
                               file_name,
                               i);
        }
    }

    {
        int status = 0;
        fits_read_key(fptr, TLONG, "BITPIX", &bitpixl, comment, &status);
        FITSIO_CHECK_ERROR(status, errmode, "File %s has no BITPIX", file_name);
    }

    int bitpix = (int) bitpixl;
    {
        int status = 0;
        fits_read_key(fptr, TDOUBLE, "BSCALE", &bscale, comment, &status);
        if(status != 0)
        {
            bscale = 1.0;
        }
    }

    {
        int status = 0;
        fits_read_key(fptr, TDOUBLE, "BZERO", &bzero, comment, &status);
        if(status != 0)
        {
            bzero = 0.0;
        }
    }

    {
        int status = 0;
        fits_set_bscale(fptr, bscale, bzero, &status);
        FITSIO_CHECK_ERROR(status, errmode, "bscake set errror");
    }

    if(1)
    {
        printf("[%ld", (long) naxes[0]);
        for(long i = 1; i < naxis; i++)
        {
            printf(",%ld", (long) naxes[i]);
        }
        printf("] %d %f %f\n", bitpix, bscale, bzero);
        fflush(stdout);
    }

    nelements = 1;
    for(long i = 0; i < naxis; i++)
    {
        nelements *= naxes[i];
    }

    /* bitpix = -32  TFLOAT */
    if(bitpix == -32)
    {
        create_image_ID(ID_name,
                        naxis,
                        naxes,
                        _DATATYPE_FLOAT,
                        data.SHARED_DFT,
                        data.NBKEYWORD_DFT,
                        0,
                        &ID);

        {
            int status = 0;
            fits_read_img(fptr,
                          data_type_code(bitpix),
                          fpixel,
                          nelements,
                          &nulval,
                          data.image[ID].array.F,
                          &anynul,
                          &status);
            FITSIO_CHECK_ERROR(status,
                               errmode,
                               "fits_read_img bitpix=%d",
                               bitpix);
        }
    }

    /* bitpix = -64  TDOUBLE */
    if(bitpix == -64)
    {
        create_image_ID(ID_name,
                        naxis,
                        naxes,
                        _DATATYPE_DOUBLE,
                        data.SHARED_DFT,
                        data.NBKEYWORD_DFT,
                        0,
                        &ID);

        {
            int status = 0;
            fits_read_img(fptr,
                          data_type_code(bitpix),
                          fpixel,
                          nelements,
                          &nulval,
                          data.image[ID].array.D,
                          &anynul,
                          &status);
            FITSIO_CHECK_ERROR(status,
                               errmode,
                               "fits_read_img bitpix=%d",
                               bitpix);
        }
    }

    /* bitpix = 16   TSHORT */
    if(bitpix == 16)
    {
        // ID = create_image_ID(ID_name, naxis, naxes, Dtype, data.SHARED_DFT, data.NBKEWORD_DFT);
        create_image_ID(ID_name,
                        naxis,
                        naxes,
                        _DATATYPE_UINT16,
                        data.SHARED_DFT,
                        data.NBKEYWORD_DFT,
                        0,
                        &ID);

        //           fits_read_img(fptr, 20, fpixel, nelements, &nulval, sarray, &anynul, &FITSIO_status);
        {
            int status = 0;
            fits_read_img(fptr,
                          20,
                          fpixel,
                          nelements,
                          &nulval,
                          data.image[ID].array.UI16,
                          &anynul,
                          &status);
            FITSIO_CHECK_ERROR(status,
                               errmode,
                               "fits_read_img bitpix=%d",
                               bitpix);
        }
    }

    /* bitpix = 32   TLONG */
    if(bitpix == 32)
    {
        create_image_ID(ID_name,
                        naxis,
                        naxes,
                        _DATATYPE_INT32,
                        data.SHARED_DFT,
                        data.NBKEYWORD_DFT,
                        0,
                        &ID);
        larray = (long *) malloc(sizeof(long) * nelements);
        if(larray == NULL)
        {
            PRINT_ERROR("malloc error");
            exit(0);
        }
        {
            int status = 0;
            fits_read_img(fptr,
                          data_type_code(bitpix),
                          fpixel,
                          nelements,
                          &nulval,
                          larray,
                          &anynul,
                          &status);
            FITSIO_CHECK_ERROR(status,
                               errmode,
                               "fits_read_img bitpix=%d",
                               bitpix);
        }

        bzero = 0.0;
        for(uint_fast64_t ii = 0; ii < (uint_fast64_t) nelements; ii++)
        {
            data.image[ID].array.SI32[ii] = larray[ii] * bscale + bzero;
        }
        free(larray);
        larray = NULL;
    }

    /* bitpix = 64   TLONG  */
    if(bitpix == 64)
    {
        create_image_ID(ID_name,
                        naxis,
                        naxes,
                        _DATATYPE_INT64,
                        data.SHARED_DFT,
                        data.NBKEYWORD_DFT,
                        0,
                        &ID);
        larray = (long *) malloc(sizeof(long) * nelements);
        if(larray == NULL)
        {
            PRINT_ERROR("malloc error");
            abort();
        }

        {
            int status = 0;
            fits_read_img(fptr,
                          data_type_code(bitpix),
                          fpixel,
                          nelements,
                          &nulval,
                          larray,
                          &anynul,
                          &status);
            FITSIO_CHECK_ERROR(status,
                               errmode,
                               "fits_read_img bitpix=%d",
                               bitpix);
        }

        bzero = 0.0;
        for(uint_fast64_t ii = 0; ii < (uint_fast64_t) nelements; ii++)
        {
            data.image[ID].array.SI64[ii] = larray[ii] * bscale + bzero;
        }
        free(larray);
        larray = NULL;
    }

    /* bitpix = 8   TBYTE */
    if(bitpix == 8)
    {
        create_image_ID(ID_name,
                        naxis,
                        naxes,
                        _DATATYPE_FLOAT,
                        data.SHARED_DFT,
                        data.NBKEYWORD_DFT,
                        0,
                        &ID);
        barray = (unsigned char *) malloc(sizeof(unsigned char) * naxes[1] *
                                          naxes[0]);
        if(barray == NULL)
        {
            PRINT_ERROR("malloc error");
            exit(0);
        }

        {
            int status = 0;
            fits_read_img(fptr,
                          data_type_code(bitpix),
                          fpixel,
                          nelements,
                          &nulval,
                          barray,
                          &anynul,
                          &status);
            FITSIO_CHECK_ERROR(status,
                               errmode,
                               "fits_read_img bitpix=%d",
                               bitpix);
        }

        for(uint_fast64_t ii = 0; ii < (uint_fast64_t) nelements; ii++)
        {
            data.image[ID].array.F[ii] = (1.0 * barray[ii] * bscale + bzero);
        }
        free(barray);
        barray = NULL;
    }

    IMGID img = makesetIMGID(ID_name, ID);

    // keywords to ignore
    char *keywordignore[] = {"BITPIX",
                             "NAXIS",
                             "SIMPLE",
                             "EXTEND",
                             "COMMENT",
                             "DATE",
                             "NAXIS1",
                             "NAXIS2",
                             "NAXIS3",
                             "NAXIS4",
                             "BSCALE",
                             "BZERO",
                             0
                            };
    printf("%d FITS keywords detected\n", nbFITSkeys);
    for(int kwnum = 0; kwnum < nbFITSkeys; kwnum++)
    {
        char keyname[9];
        char kwvaluestr[21];
        char kwcomment[81];
        {
            int status = 0;
            fits_read_keyn(fptr,
                           kwnum + 1,
                           keyname,
                           kwvaluestr,
                           kwcomment,
                           &status);
        }

        //printf("FITS KEYW %3d  %8s %20s / %s\n", kwnum, keyname, kwvaluestr, kwcomment);

        int kwignore = 0;
        int ki       = 0;
        while(keywordignore[ki])
        {
            if(strcmp(keywordignore[ki], keyname) == 0)
            {
                //printf("%3d IGNORING %s\n", kwnum, keyname);
                kwignore = 1;
                break;
            }
            ki++;
        }

        if((kwignore == 0) && (strlen(kwvaluestr) > 0))
        {
            int kwtypeOK = 0;

            // is this a long ?
            char *tailstr;
            long  kwlongval = strtol(kwvaluestr, &tailstr, 10);
            if(strlen(tailstr) == 0)
            {
                kwtypeOK = 1;
                printf("%3d FITS KEYW [L] %-8s= %20ld / %s\n",
                       kwnum,
                       keyname,
                       kwlongval,
                       kwcomment);
                image_keyword_addL(img, keyname, kwlongval, kwcomment);
            }

            if(kwtypeOK == 0)
            {
                // is this a float ?
                double kwdoubleval = strtold(kwvaluestr, &tailstr);
                if(strlen(tailstr) == 0)
                {
                    kwtypeOK = 1;
                    printf("%3d FITS KEYW [D] %-8s= %20g / %s\n",
                           kwnum,
                           keyname,
                           kwdoubleval,
                           kwcomment);
                    image_keyword_addD(img, keyname, kwdoubleval, kwcomment);
                }

                if(kwtypeOK == 0)
                {
                    // default to string
                    printf("%3d FITS KEYW [S] %-8s= %-20s / %s\n",
                           kwnum,
                           keyname,
                           kwvaluestr,
                           kwcomment);
                    // remove leading and trailing '
                    kwvaluestr[strlen(kwvaluestr) - 1] = '\0';
                    char *kwvaluestr1;
                    kwvaluestr1 = kwvaluestr + 1;
                    image_keyword_addS(img, keyname, kwvaluestr1, kwcomment);
                }
            }
        }
    }

    {
        int status = 0;
        fits_close_file(fptr, &status);
        FITSIO_CHECK_ERROR(status,
                           errmode,
                           "fits_close_file error in image %s",
                           file_name);
    }

    list_image_ID();

    if(IDout != NULL)
    {
        *IDout = ID;
    }

    DEBUG_TRACEPOINT("FOUT IDout %ld", ID);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    FUNC_CHECK_RETURN(load_fits(infilename, outimname, *FITSIOerrmode, NULL));

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t
CLIADDCMD_COREMOD_iofits__loadfits()
{
    //INSERT_STD_FPSCLIREGISTERFUNC

    int cmdi               = RegisterCLIcmd(CLIcmddata, CLIfunction);
    CLIcmddata.cmdsettings = &data.cmd[cmdi].cmdsettings;

    return RETURN_SUCCESS;
}
