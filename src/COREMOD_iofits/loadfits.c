/**
 * @file    loadfits.c
 * @brief   load FITS format files
 */


#include <stdlib.h>

#include "CommandLineInterface/CLIcore.h"


#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_iofits_common.h"

#include "data_type_code.h"
#include "check_fitsio_status.h"

#include "COREMOD_memory/image_keyword_addL.h"
#include "COREMOD_memory/image_keyword_addD.h"
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
        CLIARG_STR, ".infname", "input file", "imfname",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &infilename
    },
    {
        CLIARG_STR_NOT_IMG, ".outimname", "output image name", "outimname",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimname
    },
    {
        CLIARG_LONG, ".errcode", "FITSIO errors mode \n(0:ignore) (1:warning) (2:exit) (3:try)", "1",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &FITSIOerrmode
    }
};



// CLI function initialization data
static CLICMDDATA CLIcmddata =
{
    "loadfits",
    "load FITS format file",
    __FILE__, sizeof(farg) / sizeof(CLICMDARGDEF), farg,
    CLICMDFLAG_FPS,
    NULL
};





/// if errcode = 0, do not show error messages
/// errcode = 1: print error, continue
/// errcode = 2: exit program at error
/// errcode = 3: do not show error message, try = 1, no wait
imageID load_fits(
    const char *restrict file_name,
    const char *restrict ID_name,
    int         errcode
)
{
    fitsfile *fptr = NULL;       /* pointer to the FITS file; defined in fitsio.h */
    int FITSIOstatus;
    int       nulval, anynul;
    long      bitpixl = 0;

    uint32_t  naxes[3];
    imageID   ID;

    double    bscale;
    double    bzero;
    unsigned char *barray = NULL;
    long     *larray = NULL;
    //    unsigned short *sarray = NULL;
    //    long      NDR = 1; /* non-destructive reads */


    int fileOK;

    nulval = 0;
    anynul = 0;
    bscale = 1;
    bzero = 0;

    naxes[0] = 0;
    naxes[1] = 0;
    naxes[2] = 0;

    fileOK = 0;
    int NBtry = 3;

    for(int try = 0;
                try < NBtry;
                    try++)
                {
                    if(fileOK == 0)
                    {
                        if(fits_open_file(&fptr, file_name, READONLY,
                                          &FITSIOstatus))
                        {
                            if(FITSIOstatus != 0)
                            {
                                FITSIO_CHECK_ERROR(FITSIOstatus, errcode, "can't load %s", file_name);
                                if(errcode == 2)
                                {
                                    abort();
                                }
                                if(try != NBtry - 1) // don't wait on last try
                                        usleep(10000);
                            }
                            ID = -1;
                        }
                        else
                        {
                            fileOK = 1;
                        }
                    }
                }

    if(fileOK == 0)
    {
        PRINT_WARNING("Image \"%s\" could not be loaded from file \"%s\"",
                      ID_name,
                      file_name);
    }


    if(fileOK == 1)
    {
        char  keyword[STRINGMAXLEN_FITSKEYWORDNAME];
        long  fpixel = 1;
        char  comment[STRINGMAXLEN_FITSKEYWCOMMENT];
        long  nelements;
        long  naxis = 0;


        // Keywords
        int nbFITSkeys = 0;
        fits_get_hdrspace(fptr, &nbFITSkeys, NULL, &FITSIOstatus);
        FITSIO_CHECK_ERROR(FITSIOstatus, errcode, "fits_get_hdrspace error on %s",
                           file_name);


        fits_read_key(fptr, TLONG, "NAXIS", &naxis, comment,
                      &FITSIOstatus);
        FITSIO_CHECK_ERROR(FITSIOstatus, errcode, "File %s has no NAXIS", file_name);
        printf("naxis = %ld\n", naxis);


        for(long i = 0; i < naxis; i++)
        {
            WRITE_FITSKEYWNAME(keyword, "NAXIS%ld", i + 1);

            fits_read_key(fptr, TLONG, keyword, &naxes[i], comment,
                          &FITSIOstatus);
            FITSIO_CHECK_ERROR(FITSIOstatus, errcode, "File %s has no NAXIS%ld", file_name,
                               i);
        }

        fits_read_key(fptr, TLONG, "BITPIX", &bitpixl, comment,
                      &FITSIOstatus);
        FITSIO_CHECK_ERROR(FITSIOstatus, errcode, "File %s has no BITPIX", file_name);


        int bitpix = (int) bitpixl;
        fits_read_key(fptr, TDOUBLE, "BSCALE", &bscale, comment,
                      &FITSIOstatus);
        if(FITSIOstatus != 0)
        {
            FITSIOstatus = 0;
            bscale = 1.0;
        }

        fits_read_key(fptr, TDOUBLE, "BZERO", &bzero, comment,
                      &FITSIOstatus);
        if(FITSIOstatus != 0)
        {
            FITSIOstatus = 0;
            bzero = 0.0;
        }




        fits_set_bscale(fptr, bscale, bzero, &COREMOD_iofits_data.FITSIO_status);
        check_FITSIO_status(__FILE__, __func__, __LINE__, 1);

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
            ID = create_image_ID(ID_name, naxis, naxes, _DATATYPE_FLOAT, data.SHARED_DFT,
                                 data.NBKEYWORD_DFT, 0);

            fits_read_img(fptr, data_type_code(bitpix), fpixel, nelements, &nulval,
                          data.image[ID].array.F, &anynul, &FITSIOstatus);
            FITSIO_CHECK_ERROR(FITSIOstatus, errcode, "fits_read_img bitpix=%d", bitpix);
        }

        /* bitpix = -64  TDOUBLE */
        if(bitpix == -64)
        {
            ID = create_image_ID(ID_name, naxis, naxes, _DATATYPE_DOUBLE, data.SHARED_DFT,
                                 data.NBKEYWORD_DFT, 0);

            fits_read_img(fptr, data_type_code(bitpix), fpixel, nelements, &nulval,
                          data.image[ID].array.D, &anynul, &FITSIOstatus);
            FITSIO_CHECK_ERROR(FITSIOstatus, errcode, "fits_read_img bitpix=%d", bitpix);
        }

        /* bitpix = 16   TSHORT */
        if(bitpix == 16)
        {
            // ID = create_image_ID(ID_name, naxis, naxes, Dtype, data.SHARED_DFT, data.NBKEWORD_DFT);
            ID = create_image_ID(ID_name, naxis, naxes, _DATATYPE_UINT16, data.SHARED_DFT,
                                 data.NBKEYWORD_DFT, 0);

            //           fits_read_img(fptr, 20, fpixel, nelements, &nulval, sarray, &anynul, &FITSIO_status);
            fits_read_img(fptr, 20, fpixel, nelements, &nulval, data.image[ID].array.UI16,
                          &anynul, &FITSIOstatus);
            FITSIO_CHECK_ERROR(FITSIOstatus, errcode, "fits_read_img bitpix=%d", bitpix);
        }


        /* bitpix = 32   TLONG */
        if(bitpix == 32)
        {
            ID = create_image_ID(ID_name, naxis, naxes, _DATATYPE_INT32, data.SHARED_DFT,
                                 data.NBKEYWORD_DFT, 0);
            larray = (long *) malloc(sizeof(long) * nelements);
            if(larray == NULL)
            {
                PRINT_ERROR("malloc error");
                exit(0);
            }

            fits_read_img(fptr, data_type_code(bitpix), fpixel, nelements, &nulval, larray,
                          &anynul, &FITSIOstatus);
            FITSIO_CHECK_ERROR(FITSIOstatus, errcode, "fits_read_img bitpix=%d", bitpix);

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
            ID = create_image_ID(ID_name, naxis, naxes, _DATATYPE_INT64, data.SHARED_DFT,
                                 data.NBKEYWORD_DFT, 0);
            larray = (long *) malloc(sizeof(long) * nelements);
            if(larray == NULL)
            {
                PRINT_ERROR("malloc error");
                abort();
            }

            fits_read_img(fptr, data_type_code(bitpix), fpixel, nelements, &nulval, larray,
                          &anynul, &FITSIOstatus);
            FITSIO_CHECK_ERROR(FITSIOstatus, errcode, "fits_read_img bitpix=%d", bitpix);

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
            ID = create_image_ID(ID_name, naxis, naxes, _DATATYPE_FLOAT, data.SHARED_DFT,
                                 data.NBKEYWORD_DFT, 0);
            barray = (unsigned char *) malloc(sizeof(unsigned char) * naxes[1] * naxes[0]);
            if(barray == NULL)
            {
                PRINT_ERROR("malloc error");
                exit(0);
            }

            fits_read_img(fptr, data_type_code(bitpix), fpixel, nelements, &nulval, barray,
                          &anynul, &FITSIOstatus);
            FITSIO_CHECK_ERROR(FITSIOstatus, errcode, "fits_read_img bitpix=%d", bitpix);

            for(uint_fast64_t ii = 0; ii < (uint_fast64_t) nelements; ii++)
            {
                data.image[ID].array.F[ii] = (1.0 * barray[ii] * bscale + bzero);
            }
            free(barray);
            barray = NULL;
        }


        IMGID img = makesetIMGID(ID_name, ID);

        // keywords to ignore
        char *keywordignore[] = {"BITPIX", "NAXIS", "SIMPLE", "EXTEND", "COMMENT", "DATE", "NAXIS1", "NAXIS2", "NAXIS3", "NAXIS4", "BSCALE", "BZERO", 0};
        printf("%d FITS keywords detected\n", nbFITSkeys);
        for(int kwnum = 0; kwnum < nbFITSkeys; kwnum ++)
        {
            char keyname[9];
            char kwvaluestr[21];
            char kwcomment[81];
            fits_read_keyn(fptr, kwnum+1, keyname, kwvaluestr, kwcomment,
                           &COREMOD_iofits_data.FITSIO_status);

            //printf("FITS KEYW %3d  %8s %20s / %s\n", kwnum, keyname, kwvaluestr, kwcomment);


            int kwignore = 0;
            int ki = 0;
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
                long kwlongval = strtol(kwvaluestr, &tailstr, 10);
                if(strlen(tailstr) == 0)
                {
                    kwtypeOK = 1;
                    printf("%3d FITS KEYW [L] %-8s= %20ld / %s\n", kwnum, keyname, kwlongval, kwcomment);
                    image_keyword_addL(img, keyname, kwlongval, kwcomment);
                }

                if(kwtypeOK == 0)
                {
                    // is this a float ?
                    double kwdoubleval = strtold(kwvaluestr, &tailstr);
                    if(strlen(tailstr) == 0)
                    {
                        kwtypeOK = 1;
                        printf("%3d FITS KEYW [D] %-8s= %20g / %s\n", kwnum, keyname, kwdoubleval, kwcomment);
                        image_keyword_addD(img, keyname, kwdoubleval, kwcomment);
                    }

                    if(kwtypeOK == 0)
                    {
                        // default to string
                        printf("%3d FITS KEYW [S] %-8s= %-20s / %s\n", kwnum, keyname, kwvaluestr, kwcomment);
                        // remove leading and trailing '
                        kwvaluestr[strlen(kwvaluestr)-1] = '\0';
                        char *kwvaluestr1;
                        kwvaluestr1 = kwvaluestr+1;
                        image_keyword_addS(img, keyname, kwvaluestr1, kwcomment);
                    }

                }
            }
        }




        fits_close_file(fptr, &FITSIOstatus);
        FITSIO_CHECK_ERROR(FITSIOstatus, errcode, "fits_close_file error in image %s",
                           file_name);
    }

    return ID;
}




static errno_t compute_function()
{
    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    load_fits(
        infilename,
        outimname,
        *FITSIOerrmode
    );

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    return RETURN_SUCCESS;
}




INSERT_STD_FPSCLIfunctions


// Register function in CLI
errno_t CLIADDCMD_COREMOD_iofits__loadfits()
{
    //INSERT_STD_FPSCLIREGISTERFUNC

    int cmdi = RegisterCLIcmd(CLIcmddata, CLIfunction);
    CLIcmddata.cmdsettings = &data.cmd[cmdi].cmdsettings;

    return RETURN_SUCCESS;
}


