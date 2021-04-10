/**
 * @file    loadfits.c
 * @brief   load FITS format files
 */


#include "CommandLineInterface/CLIcore.h"


#include "COREMOD_memory/COREMOD_memory.h"

#include "COREMOD_iofits_common.h"

#include "data_type_code.h"
#include "check_fitsio_status.h"


extern COREMOD_IOFITS_DATA COREMOD_iofits_data;



// CLI function arguments and parameters
static char *infilename;
static char *outimname;
static long *errmode;


// CLI function arguments and parameters
static CLICMDARGDEF farg[] =
{
    {
        CLIARG_STR, ".infname", "input file", "imfname",
        CLICMDARG_FLAG_DEFAULT, FPTYPE_AUTO, FPFLAG_DEFAULT_INPUT,
        (void **) &infilename
    },
    {
        CLIARG_STR_NOT_IMG, ".outimname", "output image name", "outimname",
        CLICMDARG_FLAG_DEFAULT, FPTYPE_AUTO, FPFLAG_DEFAULT_INPUT,
        (void **) &outimname
    },
    {
        CLIARG_LONG, ".errcode", "input image", "0",
        CLICMDARG_FLAG_NOCLI, FPTYPE_AUTO, FPFLAG_DEFAULT_INPUT,
        (void **) &errmode
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
    int PrintErrorMsg = 1;
    int ExitOnErr = 0;

    nulval = 0;
    anynul = 0;
    bscale = 1;
    bzero = 0;

    naxes[0] = 0;
    naxes[1] = 0;
    naxes[2] = 0;




    fileOK = 0;
    int NBtry = 3;

    if(errcode == 0)
    {
        PrintErrorMsg = 0;
        ExitOnErr = 0;
    }

    if(errcode == 1)
    {
        PrintErrorMsg = 1;
        ExitOnErr = 0;
    }

    if(errcode == 2)
    {
        PrintErrorMsg = 1;
        ExitOnErr = 1;
    }

    if(errcode == 3)
    {
        NBtry = 1;
        PrintErrorMsg = 0;
        ExitOnErr = 0;
    }


    for(int try = 0;
                try < NBtry;
                    try++)
                {
                    if(fileOK == 0)
                    {
                        if(fits_open_file(&fptr, file_name, READONLY,
                                          &COREMOD_iofits_data.FITSIO_status))
                        {
                            if(check_FITSIO_status(__FILE__, __func__, __LINE__, PrintErrorMsg) != 0)
                            {
                                if(ExitOnErr == 1)
                                {
                                    exit(0);
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

        char *header;
        int nkeys;


        fits_hdr2str(fptr, 1, NULL, 0, &header, &nkeys,
                     &COREMOD_iofits_data.FITSIO_status);
        char *hptr; // pointer to header
        hptr = header;
        while(*hptr)
        {
            printf("    %.80s\n", hptr);
            hptr += 80;
        }

        fits_free_memory(header, &COREMOD_iofits_data.FITSIO_status);



        fits_read_key(fptr, TLONG, "NAXIS", &naxis, comment,
                      &COREMOD_iofits_data.FITSIO_status);
        if(errcode != 0)
        {
            if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
            {
                PRINT_ERROR("Error reading FITS key NAXIS");
                list_image_ID();
                if(errcode > 1)
                {
                    exit(0);
                }
            }
        }


        for(long i = 0; i < naxis; i++)
        {
            WRITE_FITSKEYWNAME(keyword, "NAXIS%ld", i + 1);

            fits_read_key(fptr, TLONG, keyword, &naxes[i], comment,
                          &COREMOD_iofits_data.FITSIO_status);
            if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
            {
                if(errcode != 0)
                {
                    PRINT_ERROR("Error reading FITS key NAXIS%ld", i);
                    list_image_ID();
                    if(errcode > 1)
                    {
                        abort();
                    }
                }
            }
        }

        fits_read_key(fptr, TLONG, "BITPIX", &bitpixl, comment,
                      &COREMOD_iofits_data.FITSIO_status);
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            if(errcode != 0)
            {
                PRINT_ERROR("Error reading FITS key BITPIX");
                list_image_ID();
                if(errcode > 1)
                {
                    exit(0);
                }
            }
        }



        int bitpix = (int) bitpixl;
        fits_read_key(fptr, TDOUBLE, "BSCALE", &bscale, comment,
                      &COREMOD_iofits_data.FITSIO_status);
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 0) == 1)
        {
            //fprintf(stderr,"Error reading keyword \"BSCALE\" in file \"%s\"\n",file_name);
            bscale = 1.0;
        }
        fits_read_key(fptr, TDOUBLE, "BZERO", &bzero, comment,
                      &COREMOD_iofits_data.FITSIO_status);
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 0) == 1)
        {
            //fprintf(stderr,"Error reading keyword \"BZERO\" in file \"%s\"\n",file_name);
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
                                 data.NBKEWORD_DFT);
            fits_read_img(fptr, data_type_code(bitpix), fpixel, nelements, &nulval,
                          data.image[ID].array.F, &anynul, &COREMOD_iofits_data.FITSIO_status);

            if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
            {
                if(errcode != 0)
                {
                    PRINT_ERROR("fits_read_img error");
                    list_image_ID();
                    if(errcode > 1)
                    {
                        exit(0);
                    }
                }
            }



            fits_close_file(fptr, &COREMOD_iofits_data.FITSIO_status);
            if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
            {
                if(errcode != 0)
                {
                    PRINT_ERROR("fits_close_file error");
                    list_image_ID();
                    if(errcode > 1)
                    {
                        exit(0);
                    }
                }
            }

            check_FITSIO_status(__FILE__, __func__, __LINE__, 1);
        }

        /* bitpix = -64  TDOUBLE */
        if(bitpix == -64)
        {
            ID = create_image_ID(ID_name, naxis, naxes, _DATATYPE_DOUBLE, data.SHARED_DFT,
                                 data.NBKEWORD_DFT);

            fits_read_img(fptr, data_type_code(bitpix), fpixel, nelements, &nulval,
                          data.image[ID].array.D, &anynul, &COREMOD_iofits_data.FITSIO_status);
            if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
            {
                if(errcode != 0)
                {
                    PRINT_ERROR("fits_read_img error");
                    list_image_ID();
                    if(errcode > 1)
                    {
                        exit(0);
                    }
                }
            }

            fits_close_file(fptr, &COREMOD_iofits_data.FITSIO_status);
            if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
            {
                if(errcode != 0)
                {
                    PRINT_ERROR("fits_close_file error");
                    list_image_ID();
                    if(errcode > 1)
                    {
                        exit(0);
                    }
                }
            }

            check_FITSIO_status(__FILE__, __func__, __LINE__, 1);
        }

        /* bitpix = 16   TSHORT */
        if(bitpix == 16)
        {
            // ID = create_image_ID(ID_name, naxis, naxes, Dtype, data.SHARED_DFT, data.NBKEWORD_DFT);
            ID = create_image_ID(ID_name, naxis, naxes, _DATATYPE_UINT16, data.SHARED_DFT,
                                 data.NBKEWORD_DFT);

            //           fits_read_img(fptr, 20, fpixel, nelements, &nulval, sarray, &anynul, &FITSIO_status);
            fits_read_img(fptr, 20, fpixel, nelements, &nulval, data.image[ID].array.UI16,
                          &anynul, &COREMOD_iofits_data.FITSIO_status);
            if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
            {
                if(errcode != 0)
                {
                    PRINT_ERROR("fits_read_img error");
                    list_image_ID();
                    if(errcode > 1)
                    {
                        exit(0);
                    }
                }
            }

            fits_close_file(fptr, &COREMOD_iofits_data.FITSIO_status);
            if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
            {
                if(errcode != 0)
                {
                    PRINT_ERROR("fits_close_file error");
                    list_image_ID();
                    if(errcode > 1)
                    {
                        exit(0);
                    }
                }
            }

            check_FITSIO_status(__FILE__, __func__, __LINE__, 1);
        }


        /* bitpix = 32   TLONG */
        if(bitpix == 32)
        {
            ID = create_image_ID(ID_name, naxis, naxes, _DATATYPE_INT32, data.SHARED_DFT,
                                 data.NBKEWORD_DFT);
            larray = (long *) malloc(sizeof(long) * nelements);
            if(larray == NULL)
            {
                PRINT_ERROR("malloc error");
                exit(0);
            }

            fits_read_img(fptr, data_type_code(bitpix), fpixel, nelements, &nulval, larray,
                          &anynul, &COREMOD_iofits_data.FITSIO_status);
            if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
            {
                if(errcode != 0)
                {
                    PRINT_ERROR("fits_read_img error");
                    list_image_ID();
                    if(errcode > 1)
                    {
                        abort();
                    }
                }
            }

            fits_close_file(fptr, &COREMOD_iofits_data.FITSIO_status);
            if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
            {
                if(errcode != 0)
                {
                    PRINT_ERROR("fits_close_file error");
                    list_image_ID();
                    if(errcode > 1)
                    {
                        abort();
                    }
                }
            }

            bzero = 0.0;
            for(uint64_t ii = 0; ii < nelements; ii++)
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
                                 data.NBKEWORD_DFT);
            larray = (long *) malloc(sizeof(long) * nelements);
            if(larray == NULL)
            {
                PRINT_ERROR("malloc error");
                abort();
            }

            fits_read_img(fptr, data_type_code(bitpix), fpixel, nelements, &nulval, larray,
                          &anynul, &COREMOD_iofits_data.FITSIO_status);
            if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
            {
                if(errcode != 0)
                {
                    PRINT_ERROR("fits_read_img error");
                    list_image_ID();
                    if(errcode > 1)
                    {
                        exit(0);
                    }
                }
            }

            fits_close_file(fptr, &COREMOD_iofits_data.FITSIO_status);
            if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
            {
                if(errcode != 0)
                {
                    PRINT_ERROR("fits_close_file error");
                    list_image_ID();
                    if(errcode > 1)
                    {
                        exit(0);
                    }
                }
            }

            bzero = 0.0;
            for(uint64_t ii = 0; ii < nelements; ii++)
            {
                data.image[ID].array.SI64[ii] = larray[ii] * bscale + bzero;
            }
            free(larray);
            larray = NULL;
        }




        /* bitpix = 8   TBYTE */
        if(bitpix == 8)
        {
            ID = create_image_ID(ID_name, naxis, naxes, Dtype, data.SHARED_DFT,
                                 data.NBKEWORD_DFT);
            barray = (unsigned char *) malloc(sizeof(unsigned char) * naxes[1] * naxes[0]);
            if(barray == NULL)
            {
                PRINT_ERROR("malloc error");
                exit(0);
            }

            fits_read_img(fptr, data_type_code(bitpix), fpixel, nelements, &nulval, barray,
                          &anynul, &COREMOD_iofits_data.FITSIO_status);
            if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
            {
                if(errcode != 0)
                {
                    PRINT_ERROR("fits_read_img error");
                    list_image_ID();
                    if(errcode > 1)
                    {
                        exit(0);
                    }
                }
            }

            fits_close_file(fptr, &COREMOD_iofits_data.FITSIO_status);
            if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
            {
                if(errcode != 0)
                {
                    PRINT_ERROR("fits_close_file error");
                    list_image_ID();
                    if(errcode > 1)
                    {
                        exit(0);
                    }
                }
            }


            for(uint64_t ii = 0; ii < nelements; ii++)
            {
                data.image[ID].array.F[ii] = (1.0 * barray[ii] * bscale + bzero);
            }
            free(barray);
            barray = NULL;
        }
    }

    return ID;
}




static errno_t compute_function()
{
    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    load_fits(
        infilename,
        outimname,
        *errmode
    );

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    return RETURN_SUCCESS;
}




INSERT_STD_FPSCLIfunctions



// Register function in CLI
errno_t CLIADDCMD_loadfits()
{
    INSERT_STD_FPSCLIREGISTERFUNC

    return RETURN_SUCCESS;
}


