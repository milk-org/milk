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


// ==========================================
// Forward declaration(s)
// ==========================================
imageID load_fits(
    const char *restrict file_name,
    const char *restrict ID_name,
    int         errcode
);


// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t load_fits_cli()
{
    if(
        CLI_checkarg(1, CLIARG_STR) +
        CLI_checkarg(2, CLIARG_STR_NOT_IMG)
        == 0)
    {
        load_fits(
            data.cmdargtoken[1].val.string,
            data.cmdargtoken[2].val.string,
            0);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }

    return CLICMD_SUCCESS;
}



// ==========================================
// Register CLI command(s)
// ==========================================

errno_t loadfits_addCLIcmd()
{

    RegisterCLIcommand(
        "loadfits",
        __FILE__,
        load_fits_cli,
        "load FITS format file",
        "input output",
        "loadfits im.fits im",
        "long load_fits()"
    );


    return RETURN_SUCCESS;
}






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
    long      naxis = 0;
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
                        if(fits_open_file(&fptr, file_name, READONLY, &COREMOD_iofits_data.FITSIO_status))
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
        if(PrintErrorMsg == 1)
        {
            fprintf(stderr, "%c[%d;%dm Error while calling \"fits_open_file\" %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            fprintf(stderr, "%c[%d;%dm within load_fits ( %s, %s ) %c[%d;m\n", (char) 27, 1,
                    31, ID_name, file_name, (char) 27, 0);
            fprintf(stderr, "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            list_image_ID();
        }
        else
        {
            PRINT_WARNING("Image \"%s\" could not be loaded from file \"%s\"", ID_name,
                          file_name);
        }

    }


    if(fileOK == 1)
    {
        char keyword[STRINGMAXLEN_FITSKEYWORDNAME];
        long  fpixel = 1;
        long i;
        long ii;
        char comment[STRINGMAXLEN_FITSKEYWCOMMENT];
        long  nelements;

        fits_read_key(fptr, TLONG, "NAXIS", &naxis, comment, &COREMOD_iofits_data.FITSIO_status);
        if(errcode != 0)
        {
            if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
            {
                fprintf(stderr,
                        "%c[%d;%dm Error while calling \"fits_read_key\" NAXIS %c[%d;m\n", (char) 27, 1,
                        31, (char) 27, 0);
                fprintf(stderr, "%c[%d;%dm within load_fits ( %s, %s ) %c[%d;m\n", (char) 27, 1,
                        31, ID_name, file_name, (char) 27, 0);
                fprintf(stderr, "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                        (char) 27, 1, 31, (char) 27, 0);
                list_image_ID();
                if(errcode > 1)
                {
                    exit(0);
                }
            }
        }


        for(i = 0; i < naxis; i++)
        {
            WRITE_FITSKEYWNAME(keyword, "NAXIS%ld", i + 1);
            
            fits_read_key(fptr, TLONG, keyword, &naxes[i], comment, &COREMOD_iofits_data.FITSIO_status);
            if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
            {
                if(errcode != 0)
                {
                    fprintf(stderr,
                            "%c[%d;%dm Error while calling \"fits_read_key\" NAXIS%ld %c[%d;m\n", (char) 27,
                            1, 31, i, (char) 27, 0);
                    fprintf(stderr, "%c[%d;%dm within load_fits ( %s, %s ) %c[%d;m\n", (char) 27, 1,
                            31, ID_name, file_name, (char) 27, 0);
                    fprintf(stderr, "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                            (char) 27, 1, 31, (char) 27, 0);
                    list_image_ID();
                    if(errcode > 1)
                    {
                        exit(0);
                    }
                }
            }
        }

        fits_read_key(fptr, TLONG, "BITPIX", &bitpixl, comment, &COREMOD_iofits_data.FITSIO_status);
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            if(errcode != 0)
            {
                fprintf(stderr,
                        "%c[%d;%dm Error while calling \"fits_read_key\" BITPIX %c[%d;m\n", (char) 27,
                        1, 31, (char) 27, 0);
                fprintf(stderr, "%c[%d;%dm within load_fits ( %s, %s ) %c[%d;m\n", (char) 27, 1,
                        31, ID_name, file_name, (char) 27, 0);
                fprintf(stderr, "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                        (char) 27, 1, 31, (char) 27, 0);
                list_image_ID();
                if(errcode > 1)
                {
                    exit(0);
                }
            }
        }



        int bitpix = (int) bitpixl;
        fits_read_key(fptr, TDOUBLE, "BSCALE", &bscale, comment, &COREMOD_iofits_data.FITSIO_status);
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 0) == 1)
        {
            //fprintf(stderr,"Error reading keyword \"BSCALE\" in file \"%s\"\n",file_name);
            bscale = 1.0;
        }
        fits_read_key(fptr, TDOUBLE, "BZERO", &bzero, comment, &COREMOD_iofits_data.FITSIO_status);
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
            for(i = 1; i < naxis; i++)
            {
                printf(",%ld", (long) naxes[i]);
            }
            printf("] %d %f %f\n", bitpix, bscale, bzero);
            fflush(stdout);
        }

        nelements = 1;
        for(i = 0; i < naxis; i++)
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
                    fprintf(stderr, "%c[%d;%dm Error while calling \"fits_read_img\" %c[%d;m\n",
                            (char) 27, 1, 31, (char) 27, 0);
                    fprintf(stderr, "%c[%d;%dm within load_fits ( %s, %s ) %c[%d;m\n", (char) 27, 1,
                            31, ID_name, file_name, (char) 27, 0);
                    fprintf(stderr, "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                            (char) 27, 1, 31, (char) 27, 0);
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
                    fprintf(stderr, "%c[%d;%dm Error while calling \"fits_close_file\" %c[%d;m\n",
                            (char) 27, 1, 31, (char) 27, 0);
                    fprintf(stderr, "%c[%d;%dm within load_fits ( %s, %s ) %c[%d;m\n", (char) 27, 1,
                            31, ID_name, file_name, (char) 27, 0);
                    fprintf(stderr, "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                            (char) 27, 1, 31, (char) 27, 0);
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
                    fprintf(stderr, "%c[%d;%dm Error while calling \"fits_read_img\" %c[%d;m\n",
                            (char) 27, 1, 31, (char) 27, 0);
                    fprintf(stderr, "%c[%d;%dm within load_fits ( %s, %s ) %c[%d;m\n", (char) 27, 1,
                            31, ID_name, file_name, (char) 27, 0);
                    fprintf(stderr, "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                            (char) 27, 1, 31, (char) 27, 0);
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
                    fprintf(stderr, "%c[%d;%dm Error while calling \"fits_close_file\" %c[%d;m\n",
                            (char) 27, 1, 31, (char) 27, 0);
                    fprintf(stderr, "%c[%d;%dm within load_fits ( %s, %s ) %c[%d;m\n", (char) 27, 1,
                            31, ID_name, file_name, (char) 27, 0);
                    fprintf(stderr, "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                            (char) 27, 1, 31, (char) 27, 0);
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
                    fprintf(stderr, "%c[%d;%dm Error while calling \"fits_read_img\" %c[%d;m\n",
                            (char) 27, 1, 31, (char) 27, 0);
                    fprintf(stderr, "%c[%d;%dm within load_fits ( %s, %s ) %c[%d;m\n", (char) 27, 1,
                            31, ID_name, file_name, (char) 27, 0);
                    fprintf(stderr, "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                            (char) 27, 1, 31, (char) 27, 0);
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
                    fprintf(stderr, "%c[%d;%dm Error while calling \"fits_close_file\" %c[%d;m\n",
                            (char) 27, 1, 31, (char) 27, 0);
                    fprintf(stderr, "%c[%d;%dm within load_fits ( %s, %s ) %c[%d;m\n", (char) 27, 1,
                            31, ID_name, file_name, (char) 27, 0);
                    fprintf(stderr, "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                            (char) 27, 1, 31, (char) 27, 0);
                    list_image_ID();
                    if(errcode > 1)
                    {
                        exit(0);
                    }
                }
            }

            check_FITSIO_status(__FILE__, __func__, __LINE__, 1);
            /*        for (ii = 0; ii < nelements; ii++)
                        data.image[ID].array.F[ii] = 1.0*sarray[ii];
                    free(sarray);
                    sarray = NULL;*/
        }


        /* bitpix = 32   TLONG */
        if(bitpix == 32)
        {
            /*fits_read_key(fptr, TLONG, "NDR", &NDR, comment, &FITSIO_status);
            if(check_FITSIO_status(__FILE__, __func__, __LINE__, 0) == 1) {
                NDR = 1;
            }*/
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
                    fprintf(stderr, "%c[%d;%dm Error while calling \"fits_read_img\" %c[%d;m\n",
                            (char) 27, 1, 31, (char) 27, 0);
                    fprintf(stderr, "%c[%d;%dm within load_fits ( %s, %s ) %c[%d;m\n", (char) 27, 1,
                            31, ID_name, file_name, (char) 27, 0);
                    fprintf(stderr, "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                            (char) 27, 1, 31, (char) 27, 0);
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
                    fprintf(stderr, "%c[%d;%dm Error while calling \"fits_close_file\" %c[%d;m\n",
                            (char) 27, 1, 31, (char) 27, 0);
                    fprintf(stderr, "%c[%d;%dm within load_fits ( %s, %s ) %c[%d;m\n", (char) 27, 1,
                            31, ID_name, file_name, (char) 27, 0);
                    fprintf(stderr, "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                            (char) 27, 1, 31, (char) 27, 0);
                    list_image_ID();
                    if(errcode > 1)
                    {
                        exit(0);
                    }
                }
            }

            bzero = 0.0;
            for(ii = 0; ii < nelements; ii++)
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
                exit(0);
            }

            fits_read_img(fptr, data_type_code(bitpix), fpixel, nelements, &nulval, larray,
                          &anynul, &COREMOD_iofits_data.FITSIO_status);
            if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
            {
                if(errcode != 0)
                {
                    fprintf(stderr, "%c[%d;%dm Error while calling \"fits_read_img\" %c[%d;m\n",
                            (char) 27, 1, 31, (char) 27, 0);
                    fprintf(stderr, "%c[%d;%dm within load_fits ( %s, %s ) %c[%d;m\n", (char) 27, 1,
                            31, ID_name, file_name, (char) 27, 0);
                    fprintf(stderr, "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                            (char) 27, 1, 31, (char) 27, 0);
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
                    fprintf(stderr, "%c[%d;%dm Error while calling \"fits_close_file\" %c[%d;m\n",
                            (char) 27, 1, 31, (char) 27, 0);
                    fprintf(stderr, "%c[%d;%dm within load_fits ( %s, %s ) %c[%d;m\n", (char) 27, 1,
                            31, ID_name, file_name, (char) 27, 0);
                    fprintf(stderr, "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                            (char) 27, 1, 31, (char) 27, 0);
                    list_image_ID();
                    if(errcode > 1)
                    {
                        exit(0);
                    }
                }
            }

            bzero = 0.0;
            for(ii = 0; ii < nelements; ii++)
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
                    fprintf(stderr, "%c[%d;%dm Error while calling \"fits_read_img\" %c[%d;m\n",
                            (char) 27, 1, 31, (char) 27, 0);
                    fprintf(stderr, "%c[%d;%dm within load_fits ( %s, %s ) %c[%d;m\n", (char) 27, 1,
                            31, ID_name, file_name, (char) 27, 0);
                    fprintf(stderr, "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                            (char) 27, 1, 31, (char) 27, 0);
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
                    fprintf(stderr, "%c[%d;%dm Error while calling \"fits_close_file\" %c[%d;m\n",
                            (char) 27, 1, 31, (char) 27, 0);
                    fprintf(stderr, "%c[%d;%dm within load_fits ( %s, %s ) %c[%d;m\n", (char) 27, 1,
                            31, ID_name, file_name, (char) 27, 0);
                    fprintf(stderr, "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                            (char) 27, 1, 31, (char) 27, 0);
                    list_image_ID();
                    if(errcode > 1)
                    {
                        exit(0);
                    }
                }
            }


            for(ii = 0; ii < nelements; ii++)
            {
                data.image[ID].array.F[ii] = (1.0 * barray[ii] * bscale + bzero);
            }
            free(barray);
            barray = NULL;
        }
    }

    return(ID);
}


