/**
 * @file    savefits.c
 * @brief   save FITS format files
 */


#include "CommandLineInterface/CLIcore.h"


#include <pthread.h>
#include "COREMOD_memory/COREMOD_memory.h"


#include "COREMOD_iofits_common.h"

#include "check_fitsio_status.h"
#include "file_exists.h"


extern COREMOD_IOFITS_DATA COREMOD_iofits_data;





// ==========================================
// Forward declaration(s)
// ==========================================

errno_t save_fl_fits(
    const char *restrict ID_name,
    const char *restrict file_name
);

errno_t save_db_fits(
    const char *restrict ID_name,
    const char *restrict file_name
);

errno_t save_sh16_fits(
    const char *restrict ID_name,
    const char *restrict file_name
);

errno_t save_fits(
    const char *restrict ID_name,
    const char *restrict file_name
);

errno_t save_fits_atomic(
    const char *restrict ID_name,
    const char *restrict file_name
);


// ==========================================
// Command line interface wrapper function(s)
// ==========================================



errno_t save_fl_fits_cli()
{
    char fname[STRINGMAXLEN_FILENAME];

    switch(data.cmdNBarg)
    {
        case 3:
            save_fl_fits(data.cmdargtoken[1].val.string,  data.cmdargtoken[2].val.string);
            break;
        case 2:
            WRITE_FILENAME(fname, "%s.fits", data.cmdargtoken[1].val.string);
            save_fl_fits(data.cmdargtoken[1].val.string, fname);
            break;
    }
    return CLICMD_SUCCESS;
}




errno_t save_db_fits_cli()
{
    char fname[STRINGMAXLEN_FILENAME];

    switch(data.cmdNBarg)
    {
        case 3:
            save_db_fits(data.cmdargtoken[1].val.string,  data.cmdargtoken[2].val.string);
            break;
        case 2:
            WRITE_FILENAME(fname, "%s.fits", data.cmdargtoken[1].val.string);
            save_db_fits(data.cmdargtoken[1].val.string, fname);
            break;
    }

    return CLICMD_SUCCESS;
}



errno_t save_sh16_fits_cli()
{
    char fname[STRINGMAXLEN_FILENAME];

    switch(data.cmdNBarg)
    {
        case 3:
            save_sh16_fits(data.cmdargtoken[1].val.string,  data.cmdargtoken[2].val.string);
            break;
        case 2:
            WRITE_FILENAME(fname, "%s.fits", data.cmdargtoken[1].val.string);
            save_sh16_fits(data.cmdargtoken[1].val.string, fname);
            break;
    }

    return CLICMD_SUCCESS;
}




errno_t save_fits_cli()
{
    char fname[STRINGMAXLEN_FILENAME];

    switch(data.cmdNBarg)
    {
        case 3:
            save_fits(data.cmdargtoken[1].val.string,  data.cmdargtoken[2].val.string);
            break;
        case 2:
            WRITE_FILENAME(fname, "%s.fits", data.cmdargtoken[1].val.string);
            save_fits(data.cmdargtoken[1].val.string, fname);
            break;
    }

    return CLICMD_SUCCESS;
}




// ==========================================
// Register CLI command(s)
// ==========================================

errno_t savefits_addCLIcmd()
{

    RegisterCLIcommand(
        "saveflfits",
        __FILE__,
        save_fl_fits_cli,
        "save FITS format file, float",
        "input output",
        "saveflfits im im.fits",
        "int save_fl_fits(char *ID_name, char *file_name)"
    );


    RegisterCLIcommand(
        "savedbfits",
        __FILE__,
        save_db_fits_cli,
        "save FITS format file, double",
        "input output",
        "savedbfits im im.fits",
        "int save_db_fits(char *ID_name, char *file_name)"
    );

    RegisterCLIcommand(
        "saveshfits",
        __FILE__,
        save_sh16_fits_cli,
        "save FITS format file, short",
        "input output",
        "saveshfits im im.fits",
        "int save_sh16_fits(char *ID_name, char *file_name)"
    );

    RegisterCLIcommand(
        "savefits",
        __FILE__,
        save_fits_cli,
        "save FITS format file",
        "input output",
        "savefits im im.fits",
        "int save_fits(char *ID_name, char *file_name)"
    );


    return RETURN_SUCCESS;
}







/* saves an image in a double format */

errno_t save_db_fits(
    const char *restrict ID_name,
    const char *restrict file_name
)
{
    fitsfile *fptr;
    long nelements;
    imageID ID;
    char file_name1[STRINGMAXLEN_FULLFILENAME];

    if((data.overwrite == 1) && (file_name[0] != '!')
            && (file_exists(file_name) == 1))
    {
        PRINT_WARNING("automatic overwrite on file \"%s\"\n", file_name);
        WRITE_FULLFILENAME(file_name1, "!%s", file_name);
    }
    else
    {
        WRITE_FULLFILENAME(file_name1, "!%s", file_name);
    }

    ID = image_ID(ID_name);

    if(ID != -1)
    {
        long  fpixel = 1;
        uint32_t naxes[3];
        long naxesl[3];
        double *array;
        uint8_t datatype;
        long naxis;
        long i;


        datatype = data.image[ID].md[0].datatype;
        naxis = data.image[ID].md[0].naxis;
        for(i = 0; i < naxis; i++)
        {
            naxes[i] = data.image[ID].md[0].size[i];
            naxesl[i] = (long) naxes[i];
        }



        nelements = 1;
        for(i = 0; i < naxis; i++)
        {
            nelements *= naxes[i];
        }


        if(datatype != _DATATYPE_DOUBLE)   // data conversion required
        {
            long ii;

            array = (double *) malloc(SIZEOF_DATATYPE_DOUBLE * nelements);
            if(array == NULL)
            {
                PRINT_ERROR("malloc error");
                exit(0);
            }

            switch(datatype)
            {
                case _DATATYPE_UINT8 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (double) data.image[ID].array.UI8[ii];
                    }
                    break;

                case _DATATYPE_INT8 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (double) data.image[ID].array.SI8[ii];
                    }
                    break;

                case _DATATYPE_UINT16 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (double) data.image[ID].array.UI16[ii];
                    }
                    break;

                case _DATATYPE_INT16 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (double) data.image[ID].array.SI16[ii];
                    }
                    break;

                case _DATATYPE_UINT32 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (double) data.image[ID].array.UI32[ii];
                    }
                    break;

                case _DATATYPE_INT32 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (double) data.image[ID].array.SI32[ii];
                    }
                    break;

                case _DATATYPE_UINT64 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (double) data.image[ID].array.UI64[ii];
                    }
                    break;

                case _DATATYPE_INT64 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (double) data.image[ID].array.SI64[ii];
                    }
                    break;

                case _DATATYPE_FLOAT :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (double) data.image[ID].array.F[ii];
                    }
                    break;

                default :
                    list_image_ID();
                    PRINT_ERROR("atype value not recognised");
                    printf("ID %ld  datatype = %d\n", ID, datatype);
                    free(array);
                    exit(0);
                    break;
            }
        }



        fits_create_file(&fptr, file_name1, &COREMOD_iofits_data.FITSIO_status);
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            PRINT_ERROR("fits_create_file error %s -> %s", ID_name, file_name);
            list_image_ID();
        }


        fits_create_img(fptr, DOUBLE_IMG, naxis, naxesl,
                        &COREMOD_iofits_data.FITSIO_status);
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            PRINT_ERROR("fits_create_img error %s -> %s", ID_name, file_name);
            list_image_ID();
        }

        if(datatype == _DATATYPE_DOUBLE)
        {
            fits_write_img(fptr, TDOUBLE, fpixel, nelements, data.image[ID].array.D,
                           &COREMOD_iofits_data.FITSIO_status);
        }
        else
        {
            fits_write_img(fptr, TDOUBLE, fpixel, nelements, array,
                           &COREMOD_iofits_data.FITSIO_status);
            free(array);
            array = NULL;
        }

        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            PRINT_ERROR("fits_write_img error %s -> %s", ID_name, file_name);
            list_image_ID();
        }

        fits_close_file(fptr, &COREMOD_iofits_data.FITSIO_status);
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            PRINT_ERROR("fits_close_file error %s -> %s", ID_name, file_name);
            list_image_ID();
        }
    }
    else
    {
        printf("image does not exist in memory\n");
    }

    return RETURN_SUCCESS;
}





/* saves an image in a float format */

errno_t save_fl_fits(
    const char *restrict ID_name,
    const char *restrict file_name
)
{
    fitsfile *fptr;
    long      naxis, nelements;
    float    *array = NULL;
    imageID   ID;
    long      ii;
    long      i;
    char      file_name1[STRINGMAXLEN_FULLFILENAME];

    if((data.overwrite == 1) && (file_name[0] != '!')
            && (file_exists(file_name) == 1))
    {
        PRINT_WARNING("automatic overwrite on file \"%s\"\n", file_name);
        WRITE_FULLFILENAME(file_name1, "!%s", file_name);
    }
    else
    {
        WRITE_FULLFILENAME(file_name1, "%s", file_name);
    }

    ID = image_ID(ID_name);

    if(ID != -1)
    {
        uint8_t datatype;
        long naxesl[3];
        uint32_t naxes[3];
        long  fpixel = 1;

        datatype = data.image[ID].md[0].datatype;
        naxis = data.image[ID].md[0].naxis;
        for(i = 0; i < naxis; i++)
        {
            naxes[i] = data.image[ID].md[0].size[i];
            naxesl[i] = (long) naxes[i];
        }

        nelements = 1;
        for(i = 0; i < naxis; i++)
        {
            nelements *= naxes[i];
        }


        if(datatype != _DATATYPE_FLOAT)   // data conversion required
        {
            array = (float *) malloc(SIZEOF_DATATYPE_FLOAT * nelements);
            if(array == NULL)
            {
                PRINT_ERROR("malloc error");
                exit(0);
            }

            switch(datatype)
            {
                case _DATATYPE_UINT8 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (float) data.image[ID].array.UI8[ii];
                    }
                    break;

                case _DATATYPE_INT8 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (float) data.image[ID].array.SI8[ii];
                    }
                    break;

                case _DATATYPE_UINT16 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (float) data.image[ID].array.UI16[ii];
                    }
                    break;

                case _DATATYPE_INT16 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (float) data.image[ID].array.SI16[ii];
                    }
                    break;

                case _DATATYPE_UINT32 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (float) data.image[ID].array.UI32[ii];
                    }
                    break;

                case _DATATYPE_INT32 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (float) data.image[ID].array.SI32[ii];
                    }
                    break;

                case _DATATYPE_UINT64 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (float) data.image[ID].array.UI64[ii];
                    }
                    break;

                case _DATATYPE_INT64 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (float) data.image[ID].array.SI64[ii];
                    }
                    break;

                case _DATATYPE_DOUBLE :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (float) data.image[ID].array.D[ii];
                    }
                    break;

                default :
                    list_image_ID();
                    PRINT_ERROR("atype value not recognised");
                    printf("ID %ld  datatype = %d\n", ID, datatype);
                    free(array);
                    exit(0);
                    break;
            }
        }


        COREMOD_iofits_data.FITSIO_status = 0;
        fits_create_file(&fptr, file_name1, &COREMOD_iofits_data.FITSIO_status);
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            fprintf(stderr,
                    "%c[%d;%dm Error while calling \"fits_create_file\" with filename \"%s\" %c[%d;m\n",
                    (char) 27, 1, 31, file_name1, (char) 27, 0);
            if(file_exists(file_name1) == 1)
            {
                fprintf(stderr,
                        "%c[%d;%dm File \"%s\" already exists. Make sure you remove this file before attempting to write file with identical name. %c[%d;m\n",
                        (char) 27, 1, 31, file_name1, (char) 27, 0);
                exit(0);
            }
            else
            {
                fprintf(stderr, "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                        (char) 27, 1, 31, (char) 27, 0);
                list_image_ID();
            }
        }

        fits_create_img(fptr, FLOAT_IMG, (int) naxis, naxesl,
                        &COREMOD_iofits_data.FITSIO_status);
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            fprintf(stderr, "%c[%d;%dm Error while calling \"fits_create_img\" %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            fprintf(stderr, "%c[%d;%dm within save_fl_fits ( %s, %s ) %c[%d;m\n", (char) 27,
                    1, 31, ID_name, file_name, (char) 27, 0);
            fprintf(stderr, "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            list_image_ID();
        }

        if(datatype == _DATATYPE_FLOAT)
        {
            fits_write_img(fptr, TFLOAT, fpixel, nelements, data.image[ID].array.F,
                           &COREMOD_iofits_data.FITSIO_status);
        }
        else
        {
            fits_write_img(fptr, TFLOAT, fpixel, nelements, array,
                           &COREMOD_iofits_data.FITSIO_status);
            free(array);
            array = NULL;
        }

        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            fprintf(stderr, "%c[%d;%dm Error while calling \"fits_write_img\" %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            fprintf(stderr, "%c[%d;%dm within save_fl_fits ( %s, %s ) %c[%d;m\n", (char) 27,
                    1, 31, ID_name, file_name, (char) 27, 0);
            fprintf(stderr, "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            list_image_ID();
        }

        fits_close_file(fptr, &COREMOD_iofits_data.FITSIO_status);
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            fprintf(stderr, "%c[%d;%dm Error while calling \"fits_close_file\" %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            fprintf(stderr, "%c[%d;%dm within save_fl_fits ( %s, %s ) %c[%d;m\n", (char) 27,
                    1, 31, ID_name, file_name, (char) 27, 0);
            fprintf(stderr, "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            list_image_ID();
        }
    }
    else
    {
        fprintf(stderr, "%c[%d;%dm image \"%s\" does not exist in memory %c[%d;m\n",
                (char) 27, 1, 31, ID_name, (char) 27, 0);
    }

    return RETURN_SUCCESS;
}





/* saves an image in a short int format */

errno_t save_sh16_fits(
    const char *restrict ID_name,
    const char *restrict file_name
)
{
    fitsfile *fptr;
    long  fpixel = 1, naxis, nelements;
    uint32_t naxes[3];
    long naxesl[3];
    int16_t *array = NULL;
    imageID ID;
    long ii;
    long i;
    uint8_t datatype;
    char file_name1[STRINGMAXLEN_FULLFILENAME];


    if((data.overwrite == 1) && (file_name[0] != '!')
            && (file_exists(file_name) == 1))
    {
        PRINT_WARNING("automatic overwrite on file \"%s\"\n", file_name);
        WRITE_FULLFILENAME(file_name1, "!%s", file_name);
    }
    else
    {
        WRITE_FULLFILENAME(file_name1, "%s", file_name);
    }

    ID = image_ID(ID_name);

    if(ID != -1)
    {
        datatype = data.image[ID].md[0].datatype;
        naxis = data.image[ID].md[0].naxis;
        for(i = 0; i < naxis; i++)
        {
            naxes[i] = data.image[ID].md[0].size[i];
            naxesl[i] = (long) naxes[i];
        }

        nelements = 1;
        for(i = 0; i < naxis; i++)
        {
            nelements *= naxes[i];
        }

        if(datatype != _DATATYPE_INT16)   // data conversion required
        {

            printf("Data conversion required\n"); //TEST
            fflush(stdout);


            array = (int16_t *) malloc(SIZEOF_DATATYPE_INT16 * nelements);
            if(array == NULL)
            {
                PRINT_ERROR("malloc error");
                exit(0);
            }

            switch(datatype)
            {
                case _DATATYPE_UINT8 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int16_t) data.image[ID].array.UI8[ii];
                    }
                    break;

                case _DATATYPE_INT8 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int16_t) data.image[ID].array.SI8[ii];
                    }
                    break;

                case _DATATYPE_UINT16 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int16_t) data.image[ID].array.UI16[ii];
                    }
                    break;

                case _DATATYPE_UINT32 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int16_t) data.image[ID].array.UI32[ii];
                    }
                    break;

                case _DATATYPE_INT32 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int16_t) data.image[ID].array.SI32[ii];
                    }
                    break;

                case _DATATYPE_UINT64 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int16_t) data.image[ID].array.UI64[ii];
                    }
                    break;

                case _DATATYPE_INT64 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int16_t) data.image[ID].array.SI64[ii];
                    }
                    break;

                case _DATATYPE_FLOAT :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int16_t) data.image[ID].array.F[ii];
                    }
                    break;

                case _DATATYPE_DOUBLE :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int16_t) data.image[ID].array.D[ii];
                    }
                    break;

                default :
                    list_image_ID();
                    PRINT_ERROR("atype value not recognised");
                    free(array);
                    printf("ID %ld  datatype = %d\n", ID, datatype);
                    exit(0);
                    break;
            }
        }
        else
        {
            printf("No data conversion required\n"); //TEST
            fflush(stdout);
        }


        fits_create_file(&fptr, file_name1, &COREMOD_iofits_data.FITSIO_status);
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            fprintf(stderr,
                    "%c[%d;%dm Error while calling \"fits_create_file\" with filename \"%s\" %c[%d;m\n",
                    (char) 27, 1, 31, file_name1, (char) 27, 0);
            if(file_exists(file_name1) == 1)
            {
                fprintf(stderr,
                        "%c[%d;%dm File \"%s\" already exists. Make sure you remove this file before attempting to write file with identical name. %c[%d;m\n",
                        (char) 27, 1, 31, file_name1, (char) 27, 0);
                exit(0);
            }
            else
            {
                fprintf(stderr,
                        "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                        (char) 27, 1, 31, (char) 27, 0);
                list_image_ID();
            }
        }


        //    16          short integer, I        21        TSHORT
        fits_create_img(fptr, SHORT_IMG, naxis, naxesl,
                        &COREMOD_iofits_data.FITSIO_status);
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            fprintf(stderr,
                    "%c[%d;%dm Error while calling \"fits_create_img\" %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            fprintf(stderr,
                    "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            list_image_ID();
        }

        if(datatype == _DATATYPE_INT16)
        {
            printf("Direct copy --- \n");
            fflush(stdout);

            fits_write_img(fptr, TSHORT, fpixel, nelements, data.image[ID].array.SI16,
                           &COREMOD_iofits_data.FITSIO_status);
        }
        else
        {
            fits_write_img(fptr, TSHORT, fpixel, nelements, array,
                           &COREMOD_iofits_data.FITSIO_status);
            free(array);
            array = NULL;
        }


        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            fprintf(stderr,
                    "%c[%d;%dm Error while calling \"fits_write_img\" %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            fprintf(stderr,
                    "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            list_image_ID();
        }

        fits_close_file(fptr, &COREMOD_iofits_data.FITSIO_status);
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            fprintf(stderr,
                    "%c[%d;%dm Error while calling \"fits_close_file\" %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            fprintf(stderr,
                    "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            list_image_ID();
        }
    }
    else
    {
        fprintf(stderr,
                "%c[%d;%dm image \"%s\" does not exist in memory %c[%d;m\n",
                (char) 27, 1, 31, ID_name, (char) 27, 0);
    }


    return EXIT_SUCCESS;
}






/* saves an image in a unsigned short int format */

errno_t save_ush16_fits(
    const char *restrict ID_name,
    const char *restrict file_name
)
{
    fitsfile *fptr;
    long  fpixel = 1, naxis, nelements;
    uint32_t naxes[3];
    long naxesl[3];
    uint16_t *array = NULL;
    imageID ID;
    long ii;
    long i;
    uint8_t datatype;
    char file_name1[STRINGMAXLEN_FULLFILENAME];

    if((data.overwrite == 1) && (file_name[0] != '!')
            && (file_exists(file_name) == 1))
    {
        PRINT_WARNING("automatic overwrite on file \"%s\"\n", file_name);
        WRITE_FULLFILENAME(file_name1, "!%s", file_name);
    }
    else
    {
        WRITE_FULLFILENAME(file_name1, "%s", file_name);
    }

    ID = image_ID(ID_name);

    if(ID != -1)
    {
        datatype = data.image[ID].md[0].datatype;
        naxis = data.image[ID].md[0].naxis;
        for(i = 0; i < naxis; i++)
        {
            naxes[i] = data.image[ID].md[0].size[i];
            naxesl[i] = (long) naxes[i]; // CFITSIO expects long int *
        }

        nelements = 1;
        for(i = 0; i < naxis; i++)
        {
            nelements *= naxes[i];
        }

        if(datatype != _DATATYPE_UINT16)   // data conversion required
        {
            array = (uint16_t *) malloc(SIZEOF_DATATYPE_UINT16 * nelements);
            if(array == NULL)
            {
                PRINT_ERROR("malloc error");
                exit(0);
            }

            switch(datatype)
            {
                case _DATATYPE_UINT8 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (uint16_t) data.image[ID].array.UI8[ii];
                    }
                    break;

                case _DATATYPE_INT8 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (uint16_t) data.image[ID].array.SI8[ii];
                    }
                    break;

                case _DATATYPE_INT16 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (uint16_t) data.image[ID].array.SI16[ii];
                    }
                    break;

                case _DATATYPE_UINT32 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (uint16_t) data.image[ID].array.UI32[ii];
                    }
                    break;

                case _DATATYPE_INT32 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (uint16_t) data.image[ID].array.SI32[ii];
                    }
                    break;

                case _DATATYPE_UINT64 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (uint16_t) data.image[ID].array.UI64[ii];
                    }
                    break;

                case _DATATYPE_INT64 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (uint16_t) data.image[ID].array.SI64[ii];
                    }
                    break;

                case _DATATYPE_FLOAT :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (uint16_t) data.image[ID].array.F[ii];
                    }
                    break;

                case _DATATYPE_DOUBLE :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (uint16_t) data.image[ID].array.D[ii];
                    }
                    break;

                default :
                    list_image_ID();
                    PRINT_ERROR("atype value not recognised");
                    free(array);
                    printf("ID %ld  datatype = %d\n", ID, datatype);
                    exit(0);
                    break;
            }
        }

        fits_create_file(&fptr, file_name1, &COREMOD_iofits_data.FITSIO_status);
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            fprintf(stderr,
                    "%c[%d;%dm Error while calling \"fits_create_file\" with filename \"%s\" %c[%d;m\n",
                    (char) 27, 1, 31, file_name1, (char) 27, 0);
            if(file_exists(file_name1) == 1)
            {
                fprintf(stderr,
                        "%c[%d;%dm File \"%s\" already exists. Make sure you remove this file before attempting to write file with identical name. %c[%d;m\n",
                        (char) 27, 1, 31, file_name1, (char) 27, 0);
                exit(0);
            }
            else
            {
                fprintf(stderr,
                        "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                        (char) 27, 1, 31, (char) 27, 0);
                list_image_ID();
            }
        }

        fits_create_img(fptr, USHORT_IMG, naxis, naxesl,
                        &COREMOD_iofits_data.FITSIO_status);
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            fprintf(stderr,
                    "%c[%d;%dm Error while calling \"fits_create_img\" %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            fprintf(stderr,
                    "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            list_image_ID();
        }

        if(datatype == _DATATYPE_UINT16)
        {
            fits_write_img(fptr, TUSHORT, fpixel, nelements, data.image[ID].array.UI16,
                           &COREMOD_iofits_data.FITSIO_status);
        }
        else
        {
            fits_write_img(fptr, TUSHORT, fpixel, nelements, array,
                           &COREMOD_iofits_data.FITSIO_status);
            free(array);
            array = NULL;
        }

        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            fprintf(stderr,
                    "%c[%d;%dm Error while calling \"fits_write_img\" %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            fprintf(stderr,
                    "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            list_image_ID();
        }

        fits_close_file(fptr, &COREMOD_iofits_data.FITSIO_status);
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            fprintf(stderr,
                    "%c[%d;%dm Error while calling \"fits_close_file\" %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            fprintf(stderr,
                    "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            list_image_ID();
        }
    }
    else
    {
        fprintf(stderr,
                "%c[%d;%dm image \"%s\" does not exist in memory %c[%d;m\n",
                (char) 27, 1, 31, ID_name, (char) 27, 0);
    }

    return RETURN_SUCCESS;
}










/* saves an image in a int format (32 bit) */

errno_t save_int32_fits(
    const char *restrict ID_name,
    const char *restrict file_name
)
{
    fitsfile *fptr;
    long  fpixel = 1, naxis, nelements;
    uint32_t naxes[3];
    long naxesl[3];
    int32_t *array = NULL;
    imageID ID;
    long ii;
    long i;
    uint8_t datatype;
    char file_name1[STRINGMAXLEN_FULLFILENAME];


    if((data.overwrite == 1) && (file_name[0] != '!')
            && (file_exists(file_name) == 1))
    {
        PRINT_WARNING("automatic overwrite on file \"%s\"\n", file_name);
        WRITE_FULLFILENAME(file_name1, "!%s", file_name);
    }
    else
    {
        WRITE_FULLFILENAME(file_name1, "%s", file_name);
    }

    ID = image_ID(ID_name);

    if(ID != -1)
    {
        datatype = data.image[ID].md[0].datatype;
        naxis = data.image[ID].md[0].naxis;
        for(i = 0; i < naxis; i++)
        {
            naxes[i] = data.image[ID].md[0].size[i];
            naxesl[i] = (long) naxes[i];
        }

        nelements = 1;
        for(i = 0; i < naxis; i++)
        {
            nelements *= naxes[i];
        }

        if(datatype != _DATATYPE_INT32)   // data conversion required
        {

            printf("Data conversion required\n"); //TEST
            fflush(stdout);


            array = (int32_t *) malloc(SIZEOF_DATATYPE_INT32 * nelements);
            if(array == NULL)
            {
                PRINT_ERROR("malloc error");
                exit(0);
            }

            switch(datatype)
            {
                case _DATATYPE_UINT8 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int32_t) data.image[ID].array.UI8[ii];
                    }
                    break;

                case _DATATYPE_INT8 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int32_t) data.image[ID].array.SI8[ii];
                    }
                    break;

                case _DATATYPE_UINT16 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int32_t) data.image[ID].array.UI16[ii];
                    }
                    break;

                case _DATATYPE_INT16 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int32_t) data.image[ID].array.SI16[ii];
                    }
                    break;

                case _DATATYPE_UINT32 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int32_t) data.image[ID].array.UI32[ii];
                    }
                    break;

                case _DATATYPE_UINT64 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int32_t) data.image[ID].array.UI64[ii];
                    }
                    break;

                case _DATATYPE_INT64 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int32_t) data.image[ID].array.SI64[ii];
                    }
                    break;

                case _DATATYPE_FLOAT :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int32_t) data.image[ID].array.F[ii];
                    }
                    break;

                case _DATATYPE_DOUBLE :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int32_t) data.image[ID].array.D[ii];
                    }
                    break;

                default :
                    list_image_ID();
                    PRINT_ERROR("atype value not recognised");
                    free(array);
                    printf("ID %ld  datatype = %d\n", ID, datatype);
                    exit(0);
                    break;
            }
        }
        else
        {
            printf("No data conversion required\n"); //TEST
            fflush(stdout);
        }


        fits_create_file(&fptr, file_name1, &COREMOD_iofits_data.FITSIO_status);
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            fprintf(stderr,
                    "%c[%d;%dm Error while calling \"fits_create_file\" with filename \"%s\" %c[%d;m\n",
                    (char) 27, 1, 31, file_name1, (char) 27, 0);
            if(file_exists(file_name1) == 1)
            {
                fprintf(stderr,
                        "%c[%d;%dm File \"%s\" already exists. Make sure you remove this file before attempting to write file with identical name. %c[%d;m\n",
                        (char) 27, 1, 31, file_name1, (char) 27, 0);
                exit(0);
            }
            else
            {
                fprintf(stderr,
                        "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                        (char) 27, 1, 31, (char) 27, 0);
                list_image_ID();
            }
        }


        //    32          long integer        TLONG
        fits_create_img(fptr, LONG_IMG, naxis, naxesl,
                        &COREMOD_iofits_data.FITSIO_status);
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            fprintf(stderr,
                    "%c[%d;%dm Error while calling \"fits_create_img\" %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            fprintf(stderr,
                    "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            list_image_ID();
        }

        if(datatype == _DATATYPE_INT32)
        {
            printf("Direct copy --- \n");
            fflush(stdout);

            fits_write_img(fptr, TINT, fpixel, nelements, data.image[ID].array.SI32,
                           &COREMOD_iofits_data.FITSIO_status);
        }
        else
        {
            fits_write_img(fptr, TINT, fpixel, nelements, array,
                           &COREMOD_iofits_data.FITSIO_status);
            free(array);
            array = NULL;
        }


        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            fprintf(stderr,
                    "%c[%d;%dm Error while calling \"fits_write_img\" %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            fprintf(stderr,
                    "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            list_image_ID();
        }

        fits_close_file(fptr, &COREMOD_iofits_data.FITSIO_status);
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            fprintf(stderr,
                    "%c[%d;%dm Error while calling \"fits_close_file\" %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            fprintf(stderr,
                    "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            list_image_ID();
        }
    }
    else
    {
        fprintf(stderr,
                "%c[%d;%dm image \"%s\" does not exist in memory %c[%d;m\n",
                (char) 27, 1, 31, ID_name, (char) 27, 0);
    }


    return EXIT_SUCCESS;
}






/* saves an image in a unsigned short int format */

errno_t save_uint32_fits(
    const char *restrict ID_name,
    const char *restrict file_name
)
{
    fitsfile *fptr;
    long  fpixel = 1, naxis, nelements;
    uint32_t naxes[3];
    long naxesl[3];
    uint32_t *array = NULL;
    imageID ID;
    long ii;
    long i;
    uint8_t datatype;
    char file_name1[STRINGMAXLEN_FULLFILENAME];


    if((data.overwrite == 1) && (file_name[0] != '!')
            && (file_exists(file_name) == 1))
    {
        PRINT_WARNING("automatic overwrite on file \"%s\"\n", file_name);
        WRITE_FULLFILENAME(file_name1, "!%s", file_name);
    }
    else
    {
        WRITE_FULLFILENAME(file_name1, "%s", file_name);
    }

    ID = image_ID(ID_name);

    if(ID != -1)
    {
        datatype = data.image[ID].md[0].datatype;
        naxis = data.image[ID].md[0].naxis;
        for(i = 0; i < naxis; i++)
        {
            naxes[i] = data.image[ID].md[0].size[i];
            naxesl[i] = (long) naxes[i]; // CFITSIO expects long int *
        }

        nelements = 1;
        for(i = 0; i < naxis; i++)
        {
            nelements *= naxes[i];
        }

        if(datatype != _DATATYPE_UINT32)   // data conversion required
        {
            array = (uint32_t *) malloc(SIZEOF_DATATYPE_UINT32 * nelements);
            if(array == NULL)
            {
                PRINT_ERROR("malloc error");
                exit(0);
            }

            switch(datatype)
            {
                case _DATATYPE_UINT8 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (uint32_t) data.image[ID].array.UI8[ii];
                    }
                    break;

                case _DATATYPE_INT8 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (uint32_t) data.image[ID].array.SI8[ii];
                    }
                    break;

                case _DATATYPE_UINT16 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (uint32_t) data.image[ID].array.UI16[ii];
                    }
                    break;

                case _DATATYPE_INT16 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (uint32_t) data.image[ID].array.SI16[ii];
                    }
                    break;

                case _DATATYPE_INT32 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (uint32_t) data.image[ID].array.SI32[ii];
                    }
                    break;

                case _DATATYPE_UINT64 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (uint32_t) data.image[ID].array.UI64[ii];
                    }
                    break;

                case _DATATYPE_INT64 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (uint32_t) data.image[ID].array.SI64[ii];
                    }
                    break;

                case _DATATYPE_FLOAT :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (uint32_t) data.image[ID].array.F[ii];
                    }
                    break;

                case _DATATYPE_DOUBLE :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (uint32_t) data.image[ID].array.D[ii];
                    }
                    break;

                default :
                    list_image_ID();
                    PRINT_ERROR("atype value not recognised");
                    free(array);
                    printf("ID %ld  datatype = %d\n", ID, datatype);
                    exit(0);
                    break;
            }
        }

        fits_create_file(&fptr, file_name1, &COREMOD_iofits_data.FITSIO_status);
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            fprintf(stderr,
                    "%c[%d;%dm Error while calling \"fits_create_file\" with filename \"%s\" %c[%d;m\n",
                    (char) 27, 1, 31, file_name1, (char) 27, 0);
            if(file_exists(file_name1) == 1)
            {
                fprintf(stderr,
                        "%c[%d;%dm File \"%s\" already exists. Make sure you remove this file before attempting to write file with identical name. %c[%d;m\n",
                        (char) 27, 1, 31, file_name1, (char) 27, 0);
                exit(0);
            }
            else
            {
                fprintf(stderr,
                        "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                        (char) 27, 1, 31, (char) 27, 0);
                list_image_ID();
            }
        }

        fits_create_img(fptr, USHORT_IMG, naxis, naxesl,
                        &COREMOD_iofits_data.FITSIO_status);
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            fprintf(stderr,
                    "%c[%d;%dm Error while calling \"fits_create_img\" %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            fprintf(stderr,
                    "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            list_image_ID();
        }

        if(datatype == _DATATYPE_UINT32)
        {
            fits_write_img(fptr, TUINT, fpixel, nelements, data.image[ID].array.UI32,
                           &COREMOD_iofits_data.FITSIO_status);
        }
        else
        {
            fits_write_img(fptr, TUINT, fpixel, nelements, array,
                           &COREMOD_iofits_data.FITSIO_status);
            free(array);
            array = NULL;
        }

        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            fprintf(stderr,
                    "%c[%d;%dm Error while calling \"fits_write_img\" %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            fprintf(stderr,
                    "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            list_image_ID();
        }

        fits_close_file(fptr, &COREMOD_iofits_data.FITSIO_status);
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            fprintf(stderr,
                    "%c[%d;%dm Error while calling \"fits_close_file\" %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            fprintf(stderr,
                    "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            list_image_ID();
        }
    }
    else
    {
        fprintf(stderr,
                "%c[%d;%dm image \"%s\" does not exist in memory %c[%d;m\n",
                (char) 27, 1, 31, ID_name, (char) 27, 0);
    }

    return RETURN_SUCCESS;
}






/* saves an image in a int format (64 bit) */

errno_t save_int64_fits(
    const char *restrict ID_name,
    const char *restrict file_name
)
{
    fitsfile *fptr;
    long  fpixel = 1, naxis, nelements;
    uint32_t naxes[3];
    long naxesl[3];
    int64_t *array = NULL;
    imageID ID;
    long ii;
    long i;
    uint8_t datatype;
    char file_name1[STRINGMAXLEN_FULLFILENAME];



    if((data.overwrite == 1) && (file_name[0] != '!')
            && (file_exists(file_name) == 1))
    {
        PRINT_WARNING("automatic overwrite on file \"%s\"\n", file_name);
        WRITE_FULLFILENAME(file_name1, "!%s", file_name);
    }
    else
    {
        WRITE_FULLFILENAME(file_name1, "%s", file_name);
    }

    ID = image_ID(ID_name);

    if(ID != -1)
    {
        datatype = data.image[ID].md[0].datatype;
        naxis = data.image[ID].md[0].naxis;
        for(i = 0; i < naxis; i++)
        {
            naxes[i] = data.image[ID].md[0].size[i];
            naxesl[i] = (long) naxes[i];
        }

        nelements = 1;
        for(i = 0; i < naxis; i++)
        {
            nelements *= naxes[i];
        }

        if(datatype != _DATATYPE_INT64)   // data conversion required
        {

            printf("Data conversion required\n"); //TEST
            fflush(stdout);


            array = (int64_t *) malloc(SIZEOF_DATATYPE_INT64 * nelements);
            if(array == NULL)
            {
                PRINT_ERROR("malloc error");
                exit(0);
            }

            switch(datatype)
            {
                case _DATATYPE_UINT8 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int64_t) data.image[ID].array.UI8[ii];
                    }
                    break;

                case _DATATYPE_INT8 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int64_t) data.image[ID].array.SI8[ii];
                    }
                    break;

                case _DATATYPE_UINT16 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int64_t) data.image[ID].array.UI16[ii];
                    }
                    break;

                case _DATATYPE_INT16 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int64_t) data.image[ID].array.SI16[ii];
                    }
                    break;

                case _DATATYPE_UINT32 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int64_t) data.image[ID].array.UI32[ii];
                    }
                    break;

                case _DATATYPE_INT32 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int64_t) data.image[ID].array.SI32[ii];
                    }
                    break;

                case _DATATYPE_UINT64 :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int64_t) data.image[ID].array.UI64[ii];
                    }
                    break;

                case _DATATYPE_FLOAT :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int64_t) data.image[ID].array.F[ii];
                    }
                    break;

                case _DATATYPE_DOUBLE :
                    for(ii = 0; ii < nelements; ii++)
                    {
                        array[ii] = (int64_t) data.image[ID].array.D[ii];
                    }
                    break;

                default :
                    list_image_ID();
                    PRINT_ERROR("atype value not recognised");
                    free(array);
                    printf("ID %ld  datatype = %d\n", ID, datatype);
                    exit(0);
                    break;
            }
        }
        else
        {
            printf("No data conversion required\n"); //TEST
            fflush(stdout);
        }


        fits_create_file(&fptr, file_name1, &COREMOD_iofits_data.FITSIO_status);
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            fprintf(stderr,
                    "%c[%d;%dm Error while calling \"fits_create_file\" with filename \"%s\" %c[%d;m\n",
                    (char) 27, 1, 31, file_name1, (char) 27, 0);
            if(file_exists(file_name1) == 1)
            {
                fprintf(stderr,
                        "%c[%d;%dm File \"%s\" already exists. Make sure you remove this file before attempting to write file with identical name. %c[%d;m\n",
                        (char) 27, 1, 31, file_name1, (char) 27, 0);
                exit(0);
            }
            else
            {
                fprintf(stderr,
                        "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                        (char) 27, 1, 31, (char) 27, 0);
                list_image_ID();
            }
        }


        //    32          long integer        TLONG
        fits_create_img(fptr, LONGLONG_IMG, naxis, naxesl,
                        &COREMOD_iofits_data.FITSIO_status);
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            fprintf(stderr,
                    "%c[%d;%dm Error while calling \"fits_create_img\" %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            fprintf(stderr,
                    "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            list_image_ID();
        }

        if(datatype == _DATATYPE_INT32)
        {
            printf("Direct copy --- \n");
            fflush(stdout);

            fits_write_img(fptr, TLONG, fpixel, nelements, data.image[ID].array.SI64,
                           &COREMOD_iofits_data.FITSIO_status);
        }
        else
        {
            fits_write_img(fptr, TLONG, fpixel, nelements, array,
                           &COREMOD_iofits_data.FITSIO_status);
            free(array);
            array = NULL;
        }


        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            fprintf(stderr,
                    "%c[%d;%dm Error while calling \"fits_write_img\" %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            fprintf(stderr,
                    "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            list_image_ID();
        }

        fits_close_file(fptr, &COREMOD_iofits_data.FITSIO_status);
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
        {
            fprintf(stderr,
                    "%c[%d;%dm Error while calling \"fits_close_file\" %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            fprintf(stderr,
                    "%c[%d;%dm Printing Cfits image buffer content: %c[%d;m\n",
                    (char) 27, 1, 31, (char) 27, 0);
            list_image_ID();
        }
    }
    else
    {
        fprintf(stderr,
                "%c[%d;%dm image \"%s\" does not exist in memory %c[%d;m\n",
                (char) 27, 1, 31, ID_name, (char) 27, 0);
    }


    return EXIT_SUCCESS;
}






errno_t save_fits(
    const char *restrict ID_name,
    const char *restrict file_name
)
{
    char savename[1000];
    if(file_name[0] == '!')
    {
        strcpy(savename, file_name + 1);    // avoid the leading '!'
    }
    else
    {
        strcpy(savename, file_name);
    }

    return save_fits_atomic(ID_name, savename);
}



/**
 * atomic save is a two step process:
 *
 * (1) save to temporary unique file
 * (2) change file name
 *
 */
errno_t save_fits_atomic(
    const char *restrict ID_name,
    const char *restrict file_name
)
{
    imageID ID;
    uint8_t datatype;
    char fnametmp[STRINGMAXLEN_FILENAME];
    char savename[STRINGMAXLEN_FILENAME];
    pthread_t self_id;

    ID = image_ID(ID_name);

    // get PID to include in file name, so that file name is unique
    self_id = pthread_self();

    WRITE_FILENAME(fnametmp, "_savefits_atomic_%s_%d_%ld.tmp.fits", ID_name,
                   (int) getpid(), (long) self_id);
    WRITE_FILENAME(savename, "%s", fnametmp);

    if(ID != -1)
    {
        datatype = data.image[ID].md[0].datatype;
        switch(datatype)
        {

            case _DATATYPE_UINT8:
                save_ush16_fits(ID_name, savename);
                break;

            case _DATATYPE_INT8:
                save_sh16_fits(ID_name, savename);
                break;

            case _DATATYPE_UINT16:
                save_ush16_fits(ID_name, savename);
                break;

            case _DATATYPE_INT16:
                save_sh16_fits(ID_name, savename);
                break;

            case _DATATYPE_UINT32:
                save_uint32_fits(ID_name, savename);
                break;

            case _DATATYPE_INT32:
                save_int32_fits(ID_name, savename);
                break;

            case _DATATYPE_UINT64:
                save_int64_fits(ID_name, savename);
                break;

            case _DATATYPE_INT64:
                save_int64_fits(ID_name, savename);
                break;

            case _DATATYPE_FLOAT:
                save_fl_fits(ID_name, savename);
                break;

            case _DATATYPE_DOUBLE:
                save_db_fits(ID_name, savename);
                break;
        }

        EXECUTE_SYSTEM_COMMAND("mv %s %s", fnametmp, file_name);
    }

    return RETURN_SUCCESS;
}



errno_t saveall_fits(
    const char *restrict savedirname
)
{
    char fname[STRINGMAXLEN_FULLFILENAME];

    EXECUTE_SYSTEM_COMMAND("mkdir -p %s", savedirname);

    for(long i = 0; i < data.NB_MAX_IMAGE; i++)
        if(data.image[i].used == 1)
        {

            WRITE_FULLFILENAME(fname, "./%s/%s.fits", savedirname, data.image[i].name);
            save_fits(data.image[i].name, fname);
        }

    return RETURN_SUCCESS;
}






