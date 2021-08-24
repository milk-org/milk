/**
 * @file    savefits.c
 * @brief   save FITS format files
 */


#include "CommandLineInterface/CLIcore.h"


#include <pthread.h>

// Handle old fitsios
#ifndef ULONGLONG_IMG
#define ULONGLONG_IMG (80)
#endif

#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_iofits_common.h"
#include "check_fitsio_status.h"
#include "file_exists.h"
#include "is_fits_file.h"

extern COREMOD_IOFITS_DATA COREMOD_iofits_data;





// variables local to this translation unit
static char *inimname;
static char *outfname;
static int *outbitpix;
static char *inheader; // import header from this file

// CLI function arguments and parameters
static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG, ".in_name", "input image", "im1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname
    },
    {
        CLIARG_STR, ".out_fname", "output FITS file name", "out.fits",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outfname
    },
    {
        // non-CLI parameter
        CLIARG_LONG, ".bitpix",
        "0: auto\n"
        "8 /(10) : (un)sig   8-b int\n"
        "16/(20) 32/(40) 64/(80) : (un)sig int\n"
        "-32/-64 : 32/64-b flt\n"
        , "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &outbitpix
    },
    {
        CLIARG_STR, ".in_header", "header import from this FITS file", "",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &inheader
    }
};


// CLI function initialization data
static CLICMDDATA CLIcmddata =
{
    "saveFITS",
    "save image as FITS",
    CLICMD_FIELDS_DEFAULTS
};



// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}









errno_t saveFITS(
    const char *__restrict inputimname,
    const char *__restrict outputFITSname,
    int outputbitpix,
    const char *__restrict importheaderfile
)
{
    DEBUG_TRACE_FSTART();
    DEBUG_TRACEPOINT("Saving image %s to file %s, bitpix = %d",
           inputimname,
           outputFITSname,
           outputbitpix);


    COREMOD_iofits_data.FITSIO_status = 0;

    // get PID to include in file name, so that file name is unique
    pthread_t self_id = pthread_self();

    char fnametmp[STRINGMAXLEN_FILENAME];

    WRITE_FILENAME(fnametmp,
                   "_savefits_atomic_%s_%d_%ld.tmp.fits",
                   inputimname,
                   (int) getpid(),
                   (long) self_id);



    IMGID imgin = makeIMGID(inputimname);
    resolveIMGID(&imgin, ERRMODE_WARN);
    if(imgin.ID == -1)
    {
        PRINT_WARNING("Image %s does not exist in memory - cannot save to FITS",
                      inputimname);
        DEBUG_TRACE_FEXIT();
        return RETURN_SUCCESS;
    }


    // data types
    uint8_t datatype = imgin.md->datatype;
    int FITSIOdatatype = TFLOAT;
    int bitpix = FLOAT_IMG;

    char *datainptr;

    DEBUG_TRACEPOINT("datatype = %d", (int) datatype);
    switch(datatype)
    {
    case _DATATYPE_UINT8:
        FITSIOdatatype = TBYTE;
        bitpix = BYTE_IMG;
        datainptr = (char *) imgin.im->array.UI8;
        break;

    case _DATATYPE_INT8:
        FITSIOdatatype = TSBYTE;
        bitpix = SBYTE_IMG;
        datainptr = (char *) imgin.im->array.SI8;
        break;


    case _DATATYPE_UINT16:
        FITSIOdatatype = TUSHORT;
        bitpix = SHORT_IMG;
        datainptr = (char *) imgin.im->array.UI16;
        break;

    case _DATATYPE_INT16:
        FITSIOdatatype = TUSHORT;
        bitpix = SHORT_IMG;
        datainptr = (char *) imgin.im->array.SI16;
        break;


    case _DATATYPE_UINT32:
        FITSIOdatatype = TUINT;
        bitpix = ULONG_IMG;
        datainptr = (char *) imgin.im->array.UI32;
        break;

    case _DATATYPE_INT32:
        FITSIOdatatype = TINT;
        bitpix = LONG_IMG;
        datainptr = (char *) imgin.im->array.SI32;
        break;


    case _DATATYPE_UINT64:
        FITSIOdatatype = TULONG;
        bitpix = ULONGLONG_IMG;
        datainptr = (char *) imgin.im->array.UI64;
        break;

    case _DATATYPE_INT64:
        FITSIOdatatype = TLONG;
        bitpix = LONGLONG_IMG;
        datainptr = (char *) imgin.im->array.SI64;
        break;


    case _DATATYPE_FLOAT:
        FITSIOdatatype = TFLOAT;
        bitpix = FLOAT_IMG;
        datainptr = (char *) imgin.im->array.F;
        break;

    case _DATATYPE_DOUBLE:
        FITSIOdatatype = TDOUBLE;
        bitpix = DOUBLE_IMG;
        datainptr = (char *) imgin.im->array.D;
        break;
    }

    DEBUG_TRACEPOINT("bitpix = %d", bitpix);

    switch(outputbitpix)
    {
    case 8:
        bitpix = BYTE_IMG;
        printf("    output data type: BYTE_IMG\n");
        break;
    case 10:
        bitpix = SBYTE_IMG;
        printf("    output data type: SBYTE_IMG\n");
        break;

    case 16:
        bitpix = SHORT_IMG;
        printf("    output data type: SHORT_IMG\n");
        break;
    case 20:
        bitpix = USHORT_IMG;
        printf("    output data type: USHORT_IMG\n");
        break;

    case 32:
        bitpix = LONG_IMG;
        printf("    output data type: LONG_IMG\n");
        break;
    case 40:
        bitpix = ULONG_IMG;
        printf("    output data type: ULONG_IMG\n");
        break;

    case 64:
        bitpix = LONGLONG_IMG;
        printf("    output data type: LONGLONG_IMG\n");
        break;
    case 80:
        bitpix = ULONGLONG_IMG;
        printf("    output data type: ULONGLONG_IMG\n");
        break;

    case -32:
        bitpix = FLOAT_IMG;
        printf("    output data type: FLOAT_IMG\n");
        break;
    case -64:
        bitpix = DOUBLE_IMG;
        printf("    output data type: DOUBLE_IMG\n");
        break;
    }

    DEBUG_TRACEPOINT("bitpix = %d", bitpix);


    fitsfile *fptr;
    COREMOD_iofits_data.FITSIO_status = 0;
    DEBUG_TRACEPOINT("creating FITS file %s", fnametmp);
    fits_create_file(&fptr, fnametmp, &COREMOD_iofits_data.FITSIO_status);
    DEBUG_TRACEPOINT(" ");

    if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
    {
        char errstring[200];
        if( access( fnametmp, F_OK ) == 0 ) {
            sprintf(errstring, "File already exists");
        }
        PRINT_ERROR("fits_create_file error %d on file %s %s", COREMOD_iofits_data.FITSIO_status, fnametmp, errstring);
        abort();
    }





    int naxis = imgin.md->naxis;
    long naxesl[3];
    for(int i = 0; i < naxis; i++)
    {
        naxesl[i] = (long) imgin.md->size[i];
    }

    long nelements = 1;
    for(int i = 0; i < naxis; i++)
    {
        nelements *= naxesl[i];
    }

    COREMOD_iofits_data.FITSIO_status = 0;
    fits_create_img(fptr, bitpix, naxis, naxesl,
                    &COREMOD_iofits_data.FITSIO_status);
    if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
    {
        PRINT_ERROR("fits_create_img error on file %s", fnametmp);
        EXECUTE_SYSTEM_COMMAND("rm %s", fnametmp);
        FUNC_RETURN_FAILURE(" ");
    }

    DEBUG_TRACEPOINT(" ");

    long fpixel = 1;
    COREMOD_iofits_data.FITSIO_status = 0;
    fits_write_img(fptr, FITSIOdatatype, fpixel, nelements, datainptr,
                   &COREMOD_iofits_data.FITSIO_status);
    int errcode = check_FITSIO_status(__FILE__, __func__, __LINE__, 1);
    if(errcode != 0)
    {
        if(errcode == 412)
        {
            PRINT_WARNING("data trucated");
        }
        else
        {
            PRINT_ERROR("fits_write_img error %d on file %s", errcode, fnametmp);
            EXECUTE_SYSTEM_COMMAND("rm %s", fnametmp);
            FUNC_RETURN_FAILURE(" ");
        }
    }

    COREMOD_iofits_data.FITSIO_status = 0;
    fits_write_date(fptr, &COREMOD_iofits_data.FITSIO_status);

    DEBUG_TRACEPOINT("Adding optional header");
    // HEADER


    // Add FITS keywords from image keywords
    {
        int NBkw = imgin.md->NBkw;
        int kwcnt = 0;
        for(int kw = 0; kw < NBkw; kw++)
        {
            char tmpkwvalstr[81];
            switch(imgin.im->kw[kw].type)
            {
            case 'L':
                printf("writing keyword [L] %-8s= %20ld / %s\n", imgin.im->kw[kw].name,
                       imgin.im->kw[kw].value.numl, imgin.im->kw[kw].comment);
                COREMOD_iofits_data.FITSIO_status = 0;
                fits_update_key(fptr, TLONG, imgin.im->kw[kw].name,
                                &imgin.im->kw[kw].value.numl, imgin.im->kw[kw].comment,
                                &COREMOD_iofits_data.FITSIO_status);
                kwcnt++;
                break;

            case 'D':
                printf("writing keyword [D] %-8s= %20g / %s\n", imgin.im->kw[kw].name,
                       imgin.im->kw[kw].value.numf, imgin.im->kw[kw].comment);
                COREMOD_iofits_data.FITSIO_status = 0;
                fits_update_key(fptr, TDOUBLE, imgin.im->kw[kw].name,
                                &imgin.im->kw[kw].value.numf, imgin.im->kw[kw].comment,
                                &COREMOD_iofits_data.FITSIO_status);
                kwcnt++;
                break;

            case 'S':
                sprintf(tmpkwvalstr, "'%s'", imgin.im->kw[kw].value.valstr);
                printf("writing keyword [S] %-8s= %20s / %s\n", imgin.im->kw[kw].name,
                       tmpkwvalstr, imgin.im->kw[kw].comment);
                COREMOD_iofits_data.FITSIO_status = 0;
                fits_update_key(fptr, TSTRING, imgin.im->kw[kw].name,
                                imgin.im->kw[kw].value.valstr, imgin.im->kw[kw].comment,
                                &COREMOD_iofits_data.FITSIO_status);
                kwcnt++;
                break;

            default:
                break;
            }

            if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
            {
                PRINT_ERROR("fits_write_record error on keyword %s", imgin.im->kw[kw].name);
                abort();
            }
        }
    }

    // Add FITS keywords from importheaderfile (optional)
    if(strlen(importheaderfile) > 0)
    {
        if(is_fits_file(importheaderfile) == 1)
        {
            printf("Importing FITS header entries from : %s\n", importheaderfile);

            fitsfile *fptr_header = NULL;
            int nkeys;

            char *header;

            COREMOD_iofits_data.FITSIO_status = 0;
            fits_open_file(&fptr_header, importheaderfile, READONLY,
                           &COREMOD_iofits_data.FITSIO_status);
            if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
            {
                PRINT_ERROR("fits_open_file error on file %s", importheaderfile);
                abort();
            }


            COREMOD_iofits_data.FITSIO_status = 0;
            fits_hdr2str(fptr_header, 1, NULL, 0, &header, &nkeys,
                         &COREMOD_iofits_data.FITSIO_status);
            if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
            {
                PRINT_ERROR("fits_hdr2str erroron file %s", importheaderfile);
                abort();
            }
            printf("imported %d header cards\n", nkeys);

            char *hptr; // pointer to header
            hptr = header;
            while(*hptr)
            {
                char fitscard[81];
                sprintf(fitscard, "%.80s", hptr);

                // keywords to not overwrite
                int writecard = 1;
                char *keyexcl[] = {"BITPIX", "NAXIS", "SIMPLE", "EXTEND", 0};
                int ki = 0;
                while(keyexcl[ki])
                {
                    if(strncmp(keyexcl[ki], fitscard, strlen(keyexcl[ki])) == 0)
                    {
                        printf("EXCLUDING %s\n", fitscard);
                        writecard = 0;
                        break;
                    }
                    ki++;
                }

                if(writecard == 1)
                {
                    COREMOD_iofits_data.FITSIO_status = 0;
                    fits_write_record(fptr, fitscard, &COREMOD_iofits_data.FITSIO_status);
                    if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
                    {
                        PRINT_ERROR("fits_write_record error on file %s", importheaderfile);
                        abort();
                    }
                }
                hptr += 80;
            }

            COREMOD_iofits_data.FITSIO_status = 0;
            fits_free_memory(header, &COREMOD_iofits_data.FITSIO_status);
            if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
            {
                PRINT_ERROR("fits_free_memory error on file %s", importheaderfile);
                abort();
            }

            COREMOD_iofits_data.FITSIO_status = 0;
            fits_close_file(fptr_header, &COREMOD_iofits_data.FITSIO_status);
            if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
            {
                PRINT_ERROR("fits_close_file error on file %s", importheaderfile);
                abort();
            }
        }
    }






    COREMOD_iofits_data.FITSIO_status = 0;
    fits_close_file(fptr, &COREMOD_iofits_data.FITSIO_status);
    if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) != 0)
    {
        PRINT_ERROR("fits_close_file error on file %s", fnametmp);
        EXECUTE_SYSTEM_COMMAND("rm %s", fnametmp);
        FUNC_RETURN_FAILURE(" ");
    }

    EXECUTE_SYSTEM_COMMAND_ERRCHECK("mv %s %s", fnametmp, outputFITSname);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



errno_t saveall_fits(
    const char *__restrict savedirname
)
{
    DEBUG_TRACE_FSTART();
    char fname[STRINGMAXLEN_FULLFILENAME];

    EXECUTE_SYSTEM_COMMAND("mkdir -p %s", savedirname);

    for(long i = 0; i < data.NB_MAX_IMAGE; i++)
        if(data.image[i].used == 1)
        {

            WRITE_FULLFILENAME(fname, "./%s/%s.fits", savedirname, data.image[i].name);
            saveFITS(data.image[i].name, fname, 0, "");
        }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




errno_t save_fits(
    const char *__restrict savedirname,
    const char *__restrict outputFITSname
)
{
    DEBUG_TRACE_FSTART();

    FUNC_CHECK_RETURN(
        saveFITS(savedirname, outputFITSname, 0, "")
    );

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


errno_t save_fl_fits(
    const char *__restrict savedirname,
    const char *__restrict outputFITSname
)
{
    DEBUG_TRACE_FSTART();

    FUNC_CHECK_RETURN(
        saveFITS(savedirname, outputFITSname, -32, ""));

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


errno_t save_db_fits(
    const char *__restrict savedirname,
    const char *__restrict outputFITSname
)
{
    DEBUG_TRACE_FSTART();

    FUNC_CHECK_RETURN(
        saveFITS(savedirname, outputFITSname, -64, ""));

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}






static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    saveFITS(
        inimname,
        outfname,
        *outbitpix,
        inheader
    );

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t CLIADDCMD_COREMOD_iofits__saveFITS()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
