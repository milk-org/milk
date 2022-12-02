/**
 * @file    is_fits_file.c
 */

#include "COREMOD_iofits_common.h"
#include "CommandLineInterface/CLIcore.h"
#include "check_fitsio_status.h"

extern COREMOD_IOFITS_DATA COREMOD_iofits_data;

int is_fits_file(const char *restrict file_name)
{
    int       value = 0;

    // check if string contains FITS or fits
    int fnameOK = 0;
    if(strstr(file_name, ".fits") != NULL)
    {
        fnameOK = 1;
    }
    if(strstr(file_name, ".FITS") != NULL)
    {
        fnameOK = 1;
    }

    if(fnameOK == 0)
    {
        value = 0;
    }
    else
    {
        fitsfile *fptr;

        EXECUTE_SYSTEM_COMMAND("cat fitscheck.%s", file_name);

        if(!fits_open_file(&fptr,
                           file_name,
                           READONLY,
                           &COREMOD_iofits_data.FITSIO_status))
        {
            fits_close_file(fptr, &COREMOD_iofits_data.FITSIO_status);
            value = 1;
        }
        if(check_FITSIO_status(__FILE__, __func__, __LINE__, 1) == 1)
        {
            PRINT_ERROR("Error in function is_fits_file(%s)", file_name);
        }
    }

    return (value);
}
