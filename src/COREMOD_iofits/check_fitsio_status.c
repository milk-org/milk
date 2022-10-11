/**
 * @file    ckeck_fitsio_status.c
 */

#include "COREMOD_iofits_common.h"

extern COREMOD_IOFITS_DATA COREMOD_iofits_data;

// set print to 0 if error message should not be printed to stderr
// set print to 1 if error message should be printed to stderr
int check_FITSIO_status(const char *restrict cfile,
                        const char *restrict cfunc,
                        long cline,
                        int  print)
{
    int Ferr = 0;

    if(COREMOD_iofits_data.FITSIO_status != 0)
    {
        if(print == 1)
        {
            char errstr[STRINGMAXLEN_FITSIOCHECK_ERRSTRING];
            fits_get_errstatus(COREMOD_iofits_data.FITSIO_status, errstr);
            fprintf(stderr,
                    "%c[%d;%dmFITSIO error %d [%s, %s, %ld]: %s%c[%d;m\n\a",
                    (char) 27,
                    1,
                    31,
                    COREMOD_iofits_data.FITSIO_status,
                    cfile,
                    cfunc,
                    cline,
                    errstr,
                    (char) 27,
                    0);
        }
        Ferr = COREMOD_iofits_data.FITSIO_status;
    }
    COREMOD_iofits_data.FITSIO_status = 0;

    return (Ferr);
}
