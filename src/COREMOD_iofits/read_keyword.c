/**
 * @file    read_keyword.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_iofits_common.h"
#include "check_fitsio_status.h"

extern COREMOD_IOFITS_DATA COREMOD_iofits_data;

int read_keyword(const char *restrict file_name,
                 const char *restrict KEYWORD,
                 char *restrict content)
{
    fitsfile *fptr; /* FITS file pointer, defined in fitsio.h */
    int       exists = 0;
    int       n;

    if(!fits_open_file(&fptr,
                       file_name,
                       READONLY,
                       &COREMOD_iofits_data.FITSIO_status))
    {
        char comment[STRINGMAXLEN_FITSKEYWCOMMENT];
        char str1[STRINGMAXLEN_FITSKEYWORDVALUE];

        if(fits_read_keyword(fptr,
                             KEYWORD,
                             str1,
                             comment,
                             &COREMOD_iofits_data.FITSIO_status))
        {
            PRINT_ERROR("Keyword \"%s\" does not exist in file \"%s\"",
                        KEYWORD,
                        file_name);
            exists = 1;
        }
        else
        {
            n = snprintf(content, STRINGMAXLEN_FITSKEYWORDVALUE, "%s\n", str1);
            if(n >= STRINGMAXLEN_FITSKEYWORDVALUE)
            {
                PRINT_ERROR(
                    "Attempted to write string buffer with too "
                    "many characters");
            }
        }
        fits_close_file(fptr, &COREMOD_iofits_data.FITSIO_status);
    }
    if(check_FITSIO_status(__FILE__, __func__, __LINE__, 0) == 1)
    {
        PRINT_ERROR("Error reading keyword \"%s\" in file \"%s\"",
                    KEYWORD,
                    file_name);
    }

    return (exists);
}

errno_t read_keyword_alone(const char *restrict file_name,
                           const char *restrict KEYWORD)
{
    char *content =
        (char *) malloc(sizeof(char) * STRINGMAXLEN_FITSKEYWORDVALUE);

    if(content == NULL)
    {
        PRINT_ERROR("malloc error");
        exit(0);
    }

    read_keyword(file_name, KEYWORD, content);
    printf("%s\n", content);

    free(content);
    content = NULL;

    return RETURN_SUCCESS;
}
