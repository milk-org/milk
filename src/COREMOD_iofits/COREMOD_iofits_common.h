#ifndef _IOFITS_COMMON_H
#define _IOFITS_COMMON_H

#include <fitsio.h>

typedef struct
{
    int FITSIO_status;
} COREMOD_IOFITS_DATA;

#define STRINGMAXLEN_FITSKEYWORDNAME  8
#define STRINGMAXLEN_FITSKEYWORDVALUE 68
#define STRINGMAXLEN_FITSKEYWCOMMENT  FLEN_COMMENT

#define STRINGMAXLEN_FITSIOCHECK_ERRSTRING 100

/**
 * @ingroup errcheckmacro
 * @brief Write FITS keyword to string
 *
 * Requires existing image string of len #STRINGMAXLEN_FITSKEYWORD
 *
 * Example use:
 * @code
 * char keywname[STRINGMAXLEN_FITSKEYWORD];
 * char name[]="CHAN";
 * int imindex = 34;
 * WRITE_FITSKEYWNAME(keywname, "%s_%04d", name, imindex);
 * @endcode
 *
 *
 */
#define WRITE_FITSKEYWNAME(keywname, ...)                                      \
    do                                                                         \
    {                                                                          \
        int slen =                                                             \
            snprintf(keywname, STRINGMAXLEN_FITSKEYWORDNAME, __VA_ARGS__);     \
        if (slen < 1)                                                          \
        {                                                                      \
            PRINT_ERROR("snprintf wrote <1 char");                             \
            abort();                                                           \
        }                                                                      \
        if (slen >= STRINGMAXLEN_FITSKEYWORDNAME)                              \
        {                                                                      \
            PRINT_ERROR("snprintf string truncation");                         \
            abort();                                                           \
        }                                                                      \
    } while (0)

#define FITSIO_CHECK_ERROR(status, errmode, ...)                               \
    do                                                                         \
    {                                                                          \
        if ((status != 0) && (errmode != 0))                                   \
        {                                                                      \
            char errstr[STRINGMAXLEN_FITSIOCHECK_ERRSTRING];                   \
            fits_get_errstatus(status, errstr);                                \
            fprintf(stderr,                                                    \
                    "%c[%d;%dmFITSIO error %d [%s, %s, %d]: "                  \
                    "%s%c[%d;m\n\a",                                           \
                    (char) 27,                                                 \
                    1,                                                         \
                    31,                                                        \
                    status,                                                    \
                    __FILE__,                                                  \
                    __func__,                                                  \
                    __LINE__,                                                  \
                    errstr,                                                    \
                    (char) 27,                                                 \
                    0);                                                        \
            PRINT_ERROR(__VA_ARGS__);                                          \
            if (errmode > 2)                                                   \
            {                                                                  \
                abort();                                                       \
            }                                                                  \
        }                                                                      \
        status = 0;                                                            \
    } while (0)

#endif
