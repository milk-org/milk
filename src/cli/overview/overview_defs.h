/**
 * @file overview_defs.h
 * @brief Standalone definitions for milk-CTRL TUI
 *
 * Provides all standalone macros, types, and helpers
 * needed by milk-CTRL without depending on CLIcore.
 * Mirrors the streamCTRL_defs.h pattern.
 */

#ifndef OVERVIEW_DEFS_H
#define OVERVIEW_DEFS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>
#include <time.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "processtools.h"
#include "processtools_trigger.h"

/* CLOCK_MILK — use ImageStreamIO's CLOCK_ISIO */
#ifndef CLOCK_MILK
#    define CLOCK_MILK CLOCK_ISIO
#endif

/* =========================================================
 * Basic types
 * ========================================================= */

#ifndef __STDC_LIB_EXT1__
typedef int errno_t;
#endif

typedef long imageID;

/* =========================================================
 * Return codes
 * ========================================================= */

#ifndef RETURN_SUCCESS
#    define RETURN_SUCCESS 0
#endif

#ifndef RETURN_FAILURE
#    define RETURN_FAILURE 1
#endif

/* =========================================================
 * String length constants
 * ========================================================= */

#ifndef STRINGMAXLEN_DEFAULT
#    define STRINGMAXLEN_DEFAULT 1000
#endif

#ifndef STRINGMAXLEN_COMMAND
#    define STRINGMAXLEN_COMMAND 2000
#endif

#ifndef STRINGMAXLEN_FULLFILENAME
#    define STRINGMAXLEN_FULLFILENAME 2000
#endif

#ifndef STRINGMAXLEN_IMAGE_NAME
#    define STRINGMAXLEN_IMAGE_NAME 80
#endif

#ifndef STRINGMAXLEN_DIRNAME
#    define STRINGMAXLEN_DIRNAME 200
#endif

/* =========================================================
 * Shared memory directory
 * ========================================================= */

/**
 * ov_get_shmdir - return the SHM directory path.
 *
 * Reads $MILK_SHM_DIR if set, otherwise returns "/dev/shm".
 */
static inline const char *ov_get_shmdir(void)
{
    const char *d = getenv("MILK_SHM_DIR");
    if (d != NULL && d[0] != '\0')
    {
        return d;
    }
    return "/dev/shm";
}

#define SHAREDSHMDIR (ov_get_shmdir())

/* =========================================================
 * Error printing
 * ========================================================= */

#define PRINT_ERROR(fmt, ...) \
    fprintf(stderr, "ERROR [%s %s %d]: " fmt "\n", __FILE__, __func__, __LINE__, ##__VA_ARGS__)

/* =========================================================
 * Debug tracing — no-op unless OVERVIEW_VERBOSE defined
 * ========================================================= */

#ifdef OVERVIEW_VERBOSE
#    define DEBUG_TRACEPOINT(fmt, ...) \
        fprintf(stderr, "TRACE [%s:%d] " fmt "\n", __func__, __LINE__, ##__VA_ARGS__)
#else
#    define DEBUG_TRACEPOINT(fmt, ...) \
        do                             \
        {                              \
        } while (0)
#endif

/* =========================================================
 * Timespec diff helper
 * ========================================================= */

static inline struct timespec ov_timespec_diff(struct timespec start, struct timespec end)
{
    struct timespec result;
    if ((end.tv_nsec - start.tv_nsec) < 0)
    {
        result.tv_sec  = end.tv_sec - start.tv_sec - 1;
        result.tv_nsec = end.tv_nsec - start.tv_nsec + 1000000000L;
    }
    else
    {
        result.tv_sec  = end.tv_sec - start.tv_sec;
        result.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return result;
}

/* Use ov_timespec_diff() directly; do not redefine
 * timespec_diff — it conflicts with timeutils.h */

/* =========================================================
 * Signal flags
 * ========================================================= */

extern volatile sig_atomic_t ov_sigINT;
extern volatile sig_atomic_t ov_sigTERM;

#define OV_SIG_ANY_SET() (ov_sigINT || ov_sigTERM)

#endif /* OVERVIEW_DEFS_H */
