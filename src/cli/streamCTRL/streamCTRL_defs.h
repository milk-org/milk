// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file streamCTRL_standalone_defs.h
 * @brief Standalone replacements for CLIcore macros and helpers
 *
 * When milk-streamCTRL is built as a standalone executable (without CLIcore),
 * this header provides:
 *   - String length constants matching CLIcore's STRINGMAXLEN_* values
 *   - RETURN_SUCCESS / RETURN_FAILURE
 *   - SHAREDSHMDIR  — from $MILK_SHM_DIR env var or /dev/shm fallback
 *   - PRINT_ERROR   — thin stderr wrapper
 *   - DEBUG_TRACEPOINT — no-op (or stderr if VERBOSE defined)
 *   - EXECUTE_SYSTEM_COMMAND_NOCHECK — inline system() wrapper
 *   - WRITE_FULLFILENAME — inline snprintf wrapper
 *   - errno_t / imageID types if not already defined
 */

#ifndef _STREAMCTRL_DEFS_H
#define _STREAMCTRL_DEFS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>
#include <time.h>

/* CLOCK_MILK — use ImageStreamIO's CLOCK_ISIO via the same alias */
#include "ImageStreamIO/ImageStreamIO.h"
#include "processtools.h"
#include "processtools_trigger.h"
#include "quicksort.h"
#ifndef CLOCK_MILK
#    define CLOCK_MILK CLOCK_ISIO
#endif

/* timespec_diff - compute end - start, borrowed from libprocessinfo */
static inline struct timespec sc_timespec_diff(struct timespec start, struct timespec end)
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

/*
 * Alias timespec_diff to the local inline implementation so that
 * scan.c continues to compile without linking libprocessinfo.
 */
#ifndef timespec_diff
#    define timespec_diff(s, e) sc_timespec_diff((s), (e))
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
 * (kept in sync with CLIcore's STRINGMAXLEN_* values)
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

/* =========================================================
 * Shared memory directory
 * ========================================================= */

/**
 * streamctrl_get_shmdir - return the SHM directory path.
 *
 * Reads $MILK_SHM_DIR if set, otherwise returns "/dev/shm".
 */
static inline const char *streamctrl_get_shmdir(void)
{
    const char *d = getenv("MILK_SHM_DIR");
    if (d != NULL && d[0] != '\0')
    {
        return d;
    }
    return "/dev/shm";
}

/* SHAREDSHMDIR expands to a run-time string expression */
#define SHAREDSHMDIR (streamctrl_get_shmdir())

/* =========================================================
 * Error printing
 * ========================================================= */

#define PRINT_ERROR(fmt, ...) \
    fprintf(stderr, "ERROR [%s %s %d]: " fmt "\n", __FILE__, __func__, __LINE__, ##__VA_ARGS__)

/* =========================================================
 * Debug tracing — no-op unless STREAMCTRL_VERBOSE is defined
 * ========================================================= */

#ifdef STREAMCTRL_VERBOSE
#    define DEBUG_TRACEPOINT(fmt, ...) \
        fprintf(stderr, "TRACE [%s:%d] " fmt "\n", __func__, __LINE__, ##__VA_ARGS__)
#    define DEBUG_TRACE_FSTART() fprintf(stderr, "FSTART [%s]\n", __func__)
#    define DEBUG_TRACE_FEXIT() fprintf(stderr, "FEXIT [%s]\n", __func__)
#else
#    define DEBUG_TRACEPOINT(fmt, ...) \
        do                             \
        {                              \
        } while (0)
#    define DEBUG_TRACE_FSTART() \
        do                       \
        {                        \
        } while (0)
#    define DEBUG_TRACE_FEXIT() \
        do                      \
        {                       \
        } while (0)
#endif

/* =========================================================
 * System command wrapper
 * ========================================================= */

/**
 * EXECUTE_SYSTEM_COMMAND_NOCHECK - build and run a shell command.
 *
 * Formats the command with snprintf, then passes it to system().
 * On failure, prints a warning but does not abort.
 */
#define EXECUTE_SYSTEM_COMMAND_NOCHECK(fmt, ...)                                      \
    do                                                                                \
    {                                                                                 \
        char _escmd_buf[STRINGMAXLEN_COMMAND];                                        \
        int  _escmd_n = snprintf(_escmd_buf, sizeof(_escmd_buf), fmt, ##__VA_ARGS__); \
        if (_escmd_n > 0 && _escmd_n < STRINGMAXLEN_COMMAND)                          \
        {                                                                             \
            if (system(_escmd_buf) == -1)                                             \
            {                                                                         \
                PRINT_ERROR("system() failed: %s", _escmd_buf);                       \
            }                                                                         \
        }                                                                             \
    } while (0)

/* =========================================================
 * Full filename writer
 * ========================================================= */

/**
 * WRITE_FULLFILENAME - safe snprintf into a fixed-size buffer.
 *
 * Writes the formatted path into buf.  On truncation, prints
 * an error.  The first argument must be a char array (not a
 * pointer), so sizeof() gives the correct size.
 */
#define WRITE_FULLFILENAME(buf, fmt, ...)                              \
    do                                                                 \
    {                                                                  \
        int _wff_n = snprintf((buf), sizeof(buf), fmt, ##__VA_ARGS__); \
        if (_wff_n < 1)                                                \
        {                                                              \
            PRINT_ERROR("snprintf wrote < 1 char");                    \
        }                                                              \
        else if (_wff_n >= (int) sizeof(buf))                          \
        {                                                              \
            PRINT_ERROR("snprintf: path truncated");                   \
        }                                                              \
    } while (0)

/* =========================================================
 * Signal flags (minimal subset used by streamCTRL_TUI_ansi.c)
 * ========================================================= */

extern volatile sig_atomic_t sc_sigINT;
extern volatile sig_atomic_t sc_sigTERM;

/* Check if any termination signal has been set */
#define SC_SIG_ANY_SET() (sc_sigINT || sc_sigTERM)

#endif /* _STREAMCTRL_DEFS_H */
