// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    milkDebugTools.h
 *
 * Error handling and checking
 *
 */

#ifndef _MILKDEBUGTOOLS_H
#define _MILKDEBUGTOOLS_H

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

#include "milk_compiler.h"

// define (custom) types for function return value

#ifndef __STDC_LIB_EXT1__
typedef int errno_t;
#endif

// Basic types used throughout the project
#ifndef _IMGID_H
typedef long imageID;
typedef long variableID;
#endif

// basic success/failure codes
#ifndef RETURN_SUCCESS
#    define RETURN_SUCCESS 0
#endif
#ifndef RETURN_FAILURE
#    define RETURN_FAILURE 1
#endif

// error mode
// defines function behavior on error
#define ERRMODE_NULL 0  // ignore error
#define ERRMODE_WARN 1  // issue warning and continue
#define ERRMODE_FAIL 2  // fail nicely if possible
#define ERRMODE_ABORT 3 // abort process

#define STRINGMAXLEN_DEFAULT 1000
#define STRINGMAXLEN_ERRORMSG 1000
#define STRINGMAXLEN_FILENAME 200
#define STRINGMAXLEN_DIRNAME 800
#define STRINGMAXLEN_FULLFILENAME 1000
#define STRINGMAXLEN_FUNCTIONNAME 200
#define STRINGMAXLEN_FUNCTIONARGS 1000

#define STRINGMAXLEN_CLICMDLINE 1000
#define STRINGMAXLEN_STREAMNAME 100
#define STRINGMAXLEN_IMGNAME STRINGMAXLEN_STREAMNAME
#define STRINGMAXLEN_FPSPROCESSTYPE 64
#define STRINGMAXLEN_SHMDIRNAME 200
#define STRINGMAXLEN_PROCESSNAME 100
#define STRINGMAXLEN_COMMAND 2048

#ifndef CLOCK_MILK
#    define CLOCK_MILK CLOCK_REALTIME
#endif

/**
 * @brief Print error and continue
 * @ingroup errcheckmacro
 */
#define PRINT_ERROR(format, ...)                                                        \
    do                                                                                  \
    {                                                                                   \
        fprintf(stderr, "ERROR [%s:%d %s]: " format "\n", __FILE__, __LINE__, __func__, \
                ##__VA_ARGS__);                                                         \
    } while (0)

/**
 * @brief Print warning and continue
 * @ingroup errcheckmacro
 */
#define PRINT_WARNING(format, ...)                                                        \
    do                                                                                    \
    {                                                                                     \
        fprintf(stderr, "WARNING [%s:%d %s]: " format "\n", __FILE__, __LINE__, __func__, \
                ##__VA_ARGS__);                                                           \
    } while (0)

/**
 * @brief Print info
 * @ingroup errcheckmacro
 */
#define PRINT_INFO(format, ...)                               \
    do                                                        \
    {                                                         \
        fprintf(stdout, "INFO: " format "\n", ##__VA_ARGS__); \
    } while (0)

// Helper for function return failure
#define FUNC_RETURN_FAILURE(format, ...)    \
    do                                      \
    {                                       \
        PRINT_ERROR(format, ##__VA_ARGS__); \
        return RETURN_FAILURE;              \
    } while (0)

// Helper for function return success
#define FUNC_RETURN_SUCCESS()  \
    do                         \
    {                          \
        return RETURN_SUCCESS; \
    } while (0)

// Check return value and return if failure
#define FUNC_CHECK_RETURN(ret)      \
    do                              \
    {                               \
        errno_t _ret = (ret);       \
        if (_ret != RETURN_SUCCESS) \
        {                           \
            return _ret;            \
        }                           \
    } while (0)

// Check return value and print error if failure
#define FUNC_CHECK_RETURN_PRINT(ret, format, ...) \
    do                                            \
    {                                             \
        errno_t _ret = (ret);                     \
        if (_ret != RETURN_SUCCESS)               \
        {                                         \
            PRINT_ERROR(format, ##__VA_ARGS__);   \
            return _ret;                          \
        }                                         \
    } while (0)

#define EXECUTE_SYSTEM_COMMAND_NOCHECK(format, ...)                              \
    do                                                                           \
    {                                                                            \
        char syscommandstring[STRINGMAXLEN_COMMAND];                             \
        snprintf(syscommandstring, STRINGMAXLEN_COMMAND, format, ##__VA_ARGS__); \
        int _ret = system(syscommandstring);                                     \
        (void) _ret;                                                             \
    } while (0)

#define EXECUTE_SYSTEM_COMMAND(format, ...)                                                \
    do                                                                                     \
    {                                                                                      \
        char syscommandstring[STRINGMAXLEN_COMMAND];                                       \
        snprintf(syscommandstring, STRINGMAXLEN_COMMAND, format, ##__VA_ARGS__);           \
        int _ret = system(syscommandstring);                                               \
        if (_ret != 0)                                                                     \
        {                                                                                  \
            PRINT_ERROR("System command failed with code %d: %s", _ret, syscommandstring); \
        }                                                                                  \
    } while (0)

#define WRITE_FILENAME(string, format, ...)                             \
    do                                                                  \
    {                                                                   \
        snprintf(string, STRINGMAXLEN_FILENAME, format, ##__VA_ARGS__); \
    } while (0)

#define WRITE_FULLFILENAME(string, format, ...)                             \
    do                                                                      \
    {                                                                       \
        snprintf(string, STRINGMAXLEN_FULLFILENAME, format, ##__VA_ARGS__); \
    } while (0)

#define WRITE_DIRNAME(string, format, ...)                             \
    do                                                                 \
    {                                                                  \
        snprintf(string, STRINGMAXLEN_DIRNAME, format, ##__VA_ARGS__); \
    } while (0)

#define WRITE_IMAGENAME(string, format, ...)                           \
    do                                                                 \
    {                                                                  \
        snprintf(string, STRINGMAXLEN_IMGNAME, format, ##__VA_ARGS__); \
    } while (0)

#define CREATE_IMAGENAME(string, format, ...)                          \
    do                                                                 \
    {                                                                  \
        snprintf(string, STRINGMAXLEN_IMGNAME, format, ##__VA_ARGS__); \
    } while (0)

#ifndef DEBUG_TRACEPOINT
#    define DEBUG_TRACEPOINT(...) \
        do                        \
        {                         \
            /* dummy */           \
        } while (0)
#endif

#ifndef DEBUG_TRACEPOINT_PRINT
#    define DEBUG_TRACEPOINT_PRINT(...) \
        do                              \
        {                               \
            /* dummy */                 \
        } while (0)
#endif

#ifndef DEBUG_TRACEPOINTRAW
#    define DEBUG_TRACEPOINTRAW(...) \
        do                           \
        {                            \
            /* dummy */              \
        } while (0)
#endif

#ifndef SNPRINTF_CHECK
#    define SNPRINTF_CHECK(string, maxlen, format, ...)                  \
        do                                                               \
        {                                                                \
            int _slen = snprintf(string, maxlen, format, ##__VA_ARGS__); \
            if (_slen < 1 || _slen >= (int) maxlen)                      \
            {                                                            \
                PRINT_ERROR("snprintf error or truncation");             \
            }                                                            \
        } while (0)
#endif

#ifndef DEBUG_TRACE_FSTART
#    define DEBUG_TRACE_FSTART(...) \
        do                          \
        {                           \
        } while (0)
#endif
#ifndef DEBUG_TRACE_FEXIT
#    define DEBUG_TRACE_FEXIT(...) \
        do                         \
        {                          \
        } while (0)
#endif

// Shorthands for dependency management
// when we don't want to carry mandates over from file to file

#define MILK_WEAK __attribute__((weak))
#define MILK_WEAK_FUNCDEF                                                                     \
    {                                                                                         \
        PRINT_ERROR("__attribute__((weak)) header function definition - here only because a " \
                    "MILK_CMAKE_MANDATE_ "                                                    \
                    "(dependency system) is not satisfied!");                                 \
        abort();                                                                              \
    }

#endif
