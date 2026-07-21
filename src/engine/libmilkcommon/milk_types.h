// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    milk_types.h
 * @brief   Pure POSIX types and constants for MILK
 *
 * Provides shared types, string constants, and macros
 * (like errno_t, MILK_DATA, IMGID structures)
 * independent of the CLIcore tier.
 */

#ifndef MILK_TYPES_H
#define MILK_TYPES_H

#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <sys/types.h>

/* Core data structure (MILK_DATA) */
#include "milkdata.h"

/* ImageStreamIO dependencies */
#include "ImageStreamIO.h"
#include "ImageStruct.h"

/* Process tools and timeutils */
#include "processtools.h"
#include "timeutils.h"
#include "milkDebugTools.h"

#define PI 3.14159265358979323846264338328
#define SZ_CLICOREVARRAY 1000

/* String length constants */
#define STRINGMAXLEN_CLISTARTUPFILENAME 200
#define STRINGMAXLEN_CLIPROMPT 200

#define CFITSEXIT                    \
    printf("Abnormal termination, "  \
           "File \"%s\", line %d\n", \
           __FILE__, __LINE__);      \
    exit(0)

#ifdef DEBUG
#    define nmalloc(f, type, n)                         \
        f = (type *) calloc(n, sizeof(type));           \
        if (f == NULL)                                  \
        {                                               \
            printf("ERROR: \"" #f "\" alloc failed\n"); \
            exit(0);                                    \
        }                                               \
        else                                            \
        {                                               \
            printf("\nMALLOC: \"" #f "\" allocated\n"); \
        }
#    define nfree(f) \
        free(f);     \
        printf("\nMALLOC: \"" #f "\" freed\n");
#else
#    define nmalloc(f, type, n)                         \
        f = (type *) calloc(n, sizeof(type));           \
        if (f == NULL)                                  \
        {                                               \
            printf("ERROR: \"" #f "\" alloc failed\n"); \
            exit(0);                                    \
        }
#    define nfree(f) free(f);
#endif

#define TEST_ALLOC(f)                               \
    if (f == NULL)                                  \
    {                                               \
        printf("ERROR: \"" #f "\" alloc failed\n"); \
        exit(0);                                    \
    }

#define NB_ARG_MAX 100

#endif /* MILK_TYPES_H */
