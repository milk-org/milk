/**
 * @file    processtools.h
 * @brief   Command line interface
 *
 * Command line interface (CLI) definitions and function prototypes
 *
 */

#ifndef _CACAO_DEPENDENCY_H
#define _CACAO_DEPENDENCY_H

#include <CLIcore.h>
// #include <gsl/gsl_rng.h>  // for random numbers
// #include <semaphore.h>
// #include <signal.h>
// #include <stdint.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <sys/types.h>
// #include <unistd.h>

#define SHAREDSHMDIR    "/milk/shm"  /**< default location of file mapped semaphores, can be over-ridden by env variable MILK_SHM_DIR */
#define SHAREDPROCDIR    "/milk/proc"
#define CLIPID    getpid()

// *************************** FUNCTION RETURN VALUE *********************************************
// For function returning type errno_t (= int)
//
#define RETURN_SUCCESS        0
#define RETURN_FAILURE       1   // generic error code
#define RETURN_MISSINGFILE   2

#include <time.h>
#include <errno.h>
#include "ImageStreamIO.h"

#ifndef __STDC_LIB_EXT1__
typedef int errno_t;
#endif

typedef long imageID;
typedef long variableID;

#define DEBUG_TRACEPOINT(...)

#ifdef __cplusplus
extern "C" {
#endif

struct timespec timespec_diff(struct timespec start, struct timespec end);
int print_header(const char *str, char c);
void quick_sort2l(double *array, long *array1, long count);
void quick_sort2l_double(double *array, long *array1, long count);
void quick_sort_long(long *array, long count);

#ifdef __cplusplus
}
#endif

#endif  // _CACAO_DEPENDENCY_H
