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

#include "fps_add_entry.h"
#include "fps_checkparameter.h"
#include "fps_CONFstart.h"
#include "fps_CONFstop.h"
#include "fps_connectExternalFPS.h"
#include "fps_connect.h"
#include "fps_CTRLscreen.h"
#include "fps_disconnect.h"
#include "fps_execFPScmd.h"
#include "fps_FPCONFexit.h"
#include "fps_FPCONFloopstep.h"
#include "fps_FPCONFsetup.h"
#include "fps_FPSremove.h"
#include "fps_GetFileName.h"
#include "fps_getFPSargs.h"
#include "fps_GetParamIndex.h"
#include "fps_GetTypeString.h"
#include "fps_load.h"
#include "fps_loadstream.h"
#include "fps_outlog.h"
#include "fps_paramvalue.h"
#include "fps_PrintParameterInfo.h"
#include "fps_printparameter_valuestring.h"
#include "fps_processcmdline.h"
#include "fps_process_fpsCMDarray.h"
#include "fps_processinfo_entries.h"
#include "fps_process_user_key.h"
#include "fps_read_fpsCMD_fifo.h"
#include "fps_RUNexit.h"
#include "fps_RUNstart.h"
#include "fps_RUNstop.h"
#include "fps_save2disk.h"
#include "fps_scan.h"
#include "fps_shmdirname.h"
#include "fps_tmux.h"
#include "fps_userinputsetparamvalue.h"

#ifdef __cplusplus
}
#endif

#endif  // _CACAO_DEPENDENCY_H
