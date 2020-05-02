/**
 * @file    COREMOD_tools.c
 * @brief   Frequently used tools
 *
 * Includes basic file I/O
 *
 *
 */


/* ================================================================== */
/* ================================================================== */
/*            MODULE INFO                                             */
/* ================================================================== */
/* ================================================================== */

// module default short name
// all CLI calls to this module functions will be <shortname>.<funcname>
// if set to "", then calls use <funcname>
#define MODULE_SHORTNAME_DEFAULT ""

// Module short description
#define MODULE_DESCRIPTION       "misc tools"




/* =============================================================================================== */
/* =============================================================================================== */
/*                                        HEADER FILES                                             */
/* =============================================================================================== */
/* =============================================================================================== */

#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/syscall.h>

#include <ncurses.h>

#ifdef __MACH__
#include <mach/mach_time.h>long AOloopControl_ComputeOpenLoopModes(long loop)
#define CLOCK_REALTIME 0
#define CLOCK_MONOTONIC 0
static int clock_gettime(int clk_id, struct mach_timespec *t)
{
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    uint64_t time;
    time = mach_absolute_time();
    double nseconds = ((double)time * (double)timebase.numer) / ((
                          double)timebase.denom);
    double seconds = ((double)time * (double)timebase.numer) / ((
                         double)timebase.denom * 1e9);
    t->tv_sec = seconds;
    t->tv_nsec = nseconds;
    return 0;
}
#else
#include <time.h>
#endif


#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"

#include "imdisplay3d.h"
#include "mvprocCPUset.h"
#include "statusstat.h"



/* =============================================================================================== */
/* =============================================================================================== */
/*                                      DEFINES, MACROS                                            */
/* =============================================================================================== */
/* =============================================================================================== */

#define SBUFFERSIZE 1000






INIT_MODULE_LIB(COREMOD_tools)


/* ================================================================== */
/* ================================================================== */
/*            COMMAND LINE INTERFACE (CLI) FUNCTIONS                  */
/* ================================================================== */
/* ================================================================== */

/** @name CLI bindings */










/* =============================================================================================== */
/* =============================================================================================== */
/*                                    MODULE INITIALIZATION                                        */
/* =============================================================================================== */
/* =============================================================================================== */
/** @name Module initialization */



static errno_t init_module_CLI()
{

	mvprocCPUset_addCLIcmd();

	fileutils_addCLIcmd();

	imdisplay3d_addCLIcmd();

	statusstat_addCLIcmd();

    return RETURN_SUCCESS;
}















errno_t lin_regress(
    double *a,
    double *b,
    double *Xi2,
    double *x,
    double *y,
    double *sig,
    unsigned int nb_points
)
{
    double S, Sx, Sy, Sxx, Sxy, Syy;
    unsigned int i;
    double delta;

    S = 0;
    Sx = 0;
    Sy = 0;
    Sxx = 0;
    Syy = 0;
    Sxy = 0;

    for(i = 0; i < nb_points; i++)
    {
        S += 1.0 / sig[i] / sig[i];
        Sx += x[i] / sig[i] / sig[i];
        Sy += y[i] / sig[i] / sig[i];
        Sxx += x[i] * x[i] / sig[i] / sig[i];
        Syy += y[i] * y[i] / sig[i] / sig[i];
        Sxy += x[i] * y[i] / sig[i] / sig[i];
    }

    delta = S * Sxx - Sx * Sx;
    *a = (Sxx * Sy - Sx * Sxy) / delta;
    *b = (S * Sxy - Sx * Sy) / delta;
    *Xi2 = Syy - 2 * (*a) * Sy - 2 * (*a) * (*b) * Sx + (*a) * (*a) * S + 2 *
           (*a) * (*b) * Sx - (*b) * (*b) * Sxx;

    return RETURN_SUCCESS;
}






/* test point */
errno_t tp(
    const char *word
)
{
    printf("---- Test point %s ----\n", word);
    fflush(stdout);

    return RETURN_SUCCESS;
}







