/**
 * @file    linARfilterPred.c
 * @brief   linear auto-regressive predictive filter
 *
 * Implements Empirical Orthogonal Functions
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
#define MODULE_SHORTNAME_DEFAULT "larpf"

// Module short description
#define MODULE_DESCRIPTION "Linear auto-regressive predictive filters"

#include <assert.h>
#include <ctype.h>
#include <malloc.h>
#include <math.h>
#include <sched.h>
#include <semaphore.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <fitsio.h>


#include <time.h>

#include "CLIcore.h"
#include "timeutils.h"

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"
#include "info/info.h"
#include "linopt_imtools/linopt_imtools.h"
#include "statistic/statistic.h"

#include "linARfilterPred/linARfilterPred.h"
#include "linARfilterPred_internal.h"

#include "build_linPF.h"
#include "applyPF.h"


#ifdef HAVE_CUDA
#    include "linalgebra/linalgebra.h"
#endif

/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//
static errno_t init_module_CLI()
{
    CLIADDCMD_linARfilterPred__pfloadascii();
    CLIADDCMD_linARfilterPred__mselblock();
    CLIADDCMD_linARfilterPred__imrepshiftx();
    CLIADDCMD_linARfilterPred__mkARpfilt();
    CLIADDCMD_linARfilterPred__applyARpfilt();
    CLIADDCMD_linARfilterPred__mscangain();
    CLIADDCMD_linARfilterPred__linARPFMupdate();
    CLIADDCMD_linARfilterPred__linARapplyRT();

    CLIADDCMD_LinARfilterPred__build_linPF();
    CLIADDCMD_LinARfilterPred__applyPF();

    // add atexit functions here

    return RETURN_SUCCESS;
}

MILK_MODULE(linARfilterPred, init_module_CLI, NULL);

/* =============================================================================================== */
