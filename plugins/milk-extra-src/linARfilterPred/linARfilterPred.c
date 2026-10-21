// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

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

#include "milk_runnable.h"
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


#ifdef HAVE_CUDA
#    include "linalgebra/linalgebra.h"
#endif

/* ================================================================== */
/* ================================================================== */
/*            INITIALIZE LIBRARY                                      */
/* ================================================================== */
/* ================================================================== */

// Module registry: CLI exposure = membership here.
static const MILK_RUNNABLE *const runnables[] = {
    &runnable_linARfilterPred_pfloadascii,    &runnable_linARfilterPred_mselblock,
    &runnable_linARfilterPred_imrepshiftx,    &runnable_linARfilterPred_mkARpfilt,
    &runnable_linARfilterPred_applyARpfilt,   &runnable_linARfilterPred_mscangain,
    &runnable_linARfilterPred_linARPFMupdate, &runnable_linARfilterPred_linARapplyRT,
    &runnable_linARfilterPred_build_linPF,    &runnable_linARfilterPred_applyPF,
};

MILK_MODULE_RUNNABLES(linARfilterPred, NULL, runnables);

/* =============================================================================================== */
