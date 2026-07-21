// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file get_availableCFindex.c
 * @brief Get availablecfindex module
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#    include "fps.h"
#    include "ImageStreamIO/ImageStreamIO.h"
#endif
#include "COREMOD_memory/COREMOD_memory.h"
#include "clustering_defs.h"

#include "ctree_memallocate.h"

errno_t get_availableCFindex(CLUSTERTREE *ctree, long *index)
{
    DEBUG_TRACE_FSTART();

    long nCFindex = 0;
    while ((ctree->CFarray[nCFindex].type != CLUSTER_CF_TYPE_UNUSED) && (nCFindex < ctree->NBCF))
    {
        nCFindex++;
    }
    if (nCFindex == ctree->NBCF)
    {
        FUNC_RETURN_FAILURE("No available CF index");
    }

    *index = nCFindex;

    CFmemallocate(ctree, nCFindex);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
