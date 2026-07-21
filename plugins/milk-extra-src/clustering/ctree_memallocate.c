// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file ctree_memallocate.c
 * @brief Ctree memallocate module
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

#include "CFmeminit.h"


errno_t CFmemallocate(CLUSTERTREE *ctree, long CFindex)
{
    DEBUG_TRACE_FSTART();

    if (!(ctree->CFarray[CFindex].status && CLUSTER_CF_STATUS_MEMALLOC))
    {
        ctree->CFarray[CFindex].childindex = (long *) malloc(sizeof(long) * (ctree->B + 1));
        if (ctree->CFarray[CFindex].childindex == NULL)
        {
            FUNC_RETURN_FAILURE("malloc error");
        }

        ctree->CFarray[CFindex].datasumvec = (double *) malloc(sizeof(double) * ctree->npix);
        if (ctree->CFarray[CFindex].datasumvec == NULL)
        {
            FUNC_RETURN_FAILURE("malloc error");
        }

        ctree->CFarray[CFindex].dataposvec = (double *) malloc(sizeof(double) * ctree->npix);
        if (ctree->CFarray[CFindex].dataposvec == NULL)
        {
            FUNC_RETURN_FAILURE("malloc error");
        }

        // Required to avoid infinite loop in CFmeminit upstream tracking
        ctree->CFarray[CFindex].parentindex = -1;

        CFmeminit(ctree, CFindex, 0);

        ctree->CFarray[CFindex].status |= CLUSTER_CF_STATUS_MEMALLOC;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


errno_t ctree_memallocate(CLUSTERTREE *ctree)
{
    // Allocate memory for CFs
    DEBUG_TRACE_FSTART();

    printf("Allocating CF memory. %ld CFs, size = %ld bytes\n", ctree->NBCF,
           (long) sizeof(double) * ctree->npix * ctree->NBCF);

    ctree->CFarray = (CLUSTERING_CF *) malloc(sizeof(CLUSTERING_CF) * ctree->NBCF);
    if (ctree->CFarray == NULL)
    {
        FUNC_RETURN_FAILURE("malloc error");
    }

    for (long CFindex = 0; CFindex < ctree->NBCF; CFindex++)
    {
        ctree->CFarray[CFindex].type        = CLUSTER_CF_TYPE_UNUSED;
        ctree->CFarray[CFindex].parentindex = -1;
        ctree->CFarray[CFindex].status      = 0;
    }

    // pairwise distances
    ctree->CFCFdist = (double *) malloc(sizeof(double) * ctree->NBCF * ctree->NBCF);
    if (ctree->CFCFdist == NULL)
    {
        FUNC_RETURN_FAILURE("malloc error");
    }
    // initialize all distances to -1.0 to indicate unknown
    for (long CFindex0 = 0; CFindex0 < ctree->NBCF; CFindex0++)
    {
        for (long CFindex1 = 0; CFindex1 < ctree->NBCF; CFindex1++)
        {
            ctree->CFCFdist[CFindex0 * ctree->NBCF + CFindex1] = -1.0;
        }
    }


    /*    for(long CFindex = 0; CFindex < ctree->NBCF; CFindex++)
        {

            ctree->CFarray[CFindex].childindex =
                (long *) malloc(sizeof(long) * (ctree->B + 1));
            if(ctree->CFarray[CFindex].childindex == NULL)
            {
                FUNC_RETURN_FAILURE("malloc error");
            }

            ctree->CFarray[CFindex].datasumvec =
                (double *) malloc(sizeof(double) * ctree->npix);
            if(ctree->CFarray[CFindex].datasumvec == NULL)
            {
                FUNC_RETURN_FAILURE("malloc error");
            }

            ctree->CFarray[CFindex].dataposvec =
                (double *) malloc(sizeof(double) * ctree->npix);
            if(ctree->CFarray[CFindex].dataposvec == NULL)
            {
                FUNC_RETURN_FAILURE("malloc error");
            }

            // Required to avoid infinite loop in CFmeminit upstream tracking
            ctree->CFarray[CFindex].parentindex = -1;

            CFmeminit(ctree, CFindex, 0);
        }
    */

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
