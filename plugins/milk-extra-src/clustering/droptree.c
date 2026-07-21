// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file droptree.c
 * @brief Droptree module
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

#include "get_availableCFindex.h"


errno_t droptree(CLUSTERTREE *ctree)
{
    DEBUG_TRACE_FSTART();

    for (long CFi = 0; CFi < ctree->NBCF; CFi++)
    {
        ctree->CFarray[CFi].level++;
    }
    long CFindex = 0;
    FUNC_CHECK_RETURN(get_availableCFindex(ctree, &CFindex));

    // make new root

    ctree->CFarray[CFindex].type          = CLUSTER_CF_TYPE_NODE; // default
    ctree->CFarray[CFindex].NBchild       = 1;
    ctree->CFarray[CFindex].childindex[0] = ctree->rootindex;
    //ctree->CFarray[CFindex].NBleaf        = 0;

    if (ctree->CFarray[ctree->rootindex].type == CLUSTER_CF_TYPE_LEAF)
    {
        ctree->CFarray[CFindex].type    = CLUSTER_CF_TYPE_NODE; //CLUSTER_CF_TYPE_LEAFNODE;
        ctree->CFarray[CFindex].NBchild = 1;
        //ctree->CFarray[CFindex].NBleaf       = 1;
        ctree->CFarray[CFindex].childindex[0] = ctree->rootindex;
    }

    ctree->CFarray[CFindex].level = 0;
    ctree->CFarray[CFindex].N     = ctree->CFarray[ctree->rootindex].N;
    memcpy(ctree->CFarray[CFindex].datasumvec, ctree->CFarray[ctree->rootindex].datasumvec,
           sizeof(double) * ctree->npix);
    memcpy(ctree->CFarray[CFindex].dataposvec, ctree->CFarray[ctree->rootindex].dataposvec,
           sizeof(double) * ctree->npix);
    ctree->CFarray[CFindex].datassq = ctree->CFarray[ctree->rootindex].datassq;

    ctree->CFarray[CFindex].pathcnt         = ctree->CFarray[ctree->rootindex].pathcnt;
    ctree->CFarray[CFindex].pathdistcompcnt = ctree->CFarray[ctree->rootindex].pathdistcompcnt;

    ctree->CFarray[ctree->rootindex].parentindex = CFindex;

    ctree->rootindex = CFindex;

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
