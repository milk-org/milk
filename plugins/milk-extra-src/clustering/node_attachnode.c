/**
 * @file node_attachnode.c
 * @brief attach node CFindex to CFindexupnode
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "libmilkdata/milkdata.h"
#include "milkDebugTools.h"
#include "fps.h"
#include "ImageStreamIO/ImageStreamIO.h"
#endif
#include "COREMOD_memory/COREMOD_memory.h"
#include "clustering_defs.h"

#include "addCF_to_CF.h"
#include "update_level.h"


// attach node CFindex to CFindexupnode
errno_t node_attachnode(
    CLUSTERTREE *ctree,
    long CFindex,
    long CFindexupnode
)
{
    DEBUG_TRACE_FSTART();

    ctree->CFarray[CFindexupnode].childindex[ctree->CFarray[CFindexupnode].NBchild] = CFindex;
    ctree->CFarray[CFindexupnode].NBchild++;

    ctree->CFarray[CFindex].parentindex = CFindexupnode;
    ctree->CFarray[CFindex].level = ctree->CFarray[CFindexupnode].level + 1;

    {
        long cfi = CFindexupnode;
        while(cfi != -1)
        {
            ctree->CFarray[cfi].status |= CLUSTER_CF_STATUS_UPDATE;

            int addOK = 1; // don't test radius
            addCF_to_CF(ctree,
                        ctree->CFarray[CFindex],
                        cfi,
                        &addOK);

            // move upstream to propagate change
            cfi = ctree->CFarray[cfi].parentindex;
        }
    }

    // update level of descendents
    update_level(ctree, CFindex);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
