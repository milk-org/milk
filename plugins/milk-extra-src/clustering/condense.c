// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

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
#include "update_level.h"

//#define DEBUGPRINT


/**
 * @brief Condense single node if possible
 *
 * If a node has fewer than B grandchildren, then children can be skipped
 * to reduce tree depth.
 *
 */
errno_t ctree_condense_node(CLUSTERTREE *ctree, long CFindex, int *op)
{
    DEBUG_TRACE_FSTART();

    *op = 0;

    // if we were to condense this node, how many children would it have ?
    long nbnewchild = 0;

    for (int chi = 0; chi < ctree->CFarray[CFindex].NBchild; chi++)
    {
        // scan children
        // cfic is child index
        long cfic = ctree->CFarray[CFindex].childindex[chi];

        if (ctree->CFarray[cfic].type == CLUSTER_CF_TYPE_NODE)
        {
            int ngchi = ctree->CFarray[cfic].NBchild;
            nbnewchild += ngchi;
        }
        else if (ctree->CFarray[cfic].type == CLUSTER_CF_TYPE_LEAF)
        {
            nbnewchild++;
        }
    }

    // If the total number of descendents is between 1 and B, we can condense (compress levels)
    //
    if ((nbnewchild > 0) && (nbnewchild < ctree->B) &&
        (nbnewchild > ctree->CFarray[CFindex].NBchild))
    {
        long  nchild  = 0;
        long *nchiCFI = (long *) malloc(sizeof(long) * nbnewchild);
        if (nchiCFI == NULL)
        {
            FUNC_RETURN_FAILURE("malloc error");
        }


        for (int chi = 0; chi < ctree->CFarray[CFindex].NBchild; chi++)
        {
            long cfic = ctree->CFarray[CFindex].childindex[chi];

            if (ctree->CFarray[cfic].type == CLUSTER_CF_TYPE_NODE)
            {
                int ngchi = ctree->CFarray[cfic].NBchild;
                for (int gchi = 0; gchi < ngchi; gchi++)
                {
                    long gchiCFindex = ctree->CFarray[cfic].childindex[gchi];
                    nchiCFI[nchild]  = gchiCFindex;
                    nchild++;
                }

                // remove child
                CFmeminit(ctree, cfic, 0);
            }
            else if (ctree->CFarray[cfic].type == CLUSTER_CF_TYPE_LEAF)
            {
                nchiCFI[nchild] = cfic;
                nchild++;
            }
        }


        // update children
        for (int nchi = 0; nchi < nbnewchild; nchi++)
        {
            ctree->CFarray[CFindex].childindex[nchi]  = nchiCFI[nchi];
            ctree->CFarray[nchiCFI[nchi]].parentindex = CFindex;
        }
        // update number of child
        ctree->CFarray[CFindex].NBchild = nbnewchild;

        // update level of downstream nodes
        update_level(ctree, CFindex);

        free(nchiCFI);
        // report that one condense operation has been done
        *op = 1;
    }


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/**
 * @brief Condense tree
 *
 * If a node has fewer than B grandchildren, then children can be skipped
 * to reduce tree depth.
 *
 * @param ctree     pointer to tree
 * @return errno_t
 */
errno_t ctree_condense(CLUSTERTREE *ctree, int *nbop)
{
    DEBUG_TRACE_FSTART();

#ifdef DEBUGPRINT
    printf("Condensing CF tree\n");
#endif

    *nbop = 0;

    for (long cfi = 0; cfi < ctree->NBCF; cfi++)
    {
        if (ctree->CFarray[cfi].type == CLUSTER_CF_TYPE_NODE)
        {
            ctree_condense_node(ctree, cfi, nbop);
        }

        if (*nbop > 0)
        {
            // return from function
            // only one condense operation at a time
            DEBUG_TRACE_FEXIT();
            return RETURN_SUCCESS;
        }
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
