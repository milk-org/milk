// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "CommandLineInterface/CLIcore.h"
#include "clustering_defs.h"

#include "CFmeminit.h"
#include "compute_imdistance_double.h"
#include "droptree.h"
#include "get_availableCFindex.h"
#include "leafnode_attachleaf.h"
#include "node_attachnode.h"

#include "printCFtree.h"


//#define DEBUG_TRACEPOINT DEBUG_TRACEPOINT_PRINT
//#define DEBUGPRINT


/**
 * @brief Split CF node
 *
 * @param CFarray    CF array
 * @param CFindex    Input node to be split
 * @param CFi0   Output node 0
 * @param CFi1   Output node 1
 * @return errno_t
 *
 * Input node will be released
 */
errno_t split_CF_node(CLUSTERTREE *ctree, long CFindex, long *CFi0, long *CFi1)
{
    DEBUG_TRACE_FSTART();
    DEBUG_TRACEPOINT("FARG %ld", CFindex);


    if (ctree->rootindex == CFindex)
    {
        DEBUG_TRACEPOINT("Dropping tree: previous root is %ld", ctree->rootindex);
        droptree(ctree);
        DEBUG_TRACEPOINT("Dropping tree: new root is %ld", ctree->rootindex);
    }

    long parentCFindex = ctree->CFarray[CFindex].parentindex;
    DEBUG_TRACEPOINT("Parent node : %ld, ssq = %g, pathcnt = %g", parentCFindex,
                     (double) ctree->CFarray[parentCFindex].datassq,
                     ctree->CFarray[parentCFindex].pathcnt);


#ifdef DEBUGPRINT
    printCFtree(ctree);
#endif

    // compute distances within leaf node
    double maxdist = 0.0;
    int    maxdistindex0, maxdistindex1;

    long nCF; // number of CF entries to split
    switch (ctree->CFarray[CFindex].type)
    {
    case CLUSTER_CF_TYPE_NODE:
        nCF = ctree->CFarray[CFindex].NBchild;
        break;

    default:
        FUNC_RETURN_FAILURE("type = %d not valid", ctree->CFarray[CFindex].type);
    }

    long *subCFarray = (long *) malloc(sizeof(long) * nCF);
    if (subCFarray == NULL)
    {
        FUNC_RETURN_FAILURE("malloc error");
    }


    for (long ccf = 0; ccf < nCF; ccf++)
    {
        subCFarray[ccf] = ctree->CFarray[CFindex].childindex[ccf];
    }

    double *distarray = (double *) malloc(sizeof(double) * nCF * nCF);
    if (distarray == NULL)
    {
        FUNC_RETURN_FAILURE("malloc error");
    }

    for (int ccf0 = 0; ccf0 < nCF; ccf0++)
    {
        distarray[ccf0 * nCF + ccf0] = 0.0;
        long CFindex00               = subCFarray[ccf0];
        for (int ccf1 = ccf0 + 1; ccf1 < nCF; ccf1++)
        {
            long   CFindex11 = subCFarray[ccf1];
            double distval;
            if (ctree->leafposmode == CLUSTER_CFPOS_DYNAMIC)
            {
                FUNC_CHECK_RETURN(compute_imdistance_double(
                    ctree, ctree->CFarray[CFindex00].datasumvec, ctree->CFarray[CFindex00].N,
                    ctree->CFarray[CFindex11].datasumvec, ctree->CFarray[CFindex11].N, &distval));
            }
            else
            {
                compute_CF2CF_posdistance_double(ctree, CFindex00, CFindex11, &distval);
            }
            if (distval > maxdist)
            {
                maxdist       = distval;
                maxdistindex0 = ccf0;
                maxdistindex1 = ccf1;
            }

            distarray[ccf0 * nCF + ccf1] = distval;
            distarray[ccf1 * nCF + ccf0] = distval;
        }
    }

    // use max distance pair to split
    DEBUG_TRACEPOINT("MAX dist within node: %d - %d = %g", maxdistindex0, maxdistindex1, maxdist);

    DEBUG_TRACEPOINT("CREATE NODES POINTING TO PARENT %ld", ctree->CFarray[CFindex].parentindex);


    // create two new nodes
    // find next available CFarray index
    long CFindex0 = 0;
    FUNC_CHECK_RETURN(get_availableCFindex(ctree, &CFindex0));

    DEBUG_TRACEPOINT("-> NODE INDEX %ld", CFindex0);
    CFmeminit(ctree, CFindex0, 0);
    ctree->CFarray[CFindex0].type = ctree->CFarray[CFindex].type;

    long CFindex1 = 0;
    FUNC_CHECK_RETURN(get_availableCFindex(ctree, &CFindex1));

    DEBUG_TRACEPOINT("-> NODE INDEX %ld", CFindex1);
    FUNC_CHECK_RETURN(CFmeminit(ctree, CFindex1, 0));
    ctree->CFarray[CFindex1].type = ctree->CFarray[CFindex].type;

    // CFs will be split between CFindex0 and CFindex1

    // destination CF
    long *destCF = (long *) malloc(sizeof(long) * nCF);
    if (destCF == NULL)
    {
        FUNC_RETURN_FAILURE("malloc error");
    }


    // Allocate each entry to one of the two new nodes
    //
    long cnt0 = 0;
    long cnt1 = 0;

    // look for CF with highest number of points in each of the 2 sets
    long maxN0    = 0;
    long maxN1    = 0;
    long maxNccf0 = -1;
    long maxNccf1 = -1;

    for (int ccf = 0; ccf < nCF; ccf++)
    {
        long   cfi   = ctree->CFarray[CFindex].childindex[ccf];
        double dist0 = distarray[maxdistindex0 * nCF + ccf];
        double dist1 = distarray[maxdistindex1 * nCF + ccf];

        if ((dist0 <= dist1) && (cnt0 < nCF - 1))
        {
            destCF[ccf] = CFindex0;
            long N      = ctree->CFarray[cfi].N;
            if (N > maxN0)
            {
                maxN0    = N;
                maxNccf0 = ccf;
            }
            cnt0++;
        }
        else
        {
            destCF[ccf] = CFindex1;
            long N      = ctree->CFarray[cfi].N;
            if (N > maxN1)
            {
                maxN1    = N;
                maxNccf1 = ccf;
            }
            cnt1++;
        }
    }


    long refCF0 = maxdistindex0;
    long refCF1 = maxdistindex1;

    // Using maxN yields poor clustering performance compared to maxdist
    //long refCF0 = maxNccf0;
    //long refCF1 = maxNccf1;

    // Add ref nodes first to ensure position corresponds to most
    // distant nodes.
    //
    FUNC_CHECK_RETURN(node_attachnode(ctree, ctree->CFarray[CFindex].childindex[refCF0], CFindex0));

    FUNC_CHECK_RETURN(node_attachnode(ctree, ctree->CFarray[CFindex].childindex[refCF1], CFindex1));


    for (int subindex = 0; subindex < nCF; subindex++)
    {
        if ((subindex != refCF0) && (subindex != refCF1))
        {
            FUNC_CHECK_RETURN(node_attachnode(ctree, ctree->CFarray[CFindex].childindex[subindex],
                                              destCF[subindex]));
        }
    }

    free(destCF);

    free(distarray);
    free(subCFarray);


    DEBUG_TRACEPOINT("Parent node : %ld, ssq = %g, pathcnt = %g", parentCFindex,
                     (double) ctree->CFarray[parentCFindex].datassq,
                     ctree->CFarray[parentCFindex].pathcnt);

    // release input leafnode
    if (ctree->rootindex == CFindex)
    {
        FUNC_RETURN_FAILURE("cannot release root node %ld", CFindex);
    }

    DEBUG_TRACEPOINT("release (leaf)node %ld", CFindex);

    long CFiparent = ctree->CFarray[CFindex].parentindex;
    DEBUG_TRACEPOINT("parent CF index = %ld", CFiparent);

    FUNC_CHECK_RETURN(CFmeminit(ctree, CFindex, CFMEMINIT_CFUPDATE));

    if (CFiparent != -1)
    {
        // attach node to parent
        DEBUG_TRACEPOINT("attach to parent %ld", CFiparent);

        FUNC_CHECK_RETURN(node_attachnode(ctree, CFindex0, CFiparent));

        FUNC_CHECK_RETURN(node_attachnode(ctree, CFindex1, CFiparent));
    }

    DEBUG_TRACEPOINT("output nodes %ld %ld", CFindex0, CFindex1);

    *CFi0 = CFindex0;
    *CFi1 = CFindex1;

    DEBUG_TRACEPOINT("output nodes written to pointers\n");

    DEBUG_TRACEPOINT("Parent node : %ld, ssq = %g, pathcnt = %g", parentCFindex,
                     (double) ctree->CFarray[parentCFindex].datassq,
                     ctree->CFarray[parentCFindex].pathcnt);


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
