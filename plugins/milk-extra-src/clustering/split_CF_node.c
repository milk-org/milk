
#include "CommandLineInterface/CLIcore.h"
#include "clustering_defs.h"

#include "CFmeminit.h"
#include "compute_imdistance_double.h"
#include "droptree.h"
#include "get_availableCFindex.h"
#include "leafnode_attachleaf.h"
#include "node_attachnode.h"

#include "printCFtree.h"

/**
 * @brief Split CF node or leafnode
 *
 * @param CFarray    CF array
 * @param CFindex    Input (leaf) node to be split
 * @param CFindex0   Output (leaf) node 0
 * @param CFindex1   Output (leaf) node 1
 * @return errno_t
 *
 * Input leaf node will be released
 */
errno_t split_CF_node(CLUSTERTREE *ctree, long CFindex, long *CFi0, long *CFi1)
{
    DEBUG_TRACE_FSTART();
    DEBUG_TRACEPOINT("FARG %ld", CFindex);

    if(ctree->rootindex == CFindex)
    {
        droptree(ctree);
    }

    printCFtree(ctree);

    // compute distances within leaf node
    double maxdist = 0.0;
    int    maxdistindex0, maxdistindex1;

    long nCF; // number of CF entries to split
    switch(ctree->CFarray[CFindex].type)
    {
        case CLUSTER_CF_TYPE_LEAFNODE:
            nCF = ctree->CFarray[CFindex].NBleaf;
            break;

        case CLUSTER_CF_TYPE_NODE:
            nCF = ctree->CFarray[CFindex].NBchild;
            break;

        default:
            FUNC_RETURN_FAILURE("type = %d not valid",
                                ctree->CFarray[CFindex].type);
    }

    long *subCFarray = (long *) malloc(sizeof(long) * nCF);
    if(subCFarray == NULL)
    {
        FUNC_RETURN_FAILURE("malloc error");
    }
    if(ctree->CFarray[CFindex].type == CLUSTER_CF_TYPE_LEAFNODE)
    {
        for(long i = 0; i < nCF; i++)
        {
            subCFarray[i] = ctree->CFarray[CFindex].leafindex[i];
        }
    }
    else
    {
        for(long i = 0; i < nCF; i++)
        {
            subCFarray[i] = ctree->CFarray[CFindex].childindex[i];
        }
    }

    double *distarray = (double *) malloc(sizeof(double) * nCF * nCF);
    if(distarray == NULL)
    {
        FUNC_RETURN_FAILURE("malloc error");
    }

    for(int index0 = 0; index0 < nCF; index0++)
    {
        distarray[index0 * nCF + index0] = 0.0;
        long CFindex00                   = subCFarray[index0];
        for(int index1 = index0 + 1; index1 < nCF; index1++)
        {
            long   CFindex11 = subCFarray[index1];
            double distval;
            FUNC_CHECK_RETURN(
                compute_imdistance_double(ctree,
                                          ctree->CFarray[CFindex00].datasumvec,
                                          ctree->CFarray[CFindex00].N,
                                          ctree->CFarray[CFindex11].datasumvec,
                                          ctree->CFarray[CFindex11].N,
                                          &distval));
            DEBUG_TRACEPOINT("DIST %02d %02d  %g\n", index0, index1, distval);
            if(distval > maxdist)
            {
                maxdist       = distval;
                maxdistindex0 = index0;
                maxdistindex1 = index1;
            }

            distarray[index0 * nCF + index1] = distval;
            distarray[index1 * nCF + index0] = distval;
        }
    }

    // use max distance pair to split
    DEBUG_TRACEPOINT("MAX dist within node: %d - %d = %g",
                     maxdistindex0,
                     maxdistindex1,
                     maxdist);

    DEBUG_TRACEPOINT("CREATE NODES POINTING TO PARENT %ld",
                     ctree->CFarray[CFindex].parentindex);
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

    // leafs will be split between CFindex0 and CFindex1

    // destination CF
    long *destCF = (long *) malloc(sizeof(long) * nCF);
    if(destCF == NULL)
    {
        FUNC_RETURN_FAILURE("malloc error");
    }

    long cnt0 = 0;
    long cnt1 = 0;
    for(int subindex = 0; subindex < nCF; subindex++)
    {
        double dist0 = distarray[maxdistindex0 * nCF + subindex];
        double dist1 = distarray[maxdistindex1 * nCF + subindex];

        if((dist0 <= dist1) && (cnt0 < nCF - 1))
        {
            destCF[subindex] = CFindex0;
            cnt0++;
        }
        else
        {
            destCF[subindex] = CFindex1;
            cnt1++;
        }
    }

    for(int subindex = 0; subindex < nCF; subindex++)
    {

        DEBUG_TRACEPOINT("(LEAF)NODE %2d  %12g %12g -> ADD TO (LEAF)NODE %ld",
                         subindex,
                         distarray[maxdistindex0 * nCF + subindex],
                         distarray[maxdistindex1 * nCF + subindex],
                         destCF[subindex]);

        if(ctree->CFarray[CFindex].type == CLUSTER_CF_TYPE_LEAFNODE)
        {
            FUNC_CHECK_RETURN(
                leafnode_attachleaf(ctree,
                                    ctree->CFarray[CFindex].leafindex[subindex],
                                    destCF[subindex]));
        }
        else
        {
            FUNC_CHECK_RETURN(
                node_attachnode(ctree,
                                ctree->CFarray[CFindex].childindex[subindex],
                                destCF[subindex]));
        }
    }

    free(destCF);

    free(distarray);
    free(subCFarray);

    // release input leafnode
    if(ctree->rootindex == CFindex)
    {
        FUNC_RETURN_FAILURE("cannot release root node %ld", CFindex);
    }

    DEBUG_TRACEPOINT("release (leaf)node %ld", CFindex);

    long CFiparent = ctree->CFarray[CFindex].parentindex;
    DEBUG_TRACEPOINT("parent CF index = %ld", CFiparent);

    FUNC_CHECK_RETURN(CFmeminit(ctree, CFindex, CFMEMINIT_CFUPDATE));

    if(CFiparent != -1)
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

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
