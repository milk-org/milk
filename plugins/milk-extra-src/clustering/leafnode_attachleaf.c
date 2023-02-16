
#include "CommandLineInterface/CLIcore.h"
#include "clustering_defs.h"

#include "addvector_to_CF.h"

/**
 * @brief Attach EXISITNG leaf to leafnode
 *
 * @param ctree             clustering tree
 * @param CFindexleaf       CF index of leaf
 * @param CFindexleafnode   CF index of node to which leaf should be attached
 * @return errno_t
 */
errno_t
leafnode_attachleaf(CLUSTERTREE *ctree, long CFindexleaf, long CFindexleafnode)
{
    DEBUG_TRACE_FSTART();

    ctree->CFarray[CFindexleafnode]
    .leafindex[ctree->CFarray[CFindexleafnode].NBleaf] = CFindexleaf;
    ctree->CFarray[CFindexleafnode].NBleaf++;

    ctree->CFarray[CFindexleaf].parentindex = CFindexleafnode;
    ctree->CFarray[CFindexleaf].level =
        ctree->CFarray[CFindexleafnode].level + 1;

    long cfi = CFindexleafnode;
    while(cfi != -1)
    {
        //printf("========= ADDING VECTOR TO NODE %ld (%s)\n", cfi, __FILE__);
        //fflush(stdout);

        ctree->CFarray[cfi].status |= CLUSTER_CF_STATUS_UPDATE;

        int addOK = 1; // don't test radius
        addvector_to_CF(ctree,
                        ctree->CFarray[CFindexleaf].datasumvec,
                        ctree->CFarray[CFindexleaf].datassq,
                        ctree->CFarray[CFindexleaf].N,
                        cfi,
                        &addOK);

        // move upstream to propagate change

        long cfip = ctree->CFarray[cfi].parentindex;
        if(cfi == cfip)
        {
            FUNC_RETURN_FAILURE(
                "Attaching leaf %ld to node %ld: CF parent %ld points to "
                "itself",
                CFindexleaf,
                CFindexleafnode,
                cfi);
        }

        cfi = cfip;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
