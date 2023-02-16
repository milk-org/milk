#include "CommandLineInterface/CLIcore.h"
#include "clustering_defs.h"

#include "addvector_to_CF.h"
#include "update_level.h"

errno_t node_attachnode(CLUSTERTREE *ctree, long CFindex, long CFindexupnode)
{
    DEBUG_TRACE_FSTART();

    ctree->CFarray[CFindexupnode]
    .childindex[ctree->CFarray[CFindexupnode].NBchild] = CFindex;
    ctree->CFarray[CFindexupnode].NBchild++;

    ctree->CFarray[CFindex].parentindex = CFindexupnode;
    ctree->CFarray[CFindex].level = ctree->CFarray[CFindexupnode].level + 1;

    {
        long cfi = CFindexupnode;
        while(cfi != -1)
        {
            ctree->CFarray[cfi].status |= CLUSTER_CF_STATUS_UPDATE;

            int addOK = 1; // don't test radius
            addvector_to_CF(ctree,
                            ctree->CFarray[CFindex].datasumvec,
                            ctree->CFarray[CFindex].datassq,
                            ctree->CFarray[CFindex].N,
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
