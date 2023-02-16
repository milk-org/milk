#include "CommandLineInterface/CLIcore.h"
#include "clustering_defs.h"

#include "get_availableCFindex.h"

errno_t droptree(CLUSTERTREE *ctree)
{
    DEBUG_TRACE_FSTART();

    for(long CFi = 0; CFi < ctree->NBCF; CFi++)
    {
        ctree->CFarray[CFi].level++;
    }
    long CFindex = 0;
    FUNC_CHECK_RETURN(get_availableCFindex(ctree, &CFindex));

    // make new root

    ctree->CFarray[CFindex].type          = CLUSTER_CF_TYPE_NODE; // default
    ctree->CFarray[CFindex].NBchild       = 1;
    ctree->CFarray[CFindex].childindex[0] = ctree->rootindex;
    ctree->CFarray[CFindex].NBleaf        = 0;

    if(ctree->CFarray[ctree->rootindex].type == CLUSTER_CF_TYPE_LEAF)
    {
        ctree->CFarray[CFindex].type         = CLUSTER_CF_TYPE_LEAFNODE;
        ctree->CFarray[CFindex].NBchild      = 0;
        ctree->CFarray[CFindex].NBleaf       = 1;
        ctree->CFarray[CFindex].leafindex[0] = ctree->rootindex;
    }

    ctree->CFarray[CFindex].level = 0;
    ctree->CFarray[CFindex].N     = ctree->CFarray[ctree->rootindex].N;

    ctree->CFarray[ctree->rootindex].parentindex = CFindex;

    ctree->rootindex = CFindex;

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
