/**
 * @file ctree_memfree.c
 * @brief Ctree memfree module
 */


#include "CLIcore.h"
#include "clustering_defs.h"

errno_t ctree_memfree(CLUSTERTREE *ctree)
{
    DEBUG_TRACE_FSTART();
    for(long CFindex = 0; CFindex < ctree->NBCF; CFindex++)
    {
        if ( (ctree->CFarray[CFindex].status && CLUSTER_CF_STATUS_MEMALLOC) )
        {
            free(ctree->CFarray[CFindex].childindex);
            //free(ctree->CFarray[CFindex].leafindex);
            free(ctree->CFarray[CFindex].datasumvec);
            free(ctree->CFarray[CFindex].dataposvec);
        }
    }
    free(ctree->CFarray);
    free(ctree->CFCFdist);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
