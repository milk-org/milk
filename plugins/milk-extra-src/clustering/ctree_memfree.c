
#include "CommandLineInterface/CLIcore.h"
#include "clustering_defs.h"

errno_t ctree_memfree(CLUSTERTREE *ctree)
{
    DEBUG_TRACE_FSTART();
    for(long CFindex = 0; CFindex < ctree->NBCF; CFindex++)
    {
        free(ctree->CFarray[CFindex].childindex);
        free(ctree->CFarray[CFindex].leafindex);
        free(ctree->CFarray[CFindex].datasumvec);
    }
    free(ctree->CFarray);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
