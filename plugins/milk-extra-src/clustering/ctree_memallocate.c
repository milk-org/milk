
#include "CommandLineInterface/CLIcore.h"
#include "clustering_defs.h"

#include "CFmeminit.h"

errno_t ctree_memallocate(CLUSTERTREE *ctree)
{
    // Allocate memory for CFs
    DEBUG_TRACE_FSTART();

    printf("Allocating CF memory. %ld CFs, size = %ld bytes\n",
           ctree->NBCF,
           (long) sizeof(double) * ctree->npix * ctree->NBCF);

    ctree->CFarray =
        (CLUSTERING_CF *) malloc(sizeof(CLUSTERING_CF) * ctree->NBCF);
    if(ctree->CFarray == NULL)
    {
        FUNC_RETURN_FAILURE("malloc error");
    }

    for(long CFindex = 0; CFindex < ctree->NBCF; CFindex++)
    {

        ctree->CFarray[CFindex].childindex =
            (long *) malloc(sizeof(long) * (ctree->B + 1));
        if(ctree->CFarray[CFindex].childindex == NULL)
        {
            FUNC_RETURN_FAILURE("malloc error");
        }

        ctree->CFarray[CFindex].leafindex =
            (long *) malloc(sizeof(long) * (ctree->L + 1));
        if(ctree->CFarray[CFindex].leafindex == NULL)
        {
            FUNC_RETURN_FAILURE("malloc error");
        }

        ctree->CFarray[CFindex].datasumvec =
            (double *) malloc(sizeof(double) * ctree->npix);
        if(ctree->CFarray[CFindex].datasumvec == NULL)
        {
            FUNC_RETURN_FAILURE("malloc error");
        }

        ctree->CFarray[CFindex].parentindex =
            -1; // Require to avoid infinite loop in CFmeminit upstream tracking
        CFmeminit(ctree, CFindex, 0);
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
