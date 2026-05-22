/**
 * @file ctree_memfree.c
 * @brief Ctree memfree module
 */


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

errno_t ctree_memfree(CLUSTERTREE *ctree)
{
    DEBUG_TRACE_FSTART();
    for (long CFindex = 0; CFindex < ctree->NBCF; CFindex++)
    {
        if ((ctree->CFarray[CFindex].status && CLUSTER_CF_STATUS_MEMALLOC))
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
