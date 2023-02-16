
#include "CommandLineInterface/CLIcore.h"
#include "clustering_defs.h"

errno_t update_level(CLUSTERTREE *ctree, long CFindex)
{
    DEBUG_TRACE_FSTART();

    //printf("==== UPDATELEVEL %ld\n", CFindex);

    for(int cfi = 0; cfi < ctree->CFarray[CFindex].NBchild; cfi++)
    {
        long cfic                  = ctree->CFarray[CFindex].childindex[cfi];
        ctree->CFarray[cfic].level = ctree->CFarray[CFindex].level + 1;
        update_level(ctree, cfic);
    }

    for(int lfi = 0; lfi < ctree->CFarray[CFindex].NBleaf; lfi++)
    {
        long lfic                  = ctree->CFarray[CFindex].leafindex[lfi];
        ctree->CFarray[lfic].level = ctree->CFarray[CFindex].level + 1;
        update_level(ctree, lfic);
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
