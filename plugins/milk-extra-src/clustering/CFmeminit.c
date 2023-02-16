#include "CommandLineInterface/CLIcore.h"
#include "clustering_defs.h"

#include "addvector_to_CF.h"

#include "CFmeminit.h"

errno_t CFmeminit(CLUSTERTREE *ctree, long CFindex, uint32_t mode)
{
    DEBUG_TRACE_FSTART();

    if(mode && CFMEMINIT_CFUPDATE)
    {
        // if CF has parent, remove from upstream
        long cfi = ctree->CFarray[CFindex].parentindex;

        if(cfi != -1)
        {
            // remove from list of childred or leafs
            long NBchild = ctree->CFarray[cfi].NBchild;
            int  found   = 0;
            for(int chi = 0; chi < NBchild; chi++)
            {
                if(ctree->CFarray[cfi].childindex[chi] == CFindex)
                {
                    found = 1;
                }
                if(found == 1)
                {
                    ctree->CFarray[cfi].childindex[chi] =
                        ctree->CFarray[cfi].childindex[chi + 1];
                }
            }
            if(found == 1)
            {
                ctree->CFarray[cfi].NBchild--;
            }

            long NBleaf = ctree->CFarray[cfi].NBleaf;
            found       = 0;
            for(int li = 0; li < NBleaf; li++)
            {
                if(ctree->CFarray[cfi].leafindex[li] == CFindex)
                {
                    found = 1;
                }
                if(found == 1)
                {
                    ctree->CFarray[cfi].leafindex[li] =
                        ctree->CFarray[cfi].leafindex[li + 1];
                }
            }
            if(found == 1)
            {
                ctree->CFarray[cfi].NBleaf--;
            }
        }

        while(cfi != -1)
        {
            //printf("========= SUBTRACTING NODE %ld FROM NODE %ld (%s)\n", CFindex, cfi, __FILE__);
            //fflush(stdout);

            ctree->CFarray[cfi].status |= CLUSTER_CF_STATUS_UPDATE;

            subvector_to_CF(ctree,
                            ctree->CFarray[CFindex].datasumvec,
                            ctree->CFarray[CFindex].datassq,
                            ctree->CFarray[CFindex].N,
                            cfi);

            // move upstream to propagate change
            cfi = ctree->CFarray[cfi].parentindex;
        }
    }

    ctree->CFarray[CFindex].type  = CLUSTER_CF_TYPE_UNUSED;
    ctree->CFarray[CFindex].level = 0;

    ctree->CFarray[CFindex].NBchild = 0;
    for(int cind = 0; cind < ctree->B + 1; cind++)
    {
        ctree->CFarray[CFindex].childindex[cind] = -1;
    }

    ctree->CFarray[CFindex].NBleaf = 0;
    for(int cind = 0; cind < ctree->L + 1; cind++)
    {
        ctree->CFarray[CFindex].leafindex[cind] = -1;
    }

    ctree->CFarray[CFindex].parentindex = -1;

    for(int ii = 0; ii < ctree->npix; ii++)
    {
        ctree->CFarray[CFindex].datasumvec[ii] = 0.0;
    }

    ctree->CFarray[CFindex].sum2    = 0.0;
    ctree->CFarray[CFindex].datassq = 0.0;
    ctree->CFarray[CFindex].radius2 = 0;
    ctree->CFarray[CFindex].N       = 0;

    ctree->CFarray[CFindex].status = 0;

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
