#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "clustering_defs.h"

errno_t printCFtree(CLUSTERTREE *ctree)
{
    DEBUG_TRACE_FSTART();

    printf("\n\n");

    for(int level = 0; level < 100; level++)
    {
        for(int CFindex = 0; CFindex < ctree->NBCF; CFindex++)
        {
            if((ctree->CFarray[CFindex].type != CLUSTER_CF_TYPE_UNUSED) &&
                    (ctree->CFarray[CFindex].level == level))
            {
                for(int l = 0; l < ctree->CFarray[CFindex].level; l++)
                {
                    printf("    ");
                }
                if(ctree->CFarray[CFindex].status && CLUSTER_CF_STATUS_UPDATE)
                {
                    printf("*");
                }
                else
                {
                    printf(" ");
                }
                printf("[%3d] ", CFindex);

                switch(ctree->CFarray[CFindex].type)
                {
                    case CLUSTER_CF_TYPE_ROOT:
                        printf("ROOT");
                        break;

                    case CLUSTER_CF_TYPE_NODE:
                        printf("NODE");
                        break;

                    case CLUSTER_CF_TYPE_LEAF:
                        printf("LEAF");
                        break;

                    case CLUSTER_CF_TYPE_LEAFNODE:
                        printf("LFND");
                        break;

                    default:
                        printf("????");
                }

                printf("  N=%5ld", ctree->CFarray[CFindex].N);
                printf("  %10.4g %10.4g",
                       (double) ctree->CFarray[CFindex].datassq,
                       (double) ctree->CFarray[CFindex].sum2);
                printf("    R=%6.4f ",
                       sqrt(ctree->CFarray[CFindex].radius2) / ctree->T);

                printf(" parent=%3ld", ctree->CFarray[CFindex].parentindex);

                if(ctree->CFarray[CFindex].type == CLUSTER_CF_TYPE_NODE)
                {
                    printf("  %3d children (", ctree->CFarray[CFindex].NBchild);
                    for(int chi = 0; chi < ctree->CFarray[CFindex].NBchild;
                            chi++)
                    {
                        printf(" %ld", ctree->CFarray[CFindex].childindex[chi]);
                    }
                    printf(")");
                }

                if(ctree->CFarray[CFindex].type == CLUSTER_CF_TYPE_LEAFNODE)
                {
                    printf("  %3d leaves (", ctree->CFarray[CFindex].NBleaf);
                    for(int lfi = 0; lfi < ctree->CFarray[CFindex].NBleaf;
                            lfi++)
                    {
                        printf(" %ld", ctree->CFarray[CFindex].leafindex[lfi]);
                    }
                    printf(")");
                }

                printf("\n");
            }
        }
    }
    printf("Characteristic distance = %g\n", ctree->cdist);
    printf("Minimum noise2          = %g\n", ctree->minnoise2);
    printf("Distance count      %lld\n", ctree->cdistcnt);
    printf("Negative distance   %lld (fraction = %f)\n",
           ctree->cdistnegcnt,
           1.0 * ctree->cdistnegcnt / ctree->cdistcnt);

    printf("\n\n");

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
