#include "CommandLineInterface/CLIcore.h"
#include "clustering_defs.h"

#include "CFmeminit.h"
#include "update_level.h"

#define DEBUGPRINT




/**
 * @brief Condense tree
 *
 * If a node has fewer than B grandchildren, then children can be skipped
 * to reduce tree depth.
 *
 * @param ctree     pointer to tree
 * @return errno_t
 */
errno_t ctree_condense(
    CLUSTERTREE *ctree,
    int *nbop
)
{
    DEBUG_TRACE_FSTART();

#ifdef DEBUGPRINT
    printf("Condensing CF tree\n");
#endif



    *nbop = 0;

    for(long cfi = 0; cfi < ctree->NBCF; cfi++)
    {
        if(ctree->CFarray[cfi].type == CLUSTER_CF_TYPE_NODE)
        {
#ifdef DEBUGPRINT
            printf("\nInspecting node [%3ld] NBchild = %5d\n", cfi, ctree->CFarray[cfi].NBchild);
#endif
            // if we were to condense this node, how many children would it have ?
            long nbnewchild = 0;

            // current number of child node and child leaf
            //long nbchildnode = 0;
            //long nbchildleaf = 0;

            for(int chi = 0; chi < ctree->CFarray[cfi].NBchild; chi++)
            {
                // scan children
                // cfic is child index
                long cfic = ctree->CFarray[cfi].childindex[chi];

                if(ctree->CFarray[cfic].type == CLUSTER_CF_TYPE_NODE)
                {
                    //nbchildnode++;
#ifdef DEBUGPRINT
                    printf("    [%ld] adding CF %5ld with %2d gchildren, type NODE\n", cfi, cfic, ctree->CFarray[cfic].NBchild);
#endif
                    int ngchi = ctree->CFarray[cfic].NBchild;
                    nbnewchild += ngchi;
                }
                else if (ctree->CFarray[cfic].type == CLUSTER_CF_TYPE_LEAF)
                {
#ifdef DEBUGPRINT
                    printf("    [%ld] adding CF %5ld with %2d gchildren, type LEAF\n", cfi, cfic, ctree->CFarray[cfic].NBchild);
#endif
                    //nbchildleaf++;
                    nbnewchild++;
                }
            }
#ifdef DEBUGPRINT
            printf(">> node [%3ld] nbnewchild = %5ld\n", cfi, nbnewchild);
#endif


            // If the total number of descendents is between 1 and B, we can condense (compress levels)
            //
            if((nbnewchild > 0) && (nbnewchild < ctree->B) && (nbnewchild > ctree->CFarray[cfi].NBchild))
            {
#ifdef DEBUGPRINT
                printf(
                    "CONDENSING: NODE [%5ld] LEVEL %2d    #child = %5d -> %5ld\n",
                    cfi,
                    ctree->CFarray[cfi].level,
                    ctree->CFarray[cfi].NBchild,
                    nbnewchild);
#endif

                long  nchild = 0;
                long *nchiCFI   = (long *) malloc(sizeof(long) * nbnewchild);
                if(nchiCFI == NULL)
                {
                    FUNC_RETURN_FAILURE("malloc error");
                }



                for(int chi = 0; chi < ctree->CFarray[cfi].NBchild; chi++)
                {
                    long cfic = ctree->CFarray[cfi].childindex[chi];
#ifdef DEBUGPRINT
                    printf(
                        "%s      processing node %5ld child %5d / %5d : node "
                        "%5ld\n",
                        __func__,
                        cfi,
                        chi,
                        ctree->CFarray[cfi].NBchild,
                        cfic);
#endif


                    if(ctree->CFarray[cfic].type == CLUSTER_CF_TYPE_NODE)
                    {
#ifdef DEBUGPRINT
                        printf("    [%ld] adding %d gchildren from %ld, type NODE\n", cfi, ctree->CFarray[cfic].NBchild, cfic);
#endif
                        int ngchi = ctree->CFarray[cfic].NBchild;
                        for(int gchi = 0; gchi < ngchi; gchi++)
                        {
                            long gchiCFindex = ctree->CFarray[cfic].childindex[gchi];
                            nchiCFI[nchild] = gchiCFindex;
                            nchild++;
                        }

                        // remove child
#ifdef DEBUGPRINT
                        printf("%s      removing node %5ld\n", __func__, cfic);
#endif
                        CFmeminit(ctree, cfic, 0);
                    }
                    else if (ctree->CFarray[cfic].type == CLUSTER_CF_TYPE_LEAF)
                    {
                        nchiCFI[nchild] = cfic;
                        nchild++;
                    }
                }


                // update children
                for(int nchi=0; nchi<nbnewchild; nchi++)
                {
                    ctree->CFarray[cfi].childindex[nchi] = nchiCFI[nchi];
                    ctree->CFarray[nchiCFI[nchi]].parentindex = cfi;
                }
                // update number of child
                ctree->CFarray[cfi].NBchild = nbnewchild;

                // update level of downstream nodes
                update_level(ctree, cfi);


                free(nchiCFI);
                // report that one condense operation has been done
                *nbop = 1;
            }
        }

        if(*nbop > 0)
        {
            // return from function
            // only one condense operation at a time
            DEBUG_TRACE_FEXIT();
            return RETURN_SUCCESS;
        }
    }



    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
