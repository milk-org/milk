#include "CommandLineInterface/CLIcore.h"
#include "clustering_defs.h"

#include "CFmeminit.h"
#include "update_level.h"

//#define DEBUGPRINT

/**
 * @brief Condense tree
 *
 * If a node has fewer than B grandchildren, then children can be skipped
 * to reduce tree depth.
 *
 * @param ctree     pointer to tree
 * @return errno_t
 */
errno_t ctree_condense(CLUSTERTREE *ctree, int *nbop)
{
    DEBUG_TRACE_FSTART();

    *nbop = 0;

    for(long cfi = 0; cfi < ctree->NBCF; cfi++)
    {
        if(ctree->CFarray[cfi].type == CLUSTER_CF_TYPE_NODE)
        {
            // count nb childred and grandchildren
            long nbgchild = 0;

            for(int chi = 0; chi < ctree->CFarray[cfi].NBchild; chi++)
            {
                long cfic = ctree->CFarray[cfi].childindex[chi];

                if(ctree->CFarray[cfic].type == CLUSTER_CF_TYPE_NODE)
                {
                    int ngchi = ctree->CFarray[cfic].NBchild;
                    nbgchild += ngchi;
                }
                else
                {
                    nbgchild = -1;
                    break;
                }
            }

            if((nbgchild > 0) && (nbgchild < ctree->B))
            {
                printf(
                    "CONDENSING: NODE %5ld LEVEL %d    #child=%5d -> "
                    "#gchild=%5ld\n",
                    cfi,
                    ctree->CFarray[cfi].level,
                    ctree->CFarray[cfi].NBchild,
                    nbgchild);

                long  gchildcnt = 0;
                long *gchiCFi   = (long *) malloc(sizeof(long) * nbgchild);
                if(gchiCFi == NULL)
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

                    int ngchi = ctree->CFarray[cfic].NBchild;
                    for(int gchi = 0; gchi < ngchi; gchi++)
                    {
                        long gchiCFindex =
                            ctree->CFarray[cfic].childindex[gchi];
                        gchiCFi[gchildcnt] = gchiCFindex;

                        gchildcnt++;
                    }

                    // remove child
#ifdef DEBUGPRINT
                    printf("%s      removing node %5ld\n", __func__, cfic);
#endif
                    CFmeminit(ctree, cfic, 0);
                }

                // update number of child
                ctree->CFarray[cfi].NBchild = gchildcnt;
                for(long gchi = 0; gchi < gchildcnt; gchi++)
                {
#ifdef DEBUGPRINT
                    printf("%s      node %5ld new parent is %5ld\n",
                           __func__,
                           gchiCFi[gchi],
                           cfi);
#endif
                    // point grandchild to gandparent
                    ctree->CFarray[cfi].childindex[gchi]      = gchiCFi[gchi];
                    ctree->CFarray[gchiCFi[gchi]].parentindex = cfi;
                }

                free(gchiCFi);

                // update level of downstream nodes
                update_level(ctree, cfi);

                // report that one condense operation has been done
                *nbop = 1;
            }

            if(*nbop > 0)
            {
                // return from function
                // only one condense operation at a time
                DEBUG_TRACE_FEXIT();
                return RETURN_SUCCESS;
            }
        }
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
