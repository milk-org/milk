
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "libmilkdata/milkdata.h"
#include "milkDebugTools.h"
#include "fps.h"
#include "ImageStreamIO/ImageStreamIO.h"
#endif
#include "COREMOD_memory/COREMOD_memory.h"
#include "clustering_defs.h"

#include "addCF_to_CF.h"

//#define DEBUGPRINT





/**
 * @brief Attach EXISITNG leaf to node
 *
 * @param ctree             clustering tree
 * @param CFindexleaf       CF index of leaf
 * @param CFindexnode   CF index of node to which leaf should be attached
 * @return errno_t
 */
errno_t
node_attachleaf(
    CLUSTERTREE *ctree,
    long CFindexleaf,
    long CFindexnode
)
{
    DEBUG_TRACE_FSTART();

#ifdef DEBUGPRINT
    printf("node_attachleaf  %ld %ld\n", CFindexleaf, CFindexnode);
#endif


    ctree->CFarray[CFindexnode].childindex[ctree->CFarray[CFindexnode].NBchild] = CFindexleaf;
    ctree->CFarray[CFindexnode].NBchild++;



    ctree->CFarray[CFindexleaf].parentindex = CFindexnode;
    ctree->CFarray[CFindexleaf].level = ctree->CFarray[CFindexnode].level + 1;




    long cfi = CFindexnode;
    while(cfi != -1)
    {
#ifdef DEBUGPRINT
        printf("========= ADDING VECTOR TO NODE %ld (%s)\n", cfi, __FILE__);
#endif


        ctree->CFarray[cfi].status |= CLUSTER_CF_STATUS_UPDATE;

        int addOK = 1; // don't test radius
        addCF_to_CF(ctree,
                        ctree->CFarray[CFindexleaf],
                        cfi,
                        &addOK);




        // move upstream to propagate change

        long cfip = ctree->CFarray[cfi].parentindex;
        if(cfi == cfip)
        {
#ifdef DEBUGPRINT
            printf(" cfi %ld  cfip %ld\n", cfi, cfip);
#endif

            FUNC_RETURN_FAILURE(
                "Attaching leaf %ld to node %ld: CF parent %ld points to "
                "itself",
                CFindexleaf,
                CFindexnode,
                cfi);
        }

        cfi = cfip;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
