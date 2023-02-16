
#include "CommandLineInterface/CLIcore.h"
#include "clustering_defs.h"

#include "addvector_to_CF.h"

// log all debug trace points to file
#define DEBUGLOG

/**
 * @brief Add entry to leaf
 *
 * @param ctree
 * @param datavec
 * @param ssqr
 * @param lCFindex
 * @return errno_t
 */
errno_t leaf_addentry(CLUSTERTREE *ctree,
                      double      *datavec,
                      long double  ssqr,
                      long         lCFindex,
                      int         *addOK)
{
    DEBUG_TRACE_FSTART();

    long cfi = lCFindex;
    while(cfi != -1)
    {
        addvector_to_CF(ctree, datavec, ssqr, 1, cfi, addOK);

        if(*addOK == 1)
        {
            ctree->CFarray[cfi].status |= CLUSTER_CF_STATUS_UPDATE;
            // move upstream to propagate change
            cfi = ctree->CFarray[cfi].parentindex;
        }
        else
        {
            cfi = -1;
        }
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
