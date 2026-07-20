// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "CommandLineInterface/CLIcore.h"
#include "clustering_defs.h"

#include "CFmeminit.h"
#include "get_availableCFindex.h"

// log all debug trace points to file
#define DEBUGLOG

/**
 * @brief Create a new leaf
 *
 * @param ctree
 * @param datarray
 * @param ssqr
 * @param CFindex
 * @return errno_t
 */
errno_t create_new_leaf(CLUSTERTREE *ctree, double *datarray, long double ssqr, long *CFindex)
{
    DEBUG_TRACE_FSTART();

    long CFi;
    FUNC_CHECK_RETURN(get_availableCFindex(ctree, &CFi));

    CFmeminit(ctree, CFi, CFMEMINIT_CFUPDATE);

    ctree->CFarray[CFi].type        = CLUSTER_CF_TYPE_LEAF;
    ctree->CFarray[CFi].level       = -1;
    ctree->CFarray[CFi].parentindex = -1;
    ctree->CFarray[CFi].N           = 1;
    memcpy(ctree->CFarray[CFi].datasumvec, datarray, sizeof(double) * ctree->npix);
    ctree->CFarray[CFi].datassq = ssqr;

    ctree->CFarray[CFi].sum2 = ssqr;

    // cluster radius = 0 (single point)
    ctree->CFarray[CFi].radius2 = 0.0;

    // for fixed leaf position
    // will not change as points are added
    memcpy(ctree->CFarray[CFi].dataposvec, datarray, sizeof(double) * ctree->npix);
    // position vector source ID is current CFi
    ctree->CFarray[CFi].posvecsourceID = CFi;

    ctree->CFarray[CFi].pathcnt         = 0.0;
    ctree->CFarray[CFi].pathdistcompcnt = 0.0;

    ctree->CFarray[CFi].status |= CLUSTER_CF_STATUS_CREATE;

    *CFindex = CFi;

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
