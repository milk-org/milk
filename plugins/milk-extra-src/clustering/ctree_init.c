// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "CommandLineInterface/CLIcore.h"
#include "clustering_defs.h"

#include "ctree_memallocate.h"

/**
 * @brief Initialize CF tree with first vector
 *
 * @param ctree
 * @param datavector
 * @return errno_t
 */
errno_t ctree_init(
    CLUSTERTREE *ctree,
    double *datavector,
    long double ssqr
)
{
    DEBUG_TRACE_FSTART();


    ctree->rootindex = 0;

    // root is initially a node with single child
    CFmemallocate(ctree, 0);
    ctree->CFarray[0].type          = CLUSTER_CF_TYPE_NODE;
    ctree->CFarray[0].level         = 0;
    ctree->CFarray[0].NBchild       = 1;
    ctree->CFarray[0].childindex[0] = 1;
    ctree->CFarray[0].N             = 1;

    memcpy(ctree->CFarray[0].datasumvec,
           datavector,
           sizeof(double) * ctree->npix);

    memcpy(ctree->CFarray[0].dataposvec,
           datavector,
           sizeof(double) * ctree->npix);

    ctree->CFarray[0].datassq = ssqr;
    ctree->CFarray[0].sum2 = ssqr;
    ctree->CFarray[0].pathcnt = 1.0;
    ctree->CFarray[0].pathdistcompcnt = 0.0;


    // childless leaf node, with single leaf
    CFmemallocate(ctree, 1);
    ctree->CFarray[1].type        = CLUSTER_CF_TYPE_LEAF;
    ctree->CFarray[1].level       = 1;
    ctree->CFarray[1].parentindex = 0;
    ctree->CFarray[1].NBchild     = 0;
    ctree->CFarray[1].N           = 1;


    memcpy(ctree->CFarray[1].datasumvec,
           datavector,
           sizeof(double) * ctree->npix);

    memcpy(ctree->CFarray[1].dataposvec,
           datavector,
           sizeof(double) * ctree->npix);
    ctree->CFarray[1].posvecsourceID = 1;


    ctree->CFarray[1].datassq = ssqr;
    ctree->CFarray[1].sum2 = ssqr;
    ctree->CFarray[1].pathcnt = 1.0;
    ctree->CFarray[1].pathdistcompcnt = 0.0;


    ctree->cdist = 0.0;

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
