
#include "CommandLineInterface/CLIcore.h"
#include "clustering_defs.h"

/**
 * @brief Initialize CF tree with first vector
 *
 * @param ctree
 * @param datavector
 * @return errno_t
 */
errno_t ctree_init(CLUSTERTREE *ctree, double *datavector, long double ssqr)
{
    DEBUG_TRACE_FSTART();

    ctree->rootindex = 0;

    // root is initially a node with single child
    ctree->CFarray[0].type          = CLUSTER_CF_TYPE_NODE;
    ctree->CFarray[0].level         = 0;
    ctree->CFarray[0].NBchild       = 1;
    ctree->CFarray[0].childindex[0] = 1;
    ctree->CFarray[0].NBleaf        = 0;
    ctree->CFarray[0].N             = 1;

    ctree->CFarray[1].type         = CLUSTER_CF_TYPE_LEAFNODE;
    ctree->CFarray[1].level        = 1;
    ctree->CFarray[1].parentindex  = 0;
    ctree->CFarray[1].NBchild      = 0;
    ctree->CFarray[1].NBleaf       = 1;
    ctree->CFarray[1].leafindex[0] = 2;
    ctree->CFarray[1].N            = 1;

    // childless leaf node, with single leaf
    ctree->CFarray[2].type        = CLUSTER_CF_TYPE_LEAF;
    ctree->CFarray[2].level       = 2;
    ctree->CFarray[2].parentindex = 1;
    ctree->CFarray[2].NBchild     = 0;
    ctree->CFarray[2].NBleaf      = 0;
    ctree->CFarray[2].N           = 1;

    memcpy(ctree->CFarray[0].datasumvec,
           datavector,
           sizeof(double) * ctree->npix);
    memcpy(ctree->CFarray[1].datasumvec,
           datavector,
           sizeof(double) * ctree->npix);
    memcpy(ctree->CFarray[2].datasumvec,
           datavector,
           sizeof(double) * ctree->npix);
    ctree->CFarray[0].datassq = ssqr;
    ctree->CFarray[1].datassq = ssqr;
    ctree->CFarray[2].datassq = ssqr;

    ctree->CFarray[0].sum2 = ssqr;
    ctree->CFarray[1].sum2 = ssqr;
    ctree->CFarray[2].sum2 = ssqr;

    ctree->cdist = 0.0;

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
