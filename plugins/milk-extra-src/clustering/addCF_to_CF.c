// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "CommandLineInterface/CLIcore.h"
#include "clustering_defs.h"

#include <math.h>

// #define DEBUGPRINT

/**
 * @brief Combine Cluster Feature with existing CF and compute stats
 *
 * @param ctree
 * @param CF
 * @param CFindex
 * @param combCF
 * @return errno_t
 */
static errno_t combCFcomp(CLUSTERTREE *ctree, CLUSTERING_CF CF, long CFindex, CLUSTERING_CF *combCF)
{
    DEBUG_TRACE_FSTART();

    combCF->sum2 = 0.0;

    for (long ii = 0; ii < ctree->npix; ii++)
    {
        combCF->datasumvec[ii] = ctree->CFarray[CFindex].datasumvec[ii] + CF.datasumvec[ii];
        combCF->sum2 += combCF->datasumvec[ii] * combCF->datasumvec[ii];
    }

    // new sum squared
    combCF->datassq = ctree->CFarray[CFindex].datassq + CF.datassq;

    // compute cluster radius
    // xa = average x = sumvec/N1
    // radius2 = sumsqr(xi-xa)/N1
    //         =  sumsqr(xi)/N1 - xa^2
    // with:
    // tmpv1 = sumsqr(xi)/N1
    // tmpv2 = xa^2 = sum2/N1/N1
    //
    long double tmpv1 = combCF->datassq / combCF->N;
    long double tmpv2 = combCF->sum2 / (combCF->N * combCF->N);
    combCF->radius2   = tmpv1 - tmpv2;

    combCF->pathcnt         = ctree->CFarray[CFindex].pathcnt + CF.pathcnt;
    combCF->pathdistcompcnt = ctree->CFarray[CFindex].pathdistcompcnt + CF.pathdistcompcnt;

    // printf("pathcnt:  %16f  <-  [%5ld] %16f  %16f\n",  combCF->pathcnt,
    // CFindex, ctree->CFarray[CFindex].pathcnt, CF.pathcnt);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

/**
 * @brief Add Cluster Feature to CF
 *
 * If adding a single vector, it's datasumvec is used
 *
 * @param ctree
 * @param CF
 * @param CFindex
 * @param addOK
 * @return errno_t
 */
errno_t addCF_to_CF(CLUSTERTREE *ctree, CLUSTERING_CF CF, long CFindex, int *addOK)
{
    DEBUG_TRACE_FSTART();

    // allocate position ONLY if first entry
    if (ctree->CFarray[CFindex].N == 0)
    {
        memcpy(ctree->CFarray[CFindex].dataposvec, CF.dataposvec, sizeof(double) * ctree->npix);
        ctree->CFarray[CFindex].posvecsourceID = CF.posvecsourceID;
    }

    CLUSTERING_CF combCF;

    // new cluster nb or point
    combCF.N = ctree->CFarray[CFindex].N + CF.N;

    if (ctree->leafposmode == CLUSTER_CFPOS_FIXED)
    {
        combCF.datasumvec = (double *) malloc(sizeof(double) * ctree->npix);

        combCFcomp(ctree, CF, CFindex, &combCF);

        double dist2pos2 = 0.0;
        for (long ii = 0; ii < ctree->npix; ii++)
        {
            double dval = ctree->CFarray[CFindex].dataposvec[ii] - CF.datasumvec[ii];
            dist2pos2 += dval * dval;
        }
        if ((dist2pos2 < ctree->T * ctree->T) || (*addOK == 1))
        {
            *addOK = 1;

            // if point is added, update CF stats
            // dynamic, update sumvec
            for (long ii = 0; ii < ctree->npix; ii++)
            {
                ctree->CFarray[CFindex].datasumvec[ii] = combCF.datasumvec[ii];
            }
            ctree->CFarray[CFindex].datassq = combCF.datassq;
            ctree->CFarray[CFindex].sum2    = combCF.sum2;
            ctree->CFarray[CFindex].radius2 = combCF.radius2;

            ctree->CFarray[CFindex].N = combCF.N;

            ctree->CFarray[CFindex].pathcnt         = combCF.pathcnt;
            ctree->CFarray[CFindex].pathdistcompcnt = combCF.pathdistcompcnt;
        }
        else
        {
            *addOK = 0;
        }
    }
    else
    {
        // We first assume the new CF will be added the leaf cluster,
        // recomputing the cluster features sumvec and radius2
        //
        combCF.datasumvec = (double *) malloc(sizeof(double) * ctree->npix);

        combCFcomp(ctree, CF, CFindex, &combCF);

        // Check cluster radius
        if ((combCF.radius2 < ctree->T * ctree->T) || (*addOK == 1))
        {
            *addOK = 1;

            // if point is added, update CF stats
            // dynamic, update sumvec
            for (long ii = 0; ii < ctree->npix; ii++)
            {
                ctree->CFarray[CFindex].datasumvec[ii] = combCF.datasumvec[ii];
            }
            ctree->CFarray[CFindex].datassq = combCF.datassq;
            ctree->CFarray[CFindex].sum2    = combCF.sum2;
            ctree->CFarray[CFindex].radius2 = combCF.radius2;

            ctree->CFarray[CFindex].N = combCF.N;

            ctree->CFarray[CFindex].pathcnt         = combCF.pathcnt;
            ctree->CFarray[CFindex].pathdistcompcnt = combCF.pathdistcompcnt;
        }
        else
        {
            *addOK = 0;
        }
        free(combCF.datasumvec);
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

/**
 * @brief Subtract vector from CF
 *
 * @param ctree
 * @param CF
 * @param CFindex
 * @return errno_t
 */
errno_t subvector_to_CF(CLUSTERTREE *ctree, CLUSTERING_CF CF, long CFindex)
{
    DEBUG_TRACE_FSTART();

    ctree->CFarray[CFindex].N -= CF.N;

    // subtract to vec sum
    ctree->CFarray[CFindex].sum2 = 0.0;
    for (long ii = 0; ii < ctree->npix; ii++)
    {
        ctree->CFarray[CFindex].datasumvec[ii] -= CF.datasumvec[ii];
        ctree->CFarray[CFindex].sum2 +=
            ctree->CFarray[CFindex].datasumvec[ii] * ctree->CFarray[CFindex].datasumvec[ii];
    }
    ctree->CFarray[CFindex].datassq -= CF.datassq;

    ctree->CFarray[CFindex].pathcnt -= CF.pathcnt;
    ctree->CFarray[CFindex].pathdistcompcnt -= CF.pathdistcompcnt;

    // recompute cluster radius2
    long double tmpv1 = ctree->CFarray[CFindex].datassq / ctree->CFarray[CFindex].N;
    long double tmpv2 =
        ctree->CFarray[CFindex].sum2 / ctree->CFarray[CFindex].N / ctree->CFarray[CFindex].N;
    ctree->CFarray[CFindex].radius2 = tmpv1 - tmpv2;

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
