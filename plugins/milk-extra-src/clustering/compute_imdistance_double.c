// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "CommandLineInterface/CLIcore.h"
#include "clustering_defs.h"

#include <math.h>


errno_t compute_imdistance_double(
    CLUSTERTREE *ctree,
    double      *vec1,
    long         N1,
    double      *vec2,
    long         N2,
    double      *distval
)
{
    DEBUG_TRACE_FSTART();

    long double dist2 = 0.0;

    static long double cdist2_sum    = 0.0;
    static long long   cdist2_cnt    = 0;
    static long long   dist2_neg_cnt = 0;

    static long double minnoise2_val = -1.0;

#ifdef DEBUGPRINT
    printf("[compute_imdistance_double]   Computing distance over %ld elements  %ld %ld\n",
           ctree->npix, N1, N2);
#endif

    for(long ii = 0; ii < ctree->npix; ii++)
    {
        double tmpv = vec1[ii] / N1 - vec2[ii] / N2;
        dist2 += tmpv * tmpv;
    }
#ifdef DEBUGPRINT
    printf("[compute_imdistance_double]   dist2 = %lf\n", (double) dist2);
#endif

    // keep track of minimum N-corrected distance encountered
    // assuming uncorrelated noise, distance2 is
    // sum of variance/N1 and variance/N2
    // = var * (1/N1 + 1/N2)
    double noise2val = dist2 / (1.0 / N1 + 1.0 / N2);
    if(cdist2_cnt == 1)
    {
        minnoise2_val = noise2val;
    }
    else
    {
        if(noise2val < minnoise2_val)
        {
            minnoise2_val = noise2val;
        }
    }

    dist2 -= ctree->noise2offset * (1.0 / N1 + 1.0 / N2);
    if(dist2 < 0.0)
    {
        dist2_neg_cnt++;
        dist2 = 0.0;
    }

    *distval = (double) sqrt(dist2);

#ifdef DEBUGPRINT
    printf("[compute_imdistance_double]    -> %g\n", *distval);
#endif

    cdist2_sum += dist2;
    cdist2_cnt++;

    // collect stats
    ctree->cdist       = sqrt(cdist2_sum / cdist2_cnt);
    ctree->minnoise2   = minnoise2_val;
    ctree->cdistcnt    = cdist2_cnt;
    ctree->cdistnegcnt = dist2_neg_cnt;


    ctree->stat_compdistcnt ++;

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



// Compute position vector distance between two CFs.
// Will pull from pre-computed value in CFCFdist if availabe.
//
errno_t compute_CF2CF_posdistance_double(
    CLUSTERTREE *ctree,
    long         CFI0,
    long         CFI1,
    double      *distval
)
{
    DEBUG_TRACE_FSTART();

    // Get pos indices
    long posID0 = ctree->CFarray[CFI0].posvecsourceID;
    long posID1 = ctree->CFarray[CFI1].posvecsourceID;

    // Check if distance is available
    double dval = ctree->CFCFdist[posID0*ctree->NBCF+posID1];
    if(dval < 0)
    {
        // compute distance
        compute_imdistance_double(ctree,
                                  ctree->CFarray[posID0].dataposvec,
                                  1,
                                  ctree->CFarray[posID1].dataposvec,
                                  1,
                                  &dval);
        ctree->CFCFdist[posID0*ctree->NBCF+posID1] = dval;
        ctree->CFCFdist[posID1*ctree->NBCF+posID0] = dval;
    }

    *distval = dval;


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}