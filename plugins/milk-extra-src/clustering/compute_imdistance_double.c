#include "CommandLineInterface/CLIcore.h"
#include "clustering_defs.h"

#include <math.h>

errno_t compute_imdistance_double(CLUSTERTREE *ctree,
                                  double      *vec1,
                                  long         N1,
                                  double      *vec2,
                                  long         N2,
                                  double      *distval)
{
    DEBUG_TRACE_FSTART();

    long double dist2 = 0.0;

    static long double cdist2_sum    = 0.0;
    static long long   cdist2_cnt    = 0;
    static long long   dist2_neg_cnt = 0;

    static long double minnoise2_val = -1.0;

    //printf("Computing distance over %ld elements  %ld %ld\n", NBelem, N1, N2);
    //fflush(stdout);

    for(long ii = 0; ii < ctree->npix; ii++)
    {
        double tmpv = vec1[ii] / N1 - vec2[ii] / N2;
        dist2 += tmpv * tmpv;
    }

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

    //printf("    -> %g\n", *distval);

    cdist2_sum += dist2;
    cdist2_cnt++;

    // collect stats
    ctree->cdist       = sqrt(cdist2_sum / cdist2_cnt);
    ctree->minnoise2   = minnoise2_val;
    ctree->cdistcnt    = cdist2_cnt;
    ctree->cdistnegcnt = dist2_neg_cnt;

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
