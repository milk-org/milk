
#include "CommandLineInterface/CLIcore.h"
#include "clustering_defs.h"

errno_t addvector_to_CF(CLUSTERTREE *ctree,
                        double      *datavec,
                        long double  ssqr,
                        long         N,
                        long         CFindex,
                        int         *addOK)
{
    DEBUG_TRACE_FSTART();

    double *sumvec = (double *) malloc(sizeof(double) * ctree->npix);

    long   N1   = ctree->CFarray[CFindex].N + N;
    double sum2 = 0.0;

    // add to vec sum
    for(long ii = 0; ii < ctree->npix; ii++)
    {
        sumvec[ii] = ctree->CFarray[CFindex].datasumvec[ii] + datavec[ii];
        sum2 += sumvec[ii] * sumvec[ii];
    }

    long double ssq1 = ctree->CFarray[CFindex].datassq + ssqr;

    // compute cluster radius
    long double tmpv1   = ssq1 / N1;
    long double tmpv2   = sum2 / (N1 * N1);
    double      radius2 = tmpv1 - tmpv2;

    if((radius2 < ctree->T * ctree->T) || (*addOK == 1))
    {
        *addOK = 1;

        for(long ii = 0; ii < ctree->npix; ii++)
        {
            ctree->CFarray[CFindex].datasumvec[ii] = sumvec[ii];
        }

        free(sumvec);

        ctree->CFarray[CFindex].N       = N1;
        ctree->CFarray[CFindex].datassq = ssq1;
        ctree->CFarray[CFindex].sum2    = sum2;
        ctree->CFarray[CFindex].radius2 = radius2;
    }
    else
    {
        *addOK = 0;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

errno_t subvector_to_CF(
    CLUSTERTREE *ctree, double *datavec, long double ssqr, long N, long CFindex)
{
    DEBUG_TRACE_FSTART();

    ctree->CFarray[CFindex].N -= N;

    // sub to vec sum
    ctree->CFarray[CFindex].sum2 = 0.0;
    for(long ii = 0; ii < ctree->npix; ii++)
    {
        ctree->CFarray[CFindex].datasumvec[ii] -= datavec[ii];
        ctree->CFarray[CFindex].sum2 += ctree->CFarray[CFindex].datasumvec[ii] *
                                        ctree->CFarray[CFindex].datasumvec[ii];
    }
    ctree->CFarray[CFindex].datassq -= ssqr;

    // compute cluster radius
    long double tmpv1 =
        ctree->CFarray[CFindex].datassq / ctree->CFarray[CFindex].N;
    long double tmpv2 = ctree->CFarray[CFindex].sum2 /
                        ctree->CFarray[CFindex].N / ctree->CFarray[CFindex].N;
    ctree->CFarray[CFindex].radius2 = tmpv1 - tmpv2;

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
