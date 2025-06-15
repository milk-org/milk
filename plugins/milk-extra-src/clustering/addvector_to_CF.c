
#include "CommandLineInterface/CLIcore.h"
#include "clustering_defs.h"

#include <math.h>

#define DEBUGPRINT

errno_t addvector_to_CF(
    CLUSTERTREE *ctree,
    double      *datavec,
    long double  ssqr,
    long         N,
    long         CFindex,
    int         *addOK
)
{
    DEBUG_TRACE_FSTART();

    // We first assume the new CF will be added the leaf cluster,
    // recomputing the cluster features sumvec and radius2
    //

    double *sumvec = (double *) malloc(sizeof(double) * ctree->npix);

    // new cluster nb or point
    long   N1   = ctree->CFarray[CFindex].N + N;

    double sum2 = 0.0;

    // add to vec sum
    //
    for(long ii = 0; ii < ctree->npix; ii++)
    {
        sumvec[ii] = ctree->CFarray[CFindex].datasumvec[ii] + datavec[ii];
        sum2 += sumvec[ii] * sumvec[ii];
    }

    // new sum squared
    long double ssq1 = ctree->CFarray[CFindex].datassq + ssqr;


    // compute cluster radius
    // xa = average x = sumvec/N
    // radius2 = sumsqr(xi-xa)/N
    //         =  sumsqr(xi)/N - xa^2
    // with:
    // tmpv1 = sumsqr(xi)/N
    // tmpv2 = xa^2 = sum2/N/N

    long double tmpv1   = ssq1 / N1;
    long double tmpv2   = sum2 / (N1 * N1);
    double      radius2 = tmpv1 - tmpv2;



    double dist2pos2 = 0.0;
    for(long ii = 0; ii < ctree->npix; ii++)
    {
        double dval = ctree->CFarray[CFindex].dataposvec[ii] - datavec[ii];
        dist2pos2 += dval * dval;
    }

    // Check cluster radius

#ifdef DEBUGPRINT
    printf("[CFindex %5ld / N %5ld   <-- N %5ld]  NEW CLUSTER RADIUS = %8lf / %8lf = %8lf\n",
           CFindex, ctree->CFarray[CFindex].N, N, sqrt(radius2),  ctree->T, sqrt(radius2)/ctree->T);
#endif


    {
        double tval2 = 0.0;
        if (ctree->leafposmode == 0)
        {
            // static
            tval2 = dist2pos2;
        }
        else
        {
            // dynamic
            tval2 = radius2;
        }

        if((tval2 < ctree->T * ctree->T) || (*addOK == 1))
        {
            *addOK = 1;

#ifdef DEBUGPRINT
            printf("              >>>>>> ADDING PT TO CLUSTER\n");
#endif

            // if point is added, update CF stats

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
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}







errno_t subvector_to_CF(
    CLUSTERTREE *ctree,
    double *datavec,
    long double ssqr,
    long N,
    long CFindex
)
{
    DEBUG_TRACE_FSTART();

    ctree->CFarray[CFindex].N -= N;

    // subtract to vec sum
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
