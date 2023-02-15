/**
 * @file init_fftwplan.c
 */

#include <fftw3.h>

#include "CommandLineInterface/CLIcore.h"

#include "wisdom.h"

// ==========================================
// Forward declaration(s)
// ==========================================

errno_t init_fftw_plans0();

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t init_fftwplan_addCLIcmd()
{

    RegisterCLIcommand("initfft",
                       __FILE__,
                       init_fftw_plans0,
                       "init FFTW",
                       "no argument",
                       "initfft",
                       "int init_fftw_plans0()");

    return RETURN_SUCCESS;
}

errno_t init_fftw_plans(int mode)
{
    int n;
    int size;

    fftwf_complex *inf  = NULL;
    fftwf_complex *outf = NULL;
    float         *rinf = NULL;

    fftw_complex *ind  = NULL;
    fftw_complex *outd = NULL;
    double       *rind = NULL;

    unsigned int plan_mode;

    printf("Optimization of FFTW\n");
    printf(
        "The optimization is done for 2D complex to complex FFTs, with size "
        "equal to 2^n x 2^n\n");
    printf(
        "You can kill the optimization anytime, and resume later where it "
        "previously stopped.\nAfter each size is "
        "optimized, the result is saved\n");
    printf(
        "It might be a good idea to run this overnight or when your computer "
        "is not busy\n");

    fflush(stdout);

    size = 1;

    //  plan_mode = FFTWOPTMODE;
    plan_mode = FFTW_EXHAUSTIVE;

    for(n = 0; n < 14; n++)
    {
        if(mode == 0)
        {
            printf("Optimizing 2D FFTs - size = %d\n", size);
            fflush(stdout);
        }
        rinf = (float *) fftwf_malloc(size * size * sizeof(float));
        inf =
            (fftwf_complex *) fftwf_malloc(size * size * sizeof(fftwf_complex));
        outf =
            (fftwf_complex *) fftwf_malloc(size * size * sizeof(fftwf_complex));

        fftwf_plan_dft_2d(size, size, inf, outf, FFTW_FORWARD, plan_mode);
        fftwf_plan_dft_2d(size, size, inf, outf, FFTW_BACKWARD, plan_mode);
        fftwf_plan_dft_r2c_2d(size, size, rinf, outf, plan_mode);

        fftwf_free(inf);
        fftwf_free(rinf);
        fftwf_free(outf);

        rind = (double *) fftw_malloc(size * size * sizeof(double));
        ind  = (fftw_complex *) fftw_malloc(size * size * sizeof(fftw_complex));
        outd = (fftw_complex *) fftw_malloc(size * size * sizeof(fftw_complex));

        fftw_plan_dft_2d(size, size, ind, outd, FFTW_FORWARD, plan_mode);
        fftw_plan_dft_2d(size, size, ind, outd, FFTW_BACKWARD, plan_mode);
        fftw_plan_dft_r2c_2d(size, size, rind, outd, plan_mode);

        fftw_free(ind);
        fftw_free(rind);
        fftw_free(outd);

        size *= 2;
        if(mode == 0)
        {
            export_wisdom();
        }
    }
    size = 1;
    for(n = 0; n < 15; n++)
    {
        if(mode == 0)
        {
            printf("Optimizing 1D FFTs - size = %d\n", size);
            fflush(stdout);
        }
        rinf = (float *) fftwf_malloc(size * sizeof(float));
        inf  = (fftwf_complex *) fftwf_malloc(size * sizeof(fftwf_complex));
        outf = (fftwf_complex *) fftwf_malloc(size * sizeof(fftwf_complex));

        fftwf_plan_dft_1d(size, inf, outf, FFTW_FORWARD, plan_mode);
        fftwf_plan_dft_1d(size, inf, outf, FFTW_BACKWARD, plan_mode);
        fftwf_plan_dft_r2c_1d(size, rinf, outf, plan_mode);

        fftwf_free(inf);
        fftwf_free(rinf);
        fftwf_free(outf);

        rind = (double *) fftw_malloc(size * sizeof(double));
        ind  = (fftw_complex *) fftw_malloc(size * sizeof(fftw_complex));
        outd = (fftw_complex *) fftw_malloc(size * sizeof(fftw_complex));

        fftw_plan_dft_1d(size, ind, outd, FFTW_FORWARD, plan_mode);
        fftw_plan_dft_1d(size, ind, outd, FFTW_BACKWARD, plan_mode);
        fftw_plan_dft_r2c_1d(size, rind, outd, plan_mode);

        fftw_free(ind);
        fftw_free(rind);
        fftw_free(outd);

        size *= 2;
        if(mode == 0)
        {
            export_wisdom();
        }
    }

    export_wisdom();

    return RETURN_SUCCESS;
}

errno_t init_fftw_plans0()
{
    init_fftw_plans(0);

    return RETURN_SUCCESS;
}
