/**
 * @file init_fftwplan.c
 * @brief Initialize FFTW plans
 */

#include <fftw3.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "wisdom.h"

// Forward declaration
errno_t init_fftw_plans0();

/* =========================================
 *  V2 PARAMS (no params)
 * ======================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "initfft",
    .cmdkey      = "initfft",
    .description = "init FFTW",
    .description_long =
        "Initialize and cache FFTW plans for a given image size. Pre-computing plans avoids repeated FFTW planning overhead in real-time loops."
};

#define FPS_PARAMS(X)

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING)
    {NULL, NULL, 0, 0, 0, NULL}
};
static const int nb_bindings = 0;

static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG)
};

static CLICMDDATA CLIcmddata = {
    "", "", CLICMD_FIELDS_DEFAULTS
};
static CMDSETTINGS cms = {0};

static __attribute__((constructor))
void init_cms(void)
{
    strncpy(CLIcmddata.key,
            FPS_app_info.cmdkey,
            sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description,
            FPS_app_info.description,
            sizeof(CLIcmddata.description)
            - 1);
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings = &cms;
    }
}

static MILK_HOT errno_t compute_function()
{
    init_fftw_plans0();
    return RETURN_SUCCESS;
}

#ifndef FPS_STANDALONE
static errno_t __attribute__((unused)) CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_milkfft__init_fftwplan()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif


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
