/**
 * @file lin1Dfit.c
 * @brief Lin1dfit module
 */

#include <math.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "image_fitModes.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "linopt1Dfit",
    .cmdkey      = "linopt1Dfit",
    .description = "least-square 1D fit",
    .description_long =
        "Perform a least-squares 1D polynomial fit to data. Supports configurable polynomial order."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char *infname     = NULL;
static long *NBptval     = NULL;
static long *maxorderval = NULL;
static char *outfname    = NULL;
static long *modeval     = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                      \
    X(".indat", &infname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "input file")            \
    X(".NBpt", &NBptval, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "number of sample points") \
    X(".maxorder", &maxorderval, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT,                    \
      "maximum polynomial order")                                                          \
    X(".outdat", &outfname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output file")         \
    X(".mode", &modeval, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "fit mode")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


// MODE :
// 0 : polynomial
//
errno_t linopt_compute_1Dfit(const char *fnamein,
                             long        NBpt,
                             long        MaxOrder,
                             const char *fnameout,
                             int         MODE,
                             imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    imageID IDin, IDin0;
    imageID IDmask;
    imageID IDmodes;
    long    NBmodes;

    float SVDeps = 0.0000001;

    long  NBiter = 100;
    float gain   = 1.0;
    long  iter;

    //atexit(milk_memclean);

    float *__restrict xarray;
    xarray = (float *) malloc(sizeof(float) * NBpt);
    if (xarray == NULL)
    {
        FUNC_RETURN_FAILURE("malloc returns NULL pointer");
    }
    //milk_atexit_free_add(xarray);

    float *__restrict valarray;
    valarray = (float *) malloc(sizeof(float) * NBpt);
    if (valarray == NULL)
    {
        FUNC_RETURN_FAILURE("malloc returns NULL pointer");
    }
    //milk_atexit_free_add(valarray);


    {
        FILE *fp;
        fp = fopen(fnamein, "r");
        for (int_fast32_t ii = 0; ii < NBpt; ii++)
        {
            int fscanfcnt = fscanf(fp, "%f %f\n", &xarray[ii], &valarray[ii]);

            if (fscanfcnt == EOF)
            {
                if (ferror(fp))
                {
                    perror("fscanf");
                }
                else
                {
                    fprintf(stderr, "Error: fscanf reached end of file, no matching "
                                    "characters, no matching failure\n");
                }
                exit(EXIT_FAILURE);
            }
            else if (fscanfcnt != 2)
            {
                fprintf(stderr,
                        "Error: fscanf successfully matched and assigned %i input "
                        "items, 2 expected\n",
                        fscanfcnt);
                exit(EXIT_FAILURE);
            }
        }
        fclose(fp);
    }

    FUNC_CHECK_RETURN(create_2Dimage_ID("invect", NBpt, 1, &IDin));

    FUNC_CHECK_RETURN(create_2Dimage_ID("invect0", NBpt, 1, &IDin0));

    FUNC_CHECK_RETURN(create_2Dimage_ID("inmask", NBpt, 1, &IDmask));

    for (uint_fast32_t ii = 0; ii < NBpt; ii++)
    {
        //			printf("%18.16f  %+18.16f\n", xarray[ii], valarray[ii]);
        dcimg[IDin].array.F[ii]   = valarray[ii];
        dcimg[IDin0].array.F[ii]  = valarray[ii];
        dcimg[IDmask].array.F[ii] = 1.0f;
    }

    NBmodes = MaxOrder;
    FUNC_CHECK_RETURN(create_3Dimage_ID("fitmodes", NBpt, 1, NBmodes, &IDmodes));

    imageID IDout;
    FUNC_CHECK_RETURN(create_2Dimage_ID("outcoeff", NBmodes, 1, &IDout));

    switch (MODE)
    {
    case 0:
        for (uint_fast32_t m = 0; m < NBmodes; m++)
        {
            for (uint_fast32_t ii = 0; ii < NBpt; ii++)
            {
                float v;
                if (m == 0)
                {
                    v = 1.0f;
                }
                else if (m == 1)
                {
                    v = xarray[ii];
                }
                else
                {
                    v = powf(xarray[ii], (float) m);
                }
                dcimg[IDmodes].array.F[m * NBpt + ii] = v;
            }
        }
        break;
    case 1:
        for (uint_fast32_t m = 0; m < NBmodes; m++)
        {
            for (uint_fast32_t ii = 0; ii < NBpt; ii++)
            {
                dcimg[IDmodes].array.F[m * NBpt + ii] = cos(xarray[ii] * M_PI * m);
            }
        }
        break;
    default:
        printf("ERROR: MODE = %d not supported\n", MODE);
        exit(0);
        break;
    }


    for (iter = 0; iter < NBiter; iter++)
    {
        FUNC_CHECK_RETURN(linopt_imtools_image_fitModes("invect0", "fitmodes", "inmask", SVDeps,
                                                        "outcoeffim0", 1, NULL));
        imageID IDout0 = image_ID("outcoeffim0", dcimg, dcnimg);

        for (uint_fast32_t m = 0; m < NBmodes; m++)
        {
            dcimg[IDout].array.F[m] += gain * dcimg[IDout0].array.F[m];
        }

        double err = 0.0;
        for (int_fast32_t ii = 0; ii < NBpt; ii++)
        {
            double val = 0.0;
            for (int_fast32_t m = 0; m < NBmodes; m++)
            {
                val += dcimg[IDout].array.F[m] * dcimg[IDmodes].array.F[m * NBpt + ii];
            }
            dcimg[IDin0].array.F[ii] = dcimg[IDin].array.F[ii] - val;
            err += dcimg[IDin0].array.F[ii] * dcimg[IDin0].array.F[ii];
        }
        err = sqrt(err / NBpt);
        printf("ITERATION %4ld   residual = %20g   [gain = %20g]\n", iter, err, gain);
        gain *= 0.95;
    }

    {
        FILE *fp;
        fp = fopen(fnameout, "w");
        for (long m = 0; m < NBmodes; m++)
        {
            fprintf(fp, "%4ld %+.8g\n", m, dcimg[IDout].array.F[m]);
        }
        fclose(fp);
    }

    {
        FILE *fp;
        fp         = fopen("testout.txt", "w");
        double err = 0.0;
        for (int_fast32_t ii = 0; ii < NBpt; ii++)
        {
            double val = 0.0;
            for (int_fast32_t m = 0; m < NBmodes; m++)
            {
                val += dcimg[IDout].array.F[m] * dcimg[IDmodes].array.F[m * NBpt + ii];
            }
            double vale = valarray[ii] - val;
            err += vale * vale;
            fprintf(fp, "%05ld  %18.16f  %18.16f   %18.16f\n", ii, xarray[ii], valarray[ii], val);
        }
        fclose(fp);

        err = sqrt(err / NBpt);
        printf("FIT error = %g m\n", err);
    }

    free(xarray);
    free(valarray);

    if (outID != NULL)
    {
        *outID = IDout;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    linopt_compute_1Dfit(infname, *NBptval, *maxorderval, outfname, *modeval, NULL);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#ifndef FPS_STANDALONE
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_linopt_imtools__lin1Dfits()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif
