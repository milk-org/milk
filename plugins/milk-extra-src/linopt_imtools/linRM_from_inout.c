/**
 * @file linRM_from_inout.c
 * @brief Linrm from inout module
 */

#include <math.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"

#ifdef USE_CFITSIO
#include "COREMOD_iofits/savefits.h"
#endif

#include "compute_SVDpseudoInverse.h"
#include "linalgebra/magma_compute_SVDpseudoInverse.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = { .fps_name = "lincRMiter",
                                     .cmdkey   = "lincRMiter",
                                     .description =
                                         "estimate response matrix from input and output",
                                     .description_long =
                                         "Estimate a response matrix from measured input-output "
                                         "pairs using least-squares linear regression." };


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char inputimname[FUNCTION_PARAMETER_STRMAXLEN];
static char inmaskname[FUNCTION_PARAMETER_STRMAXLEN];
static char mrespimname[FUNCTION_PARAMETER_STRMAXLEN];
static char outRMimname[FUNCTION_PARAMETER_STRMAXLEN];


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                      \
    X(".inimname", inputimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".inmaskname", inmaskname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "mask image") \
    X(".mrespimname", mrespimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,             \
      "measured response images")                                                          \
    X(".outRM", outRMimname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output RM image")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)

//
// solve for response matrix given a series of input and output
// initial value of RM should be best guess
// inmask = 0 over input that are known to produce no response
//
errno_t linopt_compute_linRM_from_inout(const char *IDinput_name,
                                        const char *IDinmask_name,
                                        const char *IDoutput_name,
                                        const char *IDRM_name,
                                        imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    imageID IDRM;
    imageID IDin;
    imageID IDinmask;
    imageID IDout;
    long    insize; // number of input
    long    xsizein, ysizein, xsizeout, ysizeout;
    double  fitval;
    long    kk, ii_in, jj_in, ii_out, jj_out;
    //double tot;
    imageID IDtmp;
    double  tmpv1;
    //long iter;
    imageID IDout1;
    //double alpha = 0.001;

    uint32_t *sizearray;
    imageID   IDpokeM; // poke matrix (input)
    //imageID IDoutM; // outputX
    double SVDeps = 1.0e-4;

    long    NBact, act;
    long   *inpixarray;
    long    spl; // sample measurement
    long    ii;
    imageID ID_rm;
    int     autoMask_MODE = 0; // if 1, automatically measure input mask based on IDinput_name image
    imageID IDpinv;
    //int use_magma = 0;

    //int ngpu;

    //ngpu = 0;
    setenv("CUDA_VISIBLE_DEVICES", "3,4", 1);

    IDin  = image_ID(IDinput_name, dcimg, dcnimg);
    IDout = image_ID(IDoutput_name, dcimg, dcnimg);
    IDRM  = image_ID(IDRM_name, dcimg, dcnimg);

    insize   = dcimg[IDin].md[0].size[2];
    xsizeout = dcimg[IDRM].md[0].size[0];
    ysizeout = dcimg[IDRM].md[0].size[1];
    xsizein  = dcimg[IDin].md[0].size[0];
    ysizein  = dcimg[IDin].md[0].size[1];

    if (autoMask_MODE == 0)
    {
        IDinmask = image_ID(IDinmask_name, dcimg, dcnimg);
    }
    else
    {
        create_2Dimage_ID("_RMmask", xsizein, ysizein, &IDinmask);
        for (spl = 0; spl < insize; spl++)
        {
            for (ii = 0; ii < xsizein * ysizein; ii++)
            {
                if (dcimg[IDin].array.F[spl * xsizein * ysizein + ii] > 0.5)
                {
                    dcimg[IDinmask].array.F[ii] = 1.0f;
                }
            }
        }
    }

    // create pokeM
    NBact = 0;
    for (ii = 0; ii < xsizein * ysizein; ii++)
    {
        if (dcimg[IDinmask].array.F[ii] > 0.5f)
        {
            NBact++;
        }
    }

    printf("NBact = %ld\n", NBact);

    inpixarray = (long *) malloc(sizeof(long) * NBact);
    if (inpixarray == NULL)
    {
        FUNC_RETURN_FAILURE("malloc returns NULL pointer");
    }

    act = 0;
    for (ii = 0; ii < xsizein * ysizein; ii++)
    {
        if (dcimg[IDinmask].array.F[ii] > 0.5f)
        {
            inpixarray[act] = ii;
            act++;
        }
    }

    sizearray = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    if (sizearray == NULL)
    {
        FUNC_RETURN_FAILURE("malloc returns NULL pointer");
    }

    sizearray[0] = NBact;
    sizearray[1] = insize; // number of measurements

    printf("NBact = %ld\n", NBact);
    for (act = 0; act < 10; act++)
    {
        printf("act %5ld -> pix %5ld\n", act, inpixarray[act]);
    }

    create_2Dimage_ID("pokeM", NBact, insize, &IDpokeM);

    for (spl = 0; spl < insize; spl++)
    {
        for (act = 0; act < NBact; act++)
        {
            dcimg[IDpokeM].array.F[NBact * spl + act] =
                dcimg[IDin].array.F[spl * xsizein * ysizein + inpixarray[act]];
        }
    }
#ifdef USE_CFITSIO
    save_fits("pokeM", "_test_pokeM.fits");
#endif

    // compute pokeM pseudo-inverse
#ifdef HAVE_MAGMA
    LINALGEBRA_magma_compute_SVDpseudoInverse("pokeM", "pokeMinv", SVDeps, insize, "VTmat", 0, 0,
                                              64,
                                              0, // GPU device
                                              NULL);
#else
    linopt_compute_SVDpseudoInverse("pokeM", "pokeMinv", SVDeps, insize, "VTmat", NULL);
#endif

    list_image_ID();
#ifdef USE_CFITSIO
    save_fits("pokeMinv", "pokeMinv.fits");
#endif
    IDpinv = image_ID("pokeMinv", dcimg, dcnimg);

    // multiply measurements by pokeMinv
    create_3Dimage_ID("_respmat", xsizeout, ysizeout, xsizein * ysizein, &ID_rm);

    for (act = 0; act < NBact; act++)
    {
        for (kk = 0; kk < insize; kk++)
        {
            for (ii = 0; ii < xsizeout * ysizeout; ii++)
            {
                dcimg[ID_rm].array.F[inpixarray[act] * xsizeout * ysizeout + ii] +=
                    dcimg[IDout].array.F[kk * xsizeout * ysizeout + ii] *
                    dcimg[IDpinv].array.F[kk * NBact + act];
            }
        }
    }
#ifdef USE_CFITSIO
    save_fits("_respmat", "_test_RM.fits");
#endif
    //exit(0);

    // COMPUTE SOLUTION QUALITY

    IDRM = image_ID("_respmat", dcimg, dcnimg);

    create_2Dimage_ID("_tmplicli", xsizeout, ysizeout, &IDtmp);
    create_3Dimage_ID("testout", xsizeout, ysizeout, insize, &IDout1);

    printf("IDin  = %ld\n", IDin);
    printf("IDout = %ld\n", IDout);
    printf("IDinmask = %ld\n", IDinmask);

    // on iteration 0, compute initial fit value
    fitval = 0.0;

    for (kk = 0; kk < insize; kk++)
    {
        printf("\r kk = %5ld / %5ld    ", kk, insize);
        fflush(stdout);

        for (ii_out = 0; ii_out < xsizeout; ii_out++)
        {
            for (jj_out = 0; jj_out < ysizeout; jj_out++)
            {
                dcimg[IDtmp].array.F[jj_out * xsizeout + ii_out] = 0.0f;
            }
        }

        for (ii_in = 0; ii_in < xsizein; ii_in++)
        {
            for (jj_in = 0; jj_in < ysizein; jj_in++)
            {
                //printf("%ld  pix %ld %ld active\n", kk, ii_in, jj_in);
                for (ii_out = 0; ii_out < xsizeout; ii_out++)
                {
                    for (jj_out = 0; jj_out < ysizeout; jj_out++)
                    {
                        dcimg[IDtmp].array.F[jj_out * xsizeout + ii_out] +=
                            dcimg[IDin].array.F[kk * xsizein * ysizein + jj_in * xsizein + ii_in] *
                            dcimg[IDRM].array.F[(jj_in * xsizein + ii_in) * xsizeout * ysizeout +
                                                jj_out * xsizeout + ii_out];
                    }
                }
            }
        }
        for (ii_out = 0; ii_out < xsizeout; ii_out++)
        {
            for (jj_out = 0; jj_out < ysizeout; jj_out++)
            {
                tmpv1 = dcimg[IDtmp].array.F[jj_out * xsizeout + ii_out] -
                        dcimg[IDout].array.F[kk * xsizeout * ysizeout + jj_out * xsizeout + ii_out];
                fitval += tmpv1 * tmpv1;
                dcimg[IDout1].array.F[kk * xsizeout * ysizeout + jj_out * xsizeout + ii_out] =
                    tmpv1; //dcimg[IDtmp].array.F[jj_out*xsizeout+ii_out];
            }
        }
    }
    printf("\n");
    printf("  %5ld    fitval = %.20f\n", kk, sqrt(fitval / xsizeout / ysizeout));

    delete_image_ID("_tmplicli", DELETE_IMAGE_ERRMODE_WARNING);

    free(sizearray);
    free(inpixarray);

    if (outID != NULL)
    {
        *outID = IDout;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    linopt_compute_linRM_from_inout(inputimname, inmaskname, mrespimname, outRMimname, NULL);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_linopt_imtools__linRM_from_inout()
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
