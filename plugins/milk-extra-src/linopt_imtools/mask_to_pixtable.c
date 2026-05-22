/**
 * @file mask_to_pixtable.c
 * @brief Mask to pixtable module
 */

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "mask2pixtable",
    .cmdkey           = "mask2pixtable",
    .description      = "make pixel tables from mask",
    .description_long = "Convert a 2D binary mask into a pixel index table. Lists coordinates of "
                        "active (non-zero) pixels for efficient vectorized access."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char *inimname      = NULL;
static char *outpixiimname = NULL;
static char *outpixmimname = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                           \
    X(".inim", &inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image")            \
    X(".outpixi", &outpixiimname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output index image") \
    X(".outpixm", &outpixmimname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output mask image")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


/**
 * Create pixel index and multiplier tables
 * from mask image for vectorized operations.
 */
errno_t linopt_imtools_mask_to_pixtable(const char *IDmask_name,
                                        const char *IDpixindex_name,
                                        const char *IDpixmult_name,
                                        long       *outNBpix)
{
    DEBUG_TRACE_FSTART();

    float eps = 1.0e-8;

    IMGID imgmask = imgid_make_from_name(IDmask_name);
    resolveIMGID(&imgmask, ERRMODE_ABORT, dcimg, dcnimg);

    long size = imgmask.md->nelement;

    long NBpix = 0;
    for (long ii = 0; ii < size; ii++)
    {
        if (imgmask.im->array.F[ii] > eps)
        {
            NBpix++;
        }
    }

    /* Create INT64 index table */
    IMGID imgpixi         = imgid_make_from_name(IDpixindex_name);
    imgpixi.mdt->naxis    = 2;
    imgpixi.mdt->size[0]  = NBpix;
    imgpixi.mdt->size[1]  = 1;
    imgpixi.mdt->datatype = _DATATYPE_INT64;
    imgpixi.mdt->shared   = 0;
    imgpixi.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgpixi);

    /* Create FLOAT multiplier table */
    IMGID imgpixm         = imgid_make_from_name(IDpixmult_name);
    imgpixm.mdt->naxis    = 2;
    imgpixm.mdt->size[0]  = NBpix;
    imgpixm.mdt->size[1]  = 1;
    imgpixm.mdt->datatype = _DATATYPE_FLOAT;
    imgpixm.mdt->shared   = 0;
    imgpixm.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgpixm);

    long k = 0;
    for (long ii = 0; ii < size; ii++)
    {
        if (imgmask.im->array.F[ii] > eps)
        {
            imgpixi.im->array.SI64[k] = ii;
            imgpixm.im->array.F[k]    = imgmask.im->array.F[ii];
            k++;
        }
    }

    if (outNBpix != NULL)
    {
        *outNBpix = NBpix;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    linopt_imtools_mask_to_pixtable(inimname, outpixiimname, outpixmimname, NULL);

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

errno_t CLIADDCMD_linopt_imtools__mask_to_pixtable()
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
