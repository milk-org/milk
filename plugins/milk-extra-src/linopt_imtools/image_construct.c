/**
 * @file image_construct.c
 * @brief Image construct module
 */


#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "imlinconstruct",
    .cmdkey           = "imlinconstruct",
    .description      = "construct image as linear sum of modes",
    .description_long = "Reconstruct an image as a linear combination of mode images. Applies "
                        "coefficient vector to a mode cube to produce the output."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char modesimname[FUNCTION_PARAMETER_STRMAXLEN];
static char invecname[FUNCTION_PARAMETER_STRMAXLEN];
static char outimname[FUNCTION_PARAMETER_STRMAXLEN];


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                        \
    X(".modes", modesimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "modes image cube") \
    X(".invec", invecname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input vector")       \
    X(".outim", outimname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


/**
 * Construct an image as linear sum of
 * modes weighted by coefficients.
 */
errno_t linopt_imtools_image_construct(const char *IDmodes_name,
                                       const char *IDcoeff_name,
                                       const char *ID_name,
                                       imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    IMGID imgmodes = imgid_make_from_name(IDmodes_name);
    resolveIMGID(&imgmodes, ERRMODE_ABORT, dcimg, dcnimg);

    uint8_t  datatype = imgmodes.md->datatype;
    uint32_t xsize    = imgmodes.md->size[0];
    uint32_t ysize    = imgmodes.md->size[1];
    uint32_t zsize    = imgmodes.md->size[2];
    uint64_t sizexy   = xsize;
    sizexy *= ysize;

    IMGID imgout         = imgid_make_from_name_2D(ID_name, xsize, ysize);
    imgout.mdt->shared   = 0;
    imgout.mdt->datatype = datatype;
    imgout.im            = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    IMGID imgcoeff = imgid_make_from_name(IDcoeff_name);
    resolveIMGID(&imgcoeff, ERRMODE_ABORT, dcimg, dcnimg);

    if (datatype == _DATATYPE_FLOAT)
    {
        memset(imgout.im->array.F, 0, sizeof(float) * imgout.md->nelement);
        for (uint32_t kk = 0; kk < zsize; kk++)
        {
            for (uint64_t ii = 0; ii < sizexy; ii++)
            {
                imgout.im->array.F[ii] +=
                    imgcoeff.im->array.F[kk] * imgmodes.im->array.F[kk * sizexy + ii];
            }
        }
    }
    else
    {
        memset(imgout.im->array.D, 0, sizeof(double) * imgout.md->nelement);
        for (uint32_t kk = 0; kk < zsize; kk++)
        {
            for (uint64_t ii = 0; ii < sizexy; ii++)
            {
                imgout.im->array.D[ii] +=
                    imgcoeff.im->array.D[kk] * imgmodes.im->array.D[kk * sizexy + ii];
            }
        }
    }

    if (outID != NULL)
    {
        *outID = imgout.ID;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    linopt_imtools_image_construct(modesimname, invecname, outimname, NULL);
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

errno_t CLIADDCMD_linopt_imtools__image_construct()
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
