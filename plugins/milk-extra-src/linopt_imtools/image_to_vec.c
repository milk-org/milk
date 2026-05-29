/**
 * @file image_to_vec.c
 * @brief Image to vec module
 */

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = { .fps_name    = "im2vec",
                                     .cmdkey      = "im2vec",
                                     .description = "remap image to vector",
                                     .description_long =
                                         "Remap a 2D image to a 1D vector using a pixel table "
                                         "(mask). Extracts active pixels into a compact array." };


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char inimname[FUNCTION_PARAMETER_STRMAXLEN]      = "";
static char inpixiname[FUNCTION_PARAMETER_STRMAXLEN]    = "";
static char inpixmultname[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char outvecname[FUNCTION_PARAMETER_STRMAXLEN]    = "";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                               \
    X(".inim", inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".inpixi", inpixiname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,            \
      "input pixel index image")                                                    \
    X(".inpixmult", inpixmultname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,      \
      "input pixel mult image")                                                     \
    X(".outvec", outvecname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output vector image")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


/**
 * Remap image to vector using pixel
 * index and multiplier tables.
 */
errno_t linopt_imtools_image_to_vec(const char *__restrict ID_name,
                                    const char *__restrict IDpixindex_name,
                                    const char *__restrict IDpixmult_name,
                                    const char *__restrict IDvec_name,
                                    imageID *outID)
{
    DEBUG_TRACE_FSTART();
    DEBUG_TRACEPOINT("FARG %s %s %s %s", ID_name, IDpixindex_name, IDpixmult_name, IDvec_name);

    IMGID imgin = imgid_make_from_name(ID_name);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }

    long    naxisin  = imgin.md->naxis;
    uint8_t datatype = imgin.md->datatype;

    IMGID imgpixi = imgid_make_from_name(IDpixindex_name);
    resolveIMGID(&imgpixi, ERRMODE_WARN, dcimg, dcnimg);
    if (imgpixi.ID == -1)
    {
        return RETURN_FAILURE;
    }

    IMGID imgpixm = imgid_make_from_name(IDpixmult_name);
    resolveIMGID(&imgpixm, ERRMODE_WARN, dcimg, dcnimg);
    if (imgpixm.ID == -1)
    {
        return RETURN_FAILURE;
    }

    long NBpix = imgpixi.md->nelement;

    IMGID imgvec;

    if (naxisin < 3)
    {
        imgvec             = imgid_make_from_name_2D(IDvec_name, NBpix, 1);
        imgvec.mdt->shared = 0;
        imgvec.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
        imgid_mkimage(&imgvec);

        for (long k = 0; k < NBpix; k++)
        {
            imgvec.im->array.F[k] =
                imgpixm.im->array.F[k] * imgin.im->array.F[imgpixi.im->array.SI64[k]];
        }
    }
    else
    {
        long sizexy = imgin.md->size[0] * imgin.md->size[1];

        if (datatype == _DATATYPE_FLOAT)
        {
            imgvec             = imgid_make_from_name_2D(IDvec_name, NBpix, imgin.md->size[2]);
            imgvec.mdt->shared = 0;
            imgvec.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
            imgid_mkimage(&imgvec);

            for (uint32_t kk = 0; kk < imgin.md->size[2]; kk++)
            {
                for (long k = 0; k < NBpix; k++)
                {
                    imgvec.im->array.F[kk * NBpix + k] =
                        imgpixm.im->array.F[k] *
                        imgin.im->array.F[kk * sizexy + imgpixi.im->array.SI64[k]];
                }
            }
        }
        if (datatype == _DATATYPE_COMPLEX_FLOAT)
        {
            imgvec             = imgid_make_from_name_2D(IDvec_name, NBpix * 2, imgin.md->size[2]);
            imgvec.mdt->shared = 0;
            imgvec.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
            imgid_mkimage(&imgvec);

            for (uint32_t kk = 0; kk < imgin.md->size[2]; kk++)
            {
                for (long k = 0; k < NBpix; k++)
                {
                    long idx = imgpixi.im->array.SI64[k];
                    imgvec.im->array.F[kk * NBpix * 2 + 2 * k] =
                        imgpixm.im->array.F[k] * imgin.im->array.CF[kk * sizexy + idx].re;
                    imgvec.im->array.F[kk * NBpix * 2 + 2 * k + 1] =
                        imgpixm.im->array.F[k] * imgin.im->array.CF[kk * sizexy + idx].im;
                }
            }
        }
    }

    if (outID != NULL)
    {
        *outID = imgvec.ID;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    linopt_imtools_image_to_vec(inimname, inpixiname, inpixmultname, outvecname, NULL);

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

errno_t CLIADDCMD_linopt_imtools__image_to_vec()
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
