/**
 * @file    image_norm.c
 * @brief   Compute per-slice norm of an image
 *
 * Uses FPS V2 framework.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore.h"
#include "fps.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "normslice",
    .cmdkey      = "normslice",
    .description = "image norm by slice"
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     *norm_inimname  = NULL;
static char     *norm_outimname = NULL;
static uint32_t *norm_sliceaxis = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".in0name", &norm_inimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input image 0") \
    X(".outname", &norm_outimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output image") \
    X(".axis", &norm_sliceaxis, \
      FPTYPE_UINT32, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "norm axis")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

/**
 * @brief Compute L2 norm per slice along axis.
 *
 * Pure computation — takes resolved IMGIDs.
 */
static errno_t image_slicenorm_IMGID(
    IMGID *inimg,
    IMGID *outimg,
    uint8_t sliceaxis)
{
    if (outimg->ID == -1) {
        imgid_copy(inimg, outimg);
    }
    for (uint8_t ax = 0;
         ax < inimg->md->naxis; ax++)
    {
        if (ax != sliceaxis) {
            outimg->mdt->size[ax] = 1;
        }
    }
    outimg->mdt->datatype = _DATATYPE_FLOAT;

    /* Create output stream */
    outimg->im =
        (IMAGE *) malloc(sizeof(IMAGE));
    strncpy(outimg->name,
            norm_outimname, 79);
    ImageStreamIO_createIm_gpu(
        outimg->im, outimg->name,
        outimg->mdt->naxis,
        outimg->mdt->size,
        outimg->mdt->datatype,
        -1, 1, 10, 0, 0, 0);
    outimg->md = outimg->im->md;

    uint32_t sizes[3] = {
        inimg->md->size[0],
        inimg->md->size[1],
        inimg->md->size[2]
    };
    if (inimg->md->naxis < 3) {
        sizes[2] = 1;
    }
    if (inimg->md->naxis < 2) {
        sizes[1] = 1;
    }
    double *normarray = (double *)
        calloc(sizes[sliceaxis],
               sizeof(double));
    for (uint32_t i = 0;
         i < sizes[0]; i++)
    {
        for (uint32_t j = 0;
             j < sizes[1]; j++)
        {
            for (uint32_t k = 0;
                 k < sizes[2]; k++)
            {
                uint64_t idx =
                    (uint64_t) k
                    * sizes[1] * sizes[0]
                    + (uint64_t) j * sizes[0]
                    + i;
                double v = 0;
                switch (
                    inimg->mdt->datatype)
                {
                case _DATATYPE_FLOAT:
                    v = inimg->im->array.F[
                        idx];
                    break;
                case _DATATYPE_DOUBLE:
                    v = inimg->im->array.D[
                        idx];
                    break;
                }
                uint32_t coords[3] =
                    {i, j, k};
                normarray[
                    coords[sliceaxis]]
                    += v * v;
            }
        }
    }
    for (uint32_t i = 0;
         i < sizes[sliceaxis]; i++)
    {
        outimg->im->array.F[i] =
            sqrt(normarray[i]);
    }
    free(normarray);
    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING)
};

static const int nb_bindings =
    sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG)
};

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
#else
static CLICMDDATA CLIcmddata = {
#endif
    "",
    "",
    CLICMD_FIELDS_DEFAULTS
};

static CMDSETTINGS default_cmdsettings = {0};

static __attribute__((constructor))
void init_cmdsettings(void)
{
    strncpy(CLIcmddata.key,
            FPS_app_info.cmdkey,
            sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description,
            FPS_app_info.description,
            sizeof(CLIcmddata.description) - 1);
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings =
            &default_cmdsettings;
    }
}


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static errno_t compute_function()
{
    if (!norm_sliceaxis) {
        return RETURN_FAILURE;
    }
    IMGID idin =
        imgid_make_from_name(norm_inimname);
    resolveIMGID(
        &idin, ERRMODE_ABORT,
        data.image, data.NB_MAX_IMAGE);
    IMGID idout =
        imgid_make_from_name(norm_outimname);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    image_slicenorm_IMGID(
        &idin, &idout, *norm_sliceaxis);
    processinfo_update_output_stream(
        processinfo, idout.im, idin.im);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&idin);
    imgid_free(&idout);
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#ifndef FPS_STANDALONE
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_COREMOD_arith__image_normslice()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    FPS_app_info,
    FPS_PARAMS,
    compute_function)
#endif