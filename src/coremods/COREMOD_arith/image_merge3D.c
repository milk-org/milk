/**
 * @file    image_merge3D.c
 * @brief   Merge images along an axis
 *
 * Uses FPS V2 framework.
 */

#include <stdlib.h>
#include <string.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "immerge",
    .cmdkey      = "immerge",
    .description = "merge images along axis",
    .description_long =
        "Concatenate multiple 2D or 3D images along a specified axis to form a higher-dimensional "
        "data cube. Input images must have matching dimensions on all other axes."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     *immerge_inimname0 = NULL;
static char     *immerge_inimname1 = NULL;
static char     *immerge_outimname = NULL;
static uint32_t *immerge_mergeaxis = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                              \
    X(".in0name", &immerge_inimname0, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image 0") \
    X(".in1name", &immerge_inimname1, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image 1") \
    X(".outname", &immerge_outimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "output image")  \
    X(".axis", &immerge_mergeaxis, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "merge axis")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

/**
 * @brief Merge two images along the specified axis.
 *
 * Creates output stream if needed and copies
 * data from both inputs into it.
 */
static MILK_HOT errno_t fpsexec(IMGID *id0, IMGID *id1, IMGID *idout)
{
    if (!immerge_mergeaxis)
    {
        return RETURN_FAILURE;
    }
    uint8_t mergeaxis = (uint8_t) *immerge_mergeaxis;

    if (idout->ID == -1)
    {
        imgid_copy(id0, idout);
    }
    if (mergeaxis < 3)
    {
        uint32_t s0 = (id0->md->size[mergeaxis] == 0) ? 1 : id0->md->size[mergeaxis];
        uint32_t s1 = (id1->md->size[mergeaxis] == 0) ? 1 : id1->md->size[mergeaxis];
        idout->mdt->size[mergeaxis] = s0 + s1;
    }
    else
    {
        return RETURN_FAILURE;
    }
    idout->mdt->naxis = (idout->mdt->size[2] > 1) ? 3 : ((idout->mdt->size[1] > 1) ? 2 : 1);

    /* Create output stream */
    idout->im = (IMAGE *) calloc(1, sizeof(IMAGE));
    strncpy(idout->name, immerge_outimname, 79);
    ImageStreamIO_createIm_gpu(idout->im, idout->name, idout->mdt->naxis, idout->mdt->size,
                               idout->mdt->datatype, -1, 1, 10, 0, 0, 0);
    idout->md = idout->im->md;

    size_t ts = ImageStreamIO_typesize(idout->mdt->datatype);

    if (mergeaxis == idout->mdt->naxis - 1)
    {
        size_t sz0 = ts * id0->md->nelement;
        __builtin_memcpy(idout->im->array.raw, id0->im->array.raw, sz0);
        __builtin_memcpy(((char *) idout->im->array.raw) + sz0, id1->im->array.raw,
                         ts * id1->md->nelement);
    }
    else
    {
        uint64_t b0 = id0->mdt->size[0];
        uint64_t b1 = id1->mdt->size[0];
        if (mergeaxis == 1)
        {
            b0 *= id0->mdt->size[1];
            b1 *= id1->mdt->size[1];
        }
        uint64_t po = 0, p0 = 0, p1 = 0;
        while (po < idout->md->nelement)
        {
            __builtin_memcpy(((char *) idout->im->array.raw) + po * ts,
                             ((char *) id0->im->array.raw) + p0 * ts, ts * b0);
            p0 += b0;
            po += b0;
            __builtin_memcpy(((char *) idout->im->array.raw) + po * ts,
                             ((char *) id1->im->array.raw) + p1 * ts, ts * b1);
            p1 += b1;
            po += b1;
        }
    }
    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    IMGID id0 = imgid_make_from_name(immerge_inimname0);
    resolveIMGID(&id0, ERRMODE_ABORT, dcimg, dcnimg);
    IMGID id1 = imgid_make_from_name(immerge_inimname1);
    resolveIMGID(&id1, ERRMODE_ABORT, dcimg, dcnimg);
    IMGID idout = imgid_make_from_name(immerge_outimname);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START fpsexec(&id0, &id1, &idout);
    processinfo_update_output_stream(processinfo, idout.im, id0.im);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END imgid_free(&id0);
    imgid_free(&id1);
    imgid_free(&idout);
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

errno_t CLIADDCMD_COREMOD_arith__image_merge()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif
