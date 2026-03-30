/**
 * @file makeCosRadModes.c
 * @brief Makecosradmodes module
 */

#include <math.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "mkcosrmodes",
    .cmdkey      = "mkcosrmodes",
    .description = "make basis of cosine radial modes"
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char   * outimname = NULL;
static long   * sizeout = NULL;
static long   * kmaxval = NULL;
static double * radiusval = NULL;
static double * radfactorlimval = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".outim", &outimname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output image") \
    X(".size", &sizeout, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "size") \
    X(".kmax", &kmaxval, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "k max") \
    X(".radius", &radiusval, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "radius [pix]") \
    X(".rfactlim", &radfactorlimval, \
      FPTYPE_FLOAT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "radius factor limit")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


/**
 * Make basis of cosine radial modes.
 */
errno_t linopt_imtools_makeCosRadModes(
    const char *ID_name,
    long        size,
    long        kmax,
    float       radius,
    float       radfactlim,
    imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    long size2 = size * size;

    IMGID imgr =
        imgid_make_from_name_2D(
            "linopt_tmpr", size, size);
    imgr.mdt->shared = 0;
    imgr.im = (IMAGE *) calloc(
        1, sizeof(IMAGE));
    imgid_mkimage(&imgr);

    FILE *fp =
        fopen("ModesExpr_CosRad.txt", "w");
    fprintf(fp,
            "# unit for r = %f pix\n",
            radius);
    fprintf(fp, "\n");
    for(long k = 0; k < kmax; k++)
    {
        fprintf(fp,
                "%5ld   cos(r*M_PI*%ld)\n",
                k, k);
    }
    fclose(fp);

    for(long ii = 0; ii < size; ii++)
    {
        float x =
            (1.0 * ii - 0.5 * size)
            / radius;
        for(long jj = 0;
            jj < size; jj++)
        {
            float y =
                (1.0 * jj - 0.5 * size)
                / radius;
            float r =
                sqrt(x * x + y * y);
            imgr.im->array.F[
                jj * size + ii] = r;
        }
    }

    IMGID imgout =
        imgid_make_from_name_3D(
            ID_name, size, size, kmax);
    imgout.mdt->shared = 0;
    imgout.im = (IMAGE *) calloc(
        1, sizeof(IMAGE));
    imgid_mkimage(&imgout);

    for(long k = 0; k < kmax; k++)
    {
        for(long ii = 0;
            ii < size2; ii++)
        {
            float r =
                imgr.im->array.F[ii];
            if(r < radfactlim)
            {
                imgout.im->array.F[
                    k * size2 + ii] =
                    cos(r * M_PI * k);
            }
        }
    }

    FUNC_CHECK_RETURN(
        delete_image_ID(
            "linopt_tmpr",
            DELETE_IMAGE_ERRMODE_WARNING));

    if(outID != NULL)
    {
        *outID = imgout.ID;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    linopt_imtools_makeCosRadModes(outimname,
                                   *sizeout,
                                   *kmaxval,
                                   *radiusval,
                                   *radfactorlimval,
                                   NULL);

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
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_linopt_imtools__makeCosRadModes()
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

