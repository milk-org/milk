/**
 * @file    image_mk_complex_from_amph.c
 * @brief   amplitude, phase -> complex
 *
 * Uses FPS V2 framework.
 */

#include <math.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info =
{
    .fps_name    = "ap2c",
    .cmdkey      = "ap2c",
    .description = "amplitude, phase -> complex",
    .description_long =
    "Construct a complex image from separate amplitude and phase images. Computes real = amp * cos(pha), imag = amp * sin(pha)."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char inampimname[
     FUNCTION_PARAMETER_STRMAXLEN] = "imamp";
static char inphaimname[
     FUNCTION_PARAMETER_STRMAXLEN] = "impha";
static char outimname[
     FUNCTION_PARAMETER_STRMAXLEN] = "imc";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".imamp_name", inampimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "amplitude image") \
    X(".impha_name", inphaimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "phase image") \
    X(".out_name", outimname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output complex image")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

errno_t mk_complex_from_amph_IMGID(
    IMGID *imginamp,
    IMGID *imginpha,
    IMGID *imgoutC)
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(
        imginamp, ERRMODE_WARN,
        dcimg, dcnimg);
    if(imginamp->ID == -1)
    {
        return RETURN_FAILURE;
    }

    resolveIMGID(
        imginpha, ERRMODE_WARN,
        dcimg, dcnimg);
    if(imginpha->ID == -1)
    {
        return RETURN_FAILURE;
    }


    uint8_t datatype_am = imginamp->md->datatype;
    uint8_t datatype_ph = imginpha->md->datatype;

    uint8_t naxisamp = imginamp->md->naxis;
    uint8_t naxispha = imginpha->md->naxis;
    uint64_t xysize = imginamp->md->size[0];
    imgoutC->mdt->size[0] =
        imginamp->md->size[0];
    imgoutC->mdt->size[1] = 1;

    uint8_t naxis = naxisamp;
    if(naxisamp > 1)
    {
        xysize *= imginamp->md->size[1];
        imgoutC->mdt->size[1] =
            imginamp->md->size[1];
    }
    if(naxispha > naxisamp)
    {
        naxis = naxispha;
    }

    uint32_t zsize    = 1;
    uint32_t zsizeamp = 1;
    uint32_t zsizepha = 1;
    if(naxisamp > 2)
    {
        zsizeamp = imginamp->md->size[2];
    }
    if(naxispha > 2)
    {
        zsizepha = imginpha->md->size[2];
    }
    zsize = zsizeamp;
    if(zsizepha > zsizeamp)
    {
        zsize = zsizepha;
    }

    imgoutC->mdt->naxis = naxis;
    imgoutC->mdt->size[2] = zsize;

    uint8_t datatype_out;

#define MK_COMPLEX_LOOP(DTYPE_OUT, TYPE_AM, TYPE_PH, TYPE_OUT, UNION_AM, UNION_PH, UNION_OUT, COS_FUNC, SIN_FUNC) \
    { \
        datatype_out = DTYPE_OUT; \
        imgoutC->mdt->datatype = datatype_out; \
        if(imgoutC->ID == -1) createimagefromIMGID(imgoutC); \
        imgoutC->md->write = 1; \
        TYPE_AM * MILK_RESTRICT ptr_am = MILK_ASSUME_ALIGNED(imginamp->im->array.UNION_AM); \
        TYPE_PH * MILK_RESTRICT ptr_ph = MILK_ASSUME_ALIGNED(imginpha->im->array.UNION_PH); \
        TYPE_OUT * MILK_RESTRICT ptr_out = MILK_ASSUME_ALIGNED(imgoutC->im->array.UNION_OUT); \
_Pragma("omp parallel if (xysize > OMP_NELEMENT_LIMIT)") \
        { \
_Pragma("omp for simd") \
            for(uint32_t kk = 0; kk < zsize; kk++) \
            { \
                uint32_t kkamp = kk; \
                if(kkamp > zsizeamp - 1) kkamp = zsizeamp - 1; \
                uint32_t kkpha = kk; \
                if(kkpha > zsizepha - 1) kkpha = zsizepha - 1; \
                for(uint64_t ii = 0; ii < xysize; ii++) \
                { \
                    ptr_out[kk*xysize + ii].re = \
                        ptr_am[kkamp*xysize + ii] * COS_FUNC(ptr_ph[kkpha*xysize + ii]); \
                    ptr_out[kk*xysize + ii].im = \
                        ptr_am[kkamp*xysize + ii] * SIN_FUNC(ptr_ph[kkpha*xysize + ii]); \
                } \
            } \
        } \
        imgoutC->md->cnt0++; \
        imgoutC->md->write = 0; \
    }

    if((datatype_am == _DATATYPE_FLOAT)
            && (datatype_ph == _DATATYPE_FLOAT))
    {
        MK_COMPLEX_LOOP(_DATATYPE_COMPLEX_FLOAT, float, float, complex_float, F, F, CF, cosf, sinf)
    }
    else if((datatype_am == _DATATYPE_FLOAT)
            && (datatype_ph == _DATATYPE_DOUBLE))
    {
        MK_COMPLEX_LOOP(_DATATYPE_COMPLEX_DOUBLE, float, double, complex_double, F, D, CD, cos, sin)
    }
    else if((datatype_am == _DATATYPE_DOUBLE)
            && (datatype_ph == _DATATYPE_FLOAT))
    {
        MK_COMPLEX_LOOP(_DATATYPE_COMPLEX_DOUBLE, double, float, complex_double, D, F, CD, cosf, sinf)
    }
    else if((datatype_am == _DATATYPE_DOUBLE)
            && (datatype_ph == _DATATYPE_DOUBLE))
    {
        MK_COMPLEX_LOOP(_DATATYPE_COMPLEX_DOUBLE, double, double, complex_double, D, D, CD, cos, sin)
    }

#undef MK_COMPLEX_LOOP
    else
    {
        PRINT_ERROR("Wrong image type(s)\n");
        return RETURN_FAILURE;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

errno_t mk_complex_from_amph(
    const char *am_name,
    const char *ph_name,
    const char *out_name,
    int        sharedmem)
{
    IMGID imgamp =
        imgid_make_from_name(am_name);
    IMGID imgpha =
        imgid_make_from_name(ph_name);
    IMGID imgoutC =
        imgid_make_from_name(out_name);
    imgoutC.mdt->shared = sharedmem;

    errno_t ret = mk_complex_from_amph_IMGID(
                      &imgamp, &imgpha, &imgoutC);
    imgid_free(&imgamp);
    imgid_free(&imgpha);
    imgid_free(&imgoutC);
    return ret;
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
    DEBUG_TRACE_FSTART();

    IMGID imgamp =
        imgid_make_from_name(inampimname);
    IMGID imgpha =
        imgid_make_from_name(inphaimname);
    IMGID imgoutC =
        imgid_make_from_name(outimname);

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        mk_complex_from_amph_IMGID(
            &imgamp, &imgpha, &imgoutC);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&imgamp);
    imgid_free(&imgpha);
    imgid_free(&imgoutC);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
               &FPS_app_info, farg, &CLIcmddata,
               my_bindings, nb_bindings,
               compute_function);
}

errno_t
CLIADDCMD_COREMOD__mk_complex_from_amph()
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
