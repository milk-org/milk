/**
 * @file    image_mk_amph_from_complex.c
 * @brief   complex -> amplitude, phase
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
    .fps_name    = "c2ap",
    .cmdkey      = "c2ap",
    .description = "complex -> ampl, pha",
    .description_long =
    "Decompose a complex image into its amplitude and phase components. Input is a complex-valued stream; outputs are two real-valued streams."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char inimname[FUNCTION_PARAMETER_STRMAXLEN] = "imc";
static char outampimname[FUNCTION_PARAMETER_STRMAXLEN] = "imamp";
static char outphaimname[FUNCTION_PARAMETER_STRMAXLEN] = "impha";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".imre_name", inimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input complex image") \
    X(".imim_name", outampimname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output amplitude image") \
    X(".out_name", outphaimname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output phase image")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

errno_t mk_amph_from_complex_IMGID(
    IMGID *imgin,
    IMGID *imgamp,
    IMGID *imgpha)
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(imgin, ERRMODE_ABORT, dcimg, dcnimg);
    uint8_t datatype = imgin->md[0].datatype;
    uint8_t naxis    = imgin->md[0].naxis;

    for(uint8_t i = 0; i < naxis; i++)
    {
        imgamp->mdt->size[i] = imgin->md[0].size[i];
        imgpha->mdt->size[i] = imgin->md[0].size[i];
    }
    imgamp->mdt->naxis = naxis;
    imgpha->mdt->naxis = naxis;

    uint64_t nelement = imgin->md[0].nelement;

#define MK_AMPH_LOOP(DTYPE_OUT, TYPE_IN, TYPE_OUT, UNION_IN, UNION_OUT, SQRT_FUNC, ATAN2_FUNC) \
    { \
        imgamp->mdt->datatype = DTYPE_OUT; \
        if(imgamp->ID == -1) createimagefromIMGID(imgamp); \
        imgpha->mdt->datatype = DTYPE_OUT; \
        if(imgpha->ID == -1) createimagefromIMGID(imgpha); \
        imgamp->md[0].write = 1; \
        imgpha->md[0].write = 1; \
        TYPE_IN * MILK_RESTRICT ptr_in = MILK_ASSUME_ALIGNED(imgin->im->array.UNION_IN); \
        TYPE_OUT * MILK_RESTRICT ptr_am = MILK_ASSUME_ALIGNED(imgamp->im->array.UNION_OUT); \
        TYPE_OUT * MILK_RESTRICT ptr_ph = MILK_ASSUME_ALIGNED(imgpha->im->array.UNION_OUT); \
_Pragma("omp parallel if (nelement > OMP_NELEMENT_LIMIT)") \
        { \
_Pragma("omp for simd") \
            for(uint64_t ii = 0; ii < nelement; ii++) \
            { \
                TYPE_OUT re_val = ptr_in[ii].re; \
                TYPE_OUT im_val = ptr_in[ii].im; \
                ptr_am[ii] = SQRT_FUNC(re_val * re_val + im_val * im_val); \
                ptr_ph[ii] = ATAN2_FUNC(im_val, re_val); \
            } \
        } \
        if(imgamp->md[0].shared == 1) COREMOD_MEMORY_image_set_sempost_byID(imgamp->ID, -1); \
        if(imgpha->md[0].shared == 1) COREMOD_MEMORY_image_set_sempost_byID(imgpha->ID, -1); \
        imgamp->md[0].cnt0++; \
        imgpha->md[0].cnt0++; \
        imgamp->md[0].write = 0; \
        imgpha->md[0].write = 0; \
    }

    if(datatype == _DATATYPE_COMPLEX_FLOAT)
    {
        MK_AMPH_LOOP(_DATATYPE_FLOAT, complex_float, float, CF, F, sqrtf, atan2f)
    }
    else if(datatype == _DATATYPE_COMPLEX_DOUBLE)
    {
        MK_AMPH_LOOP(_DATATYPE_DOUBLE, complex_double, double, CD, D, sqrt, atan2)
    }

#undef MK_AMPH_LOOP
    else
    {
        PRINT_ERROR("Wrong image type(s)\n");
        return RETURN_FAILURE;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

errno_t mk_amph_from_complex(
    const char *in_name,
    const char *am_name,
    const char *ph_name,
    int        sharedmem)
{
    IMGID imgin = imgid_make_from_name(in_name);
    IMGID imgamp = imgid_make_from_name(am_name);
    IMGID imgpha = imgid_make_from_name(ph_name);
    imgamp.mdt->shared = sharedmem;
    imgpha.mdt->shared = sharedmem;

    errno_t ret = mk_amph_from_complex_IMGID(&imgin, &imgamp, &imgpha);
    imgid_free(&imgin);
    imgid_free(&imgamp);
    imgid_free(&imgpha);
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

    IMGID imgin = imgid_make_from_name(inimname);
    IMGID imgamp = imgid_make_from_name(outampimname);
    IMGID imgpha = imgid_make_from_name(outphaimname);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START  mk_amph_from_complex_IMGID(&imgin, &imgamp, &imgpha);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END  imgid_free(&imgin);
    imgid_free(&imgamp);
    imgid_free(&imgpha);

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
               &FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings, compute_function);
}

errno_t
CLIADDCMD_COREMOD__mk_amph_from_complex()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS;
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
