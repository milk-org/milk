/**
 * @file    image_mk_reim_from_complex.c
 * @brief   complex -> re, im
 *
 * Uses FPS V2 framework.
 */


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
    .fps_name    = "c2ri",
    .cmdkey      = "c2ri",
    .description = "complex -> re, im",
    .description_long =
    "Decompose a complex image into its real and imaginary components. Input is a complex-valued stream; outputs are two real-valued streams."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char inimname[
     FUNCTION_PARAMETER_STRMAXLEN] = "imc";
static char outreimname[
     FUNCTION_PARAMETER_STRMAXLEN] = "imre";
static char outimimname[
     FUNCTION_PARAMETER_STRMAXLEN] = "imim";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".imre_name", inimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input complex image") \
    X(".imim_name", outreimname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output real image") \
    X(".out_name", outimimname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output imaginary image")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

errno_t mk_reim_from_complex_IMGID(
    IMGID *imgin,
    IMGID *imgre,
    IMGID *imgim
)
{
    DEBUG_TRACE_FSTART();

    uint8_t datatype;

    resolveIMGID(
        imgin, ERRMODE_ABORT,
        dcimg, dcnimg);
    datatype = imgin->md[0].datatype;
    uint8_t naxis = imgin->md[0].naxis;
    for(int i = 0; i < naxis; i++)
    {
        imgre->mdt->size[i] =
            imgin->md[0].size[i];
        imgim->mdt->size[i] =
            imgin->md[0].size[i];
    }
    imgre->mdt->naxis = naxis;
    imgim->mdt->naxis = naxis;

    uint64_t nelement = imgin->md[0].nelement;

#define MK_REIM_LOOP(DTYPE_OUT, TYPE_IN, TYPE_OUT, UNION_IN, UNION_OUT) \
    { \
        imgre->mdt->datatype = DTYPE_OUT; \
        if(imgre->ID == -1) createimagefromIMGID(imgre); \
        imgim->mdt->datatype = DTYPE_OUT; \
        if(imgim->ID == -1) createimagefromIMGID(imgim); \
        imgre->md[0].write = 1; \
        imgim->md[0].write = 1; \
        TYPE_IN * MILK_RESTRICT ptr_in = MILK_ASSUME_ALIGNED(imgin->im->array.UNION_IN); \
        TYPE_OUT * MILK_RESTRICT ptr_re = MILK_ASSUME_ALIGNED(imgre->im->array.UNION_OUT); \
        TYPE_OUT * MILK_RESTRICT ptr_im = MILK_ASSUME_ALIGNED(imgim->im->array.UNION_OUT); \
_Pragma("omp parallel if (nelement > OMP_NELEMENT_LIMIT)") \
        { \
_Pragma("omp for simd") \
            for(uint64_t ii = 0; ii < nelement; ii++) \
            { \
                ptr_re[ii] = ptr_in[ii].re; \
                ptr_im[ii] = ptr_in[ii].im; \
            } \
        } \
        if(imgre->md[0].shared == 1) COREMOD_MEMORY_image_set_sempost_byID(imgre->ID, -1); \
        if(imgim->md[0].shared == 1) COREMOD_MEMORY_image_set_sempost_byID(imgim->ID, -1); \
        imgre->md[0].cnt0++; \
        imgim->md[0].cnt0++; \
        imgre->md[0].write = 0; \
        imgim->md[0].write = 0; \
    }

    if(datatype == _DATATYPE_COMPLEX_FLOAT)
    {
        MK_REIM_LOOP(_DATATYPE_FLOAT, complex_float, float, CF, F)
    }
    else if(datatype == _DATATYPE_COMPLEX_DOUBLE)
    {
        MK_REIM_LOOP(_DATATYPE_DOUBLE, complex_double, double, CD, D)
    }

#undef MK_REIM_LOOP
    else
    {
        PRINT_ERROR("Wrong image type(s)\n");
        return RETURN_FAILURE;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

errno_t mk_reim_from_complex(
    const char *in_name,
    const char *re_name,
    const char *im_name,
    int        sharedmem)
{
    IMGID imgin =
        imgid_make_from_name(in_name);
    IMGID imgre =
        imgid_make_from_name(re_name);
    IMGID imgim =
        imgid_make_from_name(im_name);
    imgre.mdt->shared = sharedmem;
    imgim.mdt->shared = sharedmem;

    errno_t ret = mk_reim_from_complex_IMGID(
                      &imgin, &imgre, &imgim);
    imgid_free(&imgin);
    imgid_free(&imgre);
    imgid_free(&imgim);
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

    IMGID imgin =
        imgid_make_from_name(inimname);
    IMGID imgre =
        imgid_make_from_name(outreimname);
    IMGID imgim =
        imgid_make_from_name(outimimname);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    mk_reim_from_complex_IMGID(
        &imgin, &imgre, &imgim);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&imgin);
    imgid_free(&imgre);
    imgid_free(&imgim);

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
CLIADDCMD_COREMOD__mk_reim_from_complex()
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
