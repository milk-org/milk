/**
 * @file    stream_ave.c
 * @brief   Average stream of images
 *
 * Uses FPS V2 framework.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "streamave",
    .cmdkey      = "streamave",
    .description =
        "average stream of images"
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     streamave_inimname[
    FUNCTION_PARAMETER_STRMAXLEN] = "instream";
static char     streamave_outimave[
    FUNCTION_PARAMETER_STRMAXLEN] = "outave";
static uint32_t streamave_outimshared = 0;
static char     streamave_outimrms[
    FUNCTION_PARAMETER_STRMAXLEN] = "outrms";
static uint64_t streamave_NBcoadd     = 100;
static uint64_t streamave_cntindex    = 0;
static uint64_t streamave_compave     = 1;
static uint64_t streamave_comprms     = 0;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".in_name", streamave_inimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input image") \
    X(".outave_name", streamave_outimave, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output average image") \
    X(".out_shared", &streamave_outimshared, \
      FPTYPE_UINT32, 0, \
      FPFLAG_DEFAULT_INPUT, \
      "output shared flag") \
    X(".outrms_name", streamave_outimrms, \
      FPTYPE_STREAMNAME, 0, \
      FPFLAG_DEFAULT_INPUT, \
      "output RMS image") \
    X(".NBcoadd", &streamave_NBcoadd, \
      FPTYPE_UINT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "number of coadded") \
    X(".cntindex", &streamave_cntindex, \
      FPTYPE_UINT64, 0, \
      FPFLAG_DEFAULT_INPUT, \
      "counter index") \
    X(".comp.ave", &streamave_compave, \
      FPTYPE_ONOFF, 0, \
      FPFLAG_DEFAULT_INPUT, \
      "compute average") \
    X(".comp.rms", &streamave_comprms, \
      FPTYPE_ONOFF, 0, \
      FPFLAG_DEFAULT_INPUT, \
      "compute rms")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

/**
 * @brief Accumulate pixel values and compute
 *        average/RMS when coadd count reached.
 *
 * Uses heap-allocated double buffers for
 * accumulation to avoid overflow.
 */
static MILK_HOT errno_t fpsexec(
    IMAGE  *imgin,
    IMAGE  *imgoutave,
    IMAGE  *imgoutrms,
    double *imdataarray,
    double *imdataarrayPOW)
{
    uint64_t xysize =
        imgin->md[0].size[0]
        * imgin->md[0].size[1];

    #define STREAM_AVE_LOOP(VTYPE, ARRAY_MEMBER) \
    { \
        const VTYPE * MILK_RESTRICT in = MILK_ASSUME_ALIGNED(imgin->array.ARRAY_MEMBER); \
        if (streamave_cntindex == 0) { \
            for (uint64_t i = 0; i < xysize; i++) { \
                MILK_PREFETCH(&in[i + 8], 0, 0); \
                double v = (double)in[i]; \
                imdataarray[i] = v; \
                if (streamave_comprms) { \
                    imdataarrayPOW[i] = v * v; \
                } \
            } \
        } else { \
            for (uint64_t i = 0; i < xysize; i++) { \
                MILK_PREFETCH(&in[i + 8], 0, 0); \
                double v = (double)in[i]; \
                imdataarray[i] += v; \
                if (streamave_comprms) { \
                    imdataarrayPOW[i] += v * v; \
                } \
            } \
        } \
    }

    if (imgin->md[0].datatype == _DATATYPE_FLOAT) {
        STREAM_AVE_LOOP(float, F);
    } else if (imgin->md[0].datatype == _DATATYPE_UINT16) {
        STREAM_AVE_LOOP(uint16_t, UI16);
    } else if (imgin->md[0].datatype == _DATATYPE_UINT8) {
        STREAM_AVE_LOOP(uint8_t, UI8);
    } else if (imgin->md[0].datatype == _DATATYPE_INT8) {
        STREAM_AVE_LOOP(int8_t, SI8);
    } else if (imgin->md[0].datatype == _DATATYPE_INT16) {
        STREAM_AVE_LOOP(int16_t, SI16);
    } else if (imgin->md[0].datatype == _DATATYPE_UINT32) {
        STREAM_AVE_LOOP(uint32_t, UI32);
    } else if (imgin->md[0].datatype == _DATATYPE_INT32) {
        STREAM_AVE_LOOP(int32_t, SI32);
    } else if (imgin->md[0].datatype == _DATATYPE_UINT64) {
        STREAM_AVE_LOOP(uint64_t, UI64);
    } else if (imgin->md[0].datatype == _DATATYPE_INT64) {
        STREAM_AVE_LOOP(int64_t, SI64);
    } else if (imgin->md[0].datatype == _DATATYPE_DOUBLE) {
        STREAM_AVE_LOOP(double, D);
    }

    #undef STREAM_AVE_LOOP

    (streamave_cntindex)++;

    if (streamave_cntindex
        >= streamave_NBcoadd)
    {
        if (streamave_compave
            && imgoutave)
        {
            for (uint64_t i = 0;
                 i < xysize; i++)
            {
                imgoutave->array.F[i] =
                    imdataarray[i]
                    / (streamave_cntindex);
            }
            processinfo_update_output_stream(
                NULL, imgoutave, NULL);
        }
        if (streamave_comprms
            && imgoutrms)
        {
            for (uint64_t i = 0;
                 i < xysize; i++)
            {
                imgoutrms->array.F[i] =
                    sqrtf(imdataarrayPOW[i])
                    / (streamave_cntindex);
            }
            processinfo_update_output_stream(
                NULL, imgoutrms, NULL);
        }
        streamave_cntindex = 0;
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
    IMGID in =
        imgid_make_from_name(
            streamave_inimname);
    resolveIMGID(
        &in, ERRMODE_ABORT,
        dcimg, dcnimg);

    uint64_t xys =
        in.md[0].size[0]
        * in.md[0].size[1];
    double *d1 =
        (double *) malloc(
            sizeof(double) * xys);
    double *d2 =
        (double *) malloc(
            sizeof(double) * xys);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    fpsexec(in.im, NULL, NULL, d1, d2);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    free(d1);
    free(d2);
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

errno_t CLIADDCMD_streamaverage()
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