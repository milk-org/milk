/**
 * @file    stream_copy.c
 * @brief   copy image stream
 *
 * Uses FPS V2 framework.
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"

#include "image_ID.h"
#include "stream_sem.h"

#include "COREMOD_tools/COREMOD_tools.h"
#include "libmilkdata/milk_compiler.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "shmimcopy",
    .cmdkey      = "shmimcopy",
    .description =
        "copy in stream to existing out stream"
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char inimname[FUNCTION_PARAMETER_STRMAXLEN]
    = "imin";
static char outimname[FUNCTION_PARAMETER_STRMAXLEN]
    = "imout";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".in_sname", inimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input stream") \
    X(".out_sname", outimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output stream")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imgin =
        imgid_make_from_name(inimname);
    resolveIMGID(
        &imgin, ERRMODE_ABORT,
        dcimg, dcnimg);

    IMGID imgout =
        imgid_make_from_name(outimname);
    resolveIMGID(
        &imgout, ERRMODE_ABORT,
        dcimg, dcnimg);

    uint64_t im_in_datasize =
        ImageStreamIO_typesize(
            imgin.im->md->datatype)
        * imgin.im->md->nelement;
    uint64_t im_out_datasize =
        ImageStreamIO_typesize(
            imgin.im->md->datatype)
        * imgout.im->md->nelement;
    uint64_t byte_copy_size =
        im_in_datasize < im_out_datasize
        ? im_in_datasize : im_out_datasize;

    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    {
        ImageStreamIO_semflush(
            imgin.im,
            processinfo->triggersem);

        if (imgin.im->md->datatype == _DATATYPE_FLOAT)
        {
            float       * MILK_RESTRICT out = MILK_ASSUME_ALIGNED(imgout.im->array.F);
            const float * MILK_RESTRICT in  = MILK_ASSUME_ALIGNED(imgin.im->array.F);
            uint64_t nfloats = byte_copy_size / sizeof(float);

            #pragma omp parallel for simd if (nfloats > 20000)
            for (uint64_t i = 0; i < nfloats; i++)
            {
                MILK_PREFETCH(&in[i + 32],  0, 0); // read prefetch
                MILK_PREFETCH(&out[i + 32], 1, 0); // write prefetch
                out[i] = in[i];
            }
        }
        else
        {
            memcpy(
                imgout.im->array.F,
                imgin.im->array.F,
                byte_copy_size);
        }

        processinfo_update_output_stream(
            processinfo, imgout.im, NULL);
    }
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
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_COREMOD_memory__stream_copy()
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
