/**
 * @file    stream_diff.c
 * @brief   compute difference between two streams
 *
 * Uses FPS V2 framework.
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"

#include "create_image.h"
#include "image_ID.h"
#include "stream_sem.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "streamdiff",
    .cmdkey      = "streamdiff",
    .description =
        "compute stream difference"
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char p_stream0[FUNCTION_PARAMETER_STRMAXLEN]
    = "stream0";

static char p_stream1[FUNCTION_PARAMETER_STRMAXLEN]
    = "stream1";

static char p_mask[FUNCTION_PARAMETER_STRMAXLEN]
    = "null";

static char p_outstream[FUNCTION_PARAMETER_STRMAXLEN]
    = "outstream";

static long long p_semtrig = 3;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".in_stream0", p_stream0, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input stream 0") \
    X(".in_stream1", p_stream1, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input stream 1") \
    X(".mask", p_mask, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "optional mask (null=none)") \
    X(".out_stream", p_outstream, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_OUTPUT, \
      "output stream") \
    X(".semtrig", &p_semtrig, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "sem trigger index")


/* ================================================================
 * 4.  COMPUTATION LOGIC — forward decl
 * ============================================================= */

imageID COREMOD_MEMORY_streamDiff(
    const char *IDstream0_name,
    const char *IDstream1_name,
    const char *IDstreammask_name,
    const char *IDstreamout_name,
    long        semtrig);
/**
 * ## Purpose
 *
 * Compute difference between two 2D streams\n
 * Triggers on stream0\n
 *
 */
imageID COREMOD_MEMORY_streamDiff(const char *IDstream0_name,
                                  const char *IDstream1_name,
                                  const char *IDstreammask_name,
                                  const char *IDstreamout_name,
                                  long        semtrig)
{
    imageID            ID0;
    imageID            ID1;
    imageID            IDout;
    uint32_t           xsize;
    uint32_t           ysize;
    uint64_t           xysize;
    uint32_t          *arraysize;
    unsigned long long cnt;
    imageID            IDmask; // optional

    ID0    = image_ID(IDstream0_name, dcimg, dcnimg);
    ID1    = image_ID(IDstream1_name, dcimg, dcnimg);
    IDmask = image_ID(IDstreammask_name, dcimg, dcnimg);

    xsize  = dcimg[ID0].md[0].size[0];
    ysize  = dcimg[ID0].md[0].size[1];
    xysize = xsize * ysize;

    arraysize = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    if(arraysize == NULL)
    {
        PRINT_ERROR("malloc error");
        abort();
    }
    arraysize[0] = xsize;
    arraysize[1] = ysize;

    IDout = image_ID(IDstreamout_name, dcimg, dcnimg);
    if(IDout == -1)
    {
        create_image_ID(IDstreamout_name,
                        2,
                        arraysize,
                        _DATATYPE_FLOAT,
                        1,
                        0,
                        0,
                        &IDout);
    }

    free(arraysize);

    while(1)
    {
        // has new frame arrived ?
        if(dcimg[ID0].md[0].sem == 0)
        {
            while(cnt ==
                    dcimg[ID0].md[0].cnt0) // test if new frame exists
            {
                usleep(5);
            }
            cnt = dcimg[ID0].md[0].cnt0;
        }
        else
        {
            ImageStreamIO_semwait(dcimg+ID0, semtrig);
        }

        dcimg[IDout].md[0].write = 1;
        if(IDmask == -1)
        {
            for(uint64_t ii = 0; ii < xysize; ii++)
            {
                dcimg[IDout].array.F[ii] =
                    dcimg[ID0].array.F[ii] - dcimg[ID1].array.F[ii];
            }
        }
        else
        {
            for(uint64_t ii = 0; ii < xysize; ii++)
            {
                dcimg[IDout].array.F[ii] = (dcimg[ID0].array.F[ii] -
                                                 dcimg[ID1].array.F[ii]) *
                                                dcimg[IDmask].array.F[ii];
            }
        }
        COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);
        ;
        dcimg[IDout].md[0].cnt0++;
        dcimg[IDout].md[0].write = 0;
    }

    return IDout;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    COREMOD_MEMORY_streamDiff(
        p_stream0, p_stream1,
        p_mask, p_outstream,
        p_semtrig);

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
CLIADDCMD_COREMOD_memory__stream_diff()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);

    int cmdi = RegisterCLIcmd(
        CLIcmddata, CLIfunction);
    CLIcmddata.cmdsettings =
        &data.cmd[cmdi].cmdsettings;

    return RETURN_SUCCESS;
}
#endif
