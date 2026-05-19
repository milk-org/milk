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
#include "COREMOD_memory/COREMOD_memory.h"

#include "stream_sem.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info =
{
    .fps_name    = "streamdiff",
    .cmdkey      = "streamdiff",
    .description =
    "compute stream difference",
    .description_long =
    "Compute the pixel-wise difference between two image streams in real-time. Output stream = stream_A - stream_B. Operates as a continuous stream processor triggered by semaphore."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char p_stream0[FUNCTION_PARAMETER_STRMAXLEN] = "stream0";

static char p_stream1[FUNCTION_PARAMETER_STRMAXLEN] = "stream1";

static char p_mask[FUNCTION_PARAMETER_STRMAXLEN] = "null";

static char p_outstream[FUNCTION_PARAMETER_STRMAXLEN] = "outstream";

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
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

/**
 * Compute difference between two 2D streams.
 * Triggers on stream0.
 */
imageID MILK_HOT COREMOD_MEMORY_streamDiff(
    const char *IDstream0_name,
    const char *IDstream1_name,
    const char *IDstreammask_name,
    const char *IDstreamout_name,
    long       semtrig)
{
    IMGID img0 = imgid_make_from_name(IDstream0_name);
    resolveIMGID(&img0, ERRMODE_WARN, dcimg, dcnimg);
    if(img0.ID == -1)
    {
        return RETURN_FAILURE;
    }

    IMGID img1 = imgid_make_from_name(IDstream1_name);
    resolveIMGID(&img1, ERRMODE_WARN, dcimg, dcnimg);
    if(img1.ID == -1)
    {
        return RETURN_FAILURE;
    }

    IMGID imgmask = imgid_make_from_name(IDstreammask_name);
    resolveIMGID(&imgmask, ERRMODE_NULL, dcimg, dcnimg);

    uint32_t xsize = img0.md->size[0];
    uint32_t ysize = img0.md->size[1];
    uint64_t xysize = (uint64_t)xsize * ysize;

    IMGID imgout = imgid_make_from_name(IDstreamout_name);
    resolveIMGID(&imgout, ERRMODE_NULL, dcimg, dcnimg);
    if(imgout.ID == -1)
    {
        imgout = stream_connect_create_2D(IDstreamout_name, xsize, ysize, _DATATYPE_FLOAT);
    }

    float *MILK_RESTRICT ptr0 = MILK_ASSUME_ALIGNED(img0.im->array.F);
    float *MILK_RESTRICT ptr1 = MILK_ASSUME_ALIGNED(img1.im->array.F);
    float *MILK_RESTRICT ptrm =
        (imgmask.ID != -1) ? MILK_ASSUME_ALIGNED(imgmask.im->array.F) : NULL;
    float *MILK_RESTRICT ptro = MILK_ASSUME_ALIGNED(imgout.im->array.F);

    unsigned long long cnt = 0;

    while(1)
    {
        if(img0.md->sem == 0)
        {
            while(cnt == img0.md->cnt0)
            {
                usleep(5);
            }
            cnt = img0.md->cnt0;
        }
        else
        {
            ImageStreamIO_semwait(img0.im, semtrig);
        }

        imgout.md->write = 1;
        if(ptrm == NULL)
        {
            MILK_IVDEP for(uint64_t ii = 0;
                    ii < xysize; ii++)
            {
                ptro[ii] = ptr0[ii] - ptr1[ii];
            }
        }
        else
        {
            MILK_IVDEP for(uint64_t ii = 0;
                    ii < xysize; ii++)
            {
                ptro[ii] = (ptr0[ii] - ptr1[ii]) * ptrm[ii];
            }
        }
        COREMOD_MEMORY_image_set_sempost_byID(imgout.ID, -1);
        imgout.md->cnt0++;
        imgout.md->write = 0;
    }

    return imgout.ID;
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

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    COREMOD_MEMORY_streamDiff(p_stream0, p_stream1, p_mask, p_outstream, p_semtrig);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END  DEBUG_TRACE_FEXIT();
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
CLIADDCMD_COREMOD_memory__stream_diff()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);

    int cmdi = RegisterCLIcmd(CLIcmddata, CLIfunction);
    CLIcmddata.cmdsettings = &data.cmd[cmdi].cmdsettings;

    return RETURN_SUCCESS;
}
#endif
