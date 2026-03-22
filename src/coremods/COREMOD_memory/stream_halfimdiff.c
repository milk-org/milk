/**
 * @file    stream_halfimdiff.c
 * @brief   difference between two halves of stream
 *
 * Uses FPS V2 framework.
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"

#include "stream_sem.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "streamhalfdiff",
    .cmdkey      = "streamhalfdiff",
    .description =
        "half-image difference"
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char p_instream[FUNCTION_PARAMETER_STRMAXLEN]
    = "stream";

static char p_outstream[FUNCTION_PARAMETER_STRMAXLEN]
    = "outstream";

static long long p_semtrig = 3;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".in_stream", p_instream, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input stream") \
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
 * Compute difference between two halves of an
 * image stream. Triggers on instream.
 */
imageID MILK_HOT COREMOD_MEMORY_stream_halfimDiff(
    const char *IDstream_name,
    const char *IDstreamout_name,
    long        semtrig)
{
    IMGID img0 = imgid_make_from_name(IDstream_name);
    resolveIMGID(&img0, ERRMODE_ABORT,
                 dcimg, dcnimg);

    uint32_t xsizein = img0.md->size[0];
    uint32_t ysizein = img0.md->size[1];

    uint32_t xsize  = xsizein;
    uint32_t ysize  = ysizein / 2;
    uint64_t xysize = xsize * ysize;

    uint8_t datatype    = img0.md->datatype;
    uint8_t datatypeout = _DATATYPE_FLOAT;

    switch(datatype)
    {
    case _DATATYPE_UINT8:
        datatypeout = _DATATYPE_INT16;
        break;
    case _DATATYPE_UINT16:
        datatypeout = _DATATYPE_INT32;
        break;
    case _DATATYPE_UINT32:
    case _DATATYPE_UINT64:
        datatypeout = _DATATYPE_INT64;
        break;
    case _DATATYPE_INT8:
        datatypeout = _DATATYPE_INT16;
        break;
    case _DATATYPE_INT16:
        datatypeout = _DATATYPE_INT32;
        break;
    case _DATATYPE_INT32:
    case _DATATYPE_INT64:
        datatypeout = _DATATYPE_INT64;
        break;
    case _DATATYPE_DOUBLE:
        datatypeout = _DATATYPE_DOUBLE;
        break;
    default:
        break;
    }

    IMGID imgout =
        imgid_make_from_name(IDstreamout_name);
    resolveIMGID(&imgout, ERRMODE_NULL,
                 dcimg, dcnimg);
    if(imgout.ID == -1)
    {
        imgout = stream_connect_create_2D(
            IDstreamout_name,
            xsize, ysize, datatypeout);
    }

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
            ImageStreamIO_semwait(
                img0.im, semtrig);
        }

        imgout.md->write = 1;

        switch(datatype)
        {
        case _DATATYPE_UINT8:
        {
            uint8_t * MILK_RESTRICT pin =
                MILK_ASSUME_ALIGNED(img0.im->array.UI8);
            int16_t * MILK_RESTRICT pout =
                MILK_ASSUME_ALIGNED(imgout.im->array.SI16);
            for(uint64_t ii = 0; ii < xysize; ii++)
                pout[ii] = pin[ii] - pin[xysize + ii];
            break;
        }
        case _DATATYPE_UINT16:
        {
            uint16_t * MILK_RESTRICT pin =
                MILK_ASSUME_ALIGNED(img0.im->array.UI16);
            int32_t * MILK_RESTRICT pout =
                MILK_ASSUME_ALIGNED(imgout.im->array.SI32);
            for(uint64_t ii = 0; ii < xysize; ii++)
                pout[ii] = pin[ii] - pin[xysize + ii];
            break;
        }
        case _DATATYPE_UINT32:
        {
            uint32_t * MILK_RESTRICT pin =
                MILK_ASSUME_ALIGNED(img0.im->array.UI32);
            int64_t * MILK_RESTRICT pout =
                MILK_ASSUME_ALIGNED(imgout.im->array.SI64);
            for(uint64_t ii = 0; ii < xysize; ii++)
                pout[ii] = pin[ii] - pin[xysize + ii];
            break;
        }
        case _DATATYPE_UINT64:
        {
            uint64_t * MILK_RESTRICT pin =
                MILK_ASSUME_ALIGNED(img0.im->array.UI64);
            int64_t * MILK_RESTRICT pout =
                MILK_ASSUME_ALIGNED(imgout.im->array.SI64);
            for(uint64_t ii = 0; ii < xysize; ii++)
                pout[ii] = (int64_t)pin[ii] - (int64_t)pin[xysize + ii];
            break;
        }
        case _DATATYPE_INT8:
        {
            int8_t * MILK_RESTRICT pin =
                MILK_ASSUME_ALIGNED(img0.im->array.SI8);
            int16_t * MILK_RESTRICT pout =
                MILK_ASSUME_ALIGNED(imgout.im->array.SI16);
            for(uint64_t ii = 0; ii < xysize; ii++)
                pout[ii] = pin[ii] - pin[xysize + ii];
            break;
        }
        case _DATATYPE_INT16:
        {
            int16_t * MILK_RESTRICT pin =
                MILK_ASSUME_ALIGNED(img0.im->array.SI16);
            int32_t * MILK_RESTRICT pout =
                MILK_ASSUME_ALIGNED(imgout.im->array.SI32);
            for(uint64_t ii = 0; ii < xysize; ii++)
                pout[ii] = pin[ii] - pin[xysize + ii];
            break;
        }
        case _DATATYPE_INT32:
        {
            int32_t * MILK_RESTRICT pin =
                MILK_ASSUME_ALIGNED(img0.im->array.SI32);
            int64_t * MILK_RESTRICT pout =
                MILK_ASSUME_ALIGNED(imgout.im->array.SI64);
            for(uint64_t ii = 0; ii < xysize; ii++)
                pout[ii] = pin[ii] - pin[xysize + ii];
            break;
        }
        case _DATATYPE_INT64:
        {
            int64_t * MILK_RESTRICT pin =
                MILK_ASSUME_ALIGNED(img0.im->array.SI64);
            int64_t * MILK_RESTRICT pout =
                MILK_ASSUME_ALIGNED(imgout.im->array.SI64);
            for(uint64_t ii = 0; ii < xysize; ii++)
                pout[ii] = pin[ii] - pin[xysize + ii];
            break;
        }
        case _DATATYPE_FLOAT:
        {
            float * MILK_RESTRICT pin =
                MILK_ASSUME_ALIGNED(img0.im->array.F);
            float * MILK_RESTRICT pout =
                MILK_ASSUME_ALIGNED(imgout.im->array.F);
            for(uint64_t ii = 0; ii < xysize; ii++)
                pout[ii] = pin[ii] - pin[xysize + ii];
            break;
        }
        case _DATATYPE_DOUBLE:
        {
            double * MILK_RESTRICT pin =
                MILK_ASSUME_ALIGNED(img0.im->array.D);
            double * MILK_RESTRICT pout =
                MILK_ASSUME_ALIGNED(imgout.im->array.D);
            for(uint64_t ii = 0; ii < xysize; ii++)
                pout[ii] = pin[ii] - pin[xysize + ii];
            break;
        }
        default:
            PRINT_ERROR("unsupported datatype");
            break;
        }

        COREMOD_MEMORY_image_set_sempost_byID(
            imgout.ID, -1);
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

    COREMOD_MEMORY_stream_halfimDiff(
        p_instream, p_outstream, p_semtrig);

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
CLIADDCMD_COREMOD_memory__stream_halfimdiff()
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
