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
#include "COREMOD_memory/COREMOD_memory.h"

#include "stream_sem.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "streamhalfdiff",
    .cmdkey      = "streamhalfdiff",
    .description =
        "half-image difference",
    .description_long =
        "Compute the difference between the left and right halves of an image stream. Produces a half-width output. Used for differential measurements in optical systems."
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
 * @brief Input→output type mapping for half-image difference.
 *
 * X(DT_IN, IACC, ICT, DT_OUT, OACC, OCT)
 *   DT_IN   _DATATYPE_* constant for input
 *   IACC    input union accessor
 *   ICT     input C type
 *   DT_OUT  _DATATYPE_* constant for output
 *   OACC    output union accessor
 *   OCT     output C type
 */
#define HALFDIFF_TYPES(X)                           \
    X(_DATATYPE_UINT8,  UI8,  uint8_t,              \
      _DATATYPE_INT16,  SI16, int16_t)              \
    X(_DATATYPE_INT8,   SI8,  int8_t,               \
      _DATATYPE_INT16,  SI16, int16_t)              \
    X(_DATATYPE_UINT16, UI16, uint16_t,             \
      _DATATYPE_INT32,  SI32, int32_t)              \
    X(_DATATYPE_INT16,  SI16, int16_t,              \
      _DATATYPE_INT32,  SI32, int32_t)              \
    X(_DATATYPE_UINT32, UI32, uint32_t,             \
      _DATATYPE_INT64,  SI64, int64_t)              \
    X(_DATATYPE_INT32,  SI32, int32_t,              \
      _DATATYPE_INT64,  SI64, int64_t)              \
    X(_DATATYPE_UINT64, UI64, uint64_t,             \
      _DATATYPE_INT64,  SI64, int64_t)              \
    X(_DATATYPE_INT64,  SI64, int64_t,              \
      _DATATYPE_INT64,  SI64, int64_t)              \
    X(_DATATYPE_FLOAT,  F,    float,                \
      _DATATYPE_FLOAT,  F,    float)                \
    X(_DATATYPE_DOUBLE, D,    double,               \
      _DATATYPE_DOUBLE, D,    double)

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
    resolveIMGID(&img0, ERRMODE_WARN,
                 dcimg, dcnimg);
    if (img0.ID == -1) {
        return RETURN_FAILURE;
    }

    uint32_t xsizein = img0.md->size[0];
    uint32_t ysizein = img0.md->size[1];

    uint32_t xsize  = xsizein;
    uint32_t ysize  = ysizein / 2;
    uint64_t xysize = (uint64_t)xsize * ysize;

    uint8_t datatype    = img0.md->datatype;
    uint8_t datatypeout = _DATATYPE_FLOAT;

#define HALFDIFF_OUTTYPE(DT_IN, IACC, ICT, \
                         DT_OUT, OACC, OCT) \
    case DT_IN: datatypeout = DT_OUT; break;

    switch(datatype)
    {
    HALFDIFF_TYPES(HALFDIFF_OUTTYPE)
    default:
        break;
    }
#undef HALFDIFF_OUTTYPE

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

/*
 * Typed half-difference loop.
 * Casts operands to OCT before subtraction to
 * avoid unsigned wrap for UINT32/UINT64 inputs.
 */
#define HALFDIFF_CASE(DT_IN, IACC, ICT,              \
                      DT_OUT, OACC, OCT)             \
    case DT_IN:                                      \
    {                                                \
        ICT * MILK_RESTRICT pin =                    \
            MILK_ASSUME_ALIGNED(                     \
                img0.im->array.IACC);                \
        OCT * MILK_RESTRICT pout =                   \
            MILK_ASSUME_ALIGNED(                     \
                imgout.im->array.OACC);              \
        for (uint64_t ii = 0; ii < xysize; ii++)     \
            pout[ii] = (OCT) pin[ii]                 \
                     - (OCT) pin[xysize + ii];       \
        break;                                       \
    }

        switch(datatype)
        {
        HALFDIFF_TYPES(HALFDIFF_CASE)
        default:
            PRINT_ERROR("unsupported datatype");
            break;
        }
#undef HALFDIFF_CASE

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
