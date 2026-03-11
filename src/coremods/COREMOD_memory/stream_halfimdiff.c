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

#include "create_image.h"
#include "image_ID.h"
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

imageID COREMOD_MEMORY_stream_halfimDiff(
    const char *IDstream_name,
    const char *IDstreamout_name,
    long        semtrig);
//
// compute difference between two halves of an image stream
// triggers on instream
//
imageID COREMOD_MEMORY_stream_halfimDiff(const char *IDstream_name,
        const char *IDstreamout_name,
        long        semtrig)
{
    imageID            ID0;
    imageID            IDout;
    uint32_t           xsizein;
    uint32_t           ysizein;
    uint32_t           xsize;
    uint32_t           ysize;
    uint64_t           xysize;
    uint32_t          *arraysize;
    unsigned long long cnt;
    uint8_t            datatype;
    uint8_t            datatypeout;

    ID0 = image_ID(IDstream_name, dcimg, dcnimg);

    xsizein = dcimg[ID0].md[0].size[0];
    ysizein = dcimg[ID0].md[0].size[1];
    //    xysizein = xsizein*ysizein;

    xsize  = xsizein;
    ysize  = ysizein / 2;
    xysize = xsize * ysize;

    arraysize = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    if(arraysize == NULL)
    {
        PRINT_ERROR("malloc error");
        abort();
    }
    arraysize[0] = xsize;
    arraysize[1] = ysize;

    datatype    = dcimg[ID0].md[0].datatype;
    datatypeout = _DATATYPE_FLOAT;
    switch(datatype)
    {

        case _DATATYPE_UINT8:
            datatypeout = _DATATYPE_INT16;
            break;

        case _DATATYPE_UINT16:
            datatypeout = _DATATYPE_INT32;
            break;

        case _DATATYPE_UINT32:
            datatypeout = _DATATYPE_INT64;
            break;

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
            datatypeout = _DATATYPE_INT64;
            break;

        case _DATATYPE_INT64:
            datatypeout = _DATATYPE_INT64;
            break;

        case _DATATYPE_DOUBLE:
            datatypeout = _DATATYPE_DOUBLE;
            break;
    }

    IDout = image_ID(IDstreamout_name, dcimg, dcnimg);
    if(IDout == -1)
    {
        create_image_ID(IDstreamout_name,
                        2,
                        arraysize,
                        datatypeout,
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

        switch(datatype)
        {

            case _DATATYPE_UINT8:
                for(uint64_t ii = 0; ii < xysize; ii++)
                {
                    dcimg[IDout].array.SI16[ii] =
                        dcimg[ID0].array.UI8[ii] -
                        dcimg[ID0].array.UI8[xysize + ii];
                }
                break;

            case _DATATYPE_UINT16:
                for(uint64_t ii = 0; ii < xysize; ii++)
                {
                    dcimg[IDout].array.SI32[ii] =
                        dcimg[ID0].array.UI16[ii] -
                        dcimg[ID0].array.UI16[xysize + ii];
                }
                break;

            case _DATATYPE_UINT32:
                for(uint64_t ii = 0; ii < xysize; ii++)
                {
                    dcimg[IDout].array.SI64[ii] =
                        dcimg[ID0].array.UI32[ii] -
                        dcimg[ID0].array.UI32[xysize + ii];
                }
                break;

            case _DATATYPE_UINT64:
                for(uint64_t ii = 0; ii < xysize; ii++)
                {
                    dcimg[IDout].array.SI64[ii] =
                        dcimg[ID0].array.UI64[ii] -
                        dcimg[ID0].array.UI64[xysize + ii];
                }
                break;

            case _DATATYPE_INT8:
                for(uint64_t ii = 0; ii < xysize; ii++)
                {
                    dcimg[IDout].array.SI16[ii] =
                        dcimg[ID0].array.SI8[ii] -
                        dcimg[ID0].array.SI8[xysize + ii];
                }
                break;

            case _DATATYPE_INT16:
                for(uint64_t ii = 0; ii < xysize; ii++)
                {
                    dcimg[IDout].array.SI32[ii] =
                        dcimg[ID0].array.SI16[ii] -
                        dcimg[ID0].array.SI16[xysize + ii];
                }
                break;

            case _DATATYPE_INT32:
                for(uint64_t ii = 0; ii < xysize; ii++)
                {
                    dcimg[IDout].array.SI64[ii] =
                        dcimg[ID0].array.SI32[ii] -
                        dcimg[ID0].array.SI32[xysize + ii];
                }
                break;

            case _DATATYPE_INT64:
                for(uint64_t ii = 0; ii < xysize; ii++)
                {
                    dcimg[IDout].array.SI64[ii] =
                        dcimg[ID0].array.SI64[ii] -
                        dcimg[ID0].array.SI64[xysize + ii];
                }
                break;

            case _DATATYPE_FLOAT:
                for(uint64_t ii = 0; ii < xysize; ii++)
                {
                    dcimg[IDout].array.F[ii] =
                        dcimg[ID0].array.F[ii] -
                        dcimg[ID0].array.F[xysize + ii];
                }
                break;

            case _DATATYPE_DOUBLE:
                for(uint64_t ii = 0; ii < xysize; ii++)
                {
                    dcimg[IDout].array.D[ii] =
                        dcimg[ID0].array.D[ii] -
                        dcimg[ID0].array.D[xysize + ii];
                }
                break;
        }

        COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);
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
