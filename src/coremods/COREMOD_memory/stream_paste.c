/**
 * @file    stream_paste.c
 * @brief   paste two 2D streams into output
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
    .fps_name    = "streampaste",
    .cmdkey      = "streampaste",
    .description =
        "paste two 2D streams"
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char p_stream0[FUNCTION_PARAMETER_STRMAXLEN]
    = "stream0";

static char p_stream1[FUNCTION_PARAMETER_STRMAXLEN]
    = "stream1";

static char p_outstream[FUNCTION_PARAMETER_STRMAXLEN]
    = "outstream";

static long long p_semtrig0 = 3;
static long long p_semtrig1 = 3;
static long long p_master   = 0;


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
    X(".out_stream", p_outstream, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_OUTPUT, \
      "output stream") \
    X(".semtrig0", &p_semtrig0, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "sem trigger 0") \
    X(".semtrig1", &p_semtrig1, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "sem trigger 1") \
    X(".master", &p_master, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "master frame index")


/* ================================================================
 * 4.  COMPUTATION LOGIC — forward decl
 * ============================================================= */

imageID COREMOD_MEMORY_streamPaste(
    const char *IDstream0_name,
    const char *IDstream1_name,
    const char *IDstreamout_name,
    long        semtrig0,
    long        semtrig1,
    int         master);
//
// compute difference between two 2D streams
// triggers alternatively on stream0 and stream1
//
imageID COREMOD_MEMORY_streamPaste(const char *IDstream0_name,
                                   const char *IDstream1_name,
                                   const char *IDstreamout_name,
                                   long        semtrig0,
                                   long        semtrig1,
                                   int         master)
{
    imageID            ID0;
    imageID            ID1;
    imageID            IDout;
    imageID            IDin;
    long               Xoffset;
    uint32_t           xsize;
    uint32_t           ysize;
    uint32_t          *arraysize;
    unsigned long long cnt;
    uint8_t            datatype;
    int                FrameIndex;

    ID0 = image_ID(IDstream0_name, dcimg, dcnimg);
    ID1 = image_ID(IDstream1_name, dcimg, dcnimg);

    xsize    = dcimg[ID0].md[0].size[0];
    ysize    = dcimg[ID0].md[0].size[1];
    datatype = dcimg[ID0].md[0].datatype;

    arraysize = (uint32_t *) malloc(sizeof(uint32_t) * 2);
    if(arraysize == NULL)
    {
        PRINT_ERROR("malloc error");
        abort();
    }
    arraysize[0] = 2 * xsize;
    arraysize[1] = ysize;

    IDout = image_ID(IDstreamout_name, dcimg, dcnimg);
    if(IDout == -1)
    {
        create_image_ID(IDstreamout_name,
                        2,
                        arraysize,
                        datatype,
                        1,
                        0,
                        0,
                        &IDout);
    }
    free(arraysize);

    FrameIndex = 0;

    while(1)
    {
        if(FrameIndex == 0)
        {
            // has new frame 0 arrived ?
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
                ImageStreamIO_semwait(dcimg+ID0, semtrig0);
            }
            Xoffset = 0;
            IDin    = 0;
        }
        else
        {
            // has new frame 1 arrived ?
            if(dcimg[ID1].md[0].sem == 0)
            {
                while(cnt ==
                        dcimg[ID1].md[0].cnt0) // test if new frame exists
                {
                    usleep(5);
                }
                cnt = dcimg[ID1].md[0].cnt0;
            }
            else
            {
                ImageStreamIO_semwait(dcimg+ID1, semtrig1);
            }
            Xoffset = xsize;
            IDin    = 1;
        }

        dcimg[IDout].md[0].write = 1;

        switch(datatype)
        {
            case _DATATYPE_UINT8:
                for(uint32_t ii = 0; ii < xsize; ii++)
                    for(uint32_t jj = 0; jj < ysize; jj++)
                    {
                        dcimg[IDout].array.UI8[jj * 2 * xsize + ii + Xoffset] =
                            dcimg[IDin].array.UI8[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_UINT16:
                for(uint32_t ii = 0; ii < xsize; ii++)
                    for(uint32_t jj = 0; jj < ysize; jj++)
                    {
                        dcimg[IDout]
                        .array.UI16[jj * 2 * xsize + ii + Xoffset] =
                            dcimg[IDin].array.UI16[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_UINT32:
                for(uint32_t ii = 0; ii < xsize; ii++)
                    for(uint32_t jj = 0; jj < ysize; jj++)
                    {
                        dcimg[IDout]
                        .array.UI32[jj * 2 * xsize + ii + Xoffset] =
                            dcimg[IDin].array.UI32[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_UINT64:
                for(uint32_t ii = 0; ii < xsize; ii++)
                    for(uint32_t jj = 0; jj < ysize; jj++)
                    {
                        dcimg[IDout]
                        .array.UI64[jj * 2 * xsize + ii + Xoffset] =
                            dcimg[IDin].array.UI64[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_INT8:
                for(uint32_t ii = 0; ii < xsize; ii++)
                    for(uint32_t jj = 0; jj < ysize; jj++)
                    {
                        dcimg[IDout].array.SI8[jj * 2 * xsize + ii + Xoffset] =
                            dcimg[IDin].array.SI8[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_INT16:
                for(uint32_t ii = 0; ii < xsize; ii++)
                    for(uint32_t jj = 0; jj < ysize; jj++)
                    {
                        dcimg[IDout]
                        .array.SI16[jj * 2 * xsize + ii + Xoffset] =
                            dcimg[IDin].array.SI16[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_INT32:
                for(uint32_t ii = 0; ii < xsize; ii++)
                    for(uint32_t jj = 0; jj < ysize; jj++)
                    {
                        dcimg[IDout]
                        .array.SI32[jj * 2 * xsize + ii + Xoffset] =
                            dcimg[IDin].array.SI32[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_INT64:
                for(uint32_t ii = 0; ii < xsize; ii++)
                    for(uint32_t jj = 0; jj < ysize; jj++)
                    {
                        dcimg[IDout]
                        .array.SI64[jj * 2 * xsize + ii + Xoffset] =
                            dcimg[IDin].array.SI64[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_FLOAT:
                for(uint32_t ii = 0; ii < xsize; ii++)
                    for(uint32_t jj = 0; jj < ysize; jj++)
                    {
                        dcimg[IDout].array.F[jj * 2 * xsize + ii + Xoffset] =
                            dcimg[IDin].array.F[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_DOUBLE:
                for(uint32_t ii = 0; ii < xsize; ii++)
                    for(uint32_t jj = 0; jj < ysize; jj++)
                    {
                        dcimg[IDout].array.D[jj * 2 * xsize + ii + Xoffset] =
                            dcimg[IDin].array.D[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_COMPLEX_FLOAT:
                for(uint32_t ii = 0; ii < xsize; ii++)
                    for(uint32_t jj = 0; jj < ysize; jj++)
                    {
                        dcimg[IDout].array.CF[jj * 2 * xsize + ii + Xoffset] =
                            dcimg[IDin].array.CF[jj * xsize + ii];
                    }
                break;

            case _DATATYPE_COMPLEX_DOUBLE:
                for(uint32_t ii = 0; ii < xsize; ii++)
                    for(uint32_t jj = 0; jj < ysize; jj++)
                    {
                        dcimg[IDout].array.CD[jj * 2 * xsize + ii + Xoffset] =
                            dcimg[IDin].array.CD[jj * xsize + ii];
                    }
                break;

            default:
                printf("Unknown data type\n");
                exit(0);
                break;
        }
        if(FrameIndex == master)
        {
            COREMOD_MEMORY_image_set_sempost_byID(IDout, -1);
            ;
            dcimg[IDout].md[0].cnt0++;
        }
        dcimg[IDout].md[0].cnt1  = FrameIndex;
        dcimg[IDout].md[0].write = 0;

        if(FrameIndex == 0)
        {
            FrameIndex = 1;
        }
        else
        {
            FrameIndex = 0;
        }
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

    COREMOD_MEMORY_streamPaste(
        p_stream0, p_stream1,
        p_outstream,
        p_semtrig0, p_semtrig1,
        p_master);

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
CLIADDCMD_COREMOD_memory__stream_paste()
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
