#include "CommandLineInterface/CLIcore.h"


static char *inimname;


static float    *pixval;
static long      fpi_pixval = -1;

static uint32_t *minindex;
static long      fpi_minindex = -1;

static uint32_t *maxindex;
static long      fpi_maxindex = -1;



static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".imname",
        "input image",
        "im1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname,
        NULL
    },
    {
        CLIARG_FLOAT32,
        ".pixval",
        "pixel value",
        "3.2",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &pixval,
        &fpi_pixval
    },
    {
        CLIARG_UINT32,
        ".mini",
        "min index",
        "10",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &minindex,
        &fpi_minindex
    },
    {
        CLIARG_UINT32,
        ".maxi",
        "max index",
        "50",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &maxindex,
        &fpi_maxindex
    }
};


static CLICMDDATA CLIcmddata =
{
    "setpix1Drange",
    "set image pixel value over range",
    CLICMD_FIELDS_DEFAULTS
};



// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}





errno_t image_set_1Dpixrange(
    IMGID    inimg,
    double   value,
    uint32_t minindex,
    uint32_t maxindex
)
{
    DEBUG_TRACE_FSTART();

    long nelem = inimg.md->nelement;


    switch ( inimg.md->datatype )
    {

    case _DATATYPE_INT8 :
        for(uint_fast32_t ii = minindex; ii < maxindex; ii++)
        {
            inimg.im->array.SI8[ii] = (int8_t) value;
        }
        break;

    case _DATATYPE_UINT8 :
        for(uint_fast32_t ii = minindex; ii < maxindex; ii++)
        {
            inimg.im->array.UI8[ii] = (uint8_t) value;
        }
        break;

    case _DATATYPE_INT16 :
        for(uint_fast32_t ii = minindex; ii < maxindex; ii++)
        {
            inimg.im->array.SI16[ii] = (int16_t) value;
        }
        break;

    case _DATATYPE_UINT16 :
        for(uint_fast32_t ii = minindex; ii < maxindex; ii++)
        {
            inimg.im->array.UI16[ii] = (uint16_t) value;
        }
        break;

    case _DATATYPE_INT32 :
        for(uint_fast32_t ii = minindex; ii < maxindex; ii++)
        {
            inimg.im->array.SI32[ii] = (int32_t) value;
        }
        break;

    case _DATATYPE_UINT32 :
        for(uint_fast32_t ii = minindex; ii < maxindex; ii++)
        {
            inimg.im->array.UI32[ii] = (uint32_t) value;
        }
        break;

    case _DATATYPE_INT64 :
        for(uint_fast32_t ii = minindex; ii < maxindex; ii++)
        {
            inimg.im->array.SI64[ii] = (int64_t) value;
        }
        break;

    case _DATATYPE_UINT64 :
        for(uint_fast32_t ii = minindex; ii < maxindex; ii++)
        {
            inimg.im->array.UI64[ii] = (uint64_t) value;
        }
        break;


    case _DATATYPE_FLOAT :
        for(uint_fast32_t ii = minindex; ii < maxindex; ii++)
        {
            inimg.im->array.F[ii] = value;
        }
        break;

    case _DATATYPE_DOUBLE :
        for(uint_fast32_t ii = minindex; ii < maxindex; ii++)
        {
            inimg.im->array.D[ii] = value;
        }
        break;

    case _DATATYPE_COMPLEX_FLOAT :
        for(uint_fast32_t ii = minindex; ii < maxindex; ii++)
        {
            inimg.im->array.CF[ii].re = value;
            inimg.im->array.CF[ii].im = 0.0;
        }
        break;

    case _DATATYPE_COMPLEX_DOUBLE :
        for(uint_fast32_t ii = minindex; ii < maxindex; ii++)
        {
            inimg.im->array.CD[ii].re = value;
            inimg.im->array.CD[ii].im = 0.0;
        }
        break;
    }


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID inimg = mkIMGID_from_name(inimname);
    resolveIMGID(&inimg, ERRMODE_ABORT);


    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    {
        image_set_1Dpixrange(inimg, *pixval, *minindex, *maxindex);
        processinfo_update_output_stream(processinfo, inimg.ID);

    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}





INSERT_STD_FPSCLIfunctions



// Register function in CLI
errno_t
CLIADDCMD_COREMOD_arith__imset_1Dpixrange()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}



