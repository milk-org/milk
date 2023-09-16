
#include "CommandLineInterface/CLIcore.h"


static char *inimname;

static float    *pixval;
static long      fpi_pixval = -1;

static uint32_t *colindex;
static long      fpi_colindex = -1;



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
        ".col",
        "column index",
        "100",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &colindex,
        &fpi_colindex
    }
};


static CLICMDDATA CLIcmddata =
{
    "setcol",
    "set image column pixels values",
    CLICMD_FIELDS_DEFAULTS
};



// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}





errno_t image_set_col(
    IMGID    inimg,
    double   value,
    uint32_t colindex
)
{
    DEBUG_TRACE_FSTART();

    long nelem = inimg.md->nelement;


    switch ( inimg.md->datatype )
    {


    case _DATATYPE_INT8 :
        for(uint_fast32_t jj = 0; jj < inimg.md->size[1]; jj++)
        {
            inimg.im->array.SI8[jj * inimg.md->size[0] + colindex] = (int8_t) value;
        }
        break;

    case _DATATYPE_UINT8 :
        for(uint_fast32_t jj = 0; jj < inimg.md->size[1]; jj++)
        {
            inimg.im->array.UI8[jj * inimg.md->size[0] + colindex] = (uint8_t) value;
        }
        break;

    case _DATATYPE_INT16 :
        for(uint_fast32_t jj = 0; jj < inimg.md->size[1]; jj++)
        {
            inimg.im->array.SI16[jj * inimg.md->size[0] + colindex] = (int16_t) value;
        }
        break;

    case _DATATYPE_UINT16 :
        for(uint_fast32_t jj = 0; jj < inimg.md->size[1]; jj++)
        {
            inimg.im->array.UI16[jj * inimg.md->size[0] + colindex] = (uint16_t) value;
        }
        break;

    case _DATATYPE_INT32 :
        for(uint_fast32_t jj = 0; jj < inimg.md->size[1]; jj++)
        {
            inimg.im->array.SI32[jj * inimg.md->size[0] + colindex] = (int32_t) value;
        }
        break;

    case _DATATYPE_UINT32 :
        for(uint_fast32_t jj = 0; jj < inimg.md->size[1]; jj++)
        {
            inimg.im->array.UI32[jj * inimg.md->size[0] + colindex] = (uint32_t) value;
        }
        break;

    case _DATATYPE_INT64 :
        for(uint_fast32_t jj = 0; jj < inimg.md->size[1]; jj++)
        {
            inimg.im->array.SI64[jj * inimg.md->size[0] + colindex] = (int64_t) value;
        }
        break;

    case _DATATYPE_UINT64 :
        for(uint_fast32_t jj = 0; jj < inimg.md->size[1]; jj++)
        {
            inimg.im->array.UI64[jj * inimg.md->size[0] + colindex] = (uint64_t) value;
        }
        break;


    case _DATATYPE_FLOAT :
        for(uint_fast32_t jj = 0; jj < inimg.md->size[1]; jj++)
        {
            inimg.im->array.F[jj * inimg.md->size[0] + colindex] = value;
        }
        break;

    case _DATATYPE_DOUBLE :
        for(uint_fast32_t jj = 0; jj < inimg.md->size[1]; jj++)
        {
            inimg.im->array.D[jj * inimg.md->size[0] + colindex] = value;
        }
        break;

    case _DATATYPE_COMPLEX_FLOAT :
        for(uint_fast32_t jj = 0; jj < inimg.md->size[1]; jj++)
        {
            inimg.im->array.CF[jj * inimg.md->size[0] + colindex].re = value;
            inimg.im->array.CF[jj * inimg.md->size[0] + colindex].im = 0.0;
        }
        break;

    case _DATATYPE_COMPLEX_DOUBLE :
        for(uint_fast32_t jj = 0; jj < inimg.md->size[1]; jj++)
        {
            inimg.im->array.CD[jj * inimg.md->size[0] + colindex].re = value;
            inimg.im->array.CD[jj * inimg.md->size[0] + colindex].im = 0.0;
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
        image_set_col(inimg, *pixval, *colindex);
        processinfo_update_output_stream(processinfo, inimg.ID);

    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}





INSERT_STD_FPSCLIfunctions



// Register function in CLI
errno_t
CLIADDCMD_COREMOD_arith__imset_col()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}



