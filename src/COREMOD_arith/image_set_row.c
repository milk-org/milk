
#include "CommandLineInterface/CLIcore.h"


static char *inimname;

static float    *pixval;
static long      fpi_pixval = -1;

static uint32_t *rowindex;
static long      fpi_rowindex = -1;



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
        ".row",
        "row index",
        "100",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &rowindex,
        &fpi_rowindex
    }
};


static CLICMDDATA CLIcmddata =
{
    "setrow",
    "set image row pixels values",
    CLICMD_FIELDS_DEFAULTS
};



// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}





errno_t image_set_row(
    IMGID    inimg,
    double   value,
    uint32_t rowindex
)
{
    DEBUG_TRACE_FSTART();

    long nelem = inimg.md->nelement;


    switch ( inimg.md->datatype )
    {


    case _DATATYPE_INT8 :
        for(uint_fast32_t ii = 0; ii < inimg.md->size[0]; ii++)
        {
            inimg.im->array.SI8[rowindex * inimg.md->size[0] + ii] = (int8_t) value;
        }
        break;

    case _DATATYPE_UINT8 :
        for(uint_fast32_t ii = 0; ii < inimg.md->size[0]; ii++)
        {
            inimg.im->array.UI8[rowindex * inimg.md->size[0] + ii] = (uint8_t) value;
        }
        break;

    case _DATATYPE_INT16 :
        for(uint_fast32_t ii = 0; ii < inimg.md->size[0]; ii++)
        {
            inimg.im->array.SI16[rowindex * inimg.md->size[0] + ii] = (int16_t) value;
        }
        break;

    case _DATATYPE_UINT16 :
        for(uint_fast32_t ii = 0; ii < inimg.md->size[0]; ii++)
        {
            inimg.im->array.UI16[rowindex * inimg.md->size[0] + ii] = (uint16_t) value;
        }
        break;

    case _DATATYPE_INT32 :
        for(uint_fast32_t ii = 0; ii < inimg.md->size[0]; ii++)
        {
            inimg.im->array.SI32[rowindex * inimg.md->size[0] + ii] = (int32_t) value;
        }
        break;

    case _DATATYPE_UINT32 :
        for(uint_fast32_t ii = 0; ii < inimg.md->size[0]; ii++)
        {
            inimg.im->array.UI32[rowindex * inimg.md->size[0] + ii] = (uint32_t) value;
        }
        break;

    case _DATATYPE_INT64 :
        for(uint_fast32_t ii = 0; ii < inimg.md->size[0]; ii++)
        {
            inimg.im->array.SI64[rowindex * inimg.md->size[0] + ii] = (int64_t) value;
        }
        break;

    case _DATATYPE_UINT64 :
        for(uint_fast32_t ii = 0; ii < inimg.md->size[0]; ii++)
        {
            inimg.im->array.UI64[rowindex * inimg.md->size[0] + ii] = (uint64_t) value;
        }
        break;


    case _DATATYPE_FLOAT :
        for(uint_fast32_t ii = 0; ii < inimg.md->size[0]; ii++)
        {
            inimg.im->array.F[rowindex * inimg.md->size[0] + ii] = value;
        }
        break;

    case _DATATYPE_DOUBLE :
        for(uint_fast32_t ii = 0; ii < inimg.md->size[0]; ii++)
        {
            inimg.im->array.D[rowindex * inimg.md->size[0] + ii] = value;
        }
        break;

    case _DATATYPE_COMPLEX_FLOAT :
        for(uint_fast32_t ii = 0; ii < inimg.md->size[0]; ii++)
        {
            inimg.im->array.CF[rowindex * inimg.md->size[0] + ii].re = value;
            inimg.im->array.CF[rowindex * inimg.md->size[0] + ii].im = 0.0;
        }
        break;

    case _DATATYPE_COMPLEX_DOUBLE :
        for(uint_fast32_t ii = 0; ii < inimg.md->size[0]; ii++)
        {
            inimg.im->array.CD[rowindex * inimg.md->size[0] + ii].re = value;
            inimg.im->array.CD[rowindex * inimg.md->size[0] + ii].im = 0.0;
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
        image_set_row(inimg, *pixval, *rowindex);
        processinfo_update_output_stream(processinfo, inimg.ID);

    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}





INSERT_STD_FPSCLIfunctions



// Register function in CLI
errno_t
CLIADDCMD_COREMOD_arith__imset_row()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}



