
#include "CommandLineInterface/CLIcore.h"


static char *inimname;

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
    }
};


static CLICMDDATA CLIcmddata =
{
    "imzero",
    "set all image pixels to zero value",
    CLICMD_FIELDS_DEFAULTS
};



// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}





errno_t image_setzero(
    IMGID inimg
)
{
    DEBUG_TRACE_FSTART();

    long nelem = inimg.md->nelement;
    int typesize = ImageStreamIO_typesize(inimg.md->datatype);
    if(typesize == -1)
    {
        PRINT_ERROR("cannot detect image type for image %s",  inimg.name);
        exit(0);
    }
    memset(inimg.im->array.raw, 0, typesize * nelem);


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
        image_setzero(inimg);
        processinfo_update_output_stream(processinfo, inimg.ID);

    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}





INSERT_STD_FPSCLIfunctions



// Register function in CLI
errno_t
CLIADDCMD_COREMOD_arith__imsetzero()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}



