#include "CommandLineInterface/CLIcore.h"



// Local variables pointers
static char *outimname;
static uint32_t *imxsize;
static uint32_t *imysize;



static CLICMDARGDEF farg[] =
{
    {
        CLIARG_STR, ".out_name", "output image", "out1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimname
    },
    {
        CLIARG_LONG, ".xsize", "x size", "512",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &imxsize
    },
    {
        CLIARG_LONG, ".ysize", "y size", "512",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &imysize
    }
};

static CLICMDDATA CLIcmddata =
{
    "mk2Dim",
    "make 2D image",
    CLICMD_FIELDS_DEFAULTS
};



// detailed help
static errno_t help_function()
{
    printf(
        "Create 2D image\n"
        "File name can be appended with attribute:\n"
        "For example: \"s>im1\", indicates a shared memory image (attribute \"s\")\n"
        "Attributes list:\n"
        "    s>      shared memory\n"
        "    k20>    number of keyword=20\n"
        "    c5>     5-deep circular buffer for logging\n"
        "    tui8>   type unsigned int 8-bit\n"
        "    tsi8>   type signed int 8-bit\n"
        "    tui16>  type unsigned int 16-bit\n"
        "    tsi16>  type signed int 16-bit\n"
        "    tui32>  type unsigned int 32-bit\n"
        "    tsi32>  type signed int 32-bit\n"
        "    tui64>  type unsigned int 64-bit\n"
        "    tf32>   type float 32-bit\n"
        "    tf64>   type float 64-bit\n"
    );

    return RETURN_SUCCESS;
}



static imageID make_2Dimage(
    IMGID *img
)
{
    // Create image if needed
    imcreateIMGID(img);

    return(img->ID);
}



static errno_t compute_function()
{
    IMGID img = makeIMGID_2D(outimname, *imxsize, *imysize);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    make_2Dimage(
        &img
    );

    processinfo_update_output_stream(processinfo, img.ID);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    return RETURN_SUCCESS;
}




INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t CLIADDCMD_COREMOD_memory__mk2Dim()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

