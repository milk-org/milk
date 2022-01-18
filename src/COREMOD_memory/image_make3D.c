#include "CommandLineInterface/CLIcore.h"

// Local variables pointers
static char *outimname;
static uint32_t *imxsize;
static uint32_t *imysize;
static uint32_t *imzsize;

static CLICMDARGDEF farg[] = {
    {CLIARG_STR, ".out_name", "output image", "out1", CLIARG_VISIBLE_DEFAULT, (void **)&outimname, NULL},
    {CLIARG_LONG, ".xsize", "x size", "512", CLIARG_VISIBLE_DEFAULT, (void **)&imxsize, NULL},
    {CLIARG_LONG, ".ysize", "y size", "512", CLIARG_VISIBLE_DEFAULT, (void **)&imysize, NULL},
    {CLIARG_LONG, ".zsize", "z size", "512", CLIARG_VISIBLE_DEFAULT, (void **)&imzsize, NULL}};

static CLICMDDATA CLIcmddata = {"mk3Dim",
                                "make 3D image\n"
                                "attributes: s>    : shared\n"
                                "            k20>  : 20 keywords\n",
                                CLICMD_FIELDS_DEFAULTS};

// detailed help
static errno_t help_function() { return RETURN_SUCCESS; }

static imageID make_3Dimage(IMGID *img)
{
    // Create image if needed
    imcreateIMGID(img);

    return (img->ID);
}

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID img = makeIMGID_3D(outimname, *imxsize, *imysize, *imzsize);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    make_3Dimage(&img);

    processinfo_update_output_stream(processinfo, img.ID);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

    // Register function in CLI
    errno_t
    CLIADDCMD_COREMOD_memory__mk3Dim()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
