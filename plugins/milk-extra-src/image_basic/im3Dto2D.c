/**
 * @file im3Dto2D.c
 * @brief Collapse first 2 axis of 3D image
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"

#include "COREMOD_memory/COREMOD_memory.h"

// Forward declaration
imageID image_basic_3Dto2D(
    const char *__restrict IDname);

static char p_in[FUNCTION_PARAMETER_STRMAXLEN]
    = "im1";

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "im3Dto2D",
    .cmdkey      = "im3Dto2D",
    .description =
        "collapse first 2 axis of 3D image",
    .description_long =
        "Collapse the first two axes of a 3D image into a single axis. Reshapes (x, y, z) into (x*y, z) for vectorized processing."
};

#define FPS_PARAMS(X) \
    X(".in_name", p_in, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input image")

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING)
};
static const int nb_bindings =
    sizeof(my_bindings) /
    sizeof(FPS_CLI_BINDING);
static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG)
};
static CLICMDDATA CLIcmddata = {
    "", "", CLICMD_FIELDS_DEFAULTS
};
static CMDSETTINGS cms = {0};

static __attribute__((constructor))
void init_cms(void)
{
    strncpy(CLIcmddata.key,
            FPS_app_info.cmdkey,
            sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description,
            FPS_app_info.description,
            sizeof(CLIcmddata.description)
            - 1);
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings = &cms;
    }
}

static MILK_HOT errno_t compute_function()
{
    image_basic_3Dto2D(p_in);
    return RETURN_SUCCESS;
}

static errno_t __attribute__((unused)) CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_image_basic__im3Dto2D()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}

/* ----------------------------------------------------------------------
 *
 * turns a 3D image into a 2D image by collapsing first 2 axis
 *
 *
 * ---------------------------------------------------------------------- */

imageID image_basic_3Dto2D_byID(imageID ID)
{
    if(dcimg[ID].md[0].naxis != 3)
    {
        printf("ERROR: image needs to have 3 axis\n");
    }
    else
    {
        dcimg[ID].md[0].size[0] *= dcimg[ID].md[0].size[1];
        dcimg[ID].md[0].size[1] = dcimg[ID].md[0].size[2];
        dcimg[ID].md[0].naxis   = 2;
    }

    return ID;
}

imageID image_basic_3Dto2D(const char *__restrict IDname)
{
    imageID ID;

    ID = image_ID(IDname, dcimg, dcnimg);
    image_basic_3Dto2D_byID(ID);

    return ID;
}
