/**
 * @file    breakcube.c
 * @brief   break cube into individual 2D images
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


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "breakcube",
    .cmdkey      = "breakcube",
    .description =
        "break cube into individual images"
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char inname[FUNCTION_PARAMETER_STRMAXLEN]
    = "imc";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".in_name", inname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input cube image")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

imageID break_cube(
    const char *restrict ID_name)
{
    imageID  ID;
    uint32_t naxes[3];
    char     framename[STRINGMAXLEN_IMGNAME];

    ID       = image_ID(
        ID_name, dcimg, dcnimg);
    naxes[0] = dcimg[ID].md[0].size[0];
    naxes[1] = dcimg[ID].md[0].size[1];
    naxes[2] = dcimg[ID].md[0].size[2];

    for(uint32_t kk = 0; kk < naxes[2]; kk++)
    {
        long ID1;

        CREATE_IMAGENAME(framename,
                         "%s_%5u",
                         ID_name, kk);

        for(long i = 0;
            i < (long) strlen(framename); i++)
        {
            if(framename[i] == ' ')
            {
                framename[i] = '0';
            }
        }
        create_2Dimage_ID(framename,
                          naxes[0], naxes[1],
                          &ID1);
        for(uint32_t ii = 0;
            ii < naxes[0]; ii++)
        {
            for(uint32_t jj = 0;
                jj < naxes[1]; jj++)
            {
                dcimg[ID1].array.F[
                    jj * naxes[0] + ii] =
                    dcimg[ID].array.F[
                        kk * naxes[0] * naxes[1]
                        + jj * naxes[0] + ii];
            }
        }
    }

    return ID;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS(FPS_X_BINDING)
};

static const int nb_bindings =
    sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    FPS_PARAMS(FPS_X_FARG)
};

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
#else
static CLICMDDATA CLIcmddata = {
#endif
    "",
    "",
    CLICMD_FIELDS_DEFAULTS
};

static CMDSETTINGS default_cmdsettings = {0};

static __attribute__((constructor))
void init_cmdsettings(void)
{
    strncpy(CLIcmddata.key,
            FPS_app_info.cmdkey,
            sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description,
            FPS_app_info.description,
            sizeof(CLIcmddata.description) - 1);
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings =
            &default_cmdsettings;
    }
}


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    break_cube(inname);

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
CLIADDCMD_COREMOD_iofits__breakcube()
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
