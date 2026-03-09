
#include "CLIcore.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "imlinconstruct",
    .cmdkey      = "imlinconstruct",
    .description =
        "construct image as linear sum of modes"
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char *modesimname = NULL;
static char *invecname   = NULL;
static char *outimname   = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".modes", &modesimname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "modes image cube") \
    X(".invec", &invecname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input vector") \
    X(".outim", &outimname, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output image")


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
    "", "", CLICMD_FIELDS_DEFAULTS
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

errno_t linopt_imtools_image_construct(const char *IDmodes_name,
                                       const char *IDcoeff_name,
                                       const char *ID_name,
                                       imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    imageID ID;
    imageID IDmodes;
    imageID IDcoeff;
    uint8_t datatype;

    IDmodes  = image_ID(IDmodes_name, dcimg, dcnimg);
    datatype = dcimg[IDmodes].md[0].datatype;

    uint32_t xsize = dcimg[IDmodes].md[0].size[0];
    uint32_t ysize = dcimg[IDmodes].md[0].size[1];
    uint32_t zsize = dcimg[IDmodes].md[0].size[2];

    uint64_t sizexy = xsize;
    sizexy *= ysize;

    if(datatype == _DATATYPE_FLOAT)
    {
        FUNC_CHECK_RETURN(create_2Dimage_ID(ID_name, xsize, ysize, &ID));
    }
    else
    {
        FUNC_CHECK_RETURN(create_2Dimage_ID_double(ID_name, xsize, ysize, &ID));
    }

    IDcoeff = image_ID(IDcoeff_name, dcimg, dcnimg);

    if(datatype == _DATATYPE_FLOAT)
    {
        memset(dcimg[ID].array.F,
               0,
               sizeof(float) * dcimg[ID].md[0].nelement);
        for(uint32_t kk = 0; kk < zsize; kk++)
            for(uint64_t ii = 0; ii < sizexy; ii++)
            {
                dcimg[ID].array.F[ii] +=
                    dcimg[IDcoeff].array.F[kk] *
                    dcimg[IDmodes].array.F[kk * sizexy + ii];
            }
    }
    else
    {
        memset(dcimg[ID].array.D,
               0,
               sizeof(double) * dcimg[ID].md[0].nelement);
        for(uint32_t kk = 0; kk < zsize; kk++)
            for(uint64_t ii = 0; ii < sizexy; ii++)
            {
                dcimg[ID].array.D[ii] +=
                    dcimg[IDcoeff].array.D[kk] *
                    dcimg[IDmodes].array.D[kk * sizexy + ii];
            }
    }

    if(outID != NULL)
    {
        *outID = ID;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    linopt_imtools_image_construct(
        modesimname, invecname,
        outimname, NULL);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#ifndef FPS_STANDALONE
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_linopt_imtools__image_construct()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    FPS_app_info,
    FPS_PARAMS,
    compute_function)
#endif

