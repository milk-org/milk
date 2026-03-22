/**
 * @file    image_set_counters.c
 * @brief   set image flags / counters
 *
 * Uses FPS V2 framework.
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"

#include "image_ID.h"


/* ================================================================
 *  COMPUTATION LOGIC
 * ============================================================= */

errno_t COREMOD_MEMORY_image_set_status(
    const char *IDname, int status)
{
    imageID ID;

    ID = image_ID(IDname, dcimg, dcnimg);
    dcimg[ID].md[0].status = status;

    return RETURN_SUCCESS;
}

errno_t COREMOD_MEMORY_image_set_cnt0(
    const char *IDname, int cnt0)
{
    imageID ID;

    ID = image_ID(IDname, dcimg, dcnimg);
    dcimg[ID].md[0].cnt0 = cnt0;

    return RETURN_SUCCESS;
}

errno_t COREMOD_MEMORY_image_set_cnt1(
    const char *IDname, int cnt1)
{
    imageID ID;

    ID = image_ID(IDname, dcimg, dcnimg);
    dcimg[ID].md[0].cnt1 = cnt1;

    return RETURN_SUCCESS;
}


/* ================================================================
 *  COMMON PARAMETERS (image + int64)
 * ============================================================= */

static char p_imname[FUNCTION_PARAMETER_STRMAXLEN]
    = "im1";
static long long p_value = 2;

#define FPS_PARAMS_IMGINT(X) \
    X(".imname", p_imname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "image name") \
    X(".value", &p_value, \
      FPTYPE_INT64, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "value")


/* ================================================================
 *  CMD 1: imsetstatus
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_status = {
    .fps_name    = "imsetstatus",
    .cmdkey      = "imsetstatus",
    .description =
        "set image status variable"
};

static CLICMDDATA CLIcmddata_status = {
    "",
    "",
    CLICMD_FIELDS_NOPARAM
};

static CMDSETTINGS cms1 = {0};

static __attribute__((constructor))
void init_cms_status(void)
{
    strncpy(CLIcmddata_status.key,
            FPS_app_info_status.cmdkey,
            sizeof(CLIcmddata_status.key)
            - 1);
    strncpy(
        CLIcmddata_status.description,
        FPS_app_info_status.description,
        sizeof(
            CLIcmddata_status.description
        ) - 1);
    if (CLIcmddata_status.cmdsettings
        == NULL) {
        CLIcmddata_status.cmdsettings =
            &cms1;
    }
}

static errno_t __attribute__((unused)) compute_status()
{
    COREMOD_MEMORY_image_set_status(
        p_imname, p_value);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 2: imsetcnt0 (primary)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "imsetcnt0",
    .cmdkey      = "imsetcnt0",
    .description =
        "set image cnt0 variable"
};

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS_IMGINT(FPS_X_BINDING)
};

static const int __attribute__((unused)) nb_bindings =
    sizeof(my_bindings) /
    sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    FPS_PARAMS_IMGINT(FPS_X_FARG)
};

static CLICMDDATA CLIcmddata = {
    "",
    "",
    CLICMD_FIELDS_DEFAULTS
};

static CMDSETTINGS cms2 = {0};

static __attribute__((constructor))
void init_cms_cnt0(void)
{
    strncpy(CLIcmddata.key,
            FPS_app_info.cmdkey,
            sizeof(CLIcmddata.key) - 1);
    strncpy(CLIcmddata.description,
            FPS_app_info.description,
            sizeof(CLIcmddata.description)
            - 1);
    if (CLIcmddata.cmdsettings == NULL) {
        CLIcmddata.cmdsettings = &cms2;
    }
}

static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    COREMOD_MEMORY_image_set_cnt0(
        p_imname, p_value);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 3: imsetcnt1
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_cnt1 = {
    .fps_name    = "imsetcnt1",
    .cmdkey      = "imsetcnt1",
    .description =
        "set image cnt1 variable"
};

static CLICMDDATA CLIcmddata_cnt1 = {
    "",
    "",
    CLICMD_FIELDS_NOPARAM
};

static CMDSETTINGS cms3 = {0};

static __attribute__((constructor))
void init_cms_cnt1(void)
{
    strncpy(CLIcmddata_cnt1.key,
            FPS_app_info_cnt1.cmdkey,
            sizeof(CLIcmddata_cnt1.key)
            - 1);
    strncpy(CLIcmddata_cnt1.description,
            FPS_app_info_cnt1.description,
            sizeof(
                CLIcmddata_cnt1.description
            ) - 1);
    if (CLIcmddata_cnt1.cmdsettings
        == NULL) {
        CLIcmddata_cnt1.cmdsettings =
            &cms3;
    }
}

static errno_t __attribute__((unused)) compute_cnt1()
{
    COREMOD_MEMORY_image_set_cnt1(
        p_imname, p_value);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)

static errno_t CLIfunction_status(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_status,
        farg, &CLIcmddata_status,
        my_bindings, nb_bindings,
        compute_status);
}

static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

static errno_t CLIfunction_cnt1(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_cnt1,
        farg, &CLIcmddata_cnt1,
        my_bindings, nb_bindings,
        compute_cnt1);
}

errno_t
CLIADDCMD_COREMOD_memory__image_set_counters()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);

    {
        int cmdi = RegisterCLIcmd(
            CLIcmddata_status,
            CLIfunction_status);
        CLIcmddata_status.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    {
        int cmdi = RegisterCLIcmd(
            CLIcmddata, CLIfunction);
        CLIcmddata.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    {
        int cmdi = RegisterCLIcmd(
            CLIcmddata_cnt1,
            CLIfunction_cnt1);
        CLIcmddata_cnt1.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    return RETURN_SUCCESS;
}
#endif
