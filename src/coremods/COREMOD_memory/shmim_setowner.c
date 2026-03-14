/**
 * @file    shmim_setowner.c
 * @brief   set stream owner PID
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

/** @brief set owner to creator */
imageID shmim_setowner_creator(const char *name)
{
    imageID ID;

    ID = image_ID(name, dcimg, dcnimg);
    if(ID != -1)
    {
        dcimg[ID].md[0].ownerPID =
            dcimg[ID].md[0].creatorPID;
    }

    return ID;
}

/** @brief set owner to current PID */
imageID shmim_setowner_current(const char *name)
{
    imageID ID;

    ID = image_ID(name, dcimg, dcnimg);
    if(ID != -1)
    {
        dcimg[ID].md[0].ownerPID = getpid();
    }

    return ID;
}

/**
 * @brief set owner to init process
 *
 * Makes the stream immune to orphan purging
 */
imageID shmim_setowner_init(const char *name)
{
    imageID ID;

    ID = image_ID(name, dcimg, dcnimg);
    if(ID != -1)
    {
        dcimg[ID].md[0].ownerPID = 1;
    }

    return ID;
}


/* ================================================================
 *  COMMON PARAMETER (1 stream arg)
 * ============================================================= */

static char p_sname[FUNCTION_PARAMETER_STRMAXLEN]
    = "stream0";

#define FPS_PARAMS_1STREAM(X) \
    X(".sname", p_sname, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "stream name")


/* ================================================================
 *  CMD 1: shmimsetowncreator
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_creator = {
    .fps_name    = "shmimsetowncreator",
    .cmdkey      = "shmimsetowncreator",
    .description = "set owner to creator PID"
};

static CLICMDDATA CLIcmddata_creator = {
    "",
    "",
    CLICMD_FIELDS_NOPARAM
};

static CMDSETTINGS cms1 = {0};

static __attribute__((constructor))
void init_cms_creator(void)
{
    strncpy(CLIcmddata_creator.key,
            FPS_app_info_creator.cmdkey,
            sizeof(CLIcmddata_creator.key)
            - 1);
    strncpy(
        CLIcmddata_creator.description,
        FPS_app_info_creator.description,
        sizeof(
            CLIcmddata_creator.description
        ) - 1);
    if (CLIcmddata_creator.cmdsettings
        == NULL) {
        CLIcmddata_creator.cmdsettings =
            &cms1;
    }
}

static errno_t compute_creator()
{
    shmim_setowner_creator(p_sname);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 2: shmimsetowncurrent (primary)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "shmimsetowncurrent",
    .cmdkey      = "shmimsetowncurrent",
    .description = "set owner to current PID"
};

static FPS_CLI_BINDING my_bindings[] = {
    FPS_PARAMS_1STREAM(FPS_X_BINDING)
};

static const int nb_bindings =
    sizeof(my_bindings) / sizeof(FPS_CLI_BINDING);

static CLICMDARGDEF farg[] = {
    FPS_PARAMS_1STREAM(FPS_X_FARG)
};

static CLICMDDATA CLIcmddata = {
    "",
    "",
    CLICMD_FIELDS_DEFAULTS
};

static CMDSETTINGS cms2 = {0};

static __attribute__((constructor))
void init_cms_current(void)
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

static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();
    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    shmim_setowner_current(p_sname);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 3: shmimsetowninit
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_init = {
    .fps_name    = "shmimsetowninit",
    .cmdkey      = "shmimsetowninit",
    .description = "set owner to init PID"
};

static CLICMDDATA CLIcmddata_init = {
    "",
    "",
    CLICMD_FIELDS_NOPARAM
};

static CMDSETTINGS cms3 = {0};

static __attribute__((constructor))
void init_cms_init(void)
{
    strncpy(CLIcmddata_init.key,
            FPS_app_info_init.cmdkey,
            sizeof(CLIcmddata_init.key)
            - 1);
    strncpy(CLIcmddata_init.description,
            FPS_app_info_init.description,
            sizeof(
                CLIcmddata_init.description
            ) - 1);
    if (CLIcmddata_init.cmdsettings
        == NULL) {
        CLIcmddata_init.cmdsettings =
            &cms3;
    }
}

static errno_t compute_init()
{
    shmim_setowner_init(p_sname);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)

static errno_t CLIfunction_creator(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_creator,
        farg, &CLIcmddata_creator,
        my_bindings, nb_bindings,
        compute_creator);
}

static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

static errno_t CLIfunction_init(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info_init,
        farg, &CLIcmddata_init,
        my_bindings, nb_bindings,
        compute_init);
}

errno_t
CLIADDCMD_COREMOD_memory__shmim_setowner()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);

    {
        int cmdi = RegisterCLIcmd(
            CLIcmddata_creator,
            CLIfunction_creator);
        CLIcmddata_creator.cmdsettings =
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
            CLIcmddata_init,
            CLIfunction_init);
        CLIcmddata_init.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    return RETURN_SUCCESS;
}
#endif
