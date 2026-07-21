// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    shmim_setowner.c
 * @brief   set stream owner PID
 *
 * Uses FPS V2 framework.
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "image_ID.h"


/* ================================================================
 *  COMPUTATION LOGIC
 * ============================================================= */

/** @brief set owner to creator */
imageID shmim_setowner_creator(const char *name)
{
    IMGID img = imgid_make_from_name(name);
    resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);
    if (img.ID != -1)
    {
        img.im->md[0].ownerPID = img.im->md[0].creatorPID;
    }

    return img.ID;
}

/** @brief set owner to current PID */
imageID shmim_setowner_current(const char *name)
{
    IMGID img = imgid_make_from_name(name);
    resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);
    if (img.ID != -1)
    {
        img.im->md[0].ownerPID = getpid();
    }

    return img.ID;
}

/**
 * @brief set owner to init process
 *
 * Makes the stream immune to orphan purging
 */
imageID shmim_setowner_init(const char *name)
{
    IMGID img = imgid_make_from_name(name);
    resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);
    if (img.ID != -1)
    {
        img.im->md[0].ownerPID = 1;
    }

    return img.ID;
}


/* ================================================================
 *  COMMON PARAMETER (1 stream arg)
 * ============================================================= */

static char p_sname[FUNCTION_PARAMETER_STRMAXLEN] = "stream0";

#define FPS_PARAMS_1STREAM(X) \
    X(".sname", p_sname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "stream name")


/* ================================================================
 *  CMD 1: shmimsetowncreator
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_creator = {
    .fps_name         = "shmimsetowncreator",
    .cmdkey           = "shmimsetowncreator",
    .description      = "set owner to creator PID",
    .description_long = "Change the owner PID of a shared memory image stream. Useful for "
                        "transferring ownership when a process is restarted."
};

static CLICMDDATA CLIcmddata_creator = { "", "", CLICMD_FIELDS_NOPARAM };

FPS_CMDSETTINGS_INIT(cms1, CLIcmddata_creator, FPS_app_info_creator)

static errno_t __attribute__((unused)) compute_creator()
{
    shmim_setowner_creator(p_sname);
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 2: shmimsetowncurrent (primary)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "shmimsetowncurrent",
    .cmdkey           = "shmimsetowncurrent",
    .description      = "set owner to current PID",
    .description_long = "Change the owner PID of a shared memory image stream. Useful for "
                        "transferring ownership when a process is restarted."
};

FPS_V2_SECTION5(FPS_PARAMS_1STREAM)
static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();
    INSERT_STD_PROCINFO_COMPUTEFUNC_START shmim_setowner_current(p_sname);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END   DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 *  CMD 3: shmimsetowninit
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_init = {
    .fps_name         = "shmimsetowninit",
    .cmdkey           = "shmimsetowninit",
    .description      = "set owner to init PID",
    .description_long = "Change the owner PID of a shared memory image stream. Useful for "
                        "transferring ownership when a process is restarted."
};

static CLICMDDATA CLIcmddata_init = { "", "", CLICMD_FIELDS_NOPARAM };

FPS_CMDSETTINGS_INIT(cms3, CLIcmddata_init, FPS_app_info_init)

static errno_t __attribute__((unused)) compute_init()
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
    return safe_fps_generic_CLIfunction(&FPS_app_info_creator, farg, &CLIcmddata_creator,
                                        my_bindings, nb_bindings, compute_creator);
}

static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

static errno_t CLIfunction_init(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info_init, farg, &CLIcmddata_init, my_bindings,
                                        nb_bindings, compute_init);
}

errno_t CLIADDCMD_COREMOD_memory__shmim_setowner()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);

    {
        int cmdi                       = RegisterCLIcmd(CLIcmddata_creator, CLIfunction_creator);
        CLIcmddata_creator.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }

    {
        int cmdi               = RegisterCLIcmd(CLIcmddata, CLIfunction);
        CLIcmddata.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }

    {
        int cmdi                    = RegisterCLIcmd(CLIcmddata_init, CLIfunction_init);
        CLIcmddata_init.cmdsettings = &data.cmd[cmdi].cmdsettings;
    }

    return RETURN_SUCCESS;
}
#endif
