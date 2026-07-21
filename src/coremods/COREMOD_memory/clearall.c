// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    clearall.c
 * @brief   remove all images, variables, and FPS
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

#include "delete_image.h"
#include "delete_variable.h"
#include "image_ID.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "rmall",
    .cmdkey      = "rmall",
    .description = "remove all images",
    .description_long =
        "Remove all images and variables from the current process memory space. Frees all locally "
        "allocated image buffers but does not destroy shared memory segments."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

/* (none — zero-arg command) */


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) /* empty */


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

errno_t clearall()
{
    imageID ID;

    // clear images
    for (ID = 0; ID < dcnimg; ID++)
    {
        if (dcimg[ID].used == 1)
        {
            delete_image_ID(dcimg[ID].name, DELETE_IMAGE_ERRMODE_WARNING);
        }
    }

    // clear variables
    for (ID = 0; ID < dcnvar; ID++)
    {
        if (dcvar[ID].used == 1)
        {
            delete_variable_ID(dcvar[ID].name);
        }
    }

    // clear FPS
    for (int fpsindex = 0; fpsindex < dcnfps; fpsindex++)
    {
        DEBUG_TRACEPOINT("clear FPS %d", fpsindex);
        dcfpsarr[fpsindex].SMfd = -1;
        if (dcfpsarr[fpsindex].parray != NULL)
        {
            dcfpsarr[fpsindex].parray = NULL;
        }
        if (dcfpsarr[fpsindex].md != NULL)
        {
            dcfpsarr[fpsindex].md = NULL;
        }
    }

    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

#ifdef FPS_STANDALONE
CLICMDDATA CLIcmddata = {
#else
static CLICMDDATA CLIcmddata = {
#endif
    "", "", CLICMD_FIELDS_NOPARAM
};

FPS_CMDSETTINGS_INIT(dft, CLIcmddata, FPS_app_info)


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();

    FUNC_CHECK_RETURN(clearall());

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, NULL, &CLIcmddata, NULL, 0,
                                        compute_function);
}

errno_t CLIADDCMD_COREMOD_memory__clearall()
{
    int cmdi               = RegisterCLIcmd(CLIcmddata, CLIfunction);
    CLIcmddata.cmdsettings = &data.cmd[cmdi].cmdsettings;

    return RETURN_SUCCESS;
}
#endif
