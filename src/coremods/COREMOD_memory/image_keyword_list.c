// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_keyword_list.c
 * @brief   Image keyword list module
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


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "imkwlist",
    .cmdkey           = "imkwlist",
    .description      = "list image keywords",
    .description_long = "List all FITS-style keywords attached to an image stream, showing name, "
                        "typed value, and comment for each entry."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char inname[FUNCTION_PARAMETER_STRMAXLEN] = "im1";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".in_name", inname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

errno_t image_keywords_list(IMGID img)
{
    resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);

    int NBkw = img.md->NBkw;
    if (img.ID == -1)
    {
        return RETURN_FAILURE;
    }
    int kwcnt = 0;
    for (int kw = 0; kw < NBkw; kw++)
    {
        char tmpkwvalstr[81];
        switch (img.im->kw[kw].type)
        {
        case 'L':
            printf("[L] %-8s= %20ld / %s\n", img.im->kw[kw].name, img.im->kw[kw].value.numl,
                   img.im->kw[kw].comment);
            kwcnt++;
            break;

        case 'D':
            printf("[D] %-8s= %20g / %s\n", img.im->kw[kw].name, img.im->kw[kw].value.numf,
                   img.im->kw[kw].comment);
            kwcnt++;
            break;

        case 'S':
            snprintf(tmpkwvalstr, sizeof(tmpkwvalstr), "'%s'", img.im->kw[kw].value.valstr);
            printf("[S] %-8s= %-20s / %s\n", img.im->kw[kw].name, tmpkwvalstr,
                   img.im->kw[kw].comment);
            kwcnt++;
            break;

        default:
            break;
        }
    }

    printf("%d / %d keyword(s)\n", kwcnt, NBkw);

    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t __attribute__((unused)) compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_START image_keywords_list(imgid_make_from_name(inname));

    INSERT_STD_PROCINFO_COMPUTEFUNC_END DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_COREMOD_memory__image_keyword_list()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);

    int cmdi               = RegisterCLIcmd(CLIcmddata, CLIfunction);
    CLIcmddata.cmdsettings = &data.cmd[cmdi].cmdsettings;

    return RETURN_SUCCESS;
}
#endif
