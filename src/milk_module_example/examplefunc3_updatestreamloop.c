// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    updatestreamloop.c
 * @brief   simple procinfo+fps example - brief, no comments, uses macros
 *
 * Example 3
 * Demonstates function that updates a stream
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

// Variables local to this translation unit
static char *inimname;

static CLICMDARGDEF farg[] = { { CLIARG_IMG, ".in_sname", "input stream", "ims1",
                                 CLIARG_VISIBLE_DEFAULT, (void **) &inimname, NULL } };


static CLICMDDATA CLIcmddata = { "streamupdate", "update stream", CLICMD_FIELDS_DEFAULTS };


// Detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}


// Wrapper function, used by all CLI calls
static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID img = mkIMGID_from_name(inimname);
    resolveIMGID(&img, ERRMODE_ABORT);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    // Notify that the image is being changed.
    // This is required prior to modifying image content so that consumers can be informed.
    img.md->write = 1;

    // Insert code, or function(s) that perform operation(s) on image
    // If the code is very brief, it can be insterted right here, otherwise
    // it can be in a function, which may be made visible/accessible outside of this translation unit
    // if the function needs to be used outside of this call.

    // Call this to notify consumers that the image has been updated
    processinfo_update_output_stream(processinfo, img.ID);

    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions


    // Register function in CLI
    errno_t CLIADDCMD_milk_module_example__updatestreamloop()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
