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

#include "milk_runnable.h"
#include "examplefunc3_updatestreamloop.h"

#include "COREMOD_memory/COREMOD_memory.h"

// Variables local to this translation unit
static char inimname[FUNCTION_PARAMETER_STRMAXLEN] = "";

#define STREAMUPDATE_PARAMS(X) \
    X(".in_sname", inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input stream")

MILK_PARAMS_DEFINE(streamupdate, STREAMUPDATE_PARAMS);


// Wrapper function, used by all CLI calls
static MILK_HOT errno_t streamupdate_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    IMGID img = imgid_make_from_name(inimname);
    resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    // Notify that the image is being changed.
    // This is required prior to modifying image content so that consumers can be informed.
    img.md->write = 1;
    if (img.ID == -1)
    {
        return RETURN_FAILURE;
    }

    // Insert code, or function(s) that perform operation(s) on image
    // If the code is very brief, it can be insterted right here, otherwise
    // it can be in a function, which may be made visible/accessible outside of this translation unit
    // if the function needs to be used outside of this call.

    // Call this to notify consumers that the image has been updated
    processinfo_update_output_stream(processinfo, img.im, NULL);

    MILK_PROCINFO_END(self)

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_modex_streamupdate = {
    .kind = MILK_RUNNABLE_FPS,
    .app = { .fps_name = "streamupdate", .cmdkey = "streamupdate", .description = "update stream" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(streamupdate),
    .compute = streamupdate_compute,
};
