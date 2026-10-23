// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    imgen_segs2wfmodes.c
 * @brief   segments to WF modes
 */

#include "milk_runnable.h"
#include "image_gen.h"
#include "image_gen_internal.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */
static char    sw_pfx[FUNCTION_PARAMETER_STRMAXLEN] = "segim";
static int64_t sw_nd                                = 2;
static char    sw_out[FUNCTION_PARAMETER_STRMAXLEN] = "WFmodes";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */
#define FPS_PARAMS(X)                                                          \
    X(".prefix", sw_pfx, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "seg prefix") \
    X(".ndigit", &sw_nd, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "nb digits")   \
    X(".out_name", sw_out, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */
MILK_PARAMS_DEFINE(imgen_segs2wfmodes, FPS_PARAMS);

/* ================================================================
 * 4/6. COMPUTE FUNCTION
 * ============================================================= */
static MILK_HOT errno_t imgen_segs2wfmodes_compute(const MILK_RUNNABLE *self)
{
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    IMAGE_gen_segments2WFmodes(sw_pfx, (long) sw_nd, sw_out);

    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}

/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

const MILK_RUNNABLE runnable_image_gen_imgen_segs2wfmodes = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "segs2wfmodes",
                 .cmdkey           = "segs2wfmodes",
                 .description      = "segments to WF modes",
                 .description_long = "Convert segment piston/tip/tilt commands into wavefront "
                                     "mode images for segmented-mirror telescope simulation." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(imgen_segs2wfmodes),
    .compute = imgen_segs2wfmodes_compute,
};
EXPORT_FPSEXEC(milk, "imggen-segs2wfmodes", runnable_image_gen_imgen_segs2wfmodes);
