// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    imgen_mkfiberclpoverlap.c
 * @brief   fiber coupling overlap integral
 */

#include "milk_runnable.h"
#include "image_gen.h"
#include "image_gen_internal.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */
static char fc_n[FUNCTION_PARAMETER_STRMAXLEN] = "imdisk";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */
#define FPS_PARAMS(X) X(".out_name", fc_n, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */
MILK_PARAMS_DEFINE(imgen_mkfiberclpoverlap, FPS_PARAMS);

/* ================================================================
 * 4/6. COMPUTE FUNCTION
 * ============================================================= */
static MILK_HOT errno_t imgen_mkfiberclpoverlap_compute(const MILK_RUNNABLE *self)
{
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    make_FiberCouplingOverlap(fc_n);

    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}

/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

const MILK_RUNNABLE runnable_image_gen_imgen_mkfiberclpoverlap = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "mkfiberclpoverlap",
                 .cmdkey      = "mkfiberclpoverlap",
                 .description = "fiber coupling overlap integral",
                 .description_long =
                     "Compute the overlap integral between a PSF and a fiber mode for fiber "
                     "coupling efficiency estimation in coronagraphic instruments." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(imgen_mkfiberclpoverlap),
    .compute = imgen_mkfiberclpoverlap_compute,
};
EXPORT_FPSEXEC(milk, "imggen-mkfiberclpoverlap", runnable_image_gen_imgen_mkfiberclpoverlap);
