// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    imgen_mkhexsegpup.c
 * @brief   make hex seg pupil
 */

#include "milk_runnable.h"
#include "image_gen.h"
#include "image_gen_internal.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */
static char    hx_n[FUNCTION_PARAMETER_STRMAXLEN] = "imhex";
static int64_t hx_sz                              = 4096;
static double  hx_r                               = 200.0;
static double  hx_g                               = 2.0;
static double  hx_s                               = 46.3;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */
#define FPS_PARAMS(X)                                                      \
    X(".out_name", hx_n, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output") \
    X(".size", &hx_sz, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "size")      \
    X(".radius", &hx_r, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "radius") \
    X(".gap", &hx_g, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "gap")       \
    X(".step", &hx_s, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "step")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */
MILK_PARAMS_DEFINE(imgen_mkhexsegpup, FPS_PARAMS);

/* ================================================================
 * 4/6. COMPUTE FUNCTION
 * ============================================================= */
static MILK_HOT errno_t imgen_mkhexsegpup_compute(const MILK_RUNNABLE *self)
{
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    make_hexsegpupil(hx_n, (uint32_t) hx_sz, hx_r, hx_g, hx_s);

    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}

/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

const MILK_RUNNABLE runnable_image_gen_imgen_mkhexsegpup = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name         = "mkhexsegpup",
              .cmdkey           = "mkhexsegpup",
              .description      = "make hex seg pupil",
              .description_long = "Generate a hexagonally-segmented pupil pattern for simulating "
                                  "segmented-mirror telescopes (e.g., JWST, ELT)." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(imgen_mkhexsegpup),
    .compute = imgen_mkhexsegpup_compute,
};
EXPORT_FPSEXEC(milk, "imggen-mkhexsegpup", runnable_image_gen_imgen_mkhexsegpup);
