// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file image_keyword_add.c
 * @brief Image keyword add module
 */

#include <stdlib.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */
static char param_inimname[FUNCTION_PARAMETER_STRMAXLEN] = "im1";
static char param_kwname[FUNCTION_PARAMETER_STRMAXLEN]   = "KW1";
static char param_kwtype[FUNCTION_PARAMETER_STRMAXLEN]   = "D";
static char param_kwval[FUNCTION_PARAMETER_STRMAXLEN]    = "1.234";
static char param_comment[FUNCTION_PARAMETER_STRMAXLEN]  = "keyword comment";

/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */
#define FPS_PARAMS(X)                                                                            \
    X(".in_name", param_inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image")     \
    X(".kwname", param_kwname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "keyword name")           \
    X(".kwtype", param_kwtype, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "keyword type (L, D, S)") \
    X(".kwval", param_kwval, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "keyword value")            \
    X(".comment", param_comment, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "keyword comment")

/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */
static MILK_HOT errno_t fpsexec()
{
    char type = param_kwtype[0];
    if (type == 'L' || type == 'l')
    {
        long val = atol(param_kwval);
        image_write_keyword_L(param_inimname, param_kwname, val, param_comment);
    }
    else if (type == 'D' || type == 'd')
    {
        double val = atof(param_kwval);
        image_write_keyword_D(param_inimname, param_kwname, val, param_comment);
    }
    else if (type == 'S' || type == 's')
    {
        image_write_keyword_S(param_inimname, param_kwname, param_kwval, param_comment);
    }
    else
    {
        PRINT_ERROR("Invalid keyword type '%c'. Must be L, D, or S.", type);
        return RETURN_FAILURE;
    }
    return RETURN_SUCCESS;
}

/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */
MILK_PARAMS_DEFINE(image_keyword_add, FPS_PARAMS);

/* ================================================================
 * 6.  COMPUTE WRAPPER (processinfo loop support)
 * ============================================================= */
static MILK_HOT errno_t image_keyword_add_compute(const MILK_RUNNABLE *self)
{
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    fpsexec();

    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}

/* ================================================================
 * 7.  RUNNABLE
 * ============================================================= */
const MILK_RUNNABLE runnable_COREMOD_memory_image_keyword_add = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "imkwadd",
                 .cmdkey      = "imkwadd",
                 .description = "Add or update a keyword in an image" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(image_keyword_add),
    .compute = image_keyword_add_compute,
};
EXPORT_FPSEXEC(milk, "imkwadd", runnable_COREMOD_memory_image_keyword_add);
