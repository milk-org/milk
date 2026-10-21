// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_keyword_list.c
 * @brief   Image keyword list module
 *
 * Uses FPS V2 framework.
 */

#include "milk_runnable.h"

#include "COREMOD_memory/COREMOD_memory.h"


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

MILK_PARAMS_DEFINE(image_keyword_list, FPS_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t image_keyword_list_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    image_keywords_list(imgid_make_from_name(inname));

    MILK_PROCINFO_END(self)
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  RUNNABLE
 * ============================================================= */

const MILK_RUNNABLE runnable_COREMOD_memory_image_keyword_list = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "imkwlist",
                 .cmdkey      = "imkwlist",
                 .description = "list image keywords",
                 .description_long =
                     "List all FITS-style keywords attached to an image stream, showing "
                     "name, typed value, and comment for each entry." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(image_keyword_list),
    .compute = image_keyword_list_compute,
};
