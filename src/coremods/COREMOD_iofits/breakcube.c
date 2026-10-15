// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    breakcube.c
 * @brief   break cube into individual 2D images
 *
 * Uses FPS V2 framework.
 */

#include <stdlib.h>
#include <string.h>

#include "milk_runnable.h"
#include "computeunit.h"

#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char inname[FUNCTION_PARAMETER_STRMAXLEN] = "imc";


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".in_name", inname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input cube image")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

imageID break_cube(const char *restrict ID_name)
{
    IMGID imgin = imgid_make_from_name(ID_name);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }

    uint32_t xsize = imgin.md->size[0];
    uint32_t ysize = imgin.md->size[1];
    uint32_t nz    = imgin.md->size[2];

    for (uint32_t kk = 0; kk < nz; kk++)
    {
        char framename[STRINGMAXLEN_IMGNAME];
        CREATE_IMAGENAME(framename, "%s_%5u", ID_name, kk);
        for (long i = 0; i < (long) strlen(framename); i++)
        {
            if (framename[i] == ' ')
            {
                framename[i] = '0';
            }
        }

        IMGID imgfr       = imgid_make_from_name_2D(framename, xsize, ysize);
        imgfr.mdt->shared = 0;
        imgfr.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
        imgid_mkimage(&imgfr);

        for (uint32_t ii = 0; ii < xsize; ii++)
        {
            for (uint32_t jj = 0; jj < ysize; jj++)
            {
                imgfr.im->array.F[jj * xsize + ii] =
                    imgin.im->array.F[kk * xsize * ysize + jj * xsize + ii];
            }
        }
    }

    return imgin.ID;
}


MILK_PARAMS_DEFINE(breakcube, FPS_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t breakcube_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    break_cube(inname);

    MILK_PROCINFO_END(self)
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_COREMOD_iofits_breakcube = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "breakcube",
                 .cmdkey           = "breakcube",
                 .description      = "break cube into individual images",
                 .description_long = "Split a 3D FITS cube into individual 2D FITS files, one per "
                                     "slice along the z-axis. Output files are numbered "
                                     "sequentially." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(breakcube),
    .compute = breakcube_compute,
};
