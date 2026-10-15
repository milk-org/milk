// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_set_3Daxes.c
 * @brief   Set 3D image axes size
 *
 * Reshapes an image to 3D by setting axis sizes,
 * preserving total element count.
 * Uses FPS V2 framework.
 */

#include <stdint.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     *set3d_inimname = NULL;
static uint32_t *set3d_size0    = NULL;
static uint32_t *set3d_size1    = NULL;
static uint32_t *set3d_size2    = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                        \
    X(".imname", &set3d_inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image") \
    X(".size0", &set3d_size0, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "axis 0 size")         \
    X(".size1", &set3d_size1, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "axis 1 size")         \
    X(".size2", &set3d_size2, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "axis 2 size")


/* ================================================================
 * 4.  COMPUTATION LOGIC
 * ============================================================= */

/**
 * @brief Reshape image to 3D.
 *
 * Only applies if s0*s1*s2 == nelement.
 * Zero values inherit the current axis size.
 */
static MILK_HOT errno_t fpsexec(IMAGE *inimg)
{
    if (!set3d_size0 || !set3d_size1 || !set3d_size2)
    {
        return RETURN_FAILURE;
    }
    long nelem = inimg->md[0].nelement;

    uint32_t s0 = (*set3d_size0 == 0) ? inimg->md[0].size[0] : *set3d_size0;
    uint32_t s1 =
        (*set3d_size1 == 0) ? ((inimg->md[0].naxis < 2) ? 1 : inimg->md[0].size[1]) : *set3d_size1;
    uint32_t s2 =
        (*set3d_size2 == 0) ? ((inimg->md[0].naxis < 3) ? 1 : inimg->md[0].size[2]) : *set3d_size2;

    if ((long) s0 * s1 * s2 == nelem)
    {
        inimg->md[0].naxis   = 3;
        inimg->md[0].size[0] = s0;
        inimg->md[0].size[1] = s1;
        inimg->md[0].size[2] = s2;
    }
    return RETURN_SUCCESS;
}


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(image_set_3Daxes, FPS_PARAMS);


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t image_set_3Daxes_compute(const MILK_RUNNABLE *self)
{
    IMGID in = imgid_make_from_name(set3d_inimname);
    resolveIMGID(&in, ERRMODE_ABORT, dcimg, dcnimg);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    fpsexec(in.im);
    processinfo_update_output_stream(processinfo, in.im, NULL);

    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  RUNNABLE DESCRIPTOR
 * ============================================================= */

const MILK_RUNNABLE runnable_COREMOD_arith_image_set_3Daxes = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name         = "set3Daxes",
              .cmdkey           = "set3Daxes",
              .description      = "set 3D image axes size",
              .description_long = "Reshape a 3D image by redefining its axis dimensions without "
                                  "changing the underlying data. The total number of pixels must "
                                  "remain the same. Useful for reinterpreting data layout." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(image_set_3Daxes),
    .compute = image_set_3Daxes_compute,
};
EXPORT_FPSEXEC(milk, "arith-set3Daxes", runnable_COREMOD_arith_image_set_3Daxes);
