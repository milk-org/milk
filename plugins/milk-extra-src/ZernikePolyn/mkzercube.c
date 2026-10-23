// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file mkzercube.c
 * @brief Mkzercube module
 */

/**
 * @file mkzercube.c
 *
 */


#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include <math.h>

#include "zernike_value.h"


// zonal WFS response
//

/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     *outzcubename     = NULL;
static uint32_t *xsize            = NULL;
static uint32_t *ysize            = NULL;
static float    *xcent            = NULL;
static float    *ycent            = NULL;
static float    *radius           = NULL;
static float    *radiusmaskfactor = NULL;
static float    *TTfactor         = NULL;
static uint32_t *NBzermode        = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                        \
    X(".xsize", &xsize, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "X size")    \
    X(".ysize", &ysize, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "Y size")    \
    X(".xcent", &xcent, FPTYPE_FLOAT32, 1, FPFLAG_DEFAULT_INPUT, "X center") \
    X(".ycent", &ycent, FPTYPE_FLOAT32, 1, FPFLAG_DEFAULT_INPUT, "Y center") \
    X(".rad", &radius, FPTYPE_FLOAT32, 1, FPFLAG_DEFAULT_INPUT, "radius")    \
    X(".NBzermode", &NBzermode, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT, "Number modes")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(mkzercube, FPS_PARAMS);

static MILK_HOT errno_t mkzercube_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    zernike_init();

    IMGID imgout = imgid_make_from_name_3D(outzcubename, *xsize, *ysize, *NBzermode);
    createimagefromIMGID(&imgout);

    uint64_t xysize = *xsize;
    xysize *= *ysize;

    double *polar_r;
    double *polar_theta;


    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        polar_r = (double *) malloc(xysize * sizeof(double));
        if (polar_r == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        polar_theta = (double *) malloc(xysize * sizeof(double));
        if (polar_theta == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        if ((polar_r == NULL) || (polar_theta == NULL))
        {
            printf("error in memory allocation !!!\n");
        }

        // polar coordinates
        //
        for (uint32_t ii = 0; ii < *xsize; ii++)
        {
            float x = (*xcent) - ii;
            for (uint32_t jj = 0; jj < *ysize; jj++)
            {
                float y                     = (*ycent) - jj;
                polar_r[jj * (*xsize) + ii] = sqrt(x * x + y * y) / (*radius);

                polar_theta[jj * (*xsize) + ii] = atan2(y, x);
            }
        }


        // Make Zernikes
        //
        for (uint32_t zi = 0; zi < (*NBzermode); zi++)
        {
            float ampl = 1.0;
            if ((zi == 0) || (zi == 1))
            {
                ampl = *TTfactor;
            }
            else
            {
                ampl = 1.0;
            }
            for (uint32_t ii = 0; ii < xysize; ii++)
            {
                float r = polar_r[ii];
                if (r < (*radiusmaskfactor))
                {
                    imgout.im->array.F[zi * xysize + ii] =
                        ampl * Zernike_value(zi + 1, r, polar_theta[ii]);
                }
                else
                {
                    imgout.im->array.F[ii] = 0.0;
                }
            }
        }
    }
    MILK_PROCINFO_END(self)

    free(polar_r);
    free(polar_theta);


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  RUNNABLE DESCRIPTOR
 * ============================================================= */

const MILK_RUNNABLE runnable_ZernikePolyn_mkzercube = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "mkzerc",
                 .cmdkey      = "mkzerc",
                 .description = "make Zernike modes cube",
                 .description_long =
                     "Generate a 3D cube of Zernike polynomial mode images. Each slice is one "
                     "Zernike mode evaluated on a circular aperture." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(mkzercube),
    .compute = mkzercube_compute,
};
EXPORT_FPSEXEC(milk, "zernike-mkzercube", runnable_ZernikePolyn_mkzercube);
