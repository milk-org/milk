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

    double *polar_r     = (double *) malloc(xysize * sizeof(double));
    double *polar_theta = (double *) malloc(xysize * sizeof(double));
    if ((polar_r == NULL) || (polar_theta == NULL))
    {
        free(polar_r);
        free(polar_theta);
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        uint32_t nx     = *xsize;
        uint32_t ny     = *ysize;
        uint32_t nmodes = *NBzermode;
        float    xc     = *xcent;
        float    yc     = *ycent;
        double   invrad = 1.0 / (double) (*radius);
        float    rmask  = (radiusmaskfactor != NULL) ? *radiusmaskfactor : 1.0f;
        float    ttfact = (TTfactor != NULL) ? *TTfactor : 1.0f;

        int nth2d = milk_omp_team_size(xysize, 65536);

        // polar coordinates (row-major traversal)
#pragma omp parallel for num_threads(nth2d) if (nth2d > 1)
        for (uint32_t jj = 0; jj < ny; jj++)
        {
            double y              = (double) yc - (double) jj;
            double y2             = y * y;
            double *restrict rrow = polar_r + (uint64_t) jj * nx;
            double *restrict trow = polar_theta + (uint64_t) jj * nx;
            for (uint32_t ii = 0; ii < nx; ii++)
            {
                double x = (double) xc - (double) ii;
                rrow[ii] = sqrt(x * x + y2) * invrad;
                trow[ii] = atan2(y, x);
            }
        }

        // Make Zernikes
        float *restrict outptr = imgout.im->array.F;
        int nth3d              = milk_omp_team_size((uint64_t) nmodes * xysize, 65536);

#pragma omp parallel for num_threads(nth3d) if (nth3d > 1)
        for (uint32_t zi = 0; zi < nmodes; zi++)
        {
            float ampl            = ((zi == 0) || (zi == 1)) ? ttfact : 1.0f;
            float *restrict slice = outptr + (uint64_t) zi * xysize;
            for (uint64_t ii = 0; ii < xysize; ii++)
            {
                double r = polar_r[ii];
                if (r < (double) rmask)
                {
                    slice[ii] = (float) (ampl * Zernike_value((long) zi + 1, r, polar_theta[ii]));
                }
                else
                {
                    slice[ii] = 0.0f;
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
