// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file makeCosRadModes.c
 * @brief Makecosradmodes module
 */

#include <math.h>

#include "milk_runnable.h"
#include "linopt_imtools.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char   *outimname       = NULL;
static long   *sizeout         = NULL;
static long   *kmaxval         = NULL;
static double *radiusval       = NULL;
static double *radfactorlimval = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                 \
    X(".outim", &outimname, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output image")   \
    X(".size", &sizeout, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "size")               \
    X(".kmax", &kmaxval, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT, "k max")              \
    X(".radius", &radiusval, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "radius [pix]") \
    X(".rfactlim", &radfactorlimval, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "radius factor limit")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

MILK_PARAMS_DEFINE(makeCosRadModes, FPS_PARAMS);


/**
 * Make basis of cosine radial modes.
 */
errno_t linopt_imtools_makeCosRadModes(const char *ID_name,
                                       long        size,
                                       long        kmax,
                                       float       radius,
                                       float       radfactlim,
                                       imageID    *outID)
{
    DEBUG_TRACE_FSTART();

    long size2 = size * size;

    /* Intermediate r-coord image: registered in the global pool
     * so that delete_image_ID() can find and free it. */
    imageID IDr;
    FUNC_CHECK_RETURN(create_2Dimage_ID("linopt_tmpr", size, size, &IDr));

    FILE *fp = fopen("ModesExpr_CosRad.txt", "w");
    fprintf(fp, "# unit for r = %f pix\n", radius);
    fprintf(fp, "\n");
    for (long k = 0; k < kmax; k++)
    {
        fprintf(fp, "%5ld   cos(r*M_PI*%ld)\n", k, k);
    }
    fclose(fp);

    float *restrict rptr = dcimg[IDr].array.F;
    float invrad         = 1.0f / radius;
    float halfs          = 0.5f * (float) size;
    int   nth2d          = milk_omp_team_size((uint64_t) size2, 65536);

#pragma omp parallel for num_threads(nth2d) if (nth2d > 1)
    for (long jj = 0; jj < size; jj++)
    {
        float y             = ((float) jj - halfs) * invrad;
        float y2            = y * y;
        float *restrict row = rptr + jj * size;
#pragma omp simd
        for (long ii = 0; ii < size; ii++)
        {
            float x = ((float) ii - halfs) * invrad;
            row[ii] = sqrtf(x * x + y2);
        }
    }

    /* Output 3-D modes cube: registered in the global pool
     * so that the caller can find it with image_ID(). */
    imageID IDout;
    FUNC_CHECK_RETURN(create_3Dimage_ID(ID_name, size, size, kmax, &IDout));

    float *restrict outptr = dcimg[IDout].array.F;
    int nth3d              = milk_omp_team_size((uint64_t) kmax * (uint64_t) size2, 65536);

#pragma omp parallel for num_threads(nth3d) if (nth3d > 1)
    for (long k = 0; k < kmax; k++)
    {
        float *restrict slice = outptr + k * size2;
        float freq            = (float) M_PI * (float) k;
        for (long ii = 0; ii < size2; ii++)
        {
            float r = rptr[ii];
            if (r < radfactlim)
            {
                slice[ii] = cosf(r * freq);
            }
        }
    }

    FUNC_CHECK_RETURN(delete_image_ID("linopt_tmpr", DELETE_IMAGE_ERRMODE_WARNING));

    if (outID != NULL)
    {
        *outID = IDout;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

static MILK_HOT errno_t makeCosRadModes_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)

    linopt_imtools_makeCosRadModes(outimname, *sizeout, *kmaxval, *radiusval, *radfactorlimval,
                                   NULL);

    MILK_PROCINFO_END(self)

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

const MILK_RUNNABLE runnable_linopt_imtools_makeCosRadModes = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name    = "mkcosrmodes",
                 .cmdkey      = "mkcosrmodes",
                 .description = "make basis of cosine radial modes",
                 .description_long =
                     "Generate a basis of cosine radial modes within a circular aperture. "
                     "Useful for radial wavefront decomposition." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(makeCosRadModes),
    .compute = makeCosRadModes_compute,
};
EXPORT_FPSEXEC(milk, "linopt-mkCosRadModes", runnable_linopt_imtools_makeCosRadModes);
