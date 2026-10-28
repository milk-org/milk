// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file GramSchmidt.c
 * @brief Gramschmidt module
 */

#include <math.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_iofits/COREMOD_iofits.h"

#include "COREMOD_tools/COREMOD_tools.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char    *inmodes   = NULL;
static char    *outmodes  = NULL;
static char    *auxmat    = NULL;
static int32_t *GPUdevice = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                  \
    X(".inmodes", &inmodes, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input modes") \
    X(".outmodes", &outmodes, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output modes")


/* ================================================================
 * 5.  PARAMETER BINDINGS
 * ============================================================= */

MILK_PARAMS_DEFINE(GramSchmidt, FPS_PARAMS);

errno_t GramSchmidt(IMGID imginm, IMGID *imgoutm, IMGID imgaux, int GPUdev __attribute__((unused)))
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(&imginm, ERRMODE_WARN, dcimg, dcnimg);

    resolveIMGID(&imgaux, ERRMODE_WARN, dcimg, dcnimg);
    if (imginm.ID == -1)
    {
        return RETURN_FAILURE;
    }


    // Compute cross product on input
    //
    //IMGID imginxp  = imgid_make_from_name("_outxp");
    //computeSGEMM(imginm, imginm, &imginxp, 1, 0, GPUdev);


    // Create output
    //
    imcreatelikewiseIMGID(imgoutm, &imginm);

    uint32_t zsize;
    uint32_t xysize = imginm.md->size[0];
    if (imginm.md->naxis == 3)
    {
        zsize = imginm.md->size[2];
        xysize *= imginm.md->size[1];
    }
    else
    {
        zsize = imginm.md->size[1];
    }

    uint32_t xysizeaux = 0;
    if (imgaux.ID != -1)
    {
        xysizeaux = imgaux.md->size[0];
        if (imginm.md->naxis == 3)
        {
            xysizeaux *= imgaux.md->size[1];
        }
    }


    printf("xysize = %u, zsize = %u\n", xysize, zsize);

    int nth = milk_omp_team_size((uint64_t) xysize, 262144);

    printf("\n");
    for (uint32_t kk = 0; kk < zsize; kk++)
    {
        printf("\rGS mode %6u / %6u     ", kk, zsize);

        float *restrict vk = &imgoutm->im->array.F[(uint64_t) kk * xysize];

        // initialization
        __builtin_memcpy(vk, &imginm.im->array.F[(uint64_t) kk * xysize], sizeof(float) * xysize);

        for (uint32_t kk0 = 0; kk0 < kk; kk0++)
        {
            const float *restrict v0ptr = &imgoutm->im->array.F[(uint64_t) kk0 * xysize];

            // cross-product and square sum v0
            double xpval   = 0.0;
            double sqrsum0 = 0.0;

#pragma omp parallel for simd reduction(+ : xpval, sqrsum0) num_threads(nth) if (parallel : nth > 1)
            for (uint32_t ii = 0; ii < xysize; ii++)
            {
                float v0 = v0ptr[ii];
                float v1 = vk[ii];

                xpval += (double) v0 * (double) v1;
                sqrsum0 += (double) v0 * (double) v0;
            }

            float vcoeff = (float) (xpval / sqrsum0);

            //printf("  %5u  %5u   %f\n", kk, kk0, vcoeff);

#pragma omp parallel for simd num_threads(nth) if (parallel : nth > 1)
            for (uint32_t ii = 0; ii < xysize; ii++)
            {
                vk[ii] -= vcoeff * v0ptr[ii];
            }

            if (imgaux.ID != -1)
            {
                float *restrict auxk        = &imgaux.im->array.F[(uint64_t) kk * xysizeaux];
                const float *restrict auxk0 = &imgaux.im->array.F[(uint64_t) kk0 * xysizeaux];
#pragma omp simd
                for (uint32_t ii = 0; ii < xysizeaux; ii++)
                {
                    auxk[ii] -= vcoeff * auxk0[ii];
                }
            }
        }
    }
    printf("\n");


    list_image_ID();

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


static MILK_HOT errno_t GramSchmidt_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    IMGID imginm = imgid_make_from_name(inmodes);
    resolveIMGID(&imginm, ERRMODE_WARN, dcimg, dcnimg);


    IMGID imgoutm = imgid_make_from_name(outmodes);
    if (imginm.ID == -1)
    {
        return RETURN_FAILURE;
    }

    IMGID imgaux = imgid_make_from_name(auxmat);
    resolveIMGID(&imgaux, ERRMODE_WARN, dcimg, dcnimg);


    MILK_PROCINFO_INIT(self)


    MILK_PROCINFO_LOOPSTART(self)
    {
        GramSchmidt(imginm, &imgoutm, imgaux, *GPUdevice);
    }
    MILK_PROCINFO_END(self)


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK_RUNNABLE DESCRIPTOR AND FPSEXEC EXPORT
 * ============================================================= */

const MILK_RUNNABLE runnable_linalgebra_GramSchmidt = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "GramSchmidt",
                 .cmdkey           = "GramSchmidt",
                 .description      = "Gram-Schmidt process",
                 .description_long = "Orthogonalize a set of vectors using the Gram-Schmidt "
                                     "process. Produces an orthonormal basis from the input set." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(GramSchmidt),
    .compute = GramSchmidt_compute,
};
EXPORT_FPSEXEC(milk, "linalg-gramschmidt", runnable_linalgebra_GramSchmidt);
