// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file PCAmatch.c
 *
 * @brief match two PCA decompositions
 *
 * Find corresponding linear combination across two basis
 *
 */

#include <math.h>

#include "milk_runnable.h"
#include "ImageStreamIO/ImageStruct.h"
#include "COREMOD_memory/COREMOD_memory.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char    *incoeffM  = NULL;
static char    *outcoeffM = NULL;
static int32_t *axis      = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                            \
    X(".incoeffM", &incoeffM, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input coeffs matrix") \
    X(".outcoeffM", &outcoeffM, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output coeffs matrix")  \
    X(".axis", &axis, FPTYPE_INT32, 1, FPFLAG_DEFAULT_INPUT, "axis")


/* ================================================================
 * 5.  PARAMETER BINDINGS
 * ============================================================= */

MILK_PARAMS_DEFINE(Qexpand, FPS_PARAMS);

/**
 * @brief Quadratic expansion of vector(s)
 *
 *
 * Adds constant in front of vector, and quadractic terms behind.
 * For example, vector [a,b] -> [1,a,b,aa,ab,bb]
 *
 * @param incoeffM   input coeffs
 * @param outcoeffM  output coeffs
 * @param axis       expansion axis
 * @return errno_t
 *
 * Only handles 2D images
 */
errno_t Qexpand(IMGID imgincoeffM, IMGID *imgoutcoeffM, int axis)
{
    DEBUG_TRACE_FSTART();

    int vecindexaxis = 1;
    if (axis == 1)
    {
        vecindexaxis = 0;
    }
    uint32_t vecdim = imgincoeffM.md->size[axis];
    uint32_t NBvec  = imgincoeffM.md->size[vecindexaxis];

    printf("axis = %d\n", axis);
    printf("vecdim = %u\n", vecdim);
    printf("NBvec = %u\n", NBvec);
    fflush(stdout);


    // vecdimout is output vector dimension
    //
    uint32_t vecdimout = 1 + vecdim + vecdim * (vecdim + 1) / 2;


    // output vectors
    imgoutcoeffM->mdt->datatype = _DATATYPE_FLOAT;
    imgoutcoeffM->mdt->naxis    = 2;

    imgoutcoeffM->mdt->size[axis]         = vecdimout;
    imgoutcoeffM->mdt->size[vecindexaxis] = NBvec;

    printf("CREATING %s\n", imgoutcoeffM->name);
    fflush(stdout);
    createimagefromIMGID(imgoutcoeffM);

    {
        FILE *fp = fopen("Qmode.def.txt", "w");
        fprintf(fp, "0   \n");
        for (uint32_t ii = 0; ii < vecdim; ii++)
        {
            fprintf(fp, "%05u   %05u\n", ii + 1, ii);
        }
        uint32_t index = vecdim + 1;
        for (uint32_t ii = 0; ii < vecdim; ii++)
        {
            for (uint32_t jj = ii; jj < vecdim; jj++)
            {
                fprintf(fp, "%05d   %05u  %05u\n", index, ii, jj);
                index++;
            }
        }
        fclose(fp);
    }


    for (uint32_t vec = 0; vec < NBvec; vec++)
    {
        // copy linear part
        if (axis == 0)
        {
            imgoutcoeffM->im->array.F[vec * vecdimout] = 1.0;
            for (uint32_t ii = 0; ii < vecdim; ii++)
            {
                imgoutcoeffM->im->array.F[vec * vecdimout + ii + 1] =
                    imgincoeffM.im->array.F[vec * vecdim + ii];
            }
        }
        else
        {
            imgoutcoeffM->im->array.F[vec] = 1.0;
            for (uint32_t ii = 0; ii < vecdim; ii++)
            {
                imgoutcoeffM->im->array.F[(ii + 1) * NBvec + vec] =
                    imgincoeffM.im->array.F[ii * NBvec + vec];
            }
        }

        uint32_t index = vecdim + 1;

        for (uint32_t ii = 0; ii < vecdim; ii++)
        {
            for (uint32_t jj = ii; jj < vecdim; jj++)
            {
                if (axis == 0)
                {
                    float valii = imgincoeffM.im->array.F[vec * vecdim + ii];
                    float valjj = imgincoeffM.im->array.F[vec * vecdim + jj];
                    imgoutcoeffM->im->array.F[vec * vecdimout + index] = valii * valjj;
                }
                else
                {
                    float valii = imgincoeffM.im->array.F[ii * NBvec + vec];
                    float valjj = imgincoeffM.im->array.F[jj * NBvec + vec];
                    imgoutcoeffM->im->array.F[index * NBvec + vec] = valii * valjj;
                }
                index++;
            }
        }
    }


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


static MILK_HOT errno_t Qexpand_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    IMGID imgincoeffM = imgid_make_from_name(incoeffM);
    resolveIMGID(&imgincoeffM, ERRMODE_WARN, dcimg, dcnimg);


    fflush(stdout);
    if (imgincoeffM.ID == -1)
    {
        return RETURN_FAILURE;
    }
    IMGID imgoutcoeffM = imgid_make_from_name(outcoeffM);


    MILK_PROCINFO_INIT(self)


    MILK_PROCINFO_LOOPSTART(self)
    {
        Qexpand(imgincoeffM, &imgoutcoeffM, *axis);

        processinfo_update_output_stream(processinfo, imgoutcoeffM.im, NULL);
    }
    MILK_PROCINFO_END(self)

    imgid_free(&imgincoeffM);
    imgid_free(&imgoutcoeffM);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK_RUNNABLE DESCRIPTOR AND FPSEXEC EXPORT
 * ============================================================= */

const MILK_RUNNABLE runnable_linalgebra_Qexpand = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "Qexpand",
                 .cmdkey           = "Qexpand",
                 .description      = "quadractic expansion of vector or matrix coeffs",
                 .description_long = "Compute the quadratic expansion of a vector or matrix of "
                                     "coefficients. Generates cross-term products for nonlinear "
                                     "modeling." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(Qexpand),
    .compute = Qexpand_compute,
};
EXPORT_FPSEXEC(milk, "linalg-Qexpand", runnable_linalgebra_Qexpand);
