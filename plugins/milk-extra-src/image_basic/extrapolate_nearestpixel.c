// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file extrapolate_nearestpixel.c
 * @brief Extrapolate nearestpixel module
 */

/** @file extrapolate_nearestpixel.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#    include "fps.h"
#    include "ImageStreamIO/ImageStreamIO.h"
#endif

#include "COREMOD_memory/COREMOD_memory.h"

#include "imcontract.h"

/**
 * Extrapolate values to unmasked pixels
 * using nearest masked pixel.
 */
imageID basic_2Dextrapolate_nearestpixel(const char *__restrict IDin_name,
                                         const char *__restrict IDmask_name,
                                         const char *__restrict IDout_name)
{
    DEBUG_TRACE_FSTART();

    IMGID imgin = imgid_make_from_name(IDin_name);
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }

    IMGID imgmask = imgid_make_from_name(IDmask_name);
    resolveIMGID(&imgmask, ERRMODE_WARN, dcimg, dcnimg);
    if (imgmask.ID == -1)
    {
        return RETURN_FAILURE;
    }

    list_image_ID();
    IMGID imgmask1 = imgid_make_from_name("_mask1");
    resolveIMGID(&imgmask1, ERRMODE_WARN, dcimg, dcnimg);
    if (imgmask1.ID != -1)
    {
        printf("USING MASK\n");
    }

    long naxes[2];
    naxes[0] = imgin.md->size[0];
    naxes[1] = imgin.md->size[1];

    long NBmaskpts = 0;
    for (long ii = 0; ii < naxes[0]; ii++)
    {
        for (long jj = 0; jj < naxes[1]; jj++)
        {
            if (imgmask.im->array.F[jj * naxes[0] + ii] > 0.5)
            {
                NBmaskpts++;
            }
        }
    }

    long *maskii = (long *) malloc(sizeof(long) * NBmaskpts);
    if (maskii == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc error");
        exit(0);
    }
    maskii[0] = 0;

    long *maskjj = (long *) malloc(sizeof(long) * NBmaskpts);
    if (maskjj == NULL)
    {
        C_ERRNO = errno;
        PRINT_ERROR("malloc error");
        exit(0);
    }
    maskjj[0] = 0;

    NBmaskpts = 0;
    for (long ii = 0; ii < naxes[0]; ii++)
    {
        for (long jj = 0; jj < naxes[1]; jj++)
        {
            if (imgmask.im->array.F[jj * naxes[0] + ii] > 0.5)
            {
                maskii[NBmaskpts] = ii;
                maskjj[NBmaskpts] = jj;
                NBmaskpts++;
            }
        }
    }

    IMGID imgout       = imgid_make_from_name_2D(IDout_name, naxes[0], naxes[1]);
    imgout.mdt->shared = 0;
    imgout.im          = (IMAGE *) calloc(1, sizeof(IMAGE));
    imgid_mkimage(&imgout);
    printf("imout = %s\n", IDout_name);
    printf("\n");

    for (long ii = 0; ii < naxes[0]; ii++)
    {
        printf("\r%ld / %ld  ", ii, naxes[0]);
        fflush(stdout);

        for (long jj = 0; jj < naxes[1]; jj++)
        {
            double bdist = (double) (naxes[0] + naxes[1]);
            bdist        = bdist * bdist;
            for (long k = 0; k < NBmaskpts; k++)
            {
                long   ii1  = maskii[k];
                long   jj1  = maskjj[k];
                double dist = 1.0 * ((ii1 - ii) * (ii1 - ii) + (jj1 - jj) * (jj1 - jj));
                if (dist < bdist)
                {
                    bdist = dist;
                    imgout.im->array.F[jj * naxes[0] + ii] =
                        imgin.im->array.F[jj1 * naxes[0] + ii1];
                }
            }
        }
    }

    printf("\n");

    free(maskii);
    free(maskjj);

    DEBUG_TRACE_FEXIT();
    return imgout.ID;
}
