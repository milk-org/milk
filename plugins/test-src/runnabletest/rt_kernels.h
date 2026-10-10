// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    rt_kernels.h
 * @brief   Compute kernels shared by the runnabletest and runnabletest_legacy modules
 *
 * Pure computation on IMGID. Include after the milk headers providing IMGID / resolveIMGID.
 */

#ifndef RT_KERNELS_H
#define RT_KERNELS_H

/** @brief out = gain * in + offset, over all pixels of a float stream */
static inline errno_t rt_stream_affine(IMGID *inimg, IMGID *outimg, float gain, float offset)
{
    resolveIMGID(inimg, ERRMODE_WARN, dcimg, dcnimg);
    if (inimg->ID == -1)
    {
        return RETURN_FAILURE;
    }

    const float *in    = inimg->im->array.F;
    float       *out   = outimg->im->array.F;
    uint64_t     nbpix = inimg->md->nelement;

    outimg->md->write = 1;
    for (uint64_t ii = 0; ii < nbpix; ii++)
    {
        out[ii] = gain * in[ii] + offset;
    }

    return RETURN_SUCCESS;
}

/** @brief Print scaling * (sum of all pixels) of a float stream */
static inline errno_t rt_stream_total(IMGID *img, double scaling)
{
    resolveIMGID(img, ERRMODE_WARN, dcimg, dcnimg);
    if (img->ID == -1)
    {
        return RETURN_FAILURE;
    }

    double total = 0.0;
    for (uint64_t ii = 0; ii < img->md->nelement; ii++)
    {
        total += img->im->array.F[ii];
    }
    total *= scaling;

    printf("image %s total = %lf (scaling coeff %lf)\n", img->im->name, total, scaling);

    return RETURN_SUCCESS;
}

#endif /* RT_KERNELS_H */
