// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    imgid_arith_helpers.h
 * @brief   Shared IMGID helpers for arithmetic ops
 *
 * Extracted from imfunctions.c to eliminate
 * duplicated output-image setup boilerplate
 * across stamp macros and generic functions.
 */

#ifndef IMGID_ARITH_HELPERS_H
#define IMGID_ARITH_HELPERS_H

#include <libfps/IMGID.h>
#include "COREMOD_memory/imageID.h"

/**
 * @brief Prepare output IMGID for arithmetic
 *
 * If dst has no backing IMAGE yet (`dst->im == NULL`),
 * copies geometry from src into the unresolved output
 * description.  Then allocates or re-creates the
 * output IMAGE.  Finally registers in dcimg
 * if it is a new (unregistered) image.
 *
 * @param src   Source image for metadata copy
 * @param dst   Output image to prepare
 */
static inline void imgid_ensure_output(IMGID *src, IMGID *dst)
{
    if (dst->im == NULL)
    {
        imgid_copy(src, dst);
    }
    if (dst->im == NULL)
    {
        dst->im = (IMAGE *) calloc(1, sizeof(IMAGE));
    }
    else
    {
        if (dst->im->md && dst->im->md->shared == 1)
        {
            ImageStreamIO_closeIm(dst->im);
        }
        else
        {
            ImageStreamIO_destroyIm(dst->im);
        }
    }
    imgid_mkimage(dst);
    if (dst->ID == -1 && dst->im != NULL)
    {
        RegisterIMGID(dst, dcimg, dcnimg);
    }
}

/**
 * @brief Check if two IMGIDs have identical dimensions
 *
 * @param img1 First image
 * @param img2 Second image
 * @return 1 if dimensions match exactly, 0 otherwise
 */
static inline int imgid_same_dims(const IMGID *img1, const IMGID *img2)
{
    if (img1->md->naxis != img2->md->naxis)
    {
        return 0;
    }
    for (uint8_t a = 0; a < img1->md->naxis; a++)
    {
        if (img1->md->size[a] != img2->md->size[a])
        {
            return 0;
        }
    }
    return 1;
}

/**
 * @brief Check if first image is 3D cube and second is 2D matching slice
 *
 * @param cube  Potential 3D cube image
 * @param slice Potential 2D slice image
 * @return 1 if cube is 3D and slice is 2D with matching [x, y], 0 otherwise
 */
static inline int imgid_is_cube_slice(const IMGID *cube, const IMGID *slice)
{
    return (cube->md->naxis == 3 && slice->md->naxis == 2 &&
            cube->md->size[0] == slice->md->size[0] && cube->md->size[1] == slice->md->size[1]);
}

#endif /* IMGID_ARITH_HELPERS_H */
