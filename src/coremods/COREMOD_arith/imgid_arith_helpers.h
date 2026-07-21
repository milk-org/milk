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

#endif /* IMGID_ARITH_HELPERS_H */
