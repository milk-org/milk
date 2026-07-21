// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    stream_slice.h
 * @brief   Stream slice materialization
 *
 * Provides imgid_get_image() and imgid_put_image()
 * for transparent read/write access to sliced
 * sub-regions of a stream.
 */

#ifndef STREAM_SLICE_H
#define STREAM_SLICE_H

#include "libfps/IMGID.h"

/**
 * @brief Materialize a slice: copy source region
 *        into the local slice buffer.
 *
 * Allocates the slice buffer on first call.
 * Subsequent calls update the existing buffer.
 *
 * @param img  IMGID with slice set and im resolved
 * @return 0 on success
 */
errno_t imgid_slice_materialize(IMGID *img);


/**
 * @brief Write back the slice buffer into the
 *        source stream at the sliced offsets.
 *
 * Inverse of imgid_slice_materialize().
 *
 * @param img  IMGID with materialized slice
 * @return 0 on success
 */
errno_t imgid_slice_writeback(IMGID *img);


/**
 * @brief Get the IMAGE pointer for reading.
 *
 * For non-sliced IMGIDs, returns img->im directly
 * (zero overhead — one predicted-away branch).
 *
 * For sliced IMGIDs, materializes the slice if
 * the source has new data, then returns the
 * materialized slice IMAGE.
 *
 * @param img  IMGID (must be resolved)
 * @return IMAGE pointer for reading
 */
static inline IMAGE *imgid_get_image(IMGID *img)
{
    if (!img->slice.has_slice)
    {
        return img->im;
    }

    /* Check if source has new data */
    uint64_t cnt = img->im->md[0].cnt0;
    if (cnt != img->slice_last_cnt0 || img->slice_im == NULL)
    {
        imgid_slice_materialize(img);
        img->slice_last_cnt0 = cnt;
    }

    return img->slice_im;
}


/**
 * @brief Write back the slice buffer to source.
 *
 * For non-sliced IMGIDs, this is a no-op.
 * For sliced IMGIDs, copies the local buffer
 * back into the source stream at the correct
 * offsets.
 *
 * @param img  IMGID (must be resolved)
 */
static inline void imgid_put_image(IMGID *img)
{
    if (!img->slice.has_slice)
    {
        return;
    }
    imgid_slice_writeback(img);
}


#endif
