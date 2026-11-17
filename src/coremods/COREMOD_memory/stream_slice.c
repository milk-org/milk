// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    stream_slice.c
 * @brief   Stream slice materialization and
 *          write-back
 *
 * Implements the data-copy operations for the
 * IMGID slice system. Provides two entry points:
 *
 * - imgid_slice_materialize(): extracts a slice
 *   from the source stream into a local buffer.
 * - imgid_slice_writeback(): copies the local
 *   buffer back into the source stream at the
 *   original slice offsets.
 *
 * The local buffer is a plain malloc'd IMAGE
 * (not shared memory) by default.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#    include "COREMOD_memory/COREMOD_memory.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#    include "ImageStreamIO/ImageStreamIO.h"
#endif

#include "stream_slice.h"


/**
 * @brief Compute byte size of one element.
 *
 * Wraps ImageStreamIO_typesize which returns
 * int (bytes per element) given a datatype code.
 */
static inline int elem_size(uint8_t datatype)
{
    return ImageStreamIO_typesize(datatype);
}


/**
 * @brief Allocate or reallocate the local
 *        slice IMAGE buffer.
 *
 * Creates a local (non-shared) IMAGE with the
 * correct output dimensions. Called on first
 * materialization and when source dimensions
 * change.
 *
 * @param img      Source IMGID (slice populated)
 * @param out_sz   Output axis sizes
 * @param naxis    Output dimensionality
 * @param datatype Source data type
 * @return 0 on success
 */
static errno_t alloc_slice_buffer(IMGID *img, const uint32_t *out_sz, int naxis, uint8_t datatype)
{
    if (img->slice_im != NULL)
    {
        free(img->slice_im->array.raw);
        free(img->slice_im);
        img->slice_im = NULL;
    }

    IMAGE *sim = (IMAGE *) calloc(1, sizeof(IMAGE));
    if (sim == NULL)
    {
        return 1;
    }

    sim->md = (IMAGE_METADATA *) calloc(1, sizeof(IMAGE_METADATA));
    if (sim->md == NULL)
    {
        free(sim);
        return 1;
    }

    sim->md[0].datatype = datatype;
    sim->md[0].naxis    = (uint8_t) naxis;

    uint64_t nelement = 1;
    for (int a = 0; a < naxis; a++)
    {
        sim->md[0].size[a] = out_sz[a];
        nelement *= out_sz[a];
    }
    sim->md[0].nelement = nelement;

    int      esize  = elem_size(datatype);
    uint64_t nbytes = nelement * (uint64_t) esize;

    sim->array.raw = (char *) calloc(1, nbytes);
    if (sim->array.raw == NULL)
    {
        free(sim->md);
        free(sim);
        return 1;
    }

    img->slice_im = sim;
    return 0;
}


/**
 * @brief Copy a 2D slice region using per-row
 *        memcpy. Fast path for contiguous crops
 *        with step=1 and no flipping.
 *
 * @param src      Source pixel base pointer
 * @param dst      Destination pixel base pointer
 * @param esize    Bytes per element
 * @param src_xsz  Source X axis size
 * @param x0       Start X (inclusive)
 * @param x1       End X (inclusive)
 * @param y0       Start Y (inclusive)
 * @param y1       End Y (inclusive)
 */
static void copy_crop_2d(const char *restrict src,
                         char *restrict dst,
                         int      esize,
                         uint32_t src_xsz,
                         int32_t  x0,
                         int32_t  x1,
                         int32_t  y0,
                         int32_t  y1)
{
    int32_t out_xsz   = x1 - x0 + 1;
    int32_t row_bytes = out_xsz * esize;

    for (int32_t y = y0; y <= y1; y++)
    {
        const char *srow = src + ((uint64_t) y * src_xsz + x0) * esize;
        char       *drow = dst + ((uint64_t) (y - y0) * out_xsz) * esize;

        __builtin_memcpy(drow, srow, row_bytes);
    }
}


/**
 * @brief Copy with arbitrary stride and flip.
 *
 * General-purpose element-by-element copy that
 * handles any combination of start, end, step
 * (positive or negative) for up to 3 axes.
 *
 * @param src      Source pixel base pointer
 * @param dst      Destination pixel base pointer
 * @param esize    Bytes per element
 * @param s        Resolved slice descriptor
 * @param src_size Source axis sizes
 * @param out_size Output axis sizes
 * @param naxis    Dimensionality
 */
static void copy_general(const char *restrict src,
                         char *restrict dst,
                         int                esize,
                         const IMGID_SLICE *s,
                         const uint32_t    *src_size,
                         const uint32_t    *out_size,
                         int                naxis)
{
    /* Compute source strides (elements) */
    uint64_t src_stride[3] = { 1, 0, 0 };
    if (naxis >= 2)
    {
        src_stride[1] = src_size[0];
    }
    if (naxis >= 3)
    {
        src_stride[2] = (uint64_t) src_size[0] * src_size[1];
    }

    /* Compute dest strides (elements) */
    uint64_t dst_stride[3] = { 1, 0, 0 };
    if (naxis >= 2)
    {
        dst_stride[1] = out_size[0];
    }
    if (naxis >= 3)
    {
        dst_stride[2] = (uint64_t) out_size[0] * out_size[1];
    }

    /* Number of output elements per axis */
    uint32_t cnt[3] = { 1, 1, 1 };
    for (int a = 0; a < naxis; a++)
    {
        cnt[a] = out_size[a];
    }

    /* Step directions */
    int32_t astep[3] = { 1, 1, 1 };
    for (int a = 0; a < naxis; a++)
    {
        astep[a] = s->step[a];
    }

    /* Starting source indices (imgid_slice_output_size() puts the high index in start when step < 0) */
    int32_t astart[3] = { 0, 0, 0 };
    for (int a = 0; a < naxis; a++)
    {
        astart[a] = s->start[a];
    }

    /* Triple loop over output elements */
    for (uint32_t oz = 0; oz < cnt[2]; oz++)
    {
        int32_t sz = astart[2] + (int32_t) oz * astep[2];

        for (uint32_t oy = 0; oy < cnt[1]; oy++)
        {
            int32_t sy = astart[1] + (int32_t) oy * astep[1];

            if (astep[0] == 1) /* contiguous along x: one memcpy per row */
            {
                uint64_t si = (uint64_t) astart[0] + (uint64_t) sy * src_stride[1] +
                              (uint64_t) sz * src_stride[2];
                uint64_t di = (uint64_t) oy * dst_stride[1] + (uint64_t) oz * dst_stride[2];
                __builtin_memcpy(dst + di * esize, src + si * esize, (size_t) cnt[0] * esize);
                continue;
            }
            for (uint32_t ox = 0; ox < cnt[0]; ox++)
            {
                int32_t sx = astart[0] + (int32_t) ox * astep[0];

                uint64_t si = (uint64_t) sx * src_stride[0] + (uint64_t) sy * src_stride[1] +
                              (uint64_t) sz * src_stride[2];

                uint64_t di = (uint64_t) ox * dst_stride[0] + (uint64_t) oy * dst_stride[1] +
                              (uint64_t) oz * dst_stride[2];

                __builtin_memcpy(dst + di * esize, src + si * esize, esize);
            }
        }
    }
}


/**
 * @brief Write back: general-purpose copy from
 *        slice buffer into source stream.
 *
 * Inverse of copy_general. Iterates over the
 * output dimensions and copies each element
 * back to the correct source location.
 */
static void writeback_general(char *restrict dst_src,
                              const char *restrict src_slice,
                              int                esize,
                              const IMGID_SLICE *s,
                              const uint32_t    *src_size,
                              const uint32_t    *out_size,
                              int                naxis)
{
    uint64_t src_stride[3] = { 1, 0, 0 };
    if (naxis >= 2)
    {
        src_stride[1] = src_size[0];
    }
    if (naxis >= 3)
    {
        src_stride[2] = (uint64_t) src_size[0] * src_size[1];
    }

    uint64_t dst_stride[3] = { 1, 0, 0 };
    if (naxis >= 2)
    {
        dst_stride[1] = out_size[0];
    }
    if (naxis >= 3)
    {
        dst_stride[2] = (uint64_t) out_size[0] * out_size[1];
    }

    uint32_t cnt[3] = { 1, 1, 1 };
    for (int a = 0; a < naxis; a++)
    {
        cnt[a] = out_size[a];
    }

    int32_t astep[3] = { 1, 1, 1 };
    for (int a = 0; a < naxis; a++)
    {
        astep[a] = s->step[a];
    }

    int32_t astart[3] = { 0, 0, 0 };
    for (int a = 0; a < naxis; a++)
    {
        astart[a] = s->start[a];
    }

    for (uint32_t oz = 0; oz < cnt[2]; oz++)
    {
        int32_t sz = astart[2] + (int32_t) oz * astep[2];

        for (uint32_t oy = 0; oy < cnt[1]; oy++)
        {
            int32_t sy = astart[1] + (int32_t) oy * astep[1];

            if (astep[0] == 1) /* contiguous along x: one memcpy per row */
            {
                uint64_t si = (uint64_t) astart[0] + (uint64_t) sy * src_stride[1] +
                              (uint64_t) sz * src_stride[2];
                uint64_t di = (uint64_t) oy * dst_stride[1] + (uint64_t) oz * dst_stride[2];
                __builtin_memcpy(dst_src + si * esize, src_slice + di * esize,
                                 (size_t) cnt[0] * esize);
                continue;
            }
            for (uint32_t ox = 0; ox < cnt[0]; ox++)
            {
                int32_t sx = astart[0] + (int32_t) ox * astep[0];

                uint64_t si = (uint64_t) sx * src_stride[0] + (uint64_t) sy * src_stride[1] +
                              (uint64_t) sz * src_stride[2];

                uint64_t di = (uint64_t) ox * dst_stride[0] + (uint64_t) oy * dst_stride[1] +
                              (uint64_t) oz * dst_stride[2];

                __builtin_memcpy(dst_src + si * esize, src_slice + di * esize, esize);
            }
        }
    }
}


/** Read element @ii of a real-typed buffer as double; callers reject complex types via is_real_datatype(). */
static double pix_get(const char *base, uint8_t dt, uint64_t ii)
{
    switch (dt)
    {
    case _DATATYPE_UINT8:
        return ((const uint8_t *) base)[ii];
    case _DATATYPE_INT8:
        return ((const int8_t *) base)[ii];
    case _DATATYPE_UINT16:
        return ((const uint16_t *) base)[ii];
    case _DATATYPE_INT16:
        return ((const int16_t *) base)[ii];
    case _DATATYPE_UINT32:
        return ((const uint32_t *) base)[ii];
    case _DATATYPE_INT32:
        return ((const int32_t *) base)[ii];
    case _DATATYPE_UINT64:
        return (double) ((const uint64_t *) base)[ii];
    case _DATATYPE_INT64:
        return (double) ((const int64_t *) base)[ii];
    case _DATATYPE_FLOAT:
        return ((const float *) base)[ii];
    case _DATATYPE_DOUBLE:
        return ((const double *) base)[ii];
    default:
        return 0.0;
    }
}

/** Write @v to element @ii of a real-typed buffer. */
static void pix_set(char *base, uint8_t dt, uint64_t ii, double v)
{
    switch (dt)
    {
    case _DATATYPE_UINT8:
        ((uint8_t *) base)[ii] = (uint8_t) v;
        break;
    case _DATATYPE_INT8:
        ((int8_t *) base)[ii] = (int8_t) v;
        break;
    case _DATATYPE_UINT16:
        ((uint16_t *) base)[ii] = (uint16_t) v;
        break;
    case _DATATYPE_INT16:
        ((int16_t *) base)[ii] = (int16_t) v;
        break;
    case _DATATYPE_UINT32:
        ((uint32_t *) base)[ii] = (uint32_t) v;
        break;
    case _DATATYPE_INT32:
        ((int32_t *) base)[ii] = (int32_t) v;
        break;
    case _DATATYPE_UINT64:
        ((uint64_t *) base)[ii] = (uint64_t) v;
        break;
    case _DATATYPE_INT64:
        ((int64_t *) base)[ii] = (int64_t) v;
        break;
    case _DATATYPE_FLOAT:
        ((float *) base)[ii] = (float) v;
        break;
    case _DATATYPE_DOUBLE:
        ((double *) base)[ii] = v;
        break;
    default:
        break;
    }
}

/** True if any axis of @s is in bin (average) mode. */
static int slice_has_bin(const IMGID_SLICE *s, int naxis)
{
    for (int a = 0; a < naxis && a < IMGID_SLICE_MAXAXIS; a++)
    {
        if (s->bin[a])
        {
            return 1;
        }
    }
    return 0;
}

/** True for datatypes pix_get()/pix_set() handle (binning averages need real arithmetic). */
static int is_real_datatype(uint8_t dt)
{
    return dt == _DATATYPE_UINT8 || dt == _DATATYPE_INT8 || dt == _DATATYPE_UINT16 ||
           dt == _DATATYPE_INT16 || dt == _DATATYPE_UINT32 || dt == _DATATYPE_INT32 ||
           dt == _DATATYPE_UINT64 || dt == _DATATYPE_INT64 || dt == _DATATYPE_FLOAT ||
           dt == _DATATYPE_DOUBLE;
}

/**
 * Binned copy (@writeback = 0: each output pixel is the mean of its |step|-wide source block per bin axis) or
 * its inverse (@writeback = 1: each output pixel value is written to every source pixel of its block).
 */
static void copy_binned(char *restrict srcbuf,
                        char *restrict slcbuf,
                        uint8_t            dt,
                        const IMGID_SLICE *s,
                        const uint32_t    *src_size,
                        const uint32_t    *out_size,
                        int                naxis,
                        int                writeback)
{
    uint64_t sstride[3] = { 1, (uint64_t) src_size[0], (uint64_t) src_size[0] * src_size[1] };
    uint32_t cnt[3]     = { 1, 1, 1 };
    int32_t  st[3]      = { 0, 0, 0 };
    int32_t  step[3]    = { 1, 1, 1 };
    int32_t  dir[3]     = { 1, 1, 1 };
    int32_t  nsub[3]    = { 1, 1, 1 };
    for (int a = 0; a < naxis && a < 3; a++)
    {
        cnt[a]  = out_size[a];
        st[a]   = s->start[a];
        step[a] = s->step[a];
        dir[a]  = (s->step[a] < 0) ? -1 : 1;
        nsub[a] = s->bin[a] ? abs(s->step[a]) : 1;
    }
    double   inv = 1.0 / ((double) nsub[0] * nsub[1] * nsub[2]);
    uint64_t di  = 0;

    for (uint32_t oz = 0; oz < cnt[2]; oz++)
    {
        for (uint32_t oy = 0; oy < cnt[1]; oy++)
        {
            for (uint32_t ox = 0; ox < cnt[0]; ox++, di++)
            {
                double v   = writeback ? pix_get(slcbuf, dt, di) : 0.0;
                double sum = 0.0;
                for (int32_t kz = 0; kz < nsub[2]; kz++)
                {
                    int32_t sz = st[2] + (int32_t) oz * step[2] + kz * dir[2];
                    for (int32_t ky = 0; ky < nsub[1]; ky++)
                    {
                        int32_t sy = st[1] + (int32_t) oy * step[1] + ky * dir[1];
                        for (int32_t kx = 0; kx < nsub[0]; kx++)
                        {
                            int32_t  sx = st[0] + (int32_t) ox * step[0] + kx * dir[0];
                            uint64_t si = (uint64_t) sx + (uint64_t) sy * sstride[1] +
                                          (uint64_t) sz * sstride[2];
                            if (writeback)
                            {
                                pix_set(srcbuf, dt, si, v);
                            }
                            else
                            {
                                sum += pix_get(srcbuf, dt, si);
                            }
                        }
                    }
                }
                if (!writeback)
                {
                    pix_set(slcbuf, dt, di, sum * inv);
                }
            }
        }
    }
}


/**
 * @brief Check if a slice is a simple 2D crop
 *        (step=1, no flip, no binning).
 */
static int is_simple_crop_2d(const IMGID_SLICE *s, int naxis)
{
    if (naxis < 1 || naxis > 2)
    {
        return 0;
    }
    for (int a = 0; a < naxis; a++)
    {
        if (s->step[a] != 1 || s->bin[a])
        {
            return 0;
        }
    }
    return 1;
}


/**
 * @brief Materialize a slice from source into
 *        a local buffer.
 *
 * On first call, allocates the local IMAGE.
 * On subsequent calls, reuses the buffer
 * (reallocates only if source dimensions
 * changed).
 *
 * The copy uses a fast memcpy-per-row path
 * for simple 2D crops and falls back to a
 * general element-by-element copy for
 * strided/flipped/3D slices.
 *
 * @param img  IMGID with slice and im set
 * @return 0 on success
 */
errno_t imgid_slice_materialize(IMGID *img)
{
    if (!img->slice.has_slice)
    {
        return 0;
    }
    if (img->im == NULL)
    {
        fprintf(stderr, "ERROR: imgid_slice_materialize: "
                        "source IMAGE not resolved\n");
        return 1;
    }

    int     naxis     = (int) img->im->md[0].naxis;
    uint8_t datatype  = img->im->md[0].datatype;
    int     esize_val = elem_size(datatype);

    /* Get source dimensions */
    uint32_t src_size[3] = { 1, 1, 1 };
    for (int a = 0; a < naxis; a++)
    {
        src_size[a] = img->im->md[0].size[a];
    }

    /* Make a working copy of slice descriptor
     * (resolve negative indices) */
    IMGID_SLICE rs = img->slice;

    /* Compute output dimensions */
    uint32_t out_size[3] = { 1, 1, 1 };
    if (imgid_slice_output_size(&rs, naxis, src_size, out_size) != 0)
    {
        fprintf(stderr, "ERROR: slice output size: %s\n", rs.errmsg);
        return 1;
    }

    /* Allocate or check buffer */
    if (img->slice_im == NULL)
    {
        if (alloc_slice_buffer(img, out_size, naxis, datatype) != 0)
        {
            fprintf(stderr, "ERROR: slice buffer alloc\n");
            return 1;
        }
    }

    /* Copy data */
    const char *src = img->im->array.raw;
    char       *dst = img->slice_im->array.raw;

    if (slice_has_bin(&rs, naxis))
    {
        if (!is_real_datatype(datatype))
        {
            fprintf(stderr, "ERROR: slice binning needs a real datatype\n");
            return 1;
        }
        copy_binned((char *) src, dst, datatype, &rs, src_size, out_size, naxis, 0);
    }
    else if (is_simple_crop_2d(&rs, naxis))
    {
        /* Fast path: memcpy per row */
        int32_t y0 = (naxis >= 2) ? rs.start[1] : 0;
        int32_t y1 = (naxis >= 2) ? rs.end[1] : 0;

        copy_crop_2d(src, dst, esize_val, src_size[0], rs.start[0], rs.end[0], y0, y1);
    }
    else
    {
        /* General path */
        copy_general(src, dst, esize_val, &rs, src_size, out_size, naxis);
    }

    return 0;
}


/**
 * @brief Write the slice buffer back into the
 *        source stream at the sliced offsets.
 *
 * Inverse of imgid_slice_materialize().
 * Uses the general write-back path for all
 * slice types.
 *
 * @param img  IMGID with materialized slice
 * @return 0 on success
 */
errno_t imgid_slice_writeback(IMGID *img)
{
    if (!img->slice.has_slice)
    {
        return 0;
    }
    if (img->im == NULL || img->slice_im == NULL)
    {
        fprintf(stderr, "ERROR: imgid_slice_writeback: "
                        "source or slice not ready\n");
        return 1;
    }

    int     naxis     = (int) img->im->md[0].naxis;
    uint8_t datatype  = img->im->md[0].datatype;
    int     esize_val = elem_size(datatype);

    uint32_t src_size[3] = { 1, 1, 1 };
    for (int a = 0; a < naxis; a++)
    {
        src_size[a] = img->im->md[0].size[a];
    }

    IMGID_SLICE rs          = img->slice;
    uint32_t    out_size[3] = { 1, 1, 1 };
    if (imgid_slice_output_size(&rs, naxis, src_size, out_size) != 0)
    {
        return 1;
    }

    char       *dst_src   = img->im->array.raw;
    const char *src_slice = img->slice_im->array.raw;

    if (slice_has_bin(&rs, naxis))
    {
        if (!is_real_datatype(datatype))
        {
            fprintf(stderr, "ERROR: slice binning needs a real datatype\n");
            return 1;
        }
        copy_binned(dst_src, (char *) src_slice, datatype, &rs, src_size, out_size, naxis, 1);
    }
    else
    {
        writeback_general(dst_src, src_slice, esize_val, &rs, src_size, out_size, naxis);
    }

    return 0;
}
