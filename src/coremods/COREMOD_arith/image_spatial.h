// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file image_spatial.h
 * @brief Shared spatial, CSR segment-reduction, separable Gaussian, cumsum, and binning kernels.
 */

#ifndef COREMOD_ARITH_IMAGE_SPATIAL_H
#define COREMOD_ARITH_IMAGE_SPATIAL_H

#include <stddef.h>
#include <stdint.h>

#include "ImageStreamIO/ImageStreamIO.h"

#define ARITH_GAUSS_MAX_R 256
#define ARITH_SPATIAL_OMP_MIN 262144

/**
 * struct arith_csr_cache - Cached CSR bin-index tables for `pixunmap` and `rprof`
 */
typedef struct arith_csr_cache
{
    int          valid;
    const IMAGE *map_im;
    uint64_t     map_cnt0;
    uint32_t     ishape[3];
    int          nbins;
    double       cx;
    double       cy;
    int32_t     *row_ptr;
    int32_t     *pix_idx;
    double      *inv_cnt;
    size_t       ptr_cap;
    size_t       idx_cap;
    int32_t     *pix_bin;
    size_t       bin_cap;
    int          bin_valid;
} arith_csr_cache_t;

/**
 * arith_is_real_dt - Return 1 if @dt is a standard real integer or float32/float64 datatype
 * @dt: ImageStreamIO datatype code
 *
 * Return: 1 if real numeric datatype, 0 otherwise.
 */
static inline int arith_is_real_dt(uint8_t dt)
{
    return dt != _DATATYPE_COMPLEX_FLOAT && dt != _DATATYPE_COMPLEX_DOUBLE &&
           dt != _DATATYPE_HALF && ImageStreamIO_typesize(dt) > 0;
}

/* CSR bin-map (`pixunmap`) and radial profile (`rprof`) helpers */
void arith_csr_cache_free(arith_csr_cache_t *csr);

int arith_image_map_nbins(const IMAGE *map_im);

int arith_image_rprof_nbins(const uint32_t ishape[3], double cx, double cy);

int arith_csr_build_map(arith_csr_cache_t *csr, const IMAGE *map_im, int nbins, uint64_t nel);

int arith_csr_build_rprof(arith_csr_cache_t *csr,
                          const uint32_t     ishape[3],
                          double             cx,
                          double             cy,
                          int                nbins);

int arith_csr_ensure_rprof_bins(arith_csr_cache_t *csr,
                                const uint32_t     ishape[3],
                                double             cx,
                                double             cy,
                                int                nbins);

void arith_csr_gather(const IMAGE             *src_im,
                      const arith_csr_cache_t *csr,
                      int                      nbins,
                      uint64_t                 nel,
                      IMAGE                   *res);

/* Separable 2D Gaussian convolution kernels */
void arith_build_gauss_weights(double sigma,
                               int    max_R,
                               int   *out_R,
                               double w[ARITH_GAUSS_MAX_R + 1]);

void arith_gauss_pass_f(float        *io,
                        float        *tmp,
                        int64_t       nx,
                        int64_t       ny,
                        int64_t       nz,
                        int           Rx,
                        int           Ry,
                        const double *wx,
                        const double *wy,
                        int           nth);

void arith_gauss_pass_d(double       *io,
                        double       *tmp,
                        int64_t       nx,
                        int64_t       ny,
                        int64_t       nz,
                        int           Rx,
                        int           Ry,
                        const double *wx,
                        const double *wy,
                        int           nth);

int arith_image_gauss2d_im(const IMAGE *src,
                           IMAGE       *dst,
                           double       sx,
                           double       sy,
                           int          max_R,
                           void       **scratch_buf,
                           size_t      *scratch_cap);

/* Cumulative sum along axis 0, 1, or 2 in-place on float32/float64 image */
int arith_image_cumsum_inplace_im(IMAGE *im, int axis);

/* 2D/3D integer box binning (sum over bx x by x bz blocks) */
int arith_image_bin3d_im(const IMAGE *src, IMAGE *dst, int bx, int by, int bz);

/* 2D/3D integer nearest-neighbor expansion (replicate over bx x by x bz blocks) */
int arith_image_expand3d_im(const IMAGE *src, IMAGE *dst, int bx, int by, int bz);

/* 2D bilinear resize, rotation, radial symmetry extraction, and 1-pixel 3x3 blur */
int arith_image_resize2d_im(const IMAGE *src, IMAGE *dst);

int arith_image_rotate2d_im(const IMAGE *src, IMAGE *dst, double angle);

int arith_image_circsym_im(const IMAGE *src, IMAGE *dst, double cx, double cy, int subtract);

int arith_image_bblur1pix_im(const IMAGE *src, IMAGE *dst, double amp, int niter);

/* 1D linear LUT, 1D index gather (`pixremap`), and 2D bilinear interpolation block kernels */
void arith_lut_block_d(const IMAGE *tbl, double x0, double dx, int len, double *restrict io);

void arith_lut_block_f(const IMAGE *tbl, double x0, double dx, int len, float *restrict io);

void arith_gather_block_d(const IMAGE *im, double fill, int len, double *restrict io);

void arith_gather_block_f(const IMAGE *im, double fill, int len, float *restrict io);

void arith_interp2_block_d(const IMAGE *im,
                           int          len,
                           double *restrict xio,
                           const double *restrict ym);

void arith_interp2_block_f(const IMAGE *im, int len, float *restrict xio, const float *restrict ym);

#endif /* COREMOD_ARITH_IMAGE_SPATIAL_H */
