// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_linalg.h
 * @brief   Shared tensor shape and BLAS Level 1-3 kernels on IMAGE structs
 */

#ifndef COREMOD_ARITH_IMAGE_LINALG_H
#define COREMOD_ARITH_IMAGE_LINALG_H

#include <stdint.h>
#include "ImageStreamIO/ImageStruct.h"
#include "libmilkcommon/milk_compiler.h"

/** Size OpenBLAS's thread pool for @work multiply-adds (pass 1 inside outer worker teams). */
void arith_blas_set_threads(uint64_t work);

/** Map a 1-D, 2-D, or 3-D image to 2-D matrix dimensions `(d0, d1)`. */
void arith_image_flatten2d(const IMAGE *im, uint64_t *d0, uint64_t *d1, int *is_3d);

/** Check whether @im is a 1-D vector or a 2-D single-row/column vector. */
int arith_image_is_1d_vec(const IMAGE *im, uint64_t *len);

/** Validate whether `dst = transpose(src)` can execute directly into @dst. */
int arith_image_transpose_check(const IMAGE *src, const IMAGE *dst);

/** Transpose 1-D or 2-D image @src into pre-allocated 2-D image @dst. */
int arith_image_transpose_im(const IMAGE *src, IMAGE *dst);

/** Validate whether `dst = reshape(src, ...)` can write directly into @dst. */
int arith_image_reshape_check(const IMAGE *src, const IMAGE *dst);

/** Copy or convert pixel data from @src into reshaped image @dst. */
int arith_image_reshape_im(const IMAGE *src, IMAGE *dst);

/** Resolve matrix dimensions (`M`, `K`, `N`) and output shape for `matmul`/`gemm`. */
int arith_image_matmul_dims(const IMAGE *im1,
                            int          transA,
                            const IMAGE *im2,
                            int          transB,
                            int         *out_tA,
                            int         *out_tB,
                            uint64_t    *out_M,
                            uint64_t    *out_K,
                            uint64_t    *out_N,
                            long        *out_nax,
                            uint32_t     out_sz[3]);

/** Validate whether `dst = matmul(im1, im2)` can write directly into @dst. */
int arith_image_matmul_check(const IMAGE *im1,
                             int          transA,
                             const IMAGE *im2,
                             int          transB,
                             const IMAGE *dst,
                             int          allow_reshape);

/** Single-precision column-major GEMM `C = alpha * op(A) * op(B) + beta * C`. */
void arith_matmul_kernel_f(const float *restrict A,
                           int tA,
                           const float *restrict B,
                           int   tB,
                           float alpha,
                           float beta,
                           float *restrict C,
                           uint64_t M,
                           uint64_t K,
                           uint64_t N);

/** Double-precision column-major GEMM `C = alpha * op(A) * op(B) + beta * C`. */
void arith_matmul_kernel_d(const double *restrict A,
                           int tA,
                           const double *restrict B,
                           int    tB,
                           double alpha,
                           double beta,
                           double *restrict C,
                           uint64_t M,
                           uint64_t K,
                           uint64_t N);

/** Compute contiguous output pixel block `[off, off + len)` of `op(A) * op(B)` (float). */
void arith_matmul_block_f(const float *restrict A,
                          int tA,
                          const float *restrict B,
                          int tB,
                          float *restrict dst,
                          uint64_t M,
                          uint64_t K,
                          uint64_t N,
                          uint64_t off,
                          uint64_t len);

/** Compute contiguous output pixel block `[off, off + len)` of `op(A) * op(B)` (double). */
void arith_matmul_block_d(const double *restrict A,
                          int tA,
                          const double *restrict B,
                          int tB,
                          double *restrict dst,
                          uint64_t M,
                          uint64_t K,
                          uint64_t N,
                          uint64_t off,
                          uint64_t len);

/** Compute contiguous output pixel block `[off, off + len)` of `op(im1) * op(im2)` as double. */
void arith_image_matmul_block_d(const IMAGE *im1,
                                int          tA,
                                const IMAGE *im2,
                                int          tB,
                                double *restrict dst,
                                uint64_t M,
                                uint64_t K,
                                uint64_t N,
                                uint64_t off,
                                uint64_t len);

/** Compute `outim = alpha * op(im1) * op(im2) + beta * outim` into @outim. */
int arith_image_gemm_im(const IMAGE *im1,
                        int          transA,
                        const IMAGE *im2,
                        int          transB,
                        double       alpha,
                        double       beta,
                        IMAGE       *outim);

/** Compute matrix product `op(im1) * op(im2)` into @outim. */
int arith_image_matmul_im(const IMAGE *im1, int transA, const IMAGE *im2, int transB, IMAGE *outim);

/** Validate whether `ger` / `outer` can write directly into @dst. */
int arith_image_ger_check(const IMAGE *imX, const IMAGE *imY, const IMAGE *dst);

/** Compute rank-1 outer product `outim = alpha * x * y^T + beta * outim`. */
int arith_image_ger_im(const IMAGE *imX, const IMAGE *imY, double alpha, double beta, IMAGE *outim);

/** Validate whether `syrk` can write directly into @dst. */
int arith_image_syrk_check(const IMAGE *imA, int transA, const IMAGE *dst);

/** Compute symmetric rank-k update `outim = alpha * op(A) * op(A)^T + beta * outim`. */
int arith_image_syrk_im(const IMAGE *imA, int transA, double alpha, double beta, IMAGE *outim);

/** Compute symmetric matrix-vector product `imY = alpha * A * x + beta * imY`. */
int arith_image_symv_im(const IMAGE *imA, const IMAGE *imX, double alpha, double beta, IMAGE *imY);

/** Update `imY = alpha * imX + imY` in-place. */
int arith_image_axpy_im(double alpha, const IMAGE *imX, IMAGE *imY);

/** Scale `imX` in-place by @alpha. */
int arith_image_scal_im(double alpha, IMAGE *imX);

#endif // COREMOD_ARITH_IMAGE_LINALG_H
