// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_linalg.c
 * @brief   Shared tensor shape and BLAS Level 1-3 kernels on IMAGE structs
 */

// MILK_CMAKE_REQUEST_BLAS

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#endif

#include "image_linalg.h"
#include "image_stats.h"
#include "libmilkcommon/pixel_dispatch.h"
#include "milk_blas_lapacke.h"

static inline int is_complex_dt(uint8_t dt)
{
    return (dt == _DATATYPE_COMPLEX_FLOAT || dt == _DATATYPE_COMPLEX_DOUBLE);
}

/**
 * arith_image_flatten2d - Map a 1-D, 2-D, or 3-D image to 2-D matrix dimensions `(d0, d1)`
 * @im:    Input image
 * @d0:    Receives effective row count (`size[0]`, or `size[0]*size[1]` for 3-D cubes)
 * @d1:    Receives effective column count (`1` for 1-D, `size[1]` for 2-D, `size[2]` for 3-D)
 * @is_3d: Receives 1 when @im is 3-D, 0 otherwise
 */
void arith_image_flatten2d(const IMAGE *im, uint64_t *d0, uint64_t *d1, int *is_3d)
{
    *is_3d = 0;
    *d0    = im->md[0].size[0];
    *d1    = 1;
    if (im->md[0].naxis == 3)
    {
        *d0    = (uint64_t) im->md[0].size[0] * im->md[0].size[1];
        *d1    = im->md[0].size[2];
        *is_3d = 1;
    }
    else if (im->md[0].naxis >= 2)
    {
        *d1 = im->md[0].size[1];
    }
}

/**
 * arith_image_is_1d_vec - Check whether @im is a 1-D vector or a 2-D single-row/column vector
 * @im:  Image to inspect
 * @len: Output vector length (`nelement`)
 *
 * Return: 1 if @im is a 1-D or `N x 1` / `1 x N` real image, 0 otherwise.
 */
int arith_image_is_1d_vec(const IMAGE *im, uint64_t *len)
{
    if (im == NULL || !im->used || is_complex_dt(im->md[0].datatype))
    {
        return 0;
    }
    if (im->md[0].naxis == 1 ||
        (im->md[0].naxis == 2 && (im->md[0].size[0] == 1 || im->md[0].size[1] == 1)))
    {
        *len = im->md[0].nelement;
        return 1;
    }
    return 0;
}

/**
 * arith_image_transpose_check - Validate whether `dst = transpose(src)` can execute into @dst
 * @src: Source 1-D or 2-D real image
 * @dst: Destination 2-D float/double image
 *
 * Return: 0 if @dst is compatible for direct transpose output, -1 otherwise.
 */
int arith_image_transpose_check(const IMAGE *src, const IMAGE *dst)
{
    if (src == NULL || dst == NULL || !src->used || !dst->used || src == dst)
    {
        return -1;
    }
    if (is_complex_dt(src->md[0].datatype) || (src->md[0].naxis != 1 && src->md[0].naxis != 2))
    {
        return -1;
    }
    if (dst->md[0].datatype != _DATATYPE_FLOAT && dst->md[0].datatype != _DATATYPE_DOUBLE)
    {
        return -1;
    }
    uint64_t M = src->md[0].size[0];
    uint64_t N = (src->md[0].naxis == 2) ? src->md[0].size[1] : 1;
    if (dst->md[0].naxis != 2 || dst->md[0].size[0] != N || dst->md[0].size[1] != M)
    {
        return -1;
    }
    return 0;
}

/**
 * arith_image_transpose_im - Transpose 1-D or 2-D image @src into pre-allocated @dst
 * @src: Source image (`M x N` or 1-D `M`)
 * @dst: Destination 2-D image (`N x M`, float or double)
 *
 * Uses cache-blocked 32x32 tiling (with OpenMP for large matrices) on matching
 * float/double buffers, falling back to generic type conversion otherwise.
 *
 * Return: 0 on success.
 */
int arith_image_transpose_im(const IMAGE *src, IMAGE *dst)
{
    uint64_t M       = src->md[0].size[0];
    uint64_t N       = (src->md[0].naxis == 2) ? src->md[0].size[1] : 1;
    uint8_t  src_dt  = src->md[0].datatype;
    uint8_t  dst_dt  = dst->md[0].datatype;
    uint64_t nel     = M * N;
    uint64_t tile_sz = 32;

#define TRANSPOSE_TYPED(T, S_ARR, D_ARR)                                                           \
    do                                                                                             \
    {                                                                                              \
        const T *restrict s = (S_ARR);                                                             \
        T *restrict d       = (D_ARR);                                                             \
        if (M == 1 || N == 1)                                                                      \
        {                                                                                          \
            __builtin_memcpy(d, s, sizeof(T) * nel);                                               \
            return 0;                                                                              \
        }                                                                                          \
        _Pragma(                                                                                   \
            "omp parallel for collapse(2) schedule(static) if (nel >= 65536)") for (uint64_t m0 =  \
                                                                                        0;         \
                                                                                    m0 < M;        \
                                                                                    m0 += tile_sz) \
        {                                                                                          \
            for (uint64_t n0 = 0; n0 < N; n0 += tile_sz)                                           \
            {                                                                                      \
                uint64_t mend = (m0 + tile_sz < M) ? m0 + tile_sz : M;                             \
                uint64_t nend = (n0 + tile_sz < N) ? n0 + tile_sz : N;                             \
                for (uint64_t n = n0; n < nend; n++)                                               \
                {                                                                                  \
                    const T *restrict scol = &s[n * M];                                            \
                    for (uint64_t m = m0; m < mend; m++)                                           \
                    {                                                                              \
                        d[m * N + n] = scol[m];                                                    \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        return 0;                                                                                  \
    } while (0)

    if (src_dt == _DATATYPE_FLOAT && dst_dt == _DATATYPE_FLOAT)
    {
        TRANSPOSE_TYPED(float, src->array.F, dst->array.F);
    }
    if (src_dt == _DATATYPE_DOUBLE && dst_dt == _DATATYPE_DOUBLE)
    {
        TRANSPOSE_TYPED(double, src->array.D, dst->array.D);
    }
#undef TRANSPOSE_TYPED

    for (uint64_t m = 0; m < M; m++)
    {
        for (uint64_t n = 0; n < N; n++)
        {
            double v = arith_read_im_pix(src, src_dt, n * M + m);
            if (dst_dt == _DATATYPE_DOUBLE)
            {
                dst->array.D[m * N + n] = v;
            }
            else
            {
                dst->array.F[m * N + n] = (float) v;
            }
        }
    }
    return 0;
}

/**
 * arith_image_reshape_check - Validate whether `dst = reshape(src, ...)` can write into @dst
 * @src: Source image
 * @dst: Destination image
 *
 * Return: 0 if compatible, -1 otherwise.
 */
int arith_image_reshape_check(const IMAGE *src, const IMAGE *dst)
{
    if (src == NULL || dst == NULL || !src->used || !dst->used || src == dst ||
        src->md[0].nelement != dst->md[0].nelement)
    {
        return -1;
    }
    if (src->md[0].datatype != dst->md[0].datatype &&
        (is_complex_dt(src->md[0].datatype) || is_complex_dt(dst->md[0].datatype)))
    {
        return -1;
    }
    return 0;
}

/**
 * arith_image_reshape_im - Copy or convert pixel data from @src into reshaped image @dst
 * @src: Source image
 * @dst: Destination image with identical `nelement`
 *
 * Return: 0 on success, -1 on error.
 */
int arith_image_reshape_im(const IMAGE *src, IMAGE *dst)
{
    uint8_t src_dt = src->md[0].datatype;
    uint8_t dst_dt = dst->md[0].datatype;
    if (src_dt == dst_dt)
    {
        int tsz = ImageStreamIO_typesize(src_dt);
        if (tsz <= 0)
        {
            return -1;
        }
        __builtin_memcpy(dst->array.raw, src->array.raw,
                         (size_t) src->md[0].nelement * (size_t) tsz);
        return 0;
    }
    if (is_complex_dt(src_dt) || is_complex_dt(dst_dt))
    {
        return -1;
    }

    uint64_t nel = src->md[0].nelement;
#define _RESHAPE_DST_CASE(DT, ACC, CTYPE)                                    \
    case DT:                                                                 \
        for (uint64_t ii = 0; ii < nel; ii++)                                \
        {                                                                    \
            dst->array.ACC[ii] = (CTYPE) arith_read_im_pix(src, src_dt, ii); \
        }                                                                    \
        return 0;

    switch (dst_dt)
    {
        FOREACH_REAL_DATATYPE(_RESHAPE_DST_CASE)
    default:
        return -1;
    }
#undef _RESHAPE_DST_CASE
}

/**
 * arith_image_matmul_dims - Resolve matrix dimensions (`M`, `K`, `N`) and output shape for `matmul`
 * @im1:     Left operand image (1-D, 2-D, or 3-D cube flattened along first two axes)
 * @transA:  Non-zero if @im1 is explicitly transposed
 * @im2:     Right operand image
 * @transB:  Non-zero if @im2 is explicitly transposed
 * @out_tA:  Receives effective transpose flag for A
 * @out_tB:  Receives effective transpose flag for B
 * @out_M:   Receives output row count M
 * @out_K:   Receives inner contraction dimension K
 * @out_N:   Receives output column count N
 * @out_nax: Receives output `naxis` (1, 2, or 3)
 * @out_sz:  Receives output `size[3]`
 *
 * Return: 0 if dimensions are compatible, -1 otherwise.
 */
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
                            uint32_t     out_sz[3])
{
    uint64_t a0      = 0;
    uint64_t a1      = 0;
    int      is_3d_a = 0;
    arith_image_flatten2d(im1, &a0, &a1, &is_3d_a);

    uint64_t b0      = 0;
    uint64_t b1      = 0;
    int      is_3d_b = 0;
    arith_image_flatten2d(im2, &b0, &b1, &is_3d_b);
    if (is_3d_b && transB)
    {
        return -1;
    }
    if (is_3d_a && !transB && im2->md[0].naxis == 2 &&
        a0 == (uint64_t) im2->md[0].size[0] * im2->md[0].size[1])
    {
        b0 = a0;
        b1 = 1;
    }

    int      tA = 0;
    int      tB = 0;
    uint64_t M  = 0;
    uint64_t K  = 0;
    uint64_t N  = 0;
    if (!transA && !transB)
    {
        if (a1 == b0)
        {
            tA = 0;
            tB = 0;
            M  = a0;
            K  = a1;
            N  = b1;
        }
        else if (a0 == b0)
        {
            tA = 1;
            tB = 0;
            M  = a1;
            K  = a0;
            N  = b1;
        }
        else
        {
            return -1;
        }
    }
    else
    {
        uint64_t eff_a0 = transA ? a1 : a0;
        uint64_t eff_a1 = transA ? a0 : a1;
        uint64_t eff_b0 = transB ? b1 : b0;
        uint64_t eff_b1 = transB ? b0 : b1;
        if (eff_a1 != eff_b0)
        {
            return -1;
        }
        tA = transA;
        tB = transB;
        M  = eff_a0;
        K  = eff_a1;
        N  = eff_b1;
    }

    out_sz[0] = 1;
    out_sz[1] = 1;
    out_sz[2] = 1;
    if (tA == 0 && tB == 0 && is_3d_a)
    {
        out_sz[0] = im1->md[0].size[0];
        out_sz[1] = im1->md[0].size[1];
        *out_nax  = (N == 1) ? 2 : 3;
        out_sz[2] = (uint32_t) N;
    }
    else if (N == 1 && ((im2->md[0].naxis == 1 && !transB) || (tA == 1 && tB == 0)))
    {
        *out_nax  = 1;
        out_sz[0] = (uint32_t) M;
    }
    else
    {
        *out_nax  = 2;
        out_sz[0] = (uint32_t) M;
        out_sz[1] = (uint32_t) N;
    }
    *out_tA = tA;
    *out_tB = tB;
    *out_M  = M;
    *out_K  = K;
    *out_N  = N;
    return 0;
}

/**
 * arith_image_matmul_check - Validate whether `dst = matmul(im1, im2)` can write into @dst
 * @im1:           Left operand image
 * @transA:        Non-zero if @im1 is transposed
 * @im2:           Right operand image
 * @transB:        Non-zero if @im2 is transposed
 * @dst:           Destination image
 * @allow_reshape: Non-zero to accept any @dst shape with matching `M * N` element count
 *
 * Return: 0 if @dst can receive the product directly, -1 otherwise.
 */
int arith_image_matmul_check(const IMAGE *im1,
                             int          transA,
                             const IMAGE *im2,
                             int          transB,
                             const IMAGE *dst,
                             int          allow_reshape)
{
    if (im1 == NULL || im2 == NULL || dst == NULL || !im1->used || !im2->used || !dst->used ||
        dst == im1 || dst == im2)
    {
        return -1;
    }
    if (is_complex_dt(im1->md[0].datatype) || is_complex_dt(im2->md[0].datatype))
    {
        return -1;
    }
    if (dst->md[0].datatype != _DATATYPE_FLOAT && dst->md[0].datatype != _DATATYPE_DOUBLE)
    {
        return -1;
    }
    int      tA       = 0;
    int      tB       = 0;
    uint64_t M        = 0;
    uint64_t K        = 0;
    uint64_t N        = 0;
    long     onax     = 0;
    uint32_t outsz[3] = { 1, 1, 1 };
    if (arith_image_matmul_dims(im1, transA, im2, transB, &tA, &tB, &M, &K, &N, &onax, outsz) != 0)
    {
        return -1;
    }
    if (dst->md[0].nelement != M * N)
    {
        return -1;
    }
    if (allow_reshape)
    {
        return 0;
    }
    if (N == 1)
    {
        if (dst->md[0].size[0] != outsz[0])
        {
            return -1;
        }
    }
    else if (dst->md[0].naxis != onax || dst->md[0].size[0] != outsz[0] ||
             dst->md[0].size[1] != outsz[1] || (onax == 3 && dst->md[0].size[2] != outsz[2]))
    {
        return -1;
    }
    return 0;
}

#ifdef HAVE_BLAS
#    define MATMUL_KERNEL_BODY(T, GEMV, GEMM, MBLK)                                                \
        if (N == 1 && !tB)                                                                         \
        {                                                                                          \
            GEMV(CblasColMajor, tA ? CblasTrans : CblasNoTrans, (int) (tA ? K : M),                \
                 (int) (tA ? M : K), alpha, A, (int) (tA ? K : M), B, 1, beta, C, 1);              \
            return;                                                                                \
        }                                                                                          \
        GEMM(CblasColMajor, tA ? CblasTrans : CblasNoTrans, tB ? CblasTrans : CblasNoTrans,        \
             (int) M, (int) N, (int) K, alpha, A, (int) (tA ? K : M), B, (int) (tB ? N : K), beta, \
             C, (int) M);
#else
#    define MATMUL_OMP_PAR \
        _Pragma("omp parallel for collapse(2) schedule(static) if(M*N*K>=262144)")
#    define MATMUL_KERNEL_BODY(T, GEMV, GEMM, MBLK)                                            \
        if (!tA)                                                                               \
        {                                                                                      \
            const uint64_t mblk  = (MBLK);                                                     \
            uint64_t       nmblk = (M + mblk - 1) / mblk;                                      \
            MATMUL_OMP_PAR                                                                     \
            for (uint64_t n = 0; n < N; n++)                                                   \
            {                                                                                  \
                for (uint64_t mb = 0; mb < nmblk; mb++)                                        \
                {                                                                              \
                    uint64_t m0      = mb * mblk;                                              \
                    uint64_t mend    = (m0 + mblk < M) ? m0 + mblk : M;                        \
                    uint64_t mlen    = mend - m0;                                              \
                    T *restrict cblk = &C[n * M + m0];                                         \
                    if (beta == (T) 0)                                                         \
                    {                                                                          \
                        memset(cblk, 0, sizeof(T) * mlen);                                     \
                    }                                                                          \
                    else if (beta != (T) 1)                                                    \
                    {                                                                          \
                        _Pragma("omp simd") for (uint64_t m = 0; m < mlen; m++)                \
                        {                                                                      \
                            cblk[m] *= beta;                                                   \
                        }                                                                      \
                    }                                                                          \
                    for (uint64_t k = 0; k < K; k++)                                           \
                    {                                                                          \
                        T b_kn                 = alpha * (tB ? B[k * N + n] : B[n * K + k]);   \
                        const T *restrict colA = &A[k * M + m0];                               \
                        _Pragma("omp simd") for (uint64_t m = 0; m < mlen; m++)                \
                        {                                                                      \
                            cblk[m] += colA[m] * b_kn;                                         \
                        }                                                                      \
                    }                                                                          \
                }                                                                              \
            }                                                                                  \
        }                                                                                      \
        else if (!tB)                                                                          \
        {                                                                                      \
            MATMUL_OMP_PAR                                                                     \
            for (uint64_t n = 0; n < N; n++)                                                   \
            {                                                                                  \
                for (uint64_t m = 0; m < M; m++)                                               \
                {                                                                              \
                    const T *restrict rowA = &A[m * K];                                        \
                    const T *restrict colB = &B[n * K];                                        \
                    T acc                  = (T) 0;                                            \
                    _Pragma("omp simd reduction(+ : acc)") for (uint64_t k = 0; k < K; k++)    \
                    {                                                                          \
                        acc += rowA[k] * colB[k];                                              \
                    }                                                                          \
                    C[n * M + m] =                                                             \
                        (beta == (T) 0) ? (alpha * acc) : (alpha * acc + beta * C[n * M + m]); \
                }                                                                              \
            }                                                                                  \
        }                                                                                      \
        else                                                                                   \
        {                                                                                      \
            MATMUL_OMP_PAR                                                                     \
            for (uint64_t n = 0; n < N; n++)                                                   \
            {                                                                                  \
                for (uint64_t m = 0; m < M; m++)                                               \
                {                                                                              \
                    const T *restrict rowA = &A[m * K];                                        \
                    T acc                  = (T) 0;                                            \
                    for (uint64_t k = 0; k < K; k++)                                           \
                    {                                                                          \
                        acc += rowA[k] * B[k * N + n];                                         \
                    }                                                                          \
                    C[n * M + m] =                                                             \
                        (beta == (T) 0) ? (alpha * acc) : (alpha * acc + beta * C[n * M + m]); \
                }                                                                              \
            }                                                                                  \
        }
#endif

/**
 * arith_matmul_kernel_f - Single-precision column-major GEMM `C = alpha * op(A) * op(B) + beta * C`
 */
void arith_matmul_kernel_f(const float *restrict A,
                           int tA,
                           const float *restrict B,
                           int   tB,
                           float alpha,
                           float beta,
                           float *restrict C,
                           uint64_t M,
                           uint64_t K,
                           uint64_t N)
{
    MATMUL_KERNEL_BODY(float, cblas_sgemv, cblas_sgemm, 128)
}

/**
 * arith_matmul_kernel_d - Double-precision column-major GEMM `C = alpha * op(A) * op(B) + beta * C`
 */
void arith_matmul_kernel_d(const double *restrict A,
                           int tA,
                           const double *restrict B,
                           int    tB,
                           double alpha,
                           double beta,
                           double *restrict C,
                           uint64_t M,
                           uint64_t K,
                           uint64_t N)
{
    MATMUL_KERNEL_BODY(double, cblas_dgemv, cblas_dgemm, 64)
}
#undef MATMUL_KERNEL_BODY

/**
 * arith_image_gemm_im - Compute `outim = alpha * op(im1) * op(im2) + beta * outim`
 * @im1:    Left operand image
 * @transA: Non-zero if @im1 is transposed
 * @im2:    Right operand image
 * @transB: Non-zero if @im2 is transposed
 * @alpha:  Scalar multiplier on `op(im1) * op(im2)`
 * @beta:   Scalar multiplier on @outim (`0.0` overwrites @outim)
 * @outim:  Pre-allocated destination image (float or double)
 *
 * Return: 0 on success, -1 on dimension or allocation failure.
 */
int arith_image_gemm_im(const IMAGE *im1,
                        int          transA,
                        const IMAGE *im2,
                        int          transB,
                        double       alpha,
                        double       beta,
                        IMAGE       *outim)
{
    int      tA       = 0;
    int      tB       = 0;
    uint64_t M        = 0;
    uint64_t K        = 0;
    uint64_t N        = 0;
    long     onax     = 0;
    uint32_t outsz[3] = { 1, 1, 1 };
    if (arith_image_matmul_dims(im1, transA, im2, transB, &tA, &tB, &M, &K, &N, &onax, outsz) != 0)
    {
        return -1;
    }

    uint8_t dt1    = im1->md[0].datatype;
    uint8_t dt2    = im2->md[0].datatype;
    uint8_t out_dt = outim->md[0].datatype;

    if (dt1 == _DATATYPE_FLOAT && dt2 == _DATATYPE_FLOAT && out_dt == _DATATYPE_FLOAT)
    {
        arith_matmul_kernel_f(im1->array.F, tA, im2->array.F, tB, (float) alpha, (float) beta,
                              outim->array.F, M, K, N);
        return 0;
    }
    if (dt1 == _DATATYPE_DOUBLE && dt2 == _DATATYPE_DOUBLE && out_dt == _DATATYPE_DOUBLE)
    {
        arith_matmul_kernel_d(im1->array.D, tA, im2->array.D, tB, alpha, beta, outim->array.D, M, K,
                              N);
        return 0;
    }

    double *bufA = (double *) malloc(sizeof(double) * im1->md[0].nelement);
    double *bufB = (double *) malloc(sizeof(double) * im2->md[0].nelement);
    if (bufA == NULL || bufB == NULL)
    {
        free(bufA);
        free(bufB);
        return -1;
    }
    for (uint64_t i = 0; i < im1->md[0].nelement; i++)
    {
        bufA[i] = arith_read_im_pix(im1, dt1, i);
    }
    for (uint64_t i = 0; i < im2->md[0].nelement; i++)
    {
        bufB[i] = arith_read_im_pix(im2, dt2, i);
    }

    if (out_dt == _DATATYPE_DOUBLE)
    {
        arith_matmul_kernel_d(bufA, tA, bufB, tB, alpha, beta, outim->array.D, M, K, N);
    }
    else
    {
        double *bufC = (double *) malloc(sizeof(double) * M * N);
        if (bufC == NULL)
        {
            free(bufA);
            free(bufB);
            return -1;
        }
        if (beta != 0.0)
        {
            for (uint64_t i = 0; i < M * N; i++)
            {
                bufC[i] = (double) outim->array.F[i];
            }
        }
        arith_matmul_kernel_d(bufA, tA, bufB, tB, alpha, beta, bufC, M, K, N);
        for (uint64_t i = 0; i < M * N; i++)
        {
            outim->array.F[i] = (float) bufC[i];
        }
        free(bufC);
    }

    free(bufA);
    free(bufB);
    return 0;
}

/**
 * arith_image_matmul_im - Compute matrix product `op(im1) * op(im2)` into @outim
 */
int arith_image_matmul_im(const IMAGE *im1, int transA, const IMAGE *im2, int transB, IMAGE *outim)
{
    return arith_image_gemm_im(im1, transA, im2, transB, 1.0, 0.0, outim);
}

/**
 * arith_image_ger_check - Validate whether `ger` / `outer` can write directly into @dst
 * @imX: First vector (`M`)
 * @imY: Second vector (`N`)
 * @dst: Destination `M x N` matrix
 *
 * Return: 0 if compatible, -1 otherwise.
 */
int arith_image_ger_check(const IMAGE *imX, const IMAGE *imY, const IMAGE *dst)
{
    uint64_t M = 0;
    uint64_t N = 0;
    if (!arith_image_is_1d_vec(imX, &M) || !arith_image_is_1d_vec(imY, &N) || dst == NULL ||
        !dst->used || dst == imX || dst == imY)
    {
        return -1;
    }
    if (dst->md[0].datatype != _DATATYPE_FLOAT && dst->md[0].datatype != _DATATYPE_DOUBLE)
    {
        return -1;
    }
    if (dst->md[0].naxis != 2 || dst->md[0].size[0] != M || dst->md[0].size[1] != N)
    {
        return -1;
    }
    return 0;
}

/**
 * arith_image_ger_im - Compute rank-1 outer product `outim = alpha * x * y^T + beta * outim`
 * @imX:   Column vector `x` of length `M`
 * @imY:   Row vector `y` of length `N`
 * @alpha: Scalar multiplier on `x * y^T`
 * @beta:  Scalar multiplier on @outim (`0.0` overwrites @outim)
 * @outim: Destination `M x N` image (float or double)
 *
 * Return: 0 on success, -1 on dimension error.
 */
int arith_image_ger_im(const IMAGE *imX, const IMAGE *imY, double alpha, double beta, IMAGE *outim)
{
    uint64_t M = 0;
    uint64_t N = 0;
    if (!arith_image_is_1d_vec(imX, &M) || !arith_image_is_1d_vec(imY, &N))
    {
        return -1;
    }
    uint8_t dtx    = imX->md[0].datatype;
    uint8_t dty    = imY->md[0].datatype;
    uint8_t out_dt = outim->md[0].datatype;

    if (dtx == _DATATYPE_FLOAT && dty == _DATATYPE_FLOAT && out_dt == _DATATYPE_FLOAT)
    {
        const float *restrict X = imX->array.F;
        const float *restrict Y = imY->array.F;
        float *restrict C       = outim->array.F;
        float af                = (float) alpha;
        float bf                = (float) beta;
#ifdef HAVE_BLAS
        if (bf == 0.0f)
        {
            memset(C, 0, sizeof(float) * M * N);
        }
        else if (bf != 1.0f)
        {
            for (uint64_t i = 0; i < M * N; i++)
            {
                C[i] *= bf;
            }
        }
        cblas_sger(CblasColMajor, (int) M, (int) N, af, X, 1, Y, 1, C, (int) M);
#else
#    pragma omp parallel for schedule(static) if (M * N >= 65536)
        for (uint64_t j = 0; j < N; j++)
        {
            float yj             = af * Y[j];
            float *restrict colC = &C[j * M];
            if (bf == 0.0f)
            {
#    pragma omp simd
                for (uint64_t i = 0; i < M; i++)
                {
                    colC[i] = X[i] * yj;
                }
            }
            else
            {
#    pragma omp simd
                for (uint64_t i = 0; i < M; i++)
                {
                    colC[i] = bf * colC[i] + X[i] * yj;
                }
            }
        }
#endif
        return 0;
    }

#ifdef HAVE_BLAS
    if (dtx == _DATATYPE_DOUBLE && dty == _DATATYPE_DOUBLE && out_dt == _DATATYPE_DOUBLE)
    {
        const double *restrict X = imX->array.D;
        const double *restrict Y = imY->array.D;
        double *restrict C       = outim->array.D;
        if (beta == 0.0)
        {
            memset(C, 0, sizeof(double) * M * N);
        }
        else if (beta != 1.0)
        {
            for (uint64_t i = 0; i < M * N; i++)
            {
                C[i] *= beta;
            }
        }
        cblas_dger(CblasColMajor, (int) M, (int) N, alpha, X, 1, Y, 1, C, (int) M);
        return 0;
    }
#endif

    for (uint64_t j = 0; j < N; j++)
    {
        double yj = alpha * arith_read_im_pix(imY, dty, j);
        for (uint64_t i = 0; i < M; i++)
        {
            double xi = arith_read_im_pix(imX, dtx, i);
            if (out_dt == _DATATYPE_DOUBLE)
            {
                double prev = (beta == 0.0) ? 0.0 : (beta * outim->array.D[j * M + i]);
                outim->array.D[j * M + i] = prev + xi * yj;
            }
            else
            {
                double prev = (beta == 0.0) ? 0.0 : (beta * (double) outim->array.F[j * M + i]);
                outim->array.F[j * M + i] = (float) (prev + xi * yj);
            }
        }
    }
    return 0;
}

/**
 * arith_image_syrk_check - Validate whether `syrk` can write directly into @dst
 * @imA:    Input matrix (`N x K`, or `K x N` when @transA is set)
 * @transA: Non-zero for `A^T * A`, 0 for `A * A^T`
 * @dst:    Destination `N x N` symmetric matrix
 *
 * Return: 0 if compatible, -1 otherwise.
 */
int arith_image_syrk_check(const IMAGE *imA, int transA, const IMAGE *dst)
{
    if (imA == NULL || dst == NULL || !imA->used || !dst->used || dst == imA ||
        is_complex_dt(imA->md[0].datatype))
    {
        return -1;
    }
    if (dst->md[0].datatype != _DATATYPE_FLOAT && dst->md[0].datatype != _DATATYPE_DOUBLE)
    {
        return -1;
    }
    uint64_t a0    = 0;
    uint64_t a1    = 0;
    int      is_3d = 0;
    arith_image_flatten2d(imA, &a0, &a1, &is_3d);
    uint64_t N = transA ? a1 : a0;
    if (dst->md[0].naxis != 2 || dst->md[0].size[0] != N || dst->md[0].size[1] != N)
    {
        return -1;
    }
    return 0;
}

/**
 * arith_image_syrk_im - Compute symmetric rank-k update `alpha * op(A) * op(A)^T + beta * outim`
 * @imA:    Input matrix image
 * @transA: Non-zero for `A^T * A`, 0 for `A * A^T`
 * @alpha:  Scalar multiplier on `op(A) * op(A)^T`
 * @beta:   Scalar multiplier on @outim (`0.0` overwrites @outim)
 * @outim:  Destination `N x N` symmetric matrix image
 *
 * Return: 0 on success, -1 on error.
 */
int arith_image_syrk_im(const IMAGE *imA, int transA, double alpha, double beta, IMAGE *outim)
{
    uint64_t a0    = 0;
    uint64_t a1    = 0;
    int      is_3d = 0;
    arith_image_flatten2d(imA, &a0, &a1, &is_3d);
    uint64_t N      = transA ? a1 : a0;
    uint64_t K      = transA ? a0 : a1;
    uint8_t  dta    = imA->md[0].datatype;
    uint8_t  out_dt = outim->md[0].datatype;

#ifdef HAVE_BLAS
    if (dta == _DATATYPE_FLOAT && out_dt == _DATATYPE_FLOAT)
    {
        float *restrict C = outim->array.F;
        cblas_ssyrk(CblasColMajor, CblasUpper, transA ? CblasTrans : CblasNoTrans, (int) N, (int) K,
                    (float) alpha, imA->array.F, (int) a0, (float) beta, C, (int) N);
        for (uint64_t j = 0; j < N; j++)
        {
            for (uint64_t i = j + 1; i < N; i++)
            {
                C[j * N + i] = C[i * N + j];
            }
        }
        return 0;
    }
    if (dta == _DATATYPE_DOUBLE && out_dt == _DATATYPE_DOUBLE)
    {
        double *restrict C = outim->array.D;
        cblas_dsyrk(CblasColMajor, CblasUpper, transA ? CblasTrans : CblasNoTrans, (int) N, (int) K,
                    alpha, imA->array.D, (int) a0, beta, C, (int) N);
        for (uint64_t j = 0; j < N; j++)
        {
            for (uint64_t i = j + 1; i < N; i++)
            {
                C[j * N + i] = C[i * N + j];
            }
        }
        return 0;
    }
#else
    (void) N;
    (void) K;
    (void) dta;
    (void) out_dt;
#endif
    return arith_image_gemm_im(imA, transA, imA, !transA, alpha, beta, outim);
}

/**
 * arith_image_symv_im - Compute symmetric matrix-vector product `imY = alpha * A * x + beta * imY`
 * @imA:   Symmetric `N x N` matrix (upper triangle referenced)
 * @imX:   Input length-`N` vector
 * @alpha: Scalar multiplier on `A * x`
 * @beta:  Scalar multiplier on @imY
 * @imY:   Destination length-`N` vector (float or double)
 *
 * Return: 0 on success, -1 on error.
 */
int arith_image_symv_im(const IMAGE *imA, const IMAGE *imX, double alpha, double beta, IMAGE *imY)
{
    uint64_t N   = imA->md[0].size[0];
    uint8_t  dta = imA->md[0].datatype;
    uint8_t  dtx = imX->md[0].datatype;
    uint8_t  dty = imY->md[0].datatype;

#ifdef HAVE_BLAS
    if (dta == _DATATYPE_FLOAT && dtx == _DATATYPE_FLOAT && dty == _DATATYPE_FLOAT)
    {
        cblas_ssymv(CblasColMajor, CblasUpper, (int) N, (float) alpha, imA->array.F, (int) N,
                    imX->array.F, 1, (float) beta, imY->array.F, 1);
        return 0;
    }
    if (dta == _DATATYPE_DOUBLE && dtx == _DATATYPE_DOUBLE && dty == _DATATYPE_DOUBLE)
    {
        cblas_dsymv(CblasColMajor, CblasUpper, (int) N, alpha, imA->array.D, (int) N, imX->array.D,
                    1, beta, imY->array.D, 1);
        return 0;
    }
#endif

    for (uint64_t i = 0; i < N; i++)
    {
        double acc = 0.0;
        for (uint64_t j = 0; j < N; j++)
        {
            uint64_t r   = (i <= j) ? i : j;
            uint64_t c   = (i <= j) ? j : i;
            double   aij = arith_read_im_pix(imA, dta, c * N + r);
            acc += aij * arith_read_im_pix(imX, dtx, j);
        }
        double y0 = (beta != 0.0) ? (beta * arith_read_im_pix(imY, dty, i)) : 0.0;
        if (dty == _DATATYPE_DOUBLE)
        {
            imY->array.D[i] = alpha * acc + y0;
        }
        else
        {
            imY->array.F[i] = (float) (alpha * acc + y0);
        }
    }
    return 0;
}

/**
 * arith_image_axpy_im - Update `imY = alpha * imX + imY` in-place
 * @alpha: Scalar multiplier on @imX
 * @imX:   Source image
 * @imY:   Destination image (float or double, matching `nelement`)
 *
 * Return: 0 on success.
 */
int arith_image_axpy_im(double alpha, const IMAGE *imX, IMAGE *imY)
{
    uint64_t nel = imY->md[0].nelement;
    uint8_t  dtx = imX->md[0].datatype;
    uint8_t  dty = imY->md[0].datatype;

    if (dtx == _DATATYPE_FLOAT && dty == _DATATYPE_FLOAT)
    {
        const float *restrict X = imX->array.F;
        float *restrict Y       = imY->array.F;
        float af                = (float) alpha;
#pragma omp parallel for simd schedule(static) if (nel >= 65536)
        for (uint64_t i = 0; i < nel; i++)
        {
            Y[i] += af * X[i];
        }
        return 0;
    }
    if (dtx == _DATATYPE_DOUBLE && dty == _DATATYPE_DOUBLE)
    {
        const double *restrict X = imX->array.D;
        double *restrict Y       = imY->array.D;
#pragma omp parallel for simd schedule(static) if (nel >= 65536)
        for (uint64_t i = 0; i < nel; i++)
        {
            Y[i] += alpha * X[i];
        }
        return 0;
    }
    for (uint64_t i = 0; i < nel; i++)
    {
        double v = alpha * arith_read_im_pix(imX, dtx, i) + arith_read_im_pix(imY, dty, i);
        if (dty == _DATATYPE_DOUBLE)
        {
            imY->array.D[i] = v;
        }
        else
        {
            imY->array.F[i] = (float) v;
        }
    }
    return 0;
}

/**
 * arith_image_scal_im - Scale `imX` in-place by @alpha
 * @alpha: Scalar multiplier
 * @imX:   Target float or double image
 *
 * Return: 0 on success, -1 if @imX is not float or double.
 */
int arith_image_scal_im(double alpha, IMAGE *imX)
{
    uint64_t nel = imX->md[0].nelement;
    uint8_t  dtx = imX->md[0].datatype;
    if (dtx == _DATATYPE_FLOAT)
    {
        float *restrict X = imX->array.F;
        float af          = (float) alpha;
#pragma omp parallel for simd schedule(static) if (nel >= 65536)
        for (uint64_t i = 0; i < nel; i++)
        {
            X[i] *= af;
        }
        return 0;
    }
    if (dtx == _DATATYPE_DOUBLE)
    {
        double *restrict X = imX->array.D;
#pragma omp parallel for simd schedule(static) if (nel >= 65536)
        for (uint64_t i = 0; i < nel; i++)
        {
            X[i] *= alpha;
        }
        return 0;
    }
    return -1;
}
