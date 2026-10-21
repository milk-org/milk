// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_linalg.c
 * @brief Calculator tensor shape and matrix multiplication operations (`transpose`, `reshape`, `vec`, `matmul`).
 *
 * Role & Goal:
 *   Implements in-place and temporary-allocating matrix transpose (`transpose()`),
 *   dimension reshaping (`reshape()`), 1-D vector literal construction (`vec()`),
 *   and BLAS/OpenMP matrix-vector and matrix-matrix multiplication (`matmul()`, `@`).
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// MILK_CMAKE_REQUEST_BLAS
#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "cli_calc_internal.h"
#include "milk_blas_lapacke.h"

/**
 * cli_calc_transpose_check - Validate whether `dst = transpose(src)` can execute in-place into @dst
 * @src: Source 1-D or 2-D real image
 * @dst: Destination 2-D float/double image
 *
 * Verifies that @src and @dst are distinct real images whose dimensions are
 * transposed (`dst` is `N x M` when `src` is `M x N`).
 *
 * Return: 0 if @dst is compatible for direct transpose output, -1 otherwise.
 */
int cli_calc_transpose_check(const IMAGE *src, const IMAGE *dst)
{
    if (src == NULL || dst == NULL || !src->used || !dst->used || src == dst)
    {
        return -1;
    }
    if (cli_calc_is_complex_dt(src->md[0].datatype) ||
        (src->md[0].naxis != 1 && src->md[0].naxis != 2))
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
 * cli_calc_transpose_into - Transpose 1-D or 2-D image @src into pre-allocated @dst
 * @src: Source image (`M x N` or 1-D `M`)
 * @dst: Destination 2-D image (`N x M`, float or double)
 *
 * Uses cache-blocked 32x32 tiling (with OpenMP for large matrices) on matching
 * float/double buffers, falling back to generic type conversion otherwise.
 *
 * Return: 0 on success.
 */
int cli_calc_transpose_into(const IMAGE *src, IMAGE *dst)
{
    uint64_t M       = src->md[0].size[0];
    uint64_t N       = (src->md[0].naxis == 2) ? src->md[0].size[1] : 1;
    uint8_t  src_dt  = src->md[0].datatype;
    uint8_t  dst_dt  = dst->md[0].datatype;
    uint64_t nel     = M * N;
    uint64_t tile_sz = 32;

    if (src_dt == _DATATYPE_FLOAT && dst_dt == _DATATYPE_FLOAT)
    {
        const float *restrict s = src->array.F;
        float *restrict d       = dst->array.F;
        if (M == 1 || N == 1)
        {
            __builtin_memcpy(d, s, sizeof(float) * nel);
            return 0;
        }
#pragma omp parallel for collapse(2) schedule(static) if (nel >= 65536)
        for (uint64_t m0 = 0; m0 < M; m0 += tile_sz)
        {
            for (uint64_t n0 = 0; n0 < N; n0 += tile_sz)
            {
                uint64_t mend = (m0 + tile_sz < M) ? m0 + tile_sz : M;
                uint64_t nend = (n0 + tile_sz < N) ? n0 + tile_sz : N;
                for (uint64_t n = n0; n < nend; n++)
                {
                    const float *restrict scol = &s[n * M];
                    for (uint64_t m = m0; m < mend; m++)
                    {
                        d[m * N + n] = scol[m];
                    }
                }
            }
        }
        return 0;
    }

    if (src_dt == _DATATYPE_DOUBLE && dst_dt == _DATATYPE_DOUBLE)
    {
        const double *restrict s = src->array.D;
        double *restrict d       = dst->array.D;
        if (M == 1 || N == 1)
        {
            __builtin_memcpy(d, s, sizeof(double) * nel);
            return 0;
        }
#pragma omp parallel for collapse(2) schedule(static) if (nel >= 65536)
        for (uint64_t m0 = 0; m0 < M; m0 += tile_sz)
        {
            for (uint64_t n0 = 0; n0 < N; n0 += tile_sz)
            {
                uint64_t mend = (m0 + tile_sz < M) ? m0 + tile_sz : M;
                uint64_t nend = (n0 + tile_sz < N) ? n0 + tile_sz : N;
                for (uint64_t n = n0; n < nend; n++)
                {
                    const double *restrict scol = &s[n * M];
                    for (uint64_t m = m0; m < mend; m++)
                    {
                        d[m * N + n] = scol[m];
                    }
                }
            }
        }
        return 0;
    }

    for (uint64_t m = 0; m < M; m++)
    {
        for (uint64_t n = 0; n < N; n++)
        {
            double v = cli_calc_read_im_pix(src, src_dt, n * M + m);
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
 * cli_calc_func_transpose - Evaluate calculator `transpose(im)` call
 *
 * Parses a single 1-D or 2-D image argument, allocates a transposed temporary
 * image (`N x M`), and populates it via `cli_calc_transpose_into()`.
 *
 * Return: `val_t` holding the temporary image name, or empty string on error.
 */
val_t cli_calc_func_transpose(void)
{
    val_t arg;
    if (cli_calc_parse_func_argv(1, 1, &arg, NULL) < 0)
    {
        return mk_double(0);
    }
    if (cli_calc_reject_complex(&arg, 1))
    {
        return mk_string("");
    }
    if (arg.type != VAL_STRING || !check_image(arg.sval))
    {
        if (arg.type != VAL_STRING)
        {
            parse_errmsg("transpose: expected image argument");
        }
        return mk_string("");
    }
    IMAGE *src = &data.core.image[image_ID(arg.sval, data.core.image, data.core.NB_MAX_IMAGE)];
    if (src->md[0].naxis != 1 && src->md[0].naxis != 2)
    {
        parse_errmsg("transpose: expected 1D or 2D image");
        return mk_string("");
    }
    uint32_t outsz[3] = { (src->md[0].naxis == 2) ? src->md[0].size[1] : 1, src->md[0].size[0], 1 };
    uint8_t out_dt = (src->md[0].datatype == _DATATYPE_DOUBLE) ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT;
    const char *tmpn = alloc_tmpname();
    imageID     tid  = -1;
    create_image_ID(tmpn, 2, outsz, out_dt, 0, 10, 0, &tid);
    if (tid == -1)
    {
        parse_errmsg("transpose: failed to allocate output image");
        return mk_string("");
    }
    if (cli_calc_transpose_into(src, &data.core.image[tid]) != 0)
    {
        return mk_string("");
    }
    return mk_string(tmpn);
}

/**
 * cli_calc_reshape_check - Validate whether `dst = reshape(src, ...)` can write directly into @dst
 * @src: Source image
 * @dst: Destination image
 *
 * Ensures @src and @dst have identical total element counts and compatible real
 * or complex datatypes.
 *
 * Return: 0 if compatible, -1 otherwise.
 */
int cli_calc_reshape_check(const IMAGE *src, const IMAGE *dst)
{
    if (src == NULL || dst == NULL || !src->used || !dst->used || src == dst ||
        src->md[0].nelement != dst->md[0].nelement)
    {
        return -1;
    }
    if (src->md[0].datatype != dst->md[0].datatype &&
        (cli_calc_is_complex_dt(src->md[0].datatype) ||
         cli_calc_is_complex_dt(dst->md[0].datatype)))
    {
        return -1;
    }
    return 0;
}

/**
 * cli_calc_reshape_into - Copy pixel data from @src into reshaped image @dst
 * @src: Source image
 * @dst: Destination image with identical `nelement`
 *
 * Uses raw `memcpy` when datatypes match or converts element-wise via `cli_calc_inplace()`.
 *
 * Return: 0 on success, -1 on error.
 */
int cli_calc_reshape_into(const IMAGE *src, IMAGE *dst)
{
    if (src->md[0].datatype == dst->md[0].datatype)
    {
        int tsz = ImageStreamIO_typesize(src->md[0].datatype);
        if (tsz <= 0)
        {
            return -1;
        }
        __builtin_memcpy(dst->array.raw, src->array.raw,
                         (size_t) src->md[0].nelement * (size_t) tsz);
        return 0;
    }
    return cli_calc_inplace(dst, TOK_EQUAL, src, 0.0);
}

/**
 * cli_calc_func_reshape - Evaluate calculator `reshape(im, d0[, d1[, d2]])` call
 *
 * Supports 1-D, 2-D, or 3-D target shapes with at most one inferred `-1` dimension,
 * returning a newly allocated temporary image with the same elements as `im`.
 *
 * Return: `val_t` holding the temporary image name, or empty string on error.
 */
val_t cli_calc_func_reshape(void)
{
    val_t args[4];
    int   n = cli_calc_parse_func_argv(2, 4, args, "Expected ','");
    if (n < 0)
    {
        return mk_string("");
    }
    if (args[0].type != VAL_STRING || !check_image(args[0].sval))
    {
        if (args[0].type != VAL_STRING)
        {
            parse_errmsg("reshape: first argument must be an image");
        }
        return mk_string("");
    }
    IMAGE *src = &data.core.image[image_ID(args[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    uint64_t nel      = src->md[0].nelement;
    int      ndims    = n - 1;
    long     dims[3]  = { 1, 1, 1 };
    int      neg_idx  = -1;
    uint64_t prod_pos = 1;

    for (int i = 0; i < ndims; i++)
    {
        if (args[i + 1].type == VAL_STRING)
        {
            parse_errmsg("reshape: dimensions must be integers");
            return mk_string("");
        }
        long d =
            (args[i + 1].type == VAL_LONG) ? args[i + 1].lval : (long) llround(args[i + 1].dval);
        if (d == -1)
        {
            if (neg_idx != -1)
            {
                parse_errmsg("reshape: at most one dimension may be -1");
                return mk_string("");
            }
            neg_idx = i;
        }
        else if (d <= 0)
        {
            parse_errmsg("reshape: dimensions must be positive");
            return mk_string("");
        }
        else
        {
            dims[i] = d;
            prod_pos *= (uint64_t) d;
        }
    }
    if (neg_idx != -1)
    {
        if (prod_pos == 0 || nel % prod_pos != 0)
        {
            parse_errmsg("reshape: element count mismatch");
            return mk_string("");
        }
        dims[neg_idx] = (long) (nel / prod_pos);
        prod_pos      = nel;
    }
    if (prod_pos != nel)
    {
        parse_errmsg("reshape: element count mismatch");
        return mk_string("");
    }
    uint32_t    outsz[3] = { (uint32_t) dims[0], (uint32_t) dims[1], (uint32_t) dims[2] };
    const char *tmpn     = alloc_tmpname();
    imageID     tid      = -1;
    create_image_ID(tmpn, ndims, outsz, src->md[0].datatype, 0, 10, 0, &tid);
    if (tid == -1 || cli_calc_reshape_into(src, &data.core.image[tid]) != 0)
    {
        if (tid == -1)
        {
            parse_errmsg("reshape: failed to allocate output image");
        }
        return mk_string("");
    }
    return mk_string(tmpn);
}

/**
 * cli_calc_func_vec - Evaluate calculator `vec(v0, v1, ...)` 1-D float vector constructor
 *
 * Parses up to 256 numeric scalar arguments and packs them into a 1-D `_DATATYPE_FLOAT`
 * temporary image.
 *
 * Return: `val_t` holding the temporary image name, or empty string on error.
 */
val_t cli_calc_func_vec(void)
{
    val_t args[256];
    int   n = cli_calc_parse_func_argv(1, 256, args, "Expected ','");
    if (n < 0)
    {
        return mk_string("");
    }
    for (int i = 0; i < n; i++)
    {
        if (args[i].type == VAL_STRING)
        {
            parse_errmsg("vec: arguments must be numbers");
            return mk_string("");
        }
    }
    uint32_t    outsz[3] = { (uint32_t) n, 1, 1 };
    const char *tmpn     = alloc_tmpname();
    imageID     tid      = -1;
    create_image_ID(tmpn, 1, outsz, _DATATYPE_FLOAT, 0, 10, 0, &tid);
    if (tid == -1)
    {
        parse_errmsg("vec: failed to allocate output image");
        return mk_string("");
    }
    IMAGE *out = &data.core.image[tid];
    for (int i = 0; i < n; i++)
    {
        out->array.F[i] = (float) to_double(args[i]);
    }
    return mk_string(tmpn);
}

/**
 * cli_calc_matmul_dims - Resolve matrix dimensions (`M`, `K`, `N`) and output shape for `matmul`
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
 * Handles 1-D vector promotion, 3-D image-cube mode matrices (`(W*H) x K`), and
 * automatic transpose fallback when `a0 == b0` on untransposed operands.
 *
 * Return: 0 if dimensions are compatible, -1 otherwise.
 */
static int cli_calc_matmul_dims(const IMAGE *im1,
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
    uint64_t a0      = im1->md[0].size[0];
    uint64_t a1      = 1;
    int      is_3d_a = 0;
    if (im1->md[0].naxis == 3)
    {
        if (transA)
        {
            return -1;
        }
        a0      = (uint64_t) im1->md[0].size[0] * im1->md[0].size[1];
        a1      = im1->md[0].size[2];
        is_3d_a = 1;
    }
    else if (im1->md[0].naxis >= 2)
    {
        a1 = im1->md[0].size[1];
    }

    uint64_t b0 = im2->md[0].size[0];
    uint64_t b1 = 1;
    if (im2->md[0].naxis == 3)
    {
        if (transB)
        {
            return -1;
        }
        b0 = (uint64_t) im2->md[0].size[0] * im2->md[0].size[1];
        b1 = im2->md[0].size[2];
    }
    else if (im2->md[0].naxis >= 2 && !(is_3d_a && !transA && !transB &&
                                        a0 == (uint64_t) im2->md[0].size[0] * im2->md[0].size[1]))
    {
        b1 = im2->md[0].size[1];
    }
    else if (is_3d_a && !transA && !transB && im2->md[0].naxis == 2 &&
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
 * cli_calc_matmul_check - Validate whether `dst = matmul(im1, im2)` can write directly into @dst
 * @im1:           Left operand image
 * @transA:        Non-zero if @im1 is transposed
 * @im2:           Right operand image
 * @transB:        Non-zero if @im2 is transposed
 * @dst:           Destination image
 * @allow_reshape: Non-zero to accept any @dst shape with matching `M * N` element count
 *
 * Checks aliasing, datatype compatibility, and dimension agreement so `milk.calc`
 * and fused assignments can write `matmul` / `@` results straight into @dst without
 * allocating a temporary image.
 *
 * Return: 0 if @dst can receive the product directly, -1 otherwise.
 */
int cli_calc_matmul_check(const IMAGE *im1,
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
    if (cli_calc_is_complex_dt(im1->md[0].datatype) || cli_calc_is_complex_dt(im2->md[0].datatype))
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
    if (cli_calc_matmul_dims(im1, transA, im2, transB, &tA, &tB, &M, &K, &N, &onax, outsz) != 0)
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

/**
 * matmul_kernel_f - Single-precision column-major matrix product `C = op(A) * op(B)`
 * @A:  Left matrix buffer (`M x K` or `K x M` when @tA is set)
 * @tA: Non-zero to transpose @A
 * @B:  Right matrix buffer (`K x N` or `N x K` when @tB is set)
 * @tB: Non-zero to transpose @B
 * @C:  Output matrix buffer (`M x N`)
 * @M:  Row count of `op(A)` and @C
 * @K:  Inner contraction dimension
 * @N:  Column count of `op(B)` and @C
 *
 * Dispatches to `cblas_sgemv` / `cblas_sgemm` when BLAS is available, or a
 * cache-blocked OpenMP SIMD fallback otherwise.
 */
static void matmul_kernel_f(const float *restrict A,
                            int tA,
                            const float *restrict B,
                            int tB,
                            float *restrict C,
                            uint64_t M,
                            uint64_t K,
                            uint64_t N)
{
#ifdef HAVE_BLAS
    if (N == 1)
    {
        cblas_sgemv(CblasColMajor, tA ? CblasTrans : CblasNoTrans, (int) (tA ? K : M),
                    (int) (tA ? M : K), 1.0f, A, (int) (tA ? K : M), B, 1, 0.0f, C, 1);
        return;
    }
    cblas_sgemm(CblasColMajor, tA ? CblasTrans : CblasNoTrans, tB ? CblasTrans : CblasNoTrans,
                (int) M, (int) N, (int) K, 1.0f, A, (int) (tA ? K : M), B, (int) (tB ? N : K), 0.0f,
                C, (int) M);
#else
    if (!tA)
    {
        const uint64_t mblk  = 128;
        uint64_t       nmblk = (M + mblk - 1) / mblk;
#    pragma omp parallel for collapse(2) schedule(static) if (M * N * K >= 262144)
        for (uint64_t n = 0; n < N; n++)
        {
            for (uint64_t mb = 0; mb < nmblk; mb++)
            {
                uint64_t m0          = mb * mblk;
                uint64_t mend        = (m0 + mblk < M) ? m0 + mblk : M;
                uint64_t mlen        = mend - m0;
                float *restrict cblk = &C[n * M + m0];
                memset(cblk, 0, sizeof(float) * mlen);
                for (uint64_t k = 0; k < K; k++)
                {
                    float b_kn                 = tB ? B[k * N + n] : B[n * K + k];
                    const float *restrict colA = &A[k * M + m0];
#    pragma omp simd
                    for (uint64_t m = 0; m < mlen; m++)
                    {
                        cblk[m] += colA[m] * b_kn;
                    }
                }
            }
        }
    }
    else if (!tB)
    {
#    pragma omp      parallel for collapse(2) schedule(static) if (M * N * K >= 262144)
        for (uint64_t n = 0; n < N; n++)
        {
            for (uint64_t m = 0; m < M; m++)
            {
                const float *restrict rowA = &A[m * K];
                const float *restrict colB = &B[n * K];
                float acc                  = 0.0f;
#    pragma omp simd reduction(+ : acc)
                for (uint64_t k = 0; k < K; k++)
                {
                    acc += rowA[k] * colB[k];
                }
                C[n * M + m] = acc;
            }
        }
    }
    else
    {
#    pragma omp      parallel for collapse(2) schedule(static) if (M * N * K >= 262144)
        for (uint64_t n = 0; n < N; n++)
        {
            for (uint64_t m = 0; m < M; m++)
            {
                const float *restrict rowA = &A[m * K];
                float acc                  = 0.0f;
                for (uint64_t k = 0; k < K; k++)
                {
                    acc += rowA[k] * B[k * N + n];
                }
                C[n * M + m] = acc;
            }
        }
    }
#endif
}

/**
 * matmul_kernel_d - Double-precision column-major matrix product `C = op(A) * op(B)`
 * @A:  Left matrix buffer (`M x K` or `K x M` when @tA is set)
 * @tA: Non-zero to transpose @A
 * @B:  Right matrix buffer (`K x N` or `N x K` when @tB is set)
 * @tB: Non-zero to transpose @B
 * @C:  Output matrix buffer (`M x N`)
 * @M:  Row count of `op(A)` and @C
 * @K:  Inner contraction dimension
 * @N:  Column count of `op(B)` and @C
 *
 * Dispatches to `cblas_dgemv` / `cblas_dgemm` when BLAS is available, or a
 * cache-blocked OpenMP SIMD fallback otherwise.
 */
static void matmul_kernel_d(const double *restrict A,
                            int tA,
                            const double *restrict B,
                            int tB,
                            double *restrict C,
                            uint64_t M,
                            uint64_t K,
                            uint64_t N)
{
#ifdef HAVE_BLAS
    if (N == 1)
    {
        cblas_dgemv(CblasColMajor, tA ? CblasTrans : CblasNoTrans, (int) (tA ? K : M),
                    (int) (tA ? M : K), 1.0, A, (int) (tA ? K : M), B, 1, 0.0, C, 1);
        return;
    }
    cblas_dgemm(CblasColMajor, tA ? CblasTrans : CblasNoTrans, tB ? CblasTrans : CblasNoTrans,
                (int) M, (int) N, (int) K, 1.0, A, (int) (tA ? K : M), B, (int) (tB ? N : K), 0.0,
                C, (int) M);
#else
    if (!tA)
    {
        const uint64_t mblk  = 64;
        uint64_t       nmblk = (M + mblk - 1) / mblk;
#    pragma omp parallel for collapse(2) schedule(static) if (M * N * K >= 262144)
        for (uint64_t n = 0; n < N; n++)
        {
            for (uint64_t mb = 0; mb < nmblk; mb++)
            {
                uint64_t m0           = mb * mblk;
                uint64_t mend         = (m0 + mblk < M) ? m0 + mblk : M;
                uint64_t mlen         = mend - m0;
                double *restrict cblk = &C[n * M + m0];
                memset(cblk, 0, sizeof(double) * mlen);
                for (uint64_t k = 0; k < K; k++)
                {
                    double b_kn                 = tB ? B[k * N + n] : B[n * K + k];
                    const double *restrict colA = &A[k * M + m0];
#    pragma omp simd
                    for (uint64_t m = 0; m < mlen; m++)
                    {
                        cblk[m] += colA[m] * b_kn;
                    }
                }
            }
        }
    }
    else if (!tB)
    {
#    pragma omp      parallel for collapse(2) schedule(static) if (M * N * K >= 262144)
        for (uint64_t n = 0; n < N; n++)
        {
            for (uint64_t m = 0; m < M; m++)
            {
                const double *restrict rowA = &A[m * K];
                const double *restrict colB = &B[n * K];
                double acc                  = 0.0;
#    pragma omp simd reduction(+ : acc)
                for (uint64_t k = 0; k < K; k++)
                {
                    acc += rowA[k] * colB[k];
                }
                C[n * M + m] = acc;
            }
        }
    }
    else
    {
#    pragma omp      parallel for collapse(2) schedule(static) if (M * N * K >= 262144)
        for (uint64_t n = 0; n < N; n++)
        {
            for (uint64_t m = 0; m < M; m++)
            {
                const double *restrict rowA = &A[m * K];
                double acc                  = 0.0;
                for (uint64_t k = 0; k < K; k++)
                {
                    acc += rowA[k] * B[k * N + n];
                }
                C[n * M + m] = acc;
            }
        }
    }
#endif
}

/**
 * cli_calc_matmul_into - Compute matrix product `op(im1) * op(im2)` into @outim
 * @im1:    Left operand image
 * @transA: Non-zero if @im1 is transposed
 * @im2:    Right operand image
 * @transB: Non-zero if @im2 is transposed
 * @outim:  Pre-allocated destination image (float or double)
 *
 * Uses zero-copy `matmul_kernel_f` / `matmul_kernel_d` when both inputs and @outim
 * share `_DATATYPE_FLOAT` or `_DATATYPE_DOUBLE`, or promotes mixed/integer inputs
 * through temporary double buffers.
 *
 * Return: 0 on success, -1 on dimension or allocation failure.
 */
int cli_calc_matmul_into(const IMAGE *im1, int transA, const IMAGE *im2, int transB, IMAGE *outim)
{
    int      tA       = 0;
    int      tB       = 0;
    uint64_t M        = 0;
    uint64_t K        = 0;
    uint64_t N        = 0;
    long     onax     = 0;
    uint32_t outsz[3] = { 1, 1, 1 };
    if (cli_calc_matmul_dims(im1, transA, im2, transB, &tA, &tB, &M, &K, &N, &onax, outsz) != 0)
    {
        parse_errmsg("matmul: incompatible image dimensions");
        return -1;
    }

    uint8_t dt1    = im1->md[0].datatype;
    uint8_t dt2    = im2->md[0].datatype;
    uint8_t out_dt = outim->md[0].datatype;

    if (dt1 == _DATATYPE_FLOAT && dt2 == _DATATYPE_FLOAT && out_dt == _DATATYPE_FLOAT)
    {
        matmul_kernel_f(im1->array.F, tA, im2->array.F, tB, outim->array.F, M, K, N);
        return 0;
    }
    if (dt1 == _DATATYPE_DOUBLE && dt2 == _DATATYPE_DOUBLE && out_dt == _DATATYPE_DOUBLE)
    {
        matmul_kernel_d(im1->array.D, tA, im2->array.D, tB, outim->array.D, M, K, N);
        return 0;
    }

    double *bufA = (double *) malloc(sizeof(double) * im1->md[0].nelement);
    double *bufB = (double *) malloc(sizeof(double) * im2->md[0].nelement);
    if (bufA == NULL || bufB == NULL)
    {
        free(bufA);
        free(bufB);
        parse_errmsg("matmul: memory allocation failed");
        return -1;
    }
    for (uint64_t i = 0; i < im1->md[0].nelement; i++)
    {
        bufA[i] = cli_calc_read_im_pix(im1, dt1, i);
    }
    for (uint64_t i = 0; i < im2->md[0].nelement; i++)
    {
        bufB[i] = cli_calc_read_im_pix(im2, dt2, i);
    }

    if (out_dt == _DATATYPE_DOUBLE)
    {
        matmul_kernel_d(bufA, tA, bufB, tB, outim->array.D, M, K, N);
    }
    else
    {
        double *bufC = (double *) malloc(sizeof(double) * M * N);
        if (bufC == NULL)
        {
            free(bufA);
            free(bufB);
            parse_errmsg("matmul: memory allocation failed");
            return -1;
        }
        matmul_kernel_d(bufA, tA, bufB, tB, bufC, M, K, N);
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
 * is_top_level_transpose - Peek whether the current `matmul` argument is wrapped in `transpose(...)`
 * @end_tok: Expected token immediately after the argument (`TOK_COMMA` or `TOK_RPAREN`)
 *
 * Allows `matmul(transpose(A), B)` to fold the transpose flag into the BLAS call
 * without materializing a transposed temporary image.
 *
 * Return: 1 if the argument is a top-level `transpose(...)` call, 0 otherwise.
 */
static int is_top_level_transpose(cli_token_type end_tok)
{
    cli_token *(*cur_func)(void)     = cli_calc_cur_func();
    cli_token *(*advance_func)(void) = cli_calc_advance_func();
    if (cur_func()->type != TOK_FUNC_IMIM_IM || strcmp(cur_func()->sval, "transpose(") != 0)
    {
        return 0;
    }
    int saved = cli_calc_tok_pos();
    advance_func();
    int depth = 1;
    while (depth > 0 && cur_func()->type != TOK_EOF && cur_func()->type != TOK_NEWLINE)
    {
        cli_token_type t = cur_func()->type;
        if (t == TOK_LPAREN ||
            (t >= TOK_FUNC_D_D && t <= TOK_FUNC_COORD && strchr(cur_func()->sval, '(') != NULL))
        {
            depth++;
        }
        else if (t == TOK_RPAREN)
        {
            depth--;
        }
        advance_func();
    }
    int is_top = (depth == 0 && cur_func()->type == end_tok);
    cli_calc_tok_seek(saved);
    return is_top;
}

/**
 * parse_matmul_arg - Parse one operand of `matmul(A, B)`, unwrapping a top-level `transpose()`
 * @out:       Receives the evaluated operand value
 * @out_trans: Receives 1 if an outer `transpose(...)` was unwrapped, 0 otherwise
 * @end_tok:   Delimiter token following this operand (`TOK_COMMA` or `TOK_RPAREN`)
 *
 * Return: 0 on success, -1 on syntax or evaluation error.
 */
static int parse_matmul_arg(val_t *out, int *out_trans, cli_token_type end_tok)
{
    cli_token *(*cur_func)(void)     = cli_calc_cur_func();
    cli_token *(*advance_func)(void) = cli_calc_advance_func();
    int trans                        = 0;

    if (is_top_level_transpose(end_tok))
    {
        advance_func();
        trans = 1;
    }
    *out = parse_expr(0);
    if (parse_error || eval_error)
    {
        return -1;
    }
    if (trans)
    {
        if (cur_func()->type != TOK_RPAREN)
        {
            parse_errmsg("Expected ')'");
            return -1;
        }
        advance_func();
    }
    *out_trans = trans;
    return 0;
}

/**
 * cli_calc_func_matmul - Evaluate calculator `matmul(A, B)` call
 *
 * Parses two image operands (folding any outer `transpose(...)` wrappers into
 * BLAS transpose flags), allocates a float/double temporary image, and computes
 * the matrix product via `cli_calc_matmul_into()`.
 *
 * Return: `val_t` holding the temporary product image name, or empty string on error.
 */
val_t cli_calc_func_matmul(void)
{
    cli_token *(*cur_func)(void)     = cli_calc_cur_func();
    cli_token *(*advance_func)(void) = cli_calc_advance_func();
    val_t args[2];
    int   transA = 0;
    int   transB = 0;

    if (parse_matmul_arg(&args[0], &transA, TOK_COMMA) != 0)
    {
        return mk_double(0);
    }
    if (cur_func()->type != TOK_COMMA)
    {
        parse_errmsg("Expected ','");
        return mk_double(0);
    }
    advance_func();
    if (parse_matmul_arg(&args[1], &transB, TOK_RPAREN) != 0)
    {
        return mk_double(0);
    }
    if (cur_func()->type != TOK_RPAREN)
    {
        parse_errmsg("Expected ')'");
        return mk_double(0);
    }
    advance_func();

    if (cli_calc_reject_complex(args, 2))
    {
        return mk_string("");
    }
    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING)
    {
        parse_errmsg("matmul: expected two image arguments");
        return mk_string("");
    }
    if (!check_image(args[0].sval) || !check_image(args[1].sval))
    {
        return mk_string("");
    }

    IMAGE *im1 = &data.core.image[image_ID(args[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE *im2 = &data.core.image[image_ID(args[1].sval, data.core.image, data.core.NB_MAX_IMAGE)];

    int      tA       = 0;
    int      tB       = 0;
    uint64_t M        = 0;
    uint64_t K        = 0;
    uint64_t N        = 0;
    long     onax     = 0;
    uint32_t outsz[3] = { 1, 1, 1 };
    if (cli_calc_matmul_dims(im1, transA, im2, transB, &tA, &tB, &M, &K, &N, &onax, outsz) != 0)
    {
        parse_errmsg("matmul: incompatible image dimensions");
        return mk_string("");
    }

    uint8_t out_dt =
        (im1->md[0].datatype == _DATATYPE_DOUBLE || im2->md[0].datatype == _DATATYPE_DOUBLE)
            ? _DATATYPE_DOUBLE
            : _DATATYPE_FLOAT;

    const char *tmpn = alloc_tmpname();
    imageID     tid  = -1;
    create_image_ID(tmpn, onax, outsz, out_dt, 0, 10, 0, &tid);
    if (tid == -1)
    {
        parse_errmsg("matmul: failed to allocate output image");
        return mk_string("");
    }
    if (cli_calc_matmul_into(im1, transA, im2, transB, &data.core.image[tid]) != 0)
    {
        return mk_string("");
    }
    return mk_string(tmpn);
}
