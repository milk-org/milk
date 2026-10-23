// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cli_calc_linalg.c
 * @brief Calculator tensor shape, BLAS Level 1-3 (`matmul`, `gemm`, `gemv`, `ger`, `outer`,
 *        `syrk`, `symv`, `axpy`, `scal`), and kept matrix-assignment lowering.
 *
 * Role & Goal:
 *   Implements in-place and temporary-allocating matrix transpose (`transpose()`),
 *   dimension reshaping (`reshape()`), 1-D/2-D constructors (`vec()`, `eye()`, `diag()`),
 *   BLAS Level 1-3 operations, and direct destination lowering (`cli_calc_compile_kept_matrix()`,
 *   `cli_calc_exec_kept_matrix()`) for `dst = alpha * op(A) * op(B) + beta * dst` forms.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
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
 * cli_calc_func_eye - Evaluate `eye(N[, M])` to construct an `N x M` float32 identity matrix
 *
 * Return: `val_t` holding the temporary identity matrix image name, or empty string on error.
 */
val_t cli_calc_func_eye(void)
{
    val_t args[2];
    int   nargs = cli_calc_parse_func_argv(1, 2, args, "eye: expected ','");
    if (nargs < 0 || cli_calc_reject_complex(args, nargs))
    {
        return mk_string("");
    }
    if (args[0].type == VAL_STRING || (nargs == 2 && args[1].type == VAL_STRING))
    {
        parse_errmsg("eye: expected integer dimension arguments");
        return mk_string("");
    }
    long n = llround(to_double(args[0]));
    long m = (nargs == 2) ? llround(to_double(args[1])) : n;
    if (n <= 0 || m <= 0)
    {
        parse_errmsg("eye: dimensions must be positive");
        return mk_string("");
    }
    const char *tmpn  = alloc_tmpname();
    imageID     tid   = -1;
    uint32_t    sz[2] = { (uint32_t) n, (uint32_t) m };
    create_image_ID(tmpn, 2, sz, _DATATYPE_FLOAT, 0, 10, 0, &tid);
    if (tid == -1)
    {
        parse_errmsg("eye: failed to allocate output image");
        return mk_string("");
    }
    IMAGE   *out = &data.core.image[tid];
    uint64_t nel = (uint64_t) n * (uint64_t) m;
    memset(out->array.F, 0, sizeof(float) * nel);
    uint64_t dlen = (n < m) ? (uint64_t) n : (uint64_t) m;
    for (uint64_t i = 0; i < dlen; i++)
    {
        out->array.F[i * (uint64_t) n + i] = 1.0f;
    }
    return mk_string(tmpn);
}

/**
 * cli_calc_func_diag - Evaluate `diag(X[, k])` to extract or construct a diagonal matrix
 *
 * When @X is a 1-D vector of length `L`, builds an `(L + |k|) x (L + |k|)` diagonal
 * matrix with @X placed on diagonal @k (`0` = main diagonal). When @X is a 2-D
 * `M x N` matrix, extracts diagonal @k into a 1-D vector.
 *
 * Return: `val_t` holding the temporary output image name, or empty string on error.
 */
val_t cli_calc_func_diag(void)
{
    val_t args[2];
    int   nargs = cli_calc_parse_func_argv(1, 2, args, "diag: expected ','");
    if (nargs < 0 || cli_calc_reject_complex(args, nargs))
    {
        return mk_string("");
    }
    if (args[0].type != VAL_STRING || !check_image(args[0].sval))
    {
        parse_errmsg("diag: first argument must be an image");
        return mk_string("");
    }
    long k = 0;
    if (nargs == 2)
    {
        if (args[1].type == VAL_STRING)
        {
            parse_errmsg("diag: offset k must be a number");
            return mk_string("");
        }
        k = llround(to_double(args[1]));
    }
    const IMAGE *src =
        &data.core.image[image_ID(args[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    uint8_t src_dt = src->md[0].datatype;
    uint8_t out_dt = (src_dt == _DATATYPE_DOUBLE) ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT;
    long    absk   = (k < 0) ? -k : k;

    if (src->md[0].naxis == 1)
    {
        uint64_t    L     = src->md[0].size[0];
        uint64_t    N     = L + (uint64_t) absk;
        const char *tmpn  = alloc_tmpname();
        imageID     tid   = -1;
        uint32_t    sz[2] = { (uint32_t) N, (uint32_t) N };
        create_image_ID(tmpn, 2, sz, out_dt, 0, 10, 0, &tid);
        if (tid == -1)
        {
            parse_errmsg("diag: failed to allocate output image");
            return mk_string("");
        }
        IMAGE   *out = &data.core.image[tid];
        uint64_t r0  = (k < 0) ? (uint64_t) absk : 0;
        uint64_t c0  = (k > 0) ? (uint64_t) absk : 0;
        if (out_dt == _DATATYPE_DOUBLE)
        {
            memset(out->array.D, 0, sizeof(double) * N * N);
            for (uint64_t i = 0; i < L; i++)
            {
                out->array.D[(c0 + i) * N + (r0 + i)] = cli_calc_read_im_pix(src, src_dt, i);
            }
        }
        else
        {
            memset(out->array.F, 0, sizeof(float) * N * N);
            for (uint64_t i = 0; i < L; i++)
            {
                out->array.F[(c0 + i) * N + (r0 + i)] =
                    (float) cli_calc_read_im_pix(src, src_dt, i);
            }
        }
        return mk_string(tmpn);
    }

    if (src->md[0].naxis == 2)
    {
        uint64_t M  = src->md[0].size[0];
        uint64_t N  = src->md[0].size[1];
        uint64_t r0 = (k < 0) ? (uint64_t) absk : 0;
        uint64_t c0 = (k > 0) ? (uint64_t) absk : 0;
        if (r0 >= M || c0 >= N)
        {
            parse_errmsg("diag: diagonal offset out of range");
            return mk_string("");
        }
        uint64_t    rm   = M - r0;
        uint64_t    cn   = N - c0;
        uint64_t    L    = (rm < cn) ? rm : cn;
        const char *tmpn = alloc_tmpname();
        imageID     tid  = -1;
        uint32_t    sz   = (uint32_t) L;
        create_image_ID(tmpn, 1, &sz, out_dt, 0, 10, 0, &tid);
        if (tid == -1)
        {
            parse_errmsg("diag: failed to allocate output image");
            return mk_string("");
        }
        IMAGE *out = &data.core.image[tid];
        for (uint64_t i = 0; i < L; i++)
        {
            double v = cli_calc_read_im_pix(src, src_dt, (c0 + i) * M + (r0 + i));
            if (out_dt == _DATATYPE_DOUBLE)
            {
                out->array.D[i] = v;
            }
            else
            {
                out->array.F[i] = (float) v;
            }
        }
        return mk_string(tmpn);
    }

    parse_errmsg("diag: expected 1D or 2D image");
    return mk_string("");
}

/**
 * cli_calc_flatten2d - Map a 1-D, 2-D, or 3-D image to 2-D matrix dimensions `(d0, d1)`
 * @im:    Input image
 * @d0:    Receives effective row count (`size[0]`, or `size[0]*size[1]` for 3-D cubes)
 * @d1:    Receives effective column count (`1` for 1-D, `size[1]` for 2-D, `size[2]` for 3-D)
 * @is_3d: Receives 1 when @im is 3-D, 0 otherwise
 */
void cli_calc_flatten2d(const IMAGE *im, uint64_t *d0, uint64_t *d1, int *is_3d)
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
    uint64_t a0      = 0;
    uint64_t a1      = 0;
    int      is_3d_a = 0;
    cli_calc_flatten2d(im1, &a0, &a1, &is_3d_a);

    uint64_t b0      = 0;
    uint64_t b1      = 0;
    int      is_3d_b = 0;
    cli_calc_flatten2d(im2, &b0, &b1, &is_3d_b);
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
#    define MATMUL_KERNEL_BODY(T, GEMV, GEMM, MBLK)                                            \
        if (!tA)                                                                               \
        {                                                                                      \
            const uint64_t mblk  = (MBLK);                                                     \
            uint64_t       nmblk = (M + mblk - 1) / mblk;                                      \
            _Pragma("omp parallel for collapse(2) schedule(static) if (M * N * K >= "          \
                    "262144)") for (uint64_t n = 0; n < N; n++)                                \
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
            _Pragma("omp parallel for collapse(2) schedule(static) if (M * N * K >= "          \
                    "262144)") for (uint64_t n = 0; n < N; n++)                                \
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
            _Pragma("omp parallel for collapse(2) schedule(static) if (M * N * K >= "          \
                    "262144)") for (uint64_t n = 0; n < N; n++)                                \
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
 * matmul_kernel_f - Single-precision column-major BLAS GEMM `C = alpha * op(A) * op(B) + beta * C`
 */
static void matmul_kernel_f(const float *restrict A,
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
 * matmul_kernel_d - Double-precision column-major BLAS GEMM `C = alpha * op(A) * op(B) + beta * C`
 */
static void matmul_kernel_d(const double *restrict A,
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
 * cli_calc_gemm_into - Compute `outim = alpha * op(im1) * op(im2) + beta * outim`
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
int cli_calc_gemm_into(const IMAGE *im1,
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
        matmul_kernel_f(im1->array.F, tA, im2->array.F, tB, (float) alpha, (float) beta,
                        outim->array.F, M, K, N);
        return 0;
    }
    if (dt1 == _DATATYPE_DOUBLE && dt2 == _DATATYPE_DOUBLE && out_dt == _DATATYPE_DOUBLE)
    {
        matmul_kernel_d(im1->array.D, tA, im2->array.D, tB, alpha, beta, outim->array.D, M, K, N);
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
        matmul_kernel_d(bufA, tA, bufB, tB, alpha, beta, outim->array.D, M, K, N);
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
        if (beta != 0.0)
        {
            for (uint64_t i = 0; i < M * N; i++)
            {
                bufC[i] = (double) outim->array.F[i];
            }
        }
        matmul_kernel_d(bufA, tA, bufB, tB, alpha, beta, bufC, M, K, N);
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
 * cli_calc_matmul_into - Compute matrix product `op(im1) * op(im2)` into @outim
 * @im1:    Left operand image
 * @transA: Non-zero if @im1 is transposed
 * @im2:    Right operand image
 * @transB: Non-zero if @im2 is transposed
 * @outim:  Pre-allocated destination image (float or double)
 *
 * Return: 0 on success, -1 on dimension or allocation failure.
 */
int cli_calc_matmul_into(const IMAGE *im1, int transA, const IMAGE *im2, int transB, IMAGE *outim)
{
    return cli_calc_gemm_into(im1, transA, im2, transB, 1.0, 0.0, outim);
}

/**
 * is_top_level_transpose - Peek whether the current argument is wrapped in `transpose(...)`
 *
 * Allows `matmul(transpose(A), B)` / `gemm` / `syrk` / `trsv` / `solve` to fold
 * the transpose flag into the BLAS/LAPACK call without materializing a transposed
 * temporary image.
 *
 * Return: 1 if the argument is a top-level `transpose(...)` call, 0 otherwise.
 */
static int is_top_level_transpose(void)
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
    int is_top = (depth == 0 && (cur_func()->type == TOK_COMMA || cur_func()->type == TOK_RPAREN));
    cli_calc_tok_seek(saved);
    return is_top;
}

/**
 * cli_calc_parse_matmul_arg - Parse one matrix argument, unwrapping a top-level `transpose()`
 * @out:       Receives the evaluated operand value
 * @out_trans: Receives 1 if an outer `transpose(...)` was unwrapped, 0 otherwise
 *
 * Return: 0 on success, -1 on syntax or evaluation error.
 */
int cli_calc_parse_matmul_arg(val_t *out, int *out_trans)
{
    cli_token *(*cur_func)(void)     = cli_calc_cur_func();
    cli_token *(*advance_func)(void) = cli_calc_advance_func();
    int trans                        = 0;

    if (is_top_level_transpose())
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
 * cli_calc_func_matmul - Evaluate calculator `matmul(A, B)` 2-argument product
 *
 * Allocates a new temporary product image `op(A) * op(B)` (unwrapping top-level
 * `transpose(A)` and `transpose(B)` calls zero-copy) and populates it via
 * `cli_calc_gemm_into()`.
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

    if (cli_calc_parse_matmul_arg(&args[0], &transA) != 0)
    {
        return mk_double(0);
    }
    if (cur_func()->type != TOK_COMMA)
    {
        parse_errmsg("Expected ','");
        return mk_double(0);
    }
    advance_func();
    if (cli_calc_parse_matmul_arg(&args[1], &transB) != 0)
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
        parse_errmsg("matmul: expected image arguments");
        return mk_string("");
    }
    if (!check_image(args[0].sval) || !check_image(args[1].sval))
    {
        return mk_string("");
    }

    IMAGE *im1 = &data.core.image[image_ID(args[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE *im2 = &data.core.image[image_ID(args[1].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    int    tA  = 0;
    int    tB  = 0;
    uint64_t M = 0;
    uint64_t K = 0;
    uint64_t N = 0;
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
    IMAGE *out = &data.core.image[tid];
    if (cli_calc_gemm_into(im1, transA, im2, transB, 1.0, 0.0, out) != 0)
    {
        return mk_string("");
    }
    return mk_string(tmpn);
}

/**
 * cli_calc_func_gemm - Evaluate strict BLAS `gemm(alpha, A, B, beta, C)` or `gemv(alpha, A, x, beta, y)`
 * @is_gemv: Non-zero when invoked as `gemv(...)` (requires vector output `N == 1`)
 *
 * Updates destination matrix `C` (or vector `y`) in-place with
 * `alpha * op(A) * op(B) + beta * C` and posts output stream updates.
 *
 * Return: `val_t` holding the destination image name, or empty string on error.
 */
val_t cli_calc_func_gemm(int is_gemv)
{
    cli_token *(*cur_func)(void)     = cli_calc_cur_func();
    cli_token *(*advance_func)(void) = cli_calc_advance_func();
    val_t args[3];
    int   transA = 0;
    int   transB = 0;

    val_t av = parse_expr(0);
    if (parse_error || eval_error || av.type == VAL_STRING)
    {
        parse_errmsg(is_gemv ? "gemv: expected gemv(alpha, A, x, beta, y)"
                             : "gemm: expected gemm(alpha, A, B, beta, C)");
        return mk_string("");
    }
    double alpha = to_double(av);
    if (cur_func()->type != TOK_COMMA)
    {
        parse_errmsg("Expected ','");
        return mk_string("");
    }
    advance_func();
    if (cli_calc_parse_matmul_arg(&args[0], &transA) != 0)
    {
        return mk_string("");
    }
    if (cur_func()->type != TOK_COMMA)
    {
        parse_errmsg("Expected ','");
        return mk_string("");
    }
    advance_func();
    if (cli_calc_parse_matmul_arg(&args[1], &transB) != 0)
    {
        return mk_string("");
    }
    if (cur_func()->type != TOK_COMMA)
    {
        parse_errmsg(is_gemv ? "gemv: expected gemv(alpha, A, x, beta, y)"
                             : "gemm: expected gemm(alpha, A, B, beta, C)");
        return mk_string("");
    }
    advance_func();
    val_t bv = parse_expr(0);
    if (parse_error || eval_error || bv.type == VAL_STRING)
    {
        parse_errmsg(is_gemv ? "gemv: beta must be a scalar" : "gemm: beta must be a scalar");
        return mk_string("");
    }
    double beta = to_double(bv);
    if (cur_func()->type != TOK_COMMA)
    {
        parse_errmsg(is_gemv ? "gemv: expected destination y after beta"
                             : "gemm: expected destination C after beta");
        return mk_string("");
    }
    advance_func();
    args[2] = parse_expr(0);
    if (parse_error || eval_error)
    {
        return mk_string("");
    }
    if (cur_func()->type != TOK_RPAREN)
    {
        parse_errmsg("Expected ')'");
        return mk_string("");
    }
    advance_func();

    if (cli_calc_reject_complex(args, 3))
    {
        return mk_string("");
    }
    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING || args[2].type != VAL_STRING)
    {
        parse_errmsg(is_gemv ? "gemv: A, x, and y must be images"
                             : "gemm: A, B, and C must be images");
        return mk_string("");
    }
    if (!check_image(args[0].sval) || !check_image(args[1].sval) || !check_image(args[2].sval))
    {
        return mk_string("");
    }

    IMAGE *imA = &data.core.image[image_ID(args[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE *imB = &data.core.image[image_ID(args[1].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE *imC = &data.core.image[image_ID(args[2].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    if (imC == imA || imC == imB)
    {
        parse_errmsg(is_gemv ? "gemv: destination y must not alias A or x"
                             : "gemm: destination C must not alias A or B");
        return mk_string("");
    }

    int      tA       = 0;
    int      tB       = 0;
    uint64_t M        = 0;
    uint64_t K        = 0;
    uint64_t N        = 0;
    long     onax     = 0;
    uint32_t outsz[3] = { 1, 1, 1 };
    if (cli_calc_matmul_dims(imA, transA, imB, transB, &tA, &tB, &M, &K, &N, &onax, outsz) != 0)
    {
        parse_errmsg(is_gemv ? "gemv: incompatible image dimensions"
                             : "gemm: incompatible image dimensions");
        return mk_string("");
    }
    if (is_gemv && N != 1)
    {
        parse_errmsg("gemv: x must be a vector");
        return mk_string("");
    }
    if (cli_calc_matmul_check(imA, transA, imB, transB, imC, 0) != 0)
    {
        parse_errmsg(is_gemv ? "gemv: y dimensions or datatype do not match output vector"
                             : "gemm: C dimensions or datatype do not match product shape");
        return mk_string("");
    }

    imC->md[0].write = 1;
    if (cli_calc_gemm_into(imA, transA, imB, transB, alpha, beta, imC) != 0)
    {
        imC->md[0].write = 0;
        return mk_string("");
    }
    processinfo_update_output_stream(NULL, imC, NULL);
    return mk_string(imC->name);
}

/**
 * is_1d_vec - Check whether @im is a 1-D vector or a 2-D single-row/column vector
 * @im:  Image to inspect
 * @len: Output vector length (`nelement`)
 *
 * Return: 1 if @im is a 1-D or `N x 1` / `1 x N` real image, 0 otherwise.
 */
static int is_1d_vec(const IMAGE *im, uint64_t *len)
{
    if (im == NULL || !im->used || cli_calc_is_complex_dt(im->md[0].datatype))
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
 * cli_calc_ger_check - Validate whether `ger` / `outer` can write directly into @dst
 * @imX: First vector (`M`)
 * @imY: Second vector (`N`)
 * @dst: Destination `M x N` matrix
 *
 * Return: 0 if compatible, -1 otherwise.
 */
int cli_calc_ger_check(const IMAGE *imX, const IMAGE *imY, const IMAGE *dst)
{
    uint64_t M = 0;
    uint64_t N = 0;
    if (!is_1d_vec(imX, &M) || !is_1d_vec(imY, &N) || dst == NULL || !dst->used || dst == imX ||
        dst == imY)
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
 * cli_calc_ger_into - Compute rank-1 outer product `outim = alpha * x * y^T + beta * outim`
 * @imX:   Column vector `x` of length `M`
 * @imY:   Row vector `y` of length `N`
 * @alpha: Scalar multiplier on `x * y^T`
 * @beta:  Scalar multiplier on @outim (`0.0` overwrites @outim)
 * @outim: Destination `M x N` image (float or double)
 *
 * Return: 0 on success, -1 on dimension error.
 */
int cli_calc_ger_into(const IMAGE *imX, const IMAGE *imY, double alpha, double beta, IMAGE *outim)
{
    uint64_t M = 0;
    uint64_t N = 0;
    if (!is_1d_vec(imX, &M) || !is_1d_vec(imY, &N))
    {
        parse_errmsg("ger: expected two 1D vector arguments");
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
        double yj = alpha * cli_calc_read_im_pix(imY, dty, j);
        for (uint64_t i = 0; i < M; i++)
        {
            double xi = cli_calc_read_im_pix(imX, dtx, i);
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
 * cli_calc_func_outer - Evaluate 2-argument outer product `outer(x, y)` returning `x * y^T`
 *
 * Return: `val_t` holding the temporary `M x N` outer-product image name, or empty string on error.
 */
val_t cli_calc_func_outer(void)
{
    val_t args[2];
    int   nargs = cli_calc_parse_func_argv(2, 2, args, "outer: expected ','");
    if (nargs < 0 || cli_calc_reject_complex(args, 2))
    {
        return mk_string("");
    }
    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING || !check_image(args[0].sval) ||
        !check_image(args[1].sval))
    {
        parse_errmsg("outer: expected two 1D vector images");
        return mk_string("");
    }
    IMAGE *imX = &data.core.image[image_ID(args[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE *imY = &data.core.image[image_ID(args[1].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    uint64_t M = 0;
    uint64_t N = 0;
    if (!is_1d_vec(imX, &M) || !is_1d_vec(imY, &N))
    {
        parse_errmsg("outer: expected 1D vector arguments");
        return mk_string("");
    }
    uint8_t out_dt =
        (imX->md[0].datatype == _DATATYPE_DOUBLE || imY->md[0].datatype == _DATATYPE_DOUBLE)
            ? _DATATYPE_DOUBLE
            : _DATATYPE_FLOAT;
    const char *tmpn  = alloc_tmpname();
    imageID     tid   = -1;
    uint32_t    sz[2] = { (uint32_t) M, (uint32_t) N };
    create_image_ID(tmpn, 2, sz, out_dt, 0, 10, 0, &tid);
    if (tid == -1)
    {
        parse_errmsg("outer: failed to allocate output image");
        return mk_string("");
    }
    IMAGE *out = &data.core.image[tid];
    if (cli_calc_ger_into(imX, imY, 1.0, 0.0, out) != 0)
    {
        return mk_string("");
    }
    return mk_string(tmpn);
}

/**
 * cli_calc_func_ger - Evaluate strict BLAS rank-1 update `ger(alpha, x, y, A)`
 *
 * Updates `M x N` matrix @A in-place with `alpha * x * y^T + A`.
 *
 * Return: `val_t` holding the destination matrix @A name, or empty string on error.
 */
val_t cli_calc_func_ger(void)
{
    val_t args[4];
    int   nargs = cli_calc_parse_func_argv(4, 4, args, "ger: expected ','");
    if (nargs < 0 || cli_calc_reject_complex(args, 4))
    {
        return mk_string("");
    }
    if (args[0].type == VAL_STRING || args[1].type != VAL_STRING || args[2].type != VAL_STRING ||
        args[3].type != VAL_STRING || !check_image(args[1].sval) || !check_image(args[2].sval) ||
        !check_image(args[3].sval))
    {
        parse_errmsg("ger: expected ger(alpha, x, y, A)");
        return mk_string("");
    }
    double alpha = to_double(args[0]);
    IMAGE *imX = &data.core.image[image_ID(args[1].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE *imY = &data.core.image[image_ID(args[2].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE *imA = &data.core.image[image_ID(args[3].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    if (imA == imX || imA == imY)
    {
        parse_errmsg("ger: destination matrix A must not alias x or y");
        return mk_string("");
    }
    if (cli_calc_ger_check(imX, imY, imA) != 0)
    {
        parse_errmsg("ger: expected 1D vectors x (M), y (N) and M x N float/double matrix A");
        return mk_string("");
    }
    imA->md[0].write = 1;
    if (cli_calc_ger_into(imX, imY, alpha, 1.0, imA) != 0)
    {
        imA->md[0].write = 0;
        return mk_string("");
    }
    processinfo_update_output_stream(NULL, imA, NULL);
    return mk_string(imA->name);
}

/**
 * cli_calc_syrk_check - Validate whether `syrk` can write directly into @dst
 * @imA:    Input matrix (`N x K`, or `K x N` when @transA is set)
 * @transA: Non-zero for `A^T * A`, 0 for `A * A^T`
 * @dst:    Destination `N x N` symmetric matrix
 *
 * Return: 0 if compatible, -1 otherwise.
 */
int cli_calc_syrk_check(const IMAGE *imA, int transA, const IMAGE *dst)
{
    if (imA == NULL || dst == NULL || !imA->used || !dst->used || dst == imA ||
        cli_calc_is_complex_dt(imA->md[0].datatype))
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
    cli_calc_flatten2d(imA, &a0, &a1, &is_3d);
    uint64_t N = transA ? a1 : a0;
    if (dst->md[0].naxis != 2 || dst->md[0].size[0] != N || dst->md[0].size[1] != N)
    {
        return -1;
    }
    return 0;
}

/**
 * cli_calc_syrk_into - Compute symmetric rank-k update `outim = alpha * op(A) * op(A)^T + beta * outim`
 * @imA:    Input matrix image
 * @transA: Non-zero for `A^T * A`, 0 for `A * A^T`
 * @alpha:  Scalar multiplier on `op(A) * op(A)^T`
 * @beta:   Scalar multiplier on @outim (`0.0` overwrites @outim)
 * @outim:  Destination `N x N` symmetric matrix image
 *
 * Return: 0 on success, -1 on error.
 */
int cli_calc_syrk_into(const IMAGE *imA, int transA, double alpha, double beta, IMAGE *outim)
{
    uint64_t a0    = 0;
    uint64_t a1    = 0;
    int      is_3d = 0;
    cli_calc_flatten2d(imA, &a0, &a1, &is_3d);
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
    return cli_calc_gemm_into(imA, transA, imA, !transA, alpha, beta, outim);
}

/**
 * cli_calc_func_syrk - Evaluate strict BLAS symmetric rank-k update `syrk(alpha, A, beta, C)`
 *
 * Updates `N x N` symmetric matrix @C in-place with `alpha * A * A^T + beta * C`,
 * or `alpha * A^T * A + beta * C` when @A is wrapped in `transpose(A)`.
 *
 * Return: `val_t` holding the destination matrix @C name, or empty string on error.
 */
val_t cli_calc_func_syrk(void)
{
    cli_token *(*cur_func)(void)     = cli_calc_cur_func();
    cli_token *(*advance_func)(void) = cli_calc_advance_func();
    val_t argA;
    val_t argC;
    int   transA = 0;

    val_t av = parse_expr(0);
    if (parse_error || eval_error || av.type == VAL_STRING)
    {
        parse_errmsg("syrk: expected syrk(alpha, A, beta, C)");
        return mk_string("");
    }
    double alpha = to_double(av);
    if (cur_func()->type != TOK_COMMA)
    {
        parse_errmsg("Expected ','");
        return mk_string("");
    }
    advance_func();
    if (cli_calc_parse_matmul_arg(&argA, &transA) != 0)
    {
        return mk_string("");
    }
    if (cur_func()->type != TOK_COMMA)
    {
        parse_errmsg("syrk: expected syrk(alpha, A, beta, C)");
        return mk_string("");
    }
    advance_func();
    val_t bv = parse_expr(0);
    if (parse_error || eval_error || bv.type == VAL_STRING)
    {
        parse_errmsg("syrk: beta must be a scalar");
        return mk_string("");
    }
    double beta = to_double(bv);
    if (cur_func()->type != TOK_COMMA)
    {
        parse_errmsg("syrk: expected destination C after beta");
        return mk_string("");
    }
    advance_func();
    argC = parse_expr(0);
    if (parse_error || eval_error)
    {
        return mk_string("");
    }
    if (cur_func()->type != TOK_RPAREN)
    {
        parse_errmsg("Expected ')'");
        return mk_string("");
    }
    advance_func();

    if (argA.type != VAL_STRING || argC.type != VAL_STRING || !check_image(argA.sval) ||
        !check_image(argC.sval) || cli_calc_reject_complex(&argA, 1) ||
        cli_calc_reject_complex(&argC, 1))
    {
        parse_errmsg("syrk: A and C must be real matrix images");
        return mk_string("");
    }
    IMAGE *imA = &data.core.image[image_ID(argA.sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE *imC = &data.core.image[image_ID(argC.sval, data.core.image, data.core.NB_MAX_IMAGE)];
    if (imC == imA)
    {
        parse_errmsg("syrk: destination C must not alias A");
        return mk_string("");
    }
    if (cli_calc_syrk_check(imA, transA, imC) != 0)
    {
        parse_errmsg("syrk: C must be an N x N float/double matrix");
        return mk_string("");
    }
    imC->md[0].write = 1;
    if (cli_calc_syrk_into(imA, transA, alpha, beta, imC) != 0)
    {
        imC->md[0].write = 0;
        return mk_string("");
    }
    processinfo_update_output_stream(NULL, imC, NULL);
    return mk_string(imC->name);
}

/**
 * cli_calc_func_symv - Evaluate strict BLAS symmetric matrix-vector product `symv(alpha, A, x, beta, y)`
 *
 * Updates length-`N` vector @y in-place with `alpha * A * x + beta * y` using
 * the upper triangle of symmetric `N x N` matrix @A.
 *
 * Return: `val_t` holding the destination vector @y name, or empty string on error.
 */
val_t cli_calc_func_symv(void)
{
    val_t args[5];
    int   nargs = cli_calc_parse_func_argv(5, 5, args, "symv: expected ','");
    if (nargs < 0 || cli_calc_reject_complex(args, 5))
    {
        return mk_string("");
    }
    if (args[0].type == VAL_STRING || args[1].type != VAL_STRING || args[2].type != VAL_STRING ||
        args[3].type == VAL_STRING || args[4].type != VAL_STRING || !check_image(args[1].sval) ||
        !check_image(args[2].sval) || !check_image(args[4].sval))
    {
        parse_errmsg("symv: expected symv(alpha, A, x, beta, y)");
        return mk_string("");
    }
    double alpha = to_double(args[0]);
    double beta  = to_double(args[3]);
    IMAGE *imA = &data.core.image[image_ID(args[1].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE *imX = &data.core.image[image_ID(args[2].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE *imY = &data.core.image[image_ID(args[4].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    if (imY == imA || imY == imX)
    {
        parse_errmsg("symv: destination y must not alias A or x");
        return mk_string("");
    }
    uint64_t N   = imA->md[0].size[0];
    uint64_t lx  = 0;
    uint64_t ly  = 0;
    uint8_t  dty = imY->md[0].datatype;
    if (imA->md[0].naxis != 2 || imA->md[0].size[1] != N || !is_1d_vec(imX, &lx) || lx != N ||
        !is_1d_vec(imY, &ly) || ly != N || (dty != _DATATYPE_FLOAT && dty != _DATATYPE_DOUBLE))
    {
        parse_errmsg("symv: expected N x N matrix A and length-N vectors x, y (float/double)");
        return mk_string("");
    }
    uint8_t dta      = imA->md[0].datatype;
    uint8_t dtx      = imX->md[0].datatype;
    imY->md[0].write = 1;

#ifdef HAVE_BLAS
    if (dta == _DATATYPE_FLOAT && dtx == _DATATYPE_FLOAT && dty == _DATATYPE_FLOAT)
    {
        cblas_ssymv(CblasColMajor, CblasUpper, (int) N, (float) alpha, imA->array.F, (int) N,
                    imX->array.F, 1, (float) beta, imY->array.F, 1);
        processinfo_update_output_stream(NULL, imY, NULL);
        return mk_string(imY->name);
    }
    if (dta == _DATATYPE_DOUBLE && dtx == _DATATYPE_DOUBLE && dty == _DATATYPE_DOUBLE)
    {
        cblas_dsymv(CblasColMajor, CblasUpper, (int) N, alpha, imA->array.D, (int) N, imX->array.D,
                    1, beta, imY->array.D, 1);
        processinfo_update_output_stream(NULL, imY, NULL);
        return mk_string(imY->name);
    }
#endif

    for (uint64_t i = 0; i < N; i++)
    {
        double acc = 0.0;
        for (uint64_t j = 0; j < N; j++)
        {
            uint64_t r   = (i <= j) ? i : j;
            uint64_t c   = (i <= j) ? j : i;
            double   aij = cli_calc_read_im_pix(imA, dta, c * N + r);
            acc += aij * cli_calc_read_im_pix(imX, dtx, j);
        }
        double y0 = (beta != 0.0) ? (beta * cli_calc_read_im_pix(imY, dty, i)) : 0.0;
        if (dty == _DATATYPE_DOUBLE)
        {
            imY->array.D[i] = alpha * acc + y0;
        }
        else
        {
            imY->array.F[i] = (float) (alpha * acc + y0);
        }
    }
    processinfo_update_output_stream(NULL, imY, NULL);
    return mk_string(imY->name);
}

/**
 * cli_calc_func_axpy - Evaluate strict BLAS `axpy(alpha, x, y)` updating `y = alpha * x + y` in-place
 *
 * Return: `val_t` holding the destination image @y name, or empty string on error.
 */
val_t cli_calc_func_axpy(void)
{
    val_t args[3];
    if (cli_calc_parse_func_argv(3, 3, args, "axpy: expected ','") != 3 ||
        cli_calc_reject_complex(args, 3))
    {
        return mk_string("");
    }
    if (args[0].type == VAL_STRING || args[1].type != VAL_STRING || args[2].type != VAL_STRING ||
        !check_image(args[1].sval) || !check_image(args[2].sval))
    {
        parse_errmsg("axpy: expected axpy(alpha, x, y)");
        return mk_string("");
    }
    double alpha = to_double(args[0]);
    IMAGE *imX = &data.core.image[image_ID(args[1].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE *imY = &data.core.image[image_ID(args[2].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    if (imY == imX)
    {
        parse_errmsg("axpy: destination y must not alias x");
        return mk_string("");
    }
    uint64_t nel = imY->md[0].nelement;
    uint8_t  dtx = imX->md[0].datatype;
    uint8_t  dty = imY->md[0].datatype;
    if (imX->md[0].nelement != nel || (dty != _DATATYPE_FLOAT && dty != _DATATYPE_DOUBLE))
    {
        parse_errmsg("axpy: x and y must have matching element counts and float/double y");
        return mk_string("");
    }
    imY->md[0].write = 1;
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
        processinfo_update_output_stream(NULL, imY, NULL);
        return mk_string(imY->name);
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
        processinfo_update_output_stream(NULL, imY, NULL);
        return mk_string(imY->name);
    }
    for (uint64_t i = 0; i < nel; i++)
    {
        double v = alpha * cli_calc_read_im_pix(imX, dtx, i) + cli_calc_read_im_pix(imY, dty, i);
        if (dty == _DATATYPE_DOUBLE)
        {
            imY->array.D[i] = v;
        }
        else
        {
            imY->array.F[i] = (float) v;
        }
    }
    processinfo_update_output_stream(NULL, imY, NULL);
    return mk_string(imY->name);
}

/**
 * cli_calc_func_scal - Evaluate strict BLAS `scal(alpha, x)` scaling @x in-place by @alpha
 *
 * Return: `val_t` holding the destination image @x name, or empty string on error.
 */
val_t cli_calc_func_scal(void)
{
    val_t args[2];
    if (cli_calc_parse_func_argv(2, 2, args, "scal: expected ','") != 2 ||
        cli_calc_reject_complex(args, 2))
    {
        return mk_string("");
    }
    if (args[0].type == VAL_STRING || args[1].type != VAL_STRING || !check_image(args[1].sval))
    {
        parse_errmsg("scal: expected scal(alpha, x)");
        return mk_string("");
    }
    double alpha = to_double(args[0]);
    IMAGE *imX = &data.core.image[image_ID(args[1].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    uint64_t nel = imX->md[0].nelement;
    uint8_t  dtx = imX->md[0].datatype;
    if (dtx != _DATATYPE_FLOAT && dtx != _DATATYPE_DOUBLE)
    {
        parse_errmsg("scal: x must be a float or double image");
        return mk_string("");
    }
    imX->md[0].write = 1;
    if (dtx == _DATATYPE_FLOAT)
    {
        float *restrict X = imX->array.F;
        float af          = (float) alpha;
#pragma omp parallel for simd schedule(static) if (nel >= 65536)
        for (uint64_t i = 0; i < nel; i++)
        {
            X[i] *= af;
        }
    }
    else
    {
        double *restrict X = imX->array.D;
#pragma omp parallel for simd schedule(static) if (nel >= 65536)
        for (uint64_t i = 0; i < nel; i++)
        {
            X[i] *= alpha;
        }
    }
    processinfo_update_output_stream(NULL, imX, NULL);
    return mk_string(imX->name);
}

/**
 * cli_calc_func_gemm_op - Zero-argument builtin entry point for `gemm(alpha, A, B, beta, C)`
 */
val_t cli_calc_func_gemm_op(void)
{
    return cli_calc_func_gemm(0);
}

/**
 * cli_calc_func_gemv_op - Zero-argument builtin entry point for `gemv(alpha, A, x, beta, y)`
 */
val_t cli_calc_func_gemv_op(void)
{
    return cli_calc_func_gemm(1);
}

/* ------------------------------------------- kept matrix assignment lowering */

enum
{
    FKEPT_ELEMWISE  = 0,
    FKEPT_MATMUL    = 1,
    FKEPT_TRANSPOSE = 2,
    FKEPT_RESHAPE   = 3,
    FKEPT_GER       = 4,
    FKEPT_SYRK      = 5
};

/**
 * parse_kept_img_or_trans - Parse `im` or `transpose(im)` as an operand of a kept matrix op
 * @out_im:    Output pointer to the resolved `IMAGE`
 * @out_trans: Output flag set to 1 if wrapped in `transpose(...)`, 0 otherwise
 *
 * Return: 0 on success, -1 if the tokens do not match `im` or `transpose(im)`
 */
static int parse_kept_img_or_trans(IMAGE **out_im, int *out_trans)
{
    int trans = 0;
    if (cur_eval()->type == TOK_FUNC_IMIM_IM && strcmp(cur_eval()->sval, "transpose(") == 0)
    {
        advance_eval();
        trans = 1;
    }
    if (cur_eval()->type != TOK_IMAGE || strchr(cur_eval()->sval, '[') != NULL)
    {
        return -1;
    }
    imageID id = image_ID(cur_eval()->sval, data.core.image, data.core.NB_MAX_IMAGE);
    if (id == -1)
    {
        return -1;
    }
    advance_eval();
    if (trans)
    {
        if (cur_eval()->type != TOK_RPAREN)
        {
            return -1;
        }
        advance_eval();
    }
    *out_im    = &data.core.image[id];
    *out_trans = trans;
    return 0;
}

/**
 * parse_kept_scalar_factor - Parse a scalar constant or live variable factor for BLAS alpha/beta
 * @keep_vars: Non-zero to record live variable indices instead of folding their current value
 * @out_c:     Output constant multiplier
 * @out_s1:    Output 1-based live variable slot (`0` when constant)
 *
 * Return: 0 on success, -1 if the current token is not a scalar factor.
 */
static int parse_kept_scalar_factor(int keep_vars, double *out_c, int16_t *out_s1)
{
    int    pos  = cli_calc_tok_pos();
    double sign = 1.0;
    while (cur_eval()->type == TOK_OP_MINUS)
    {
        advance_eval();
        sign = -sign;
    }
    if (cur_eval()->type == TOK_LONG)
    {
        *out_c  = sign * (double) cur_eval()->val_l;
        *out_s1 = 0;
        advance_eval();
        return 0;
    }
    if (cur_eval()->type == TOK_DOUBLE)
    {
        *out_c  = sign * cur_eval()->val_d;
        *out_s1 = 0;
        advance_eval();
        return 0;
    }
    if ((cur_eval()->type == TOK_VAR || cur_eval()->type == TOK_NVAR) && !cur_eval()->literal &&
        image_ID(cur_eval()->sval, data.core.image, data.core.NB_MAX_IMAGE) == -1)
    {
        if (keep_vars)
        {
            long vid = variable_ID(cur_eval()->sval);
            if (vid >= 0 && data.core.variable[vid].type != 2)
            {
                *out_c  = sign;
                *out_s1 = (int16_t) (vid + 1);
                advance_eval();
                return 0;
            }
        }
        if (!(cur_eval()[1].type >= TOK_EQUAL && cur_eval()[1].type <= TOK_OP_SLASH_EQ))
        {
            val_t v = eval_existing_var_ref(cur_eval()->sval);
            if (v.type == VAL_LONG || v.type == VAL_DOUBLE)
            {
                *out_c  = sign * to_double(v);
                *out_s1 = 0;
                advance_eval();
                return 0;
            }
        }
    }
    cli_calc_tok_seek(pos);
    return -1;
}

/**
 * parse_kept_dst_term - Parse `[scalar *] dst [* scalar]` representing `beta * dst`
 */
static int parse_kept_dst_term(int          keep_vars,
                               const IMAGE *dst,
                               double       sign,
                               double      *beta_c,
                               int16_t     *beta_s1)
{
    int pos = cli_calc_tok_pos();
    if (cur_eval()->type == TOK_OP_MINUS)
    {
        advance_eval();
        sign = -sign;
    }
    if (cur_eval()->type == TOK_IMAGE && strcmp(cur_eval()->sval, dst->name) == 0)
    {
        advance_eval();
        double  c  = 1.0;
        int16_t s1 = 0;
        if (cur_eval()->type == TOK_OP_STAR)
        {
            advance_eval();
            if (parse_kept_scalar_factor(keep_vars, &c, &s1) != 0)
            {
                cli_calc_tok_seek(pos);
                return -1;
            }
        }
        *beta_c  = sign * c;
        *beta_s1 = s1;
        return 0;
    }
    double  c  = 1.0;
    int16_t s1 = 0;
    if (parse_kept_scalar_factor(keep_vars, &c, &s1) == 0 && cur_eval()->type == TOK_OP_STAR)
    {
        advance_eval();
        if (cur_eval()->type == TOK_IMAGE && strcmp(cur_eval()->sval, dst->name) == 0)
        {
            advance_eval();
            *beta_c  = sign * c;
            *beta_s1 = s1;
            return 0;
        }
    }
    cli_calc_tok_seek(pos);
    return -1;
}

/**
 * parse_kept_alpha_arg - Parse leading `alpha,` argument inside `gemm(`, `gemv(`, `ger(`, or `syrk(`
 */
static int parse_kept_alpha_arg(int keep_vars, double *alpha_c, int16_t *alpha_s1)
{
    double  ac  = 1.0;
    int16_t as1 = 0;
    if (parse_kept_scalar_factor(keep_vars, &ac, &as1) != 0 || (*alpha_s1 != 0 && as1 != 0) ||
        cur_eval()->type != TOK_COMMA)
    {
        return -1;
    }
    *alpha_c *= ac;
    if (as1 != 0)
    {
        *alpha_s1 = as1;
    }
    advance_eval();
    return 0;
}

/**
 * parse_kept_beta_dst_args - Parse trailing `[beta,] dst)` arguments of `gemm`, `gemv`, `ger`, or `syrk`
 */
static int parse_kept_beta_dst_args(cli_calc_kept_matrix *m,
                                    const IMAGE          *dst,
                                    int                   keep_vars,
                                    int                   has_beta)
{
    double  bc  = 1.0;
    int16_t bs1 = 0;
    if (has_beta)
    {
        if (parse_kept_scalar_factor(keep_vars, &bc, &bs1) != 0 || cur_eval()->type != TOK_COMMA)
        {
            return -1;
        }
        advance_eval();
    }
    if (cur_eval()->type != TOK_IMAGE || strcmp(cur_eval()->sval, dst->name) != 0 ||
        m->beta_c != 0.0 || m->beta_s1 != 0)
    {
        return -1;
    }
    advance_eval();
    if (cur_eval()->type != TOK_RPAREN)
    {
        return -1;
    }
    advance_eval();
    m->beta_c  = bc;
    m->beta_s1 = bs1;
    return 0;
}

/**
 * parse_kept_blas_term - Parse `[alpha *] (matmul|gemm|gemv|ger|outer|syrk)(...) [* alpha2]`
 */
static int parse_kept_blas_term(cli_calc_kept_matrix *m,
                                const IMAGE          *dst,
                                double                sign,
                                int                   keep_vars)
{
    int pos = cli_calc_tok_pos();
    while (cur_eval()->type == TOK_OP_MINUS)
    {
        advance_eval();
        sign = -sign;
    }
    double  alpha_c  = sign;
    int16_t alpha_s1 = 0;
    int     p2       = cli_calc_tok_pos();
    double  fc       = 1.0;
    int16_t fs1      = 0;
    if (parse_kept_scalar_factor(keep_vars, &fc, &fs1) == 0)
    {
        if (cur_eval()->type == TOK_OP_STAR)
        {
            advance_eval();
            alpha_c *= fc;
            alpha_s1 = fs1;
        }
        else
        {
            cli_calc_tok_seek(p2);
        }
    }
    while (cur_eval()->type == TOK_OP_MINUS)
    {
        advance_eval();
        alpha_c = -alpha_c;
    }
    if (cur_eval()->type != TOK_FUNC_IMIM_IM)
    {
        cli_calc_tok_seek(pos);
        return -1;
    }
    const char *fn = cur_eval()->sval;
    if (strcmp(fn, "matmul(") == 0)
    {
        advance_eval();
        IMAGE *imA    = NULL;
        IMAGE *imB    = NULL;
        int    transA = 0;
        int    transB = 0;
        if (parse_kept_img_or_trans(&imA, &transA) != 0 || cur_eval()->type != TOK_COMMA)
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
        advance_eval();
        if (parse_kept_img_or_trans(&imB, &transB) != 0 || cur_eval()->type != TOK_RPAREN)
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
        advance_eval();
        if (imA == imB && transA != transB && cli_calc_syrk_check(imA, transA, dst) == 0)
        {
            m->kind   = FKEPT_SYRK;
            m->transA = transA;
            m->img[0] = imA;
            m->nimg   = 1;
        }
        else if (cli_calc_matmul_check(imA, transA, imB, transB, dst, 0) == 0)
        {
            m->kind          = FKEPT_MATMUL;
            m->transA        = transA;
            m->transB        = transB;
            m->allow_reshape = 0;
            m->img[0]        = imA;
            m->img[1]        = imB;
            m->nimg          = 2;
        }
        else
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
    }
    else if (strcmp(fn, "outer(") == 0)
    {
        advance_eval();
        IMAGE *imX = NULL;
        IMAGE *imY = NULL;
        int    tX  = 0;
        int    tY  = 0;
        if (parse_kept_img_or_trans(&imX, &tX) != 0 || tX || cur_eval()->type != TOK_COMMA)
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
        advance_eval();
        if (parse_kept_img_or_trans(&imY, &tY) != 0 || tY || cur_eval()->type != TOK_RPAREN ||
            cli_calc_ger_check(imX, imY, dst) != 0)
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
        advance_eval();
        m->kind   = FKEPT_GER;
        m->img[0] = imX;
        m->img[1] = imY;
        m->nimg   = 2;
    }
    else if (strcmp(fn, "gemm(") == 0 || strcmp(fn, "gemv(") == 0)
    {
        int is_gemv = (strcmp(fn, "gemv(") == 0);
        advance_eval();
        IMAGE *imA    = NULL;
        IMAGE *imB    = NULL;
        int    transA = 0;
        int    transB = 0;
        if (parse_kept_alpha_arg(keep_vars, &alpha_c, &alpha_s1) != 0 ||
            parse_kept_img_or_trans(&imA, &transA) != 0 || cur_eval()->type != TOK_COMMA)
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
        advance_eval();
        if (parse_kept_img_or_trans(&imB, &transB) != 0 || cur_eval()->type != TOK_COMMA)
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
        advance_eval();
        if (parse_kept_beta_dst_args(m, dst, keep_vars, 1) != 0 ||
            cli_calc_matmul_check(imA, transA, imB, transB, dst, 0) != 0 ||
            (is_gemv && dst->md[0].naxis == 2 && dst->md[0].size[1] != 1 && imA->md[0].naxis != 3))
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
        m->kind          = FKEPT_MATMUL;
        m->transA        = transA;
        m->transB        = transB;
        m->allow_reshape = 0;
        m->img[0]        = imA;
        m->img[1]        = imB;
        m->nimg          = 2;
    }
    else if (strcmp(fn, "ger(") == 0)
    {
        advance_eval();
        IMAGE *imX = NULL;
        IMAGE *imY = NULL;
        int    tX  = 0;
        int    tY  = 0;
        if (parse_kept_alpha_arg(keep_vars, &alpha_c, &alpha_s1) != 0 ||
            parse_kept_img_or_trans(&imX, &tX) != 0 || tX || cur_eval()->type != TOK_COMMA)
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
        advance_eval();
        if (parse_kept_img_or_trans(&imY, &tY) != 0 || tY || cur_eval()->type != TOK_COMMA)
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
        advance_eval();
        if (parse_kept_beta_dst_args(m, dst, keep_vars, 0) != 0 ||
            cli_calc_ger_check(imX, imY, dst) != 0)
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
        m->kind   = FKEPT_GER;
        m->img[0] = imX;
        m->img[1] = imY;
        m->nimg   = 2;
    }
    else if (strcmp(fn, "syrk(") == 0)
    {
        advance_eval();
        IMAGE *imA    = NULL;
        int    transA = 0;
        if (parse_kept_alpha_arg(keep_vars, &alpha_c, &alpha_s1) != 0 ||
            parse_kept_img_or_trans(&imA, &transA) != 0 || cur_eval()->type != TOK_COMMA)
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
        advance_eval();
        if (parse_kept_beta_dst_args(m, dst, keep_vars, 1) != 0 ||
            cli_calc_syrk_check(imA, transA, dst) != 0)
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
        m->kind   = FKEPT_SYRK;
        m->transA = transA;
        m->img[0] = imA;
        m->nimg   = 1;
    }
    else
    {
        cli_calc_tok_seek(pos);
        return -1;
    }

    if (cur_eval()->type == TOK_OP_STAR)
    {
        advance_eval();
        double  tc  = 1.0;
        int16_t ts1 = 0;
        if (parse_kept_scalar_factor(keep_vars, &tc, &ts1) != 0 || (alpha_s1 != 0 && ts1 != 0))
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
        alpha_c *= tc;
        if (ts1 != 0)
        {
            alpha_s1 = ts1;
        }
    }
    m->alpha_c  = alpha_c;
    m->alpha_s1 = alpha_s1;
    return 0;
}

/**
 * parse_kept_reshape_dims - Parse and validate target dimensions of a kept `reshape(...)` call
 */
static int parse_kept_reshape_dims(uint64_t nel, const IMAGE *dst)
{
    int      ndims    = 0;
    long     dims[3]  = { 1, 1, 1 };
    int      neg_idx  = -1;
    uint64_t prod_pos = 1;

    while (cur_eval()->type == TOK_COMMA && ndims < 3)
    {
        advance_eval();
        double dv = 0.0;
        if (cli_calc_eval_const_scalar(&dv) != 0)
        {
            return -1;
        }
        long d = llround(dv);
        if (d == -1)
        {
            if (neg_idx != -1)
            {
                return -1;
            }
            neg_idx = ndims;
        }
        else if (d <= 0)
        {
            return -1;
        }
        else
        {
            dims[ndims] = d;
            prod_pos *= (uint64_t) d;
        }
        ndims++;
    }
    if (ndims == 0 || cur_eval()->type != TOK_RPAREN)
    {
        return -1;
    }
    advance_eval();
    if (cur_eval()->type != TOK_EOF && cur_eval()->type != TOK_NEWLINE)
    {
        return -1;
    }
    if (neg_idx != -1)
    {
        if (prod_pos == 0 || nel % prod_pos != 0)
        {
            return -1;
        }
        dims[neg_idx] = (long) (nel / prod_pos);
        prod_pos      = nel;
    }
    if (prod_pos != nel || dst->md[0].naxis != ndims)
    {
        return -1;
    }
    for (int i = 0; i < ndims; i++)
    {
        if (dst->md[0].size[i] != (uint32_t) dims[i])
        {
            return -1;
        }
    }
    return 0;
}

/**
 * cli_calc_compile_kept_matrix - Try to compile a top-level matrix or reshape assignment into @m
 * @m:               Kept-matrix descriptor to populate
 * @dst:             Destination image assigned by the statement
 * @aop:             Assignment operator (`=`, `+=`, or `-=`)
 * @keep_vars:       Non-zero to record live scalar variable indices for `alpha` / `beta`
 * @strict_local_dt: Non-zero to require local @dst datatype to match natural output precision
 *
 * Recognizes `dst = transpose(A)`, `dst = reshape(...)`, and BLAS forms
 * (`dst = alpha * matmul(A, B) + beta * dst`, `dst += alpha * matmul(A, B)`,
 * `ger`, `syrk`, `gemm`, `gemv`) so both one-shot `@calc` and `milk.calc`
 * execute them directly into @dst without temporary allocations.
 *
 * Return: 0 on successful compilation, -1 if the RHS is not a supported matrix/reshape op
 */
int cli_calc_compile_kept_matrix(cli_calc_kept_matrix *m,
                                 IMAGE                *dst,
                                 cli_token_type        aop,
                                 int                   keep_vars,
                                 int                   strict_local_dt)
{
    memset(m, 0, sizeof(*m));
    int pos = cli_calc_tok_pos();
    if (aop != TOK_EQUAL && aop != TOK_OP_PLUS_EQ && aop != TOK_OP_MINUS_EQ)
    {
        return -1;
    }
    if (aop == TOK_EQUAL && cur_eval()->type == TOK_FUNC_IMIM_IM)
    {
        if (strcmp(cur_eval()->sval, "transpose(") == 0)
        {
            advance_eval();
            if (cur_eval()->type != TOK_IMAGE || strchr(cur_eval()->sval, '[') != NULL)
            {
                cli_calc_tok_seek(pos);
                return -1;
            }
            imageID id = image_ID(cur_eval()->sval, data.core.image, data.core.NB_MAX_IMAGE);
            advance_eval();
            if (id == -1 || cur_eval()->type != TOK_RPAREN)
            {
                cli_calc_tok_seek(pos);
                return -1;
            }
            advance_eval();
            IMAGE  *src = &data.core.image[id];
            uint8_t exp_dt =
                (src->md[0].datatype == _DATATYPE_DOUBLE) ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT;
            if ((cur_eval()->type != TOK_EOF && cur_eval()->type != TOK_NEWLINE) ||
                cli_calc_transpose_check(src, dst) != 0 ||
                (strict_local_dt && dst->md[0].datatype != exp_dt))
            {
                cli_calc_tok_seek(pos);
                return -1;
            }
            m->kind   = FKEPT_TRANSPOSE;
            m->img[0] = src;
            m->nimg   = 1;
            return 0;
        }
        if (strcmp(cur_eval()->sval, "reshape(") == 0)
        {
            advance_eval();
            if (cur_eval()->type == TOK_FUNC_IMIM_IM && strcmp(cur_eval()->sval, "matmul(") == 0)
            {
                advance_eval();
                IMAGE *imA    = NULL;
                IMAGE *imB    = NULL;
                int    transA = 0;
                int    transB = 0;
                if (parse_kept_img_or_trans(&imA, &transA) != 0 || cur_eval()->type != TOK_COMMA)
                {
                    cli_calc_tok_seek(pos);
                    return -1;
                }
                advance_eval();
                if (parse_kept_img_or_trans(&imB, &transB) != 0 || cur_eval()->type != TOK_RPAREN)
                {
                    cli_calc_tok_seek(pos);
                    return -1;
                }
                advance_eval();
                uint8_t exp_dt = (imA->md[0].datatype == _DATATYPE_DOUBLE ||
                                  imB->md[0].datatype == _DATATYPE_DOUBLE)
                                     ? _DATATYPE_DOUBLE
                                     : _DATATYPE_FLOAT;
                if (cli_calc_matmul_check(imA, transA, imB, transB, dst, 1) != 0 ||
                    parse_kept_reshape_dims(dst->md[0].nelement, dst) != 0 ||
                    (strict_local_dt && dst->md[0].datatype != exp_dt))
                {
                    cli_calc_tok_seek(pos);
                    return -1;
                }
                m->kind          = FKEPT_MATMUL;
                m->transA        = transA;
                m->transB        = transB;
                m->allow_reshape = 1;
                m->alpha_c       = 1.0;
                m->beta_c        = 0.0;
                m->img[0]        = imA;
                m->img[1]        = imB;
                m->nimg          = 2;
                return 0;
            }
            if (cur_eval()->type == TOK_IMAGE && strchr(cur_eval()->sval, '[') == NULL)
            {
                imageID id = image_ID(cur_eval()->sval, data.core.image, data.core.NB_MAX_IMAGE);
                advance_eval();
                if (id == -1 || cli_calc_reshape_check(&data.core.image[id], dst) != 0 ||
                    parse_kept_reshape_dims(dst->md[0].nelement, dst) != 0 ||
                    (strict_local_dt && dst->md[0].datatype != data.core.image[id].md[0].datatype))
                {
                    cli_calc_tok_seek(pos);
                    return -1;
                }
                m->kind   = FKEPT_RESHAPE;
                m->img[0] = &data.core.image[id];
                m->nimg   = 1;
                return 0;
            }
            cli_calc_tok_seek(pos);
            return -1;
        }
    }

    cli_calc_tok_seek(pos);
    m->alpha_c  = 1.0;
    m->alpha_s1 = 0;
    m->beta_c   = (aop == TOK_EQUAL) ? 0.0 : 1.0;
    m->beta_s1  = 0;
    double sign = (aop == TOK_OP_MINUS_EQ) ? -1.0 : 1.0;

    if (aop == TOK_EQUAL)
    {
        double  bc  = 0.0;
        int16_t bs1 = 0;
        if (parse_kept_dst_term(keep_vars, dst, 1.0, &bc, &bs1) == 0)
        {
            if (cur_eval()->type != TOK_OP_PLUS && cur_eval()->type != TOK_OP_MINUS)
            {
                cli_calc_tok_seek(pos);
                return -1;
            }
            sign = (cur_eval()->type == TOK_OP_MINUS) ? -1.0 : 1.0;
            advance_eval();
            m->beta_c  = bc;
            m->beta_s1 = bs1;
            if (parse_kept_blas_term(m, dst, sign, keep_vars) != 0)
            {
                cli_calc_tok_seek(pos);
                return -1;
            }
        }
        else
        {
            if (parse_kept_blas_term(m, dst, 1.0, keep_vars) != 0)
            {
                cli_calc_tok_seek(pos);
                return -1;
            }
            if ((cur_eval()->type == TOK_OP_PLUS || cur_eval()->type == TOK_OP_MINUS) &&
                m->beta_c == 0.0 && m->beta_s1 == 0)
            {
                double bsign = (cur_eval()->type == TOK_OP_MINUS) ? -1.0 : 1.0;
                advance_eval();
                if (parse_kept_dst_term(keep_vars, dst, bsign, &m->beta_c, &m->beta_s1) != 0)
                {
                    cli_calc_tok_seek(pos);
                    return -1;
                }
            }
        }
    }
    else
    {
        if (parse_kept_blas_term(m, dst, sign, keep_vars) != 0)
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
    }

    if (cur_eval()->type != TOK_EOF && cur_eval()->type != TOK_NEWLINE)
    {
        cli_calc_tok_seek(pos);
        return -1;
    }
    if (strict_local_dt)
    {
        int any_d = 0;
        for (int i = 0; i < m->nimg; i++)
        {
            if (m->img[i]->md[0].datatype == _DATATYPE_DOUBLE)
            {
                any_d = 1;
            }
        }
        uint8_t exp_dt = any_d ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT;
        if (dst->md[0].datatype != exp_dt)
        {
            cli_calc_tok_seek(pos);
            return -1;
        }
    }
    return 0;
}

static double read_kept_var(int16_t s1)
{
    const VARIABLE *v = &data.core.variable[s1 - 1];
    return (v->type == 1) ? (double) v->value.l : v->value.f;
}

/**
 * cli_calc_exec_kept_matrix - Execute a compiled matrix or reshape operation directly into @dst
 * @m:   Compiled `cli_calc_kept_matrix` descriptor
 * @dst: Destination image to write
 *
 * Return: 0 on success, -1 if an image changed shape/type or execution failed.
 */
int cli_calc_exec_kept_matrix(const cli_calc_kept_matrix *m, IMAGE *dst)
{
    double      alpha  = (m->alpha_s1 > 0) ? (read_kept_var(m->alpha_s1) * m->alpha_c) : m->alpha_c;
    double      beta   = (m->beta_s1 > 0) ? (read_kept_var(m->beta_s1) * m->beta_c) : m->beta_c;
    const char *opname = "matrix";
    int         chk    = -1;

    switch (m->kind)
    {
    case FKEPT_MATMUL:
        opname = "matmul";
        chk    = cli_calc_matmul_check(m->img[0], m->transA, m->img[1], m->transB, dst,
                                       m->allow_reshape);
        break;
    case FKEPT_GER:
        opname = "ger";
        chk    = cli_calc_ger_check(m->img[0], m->img[1], dst);
        break;
    case FKEPT_SYRK:
        opname = "syrk";
        chk    = cli_calc_syrk_check(m->img[0], m->transA, dst);
        break;
    case FKEPT_TRANSPOSE:
        opname = "transpose";
        chk    = cli_calc_transpose_check(m->img[0], dst);
        break;
    case FKEPT_RESHAPE:
        opname = "reshape";
        chk    = cli_calc_reshape_check(m->img[0], dst);
        break;
    default:
        return -1;
    }
    if (chk != 0)
    {
        snprintf(cli_calc_lasterr, sizeof(cli_calc_lasterr), "%s: %s or input changed size or type",
                 opname, dst->name);
        return -1;
    }

    dst->md[0].write = 1;
    int rc           = -1;
    switch (m->kind)
    {
    case FKEPT_MATMUL:
        rc = cli_calc_gemm_into(m->img[0], m->transA, m->img[1], m->transB, alpha, beta, dst);
        break;
    case FKEPT_GER:
        rc = cli_calc_ger_into(m->img[0], m->img[1], alpha, beta, dst);
        break;
    case FKEPT_SYRK:
        rc = cli_calc_syrk_into(m->img[0], m->transA, alpha, beta, dst);
        break;
    case FKEPT_TRANSPOSE:
        rc = cli_calc_transpose_into(m->img[0], dst);
        break;
    case FKEPT_RESHAPE:
        rc = cli_calc_reshape_into(m->img[0], dst);
        break;
    default:
        break;
    }
    if (rc != 0)
    {
        dst->md[0].write = 0;
        return -1;
    }
    processinfo_update_output_stream(NULL, dst, NULL);
    return 0;
}

#include "cli_calc_lapack.h"
