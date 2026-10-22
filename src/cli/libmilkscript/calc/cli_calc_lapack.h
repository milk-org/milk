/**
 * @file    cli_calc_lapack.h
 * @brief   BLAS triangular solvers and LAPACK linear system / inverse / SVD functions
 *
 * Implements triangular solvers (`trsv`, `trsm`), general LU solver (`gesv`),
 * symmetric positive-definite Cholesky solver (`posv`), QR/LQ least-squares solver
 * (`gels`), unified solver (`solve` with optional Tikhonov regularization), matrix
 * inverse (`inv`), and SVD Moore-Penrose pseudo-inverse (`pinv`).
 */

#ifndef CLI_CALC_LAPACK_H
#define CLI_CALC_LAPACK_H

#include <ctype.h>

/**
 * copy_im_to_dmat - Copy an image into a newly allocated column-major `double` matrix
 * @im:     Source image
 * @rows:   Matrix row count (before optional transpose)
 * @cols:   Matrix column count (before optional transpose)
 * @trans:  Non-zero to transpose during copy (producing a `cols x rows` matrix)
 *
 * Return: Newly allocated `double` buffer of `rows * cols` elements, or NULL on failure.
 */
static double *copy_im_to_dmat(const IMAGE *im, uint64_t rows, uint64_t cols, int trans)
{
    uint64_t nel = rows * cols;
    double  *buf = (double *) malloc((size_t) (nel > 0 ? nel : 1) * sizeof(double));
    if (!buf)
    {
        return NULL;
    }
    uint8_t dt = im->md[0].datatype;
    if (!trans)
    {
        if (dt == _DATATYPE_FLOAT)
        {
            const float *restrict src = im->array.F;
            for (uint64_t i = 0; i < nel; i++)
            {
                buf[i] = (double) src[i];
            }
        }
        else if (dt == _DATATYPE_DOUBLE)
        {
            __builtin_memcpy(buf, im->array.D, (size_t) nel * sizeof(double));
        }
        else
        {
            for (uint64_t i = 0; i < nel; i++)
            {
                buf[i] = cli_calc_read_im_pix(im, dt, i);
            }
        }
    }
    else
    {
        for (uint64_t r = 0; r < rows; r++)
        {
            for (uint64_t c = 0; c < cols; c++)
            {
                buf[r * cols + c] = cli_calc_read_im_pix(im, dt, c * rows + r);
            }
        }
    }
    return buf;
}

/**
 * write_dmat_to_new_im - Create a temporary output image and populate it from a `double` buffer
 * @src:     Source `double` array of `nel` elements
 * @nel:     Total number of elements
 * @naxis:   Output image `naxis` (1, 2, or 3)
 * @outsz:   Output image dimensions `size[3]`
 * @out_dt:  Output datatype (`_DATATYPE_FLOAT` or `_DATATYPE_DOUBLE`)
 * @fn_name: Function name for error reporting
 *
 * Return: `val_t` holding the new temporary image name, or empty string on error.
 */
static val_t write_dmat_to_new_im(const double  *src,
                                  uint64_t       nel,
                                  long           naxis,
                                  const uint32_t outsz[3],
                                  uint8_t        out_dt,
                                  const char    *fn_name)
{
    const char *tmpn  = alloc_tmpname();
    imageID     tid   = -1;
    uint32_t    sz[3] = { outsz[0], outsz[1], outsz[2] };
    create_image_ID(tmpn, naxis, sz, out_dt, 0, 10, 0, &tid);
    if (tid == -1)
    {
        char msg[96];
        snprintf(msg, sizeof(msg), "%s: failed to allocate output image", fn_name);
        parse_errmsg(msg);
        return mk_string("");
    }
    IMAGE *out = &data.core.image[tid];
    if (out_dt == _DATATYPE_DOUBLE)
    {
        __builtin_memcpy(out->array.D, src, (size_t) nel * sizeof(double));
    }
    else
    {
        float *restrict dst = out->array.F;
        for (uint64_t i = 0; i < nel; i++)
        {
            dst[i] = (float) src[i];
        }
    }
    return mk_string(tmpn);
}

/**
 * resolve_rhs_dims - Validate and extract `(M, NRHS)` and output shape for solver RHS `B`
 * @imA:     Left-hand matrix image (already flattened to `M x N` after `transA`)
 * @is_3d_a: Non-zero if @imA is a 3-D mode cube `(W, H, N)` with `transA == 0`
 * @M:       Expected row count of the linear system
 * @N:       Solution row count (columns of $op(A)$)
 * @imB:     Right-hand side image
 * @out_rhs: Receives number of right-hand side columns `NRHS`
 * @out_nax: Receives output `naxis` for solution `X`
 * @out_sz:  Receives output `size[3]` for solution `X`
 *
 * Return: 0 if dimensions match, -1 on mismatch.
 */
static int resolve_rhs_dims(const IMAGE *imA,
                            int          is_3d_a,
                            uint64_t     M,
                            uint64_t     N,
                            const IMAGE *imB,
                            uint64_t    *out_rhs,
                            long        *out_nax,
                            uint32_t     out_sz[3])
{
    uint64_t b0      = 0;
    uint64_t b1      = 0;
    int      is_3d_b = 0;
    cli_calc_flatten2d(imB, &b0, &b1, &is_3d_b);

    if (is_3d_a && imB->md[0].naxis == 2 && M == (uint64_t) imB->md[0].size[0] * imB->md[0].size[1])
    {
        b0 = M;
        b1 = 1;
    }
    if (b0 != M || M == 0 || N == 0 || b1 == 0)
    {
        return -1;
    }
    *out_rhs  = b1;
    out_sz[0] = (uint32_t) N;
    out_sz[1] = (uint32_t) b1;
    out_sz[2] = 1;
    if (b1 == 1 && (imB->md[0].naxis == 1 || (is_3d_a && imB->md[0].naxis == 2)))
    {
        *out_nax = 1;
    }
    else
    {
        *out_nax = 2;
    }
    (void) imA;
    return 0;
}

/**
 * detect_uplo - Auto-detect whether a square `N x N` matrix is upper or lower triangular
 * @A: Column-major `N x N` matrix
 * @N: Matrix dimension
 *
 * Compares the L1 weight of the strict lower triangle against the strict upper
 * triangle. Defaults to upper (`1`) when both triangles are zero (diagonal matrix).
 *
 * Return: 1 if upper triangular, 0 if lower triangular.
 */
static int detect_uplo(const double *A, uint64_t N)
{
    double upper_sum = 0.0;
    double lower_sum = 0.0;
    for (uint64_t c = 0; c < N; c++)
    {
        const double *col = A + c * N;
        for (uint64_t r = 0; r < c; r++)
        {
            upper_sum += fabs(col[r]);
        }
        for (uint64_t r = c + 1; r < N; r++)
        {
            lower_sum += fabs(col[r]);
        }
    }
    return (lower_sum <= upper_sum) ? 1 : 0;
}

/**
 * parse_uplo_arg - Parse an optional BLAS `uplo` argument (`"U"`/`"L"` or numeric)
 * @v:        Argument value
 * @out_uplo: Receives 1 for upper (`'U'`), 0 for lower (`'L'`)
 *
 * Return: 0 on valid specification, -1 on error.
 */
static int parse_uplo_arg(val_t v, int *out_uplo)
{
    if (v.type == VAL_STRING)
    {
        char c = (char) toupper((unsigned char) v.sval[0]);
        if (c == 'U')
        {
            *out_uplo = 1;
            return 0;
        }
        if (c == 'L')
        {
            *out_uplo = 0;
            return 0;
        }
        return -1;
    }
    double d  = to_double(v);
    *out_uplo = (d > 0.0) ? 1 : 0;
    return 0;
}

/**
 * parse_diag_arg - Parse an optional BLAS `diag` argument (`"U"`/`"N"` or numeric)
 * @v:        Argument value
 * @out_unit: Receives 1 for unit diagonal (`'U'`), 0 for non-unit (`'N'`)
 *
 * Return: 0 on valid specification, -1 on error.
 */
static int parse_diag_arg(val_t v, int *out_unit)
{
    if (v.type == VAL_STRING)
    {
        char c = (char) toupper((unsigned char) v.sval[0]);
        if (c == 'U')
        {
            *out_unit = 1;
            return 0;
        }
        if (c == 'N')
        {
            *out_unit = 0;
            return 0;
        }
        return -1;
    }
    double d  = to_double(v);
    *out_unit = (d != 0.0) ? 1 : 0;
    return 0;
}

/**
 * solve_triangular_inplace - Solve $op(A) X = \alpha B$ in place in `X` (`N x NRHS`)
 * @A:        Column-major `N x N` triangular matrix
 * @N:        Matrix dimension
 * @transA:   Non-zero to solve $A^T X = \alpha B$ instead of $A X = \alpha B$
 * @is_upper: Non-zero if @A stores an upper-triangular matrix, 0 if lower-triangular
 * @is_unit:  Non-zero if @A has an implicit unit diagonal
 * @alpha:    Scalar multiplier applied to `X` before solving
 * @X:        Column-major `N x NRHS` right-hand side buffer, overwritten with solution
 * @NRHS:     Number of right-hand side columns
 *
 * Return: 0 on success, -1 if a zero diagonal element is encountered when `!is_unit`.
 */
static int solve_triangular_inplace(const double *restrict A,
                                    uint64_t N,
                                    int      transA,
                                    int      is_upper,
                                    int      is_unit,
                                    double   alpha,
                                    double *restrict X,
                                    uint64_t NRHS)
{
    if (alpha != 1.0)
    {
        uint64_t total = N * NRHS;
        for (uint64_t i = 0; i < total; i++)
        {
            X[i] *= alpha;
        }
    }
    if (!is_unit)
    {
        for (uint64_t i = 0; i < N; i++)
        {
            if (A[i * N + i] == 0.0)
            {
                return -1;
            }
        }
    }

    int eff_upper = transA ? !is_upper : is_upper;
    for (uint64_t rhs = 0; rhs < NRHS; rhs++)
    {
        double *restrict x = X + rhs * N;
        if (!eff_upper)
        {
            /* Forward substitution: row i = 0 .. N-1 */
            if (!transA)
            {
                /* A is lower triangular (column-major): update x[i+1..N-1] per column j */
                for (uint64_t j = 0; j < N; j++)
                {
                    if (!is_unit)
                    {
                        x[j] /= A[j * N + j];
                    }
                    double xj                  = x[j];
                    const double *restrict col = A + j * N;
                    if (xj != 0.0)
                    {
                        for (uint64_t i = j + 1; i < N; i++)
                        {
                            x[i] -= col[i] * xj;
                        }
                    }
                }
            }
            else
            {
                /* A is upper triangular, solving A^T x = b: row i of A^T is col i of A */
                for (uint64_t i = 0; i < N; i++)
                {
                    const double *restrict col_i = A + i * N;
                    double sum                   = x[i];
                    for (uint64_t j = 0; j < i; j++)
                    {
                        sum -= col_i[j] * x[j];
                    }
                    x[i] = is_unit ? sum : (sum / col_i[i]);
                }
            }
        }
        else
        {
            /* Backward substitution: row i = N-1 .. 0 */
            if (!transA)
            {
                /* A is upper triangular (column-major): update x[0..j-1] per column j */
                for (uint64_t jj = N; jj > 0; jj--)
                {
                    uint64_t j = jj - 1;
                    if (!is_unit)
                    {
                        x[j] /= A[j * N + j];
                    }
                    double xj                  = x[j];
                    const double *restrict col = A + j * N;
                    if (xj != 0.0)
                    {
                        for (uint64_t i = 0; i < j; i++)
                        {
                            x[i] -= col[i] * xj;
                        }
                    }
                }
            }
            else
            {
                /* A is lower triangular, solving A^T x = b: row i of A^T is col i of A */
                for (uint64_t ii = N; ii > 0; ii--)
                {
                    uint64_t i                   = ii - 1;
                    const double *restrict col_i = A + i * N;
                    double sum                   = x[i];
                    for (uint64_t j = i + 1; j < N; j++)
                    {
                        sum -= col_i[j] * x[j];
                    }
                    x[i] = is_unit ? sum : (sum / col_i[i]);
                }
            }
        }
    }
    return 0;
}

/**
 * cli_calc_func_trsv - Evaluate `trsv(A, b[, uplo[, diag]])` and `trsm(A, B[, uplo[, diag[, alpha]]])`
 * @is_trsm: Non-zero when invoked as `trsm`, 0 when invoked as `trsv`
 *
 * Solves $op(A) X = \alpha B$ where `A` is an `N x N` upper or lower triangular matrix
 * (optionally wrapped in `transpose(A)`). If `uplo` is omitted, upper vs. lower
 * triangularity is auto-detected from `A`.
 *
 * Return: `val_t` holding the solution image name, or empty string on error.
 */
val_t cli_calc_func_trsv(int is_trsm)
{
    const char *fn = is_trsm ? "trsm" : "trsv";
    val_t       argA;
    int         transA = 0;
    if (cli_calc_parse_matmul_arg(&argA, &transA) != 0)
    {
        return mk_string("");
    }
    if (cli_calc_cur_func()()->type != TOK_COMMA)
    {
        parse_errmsg(is_trsm ? "trsm: expected ','" : "trsv: expected ','");
        return mk_string("");
    }
    cli_calc_advance_func()();

    val_t rest[4];
    int   max_rest = is_trsm ? 4 : 3;
    int   nrest    = cli_calc_parse_func_argv(1, max_rest, rest, "Expected ','");
    if (nrest < 0 || cli_calc_reject_complex(&argA, 1) || cli_calc_reject_complex(rest, nrest))
    {
        return mk_string("");
    }
    if (argA.type != VAL_STRING || rest[0].type != VAL_STRING || !check_image(argA.sval) ||
        !check_image(rest[0].sval))
    {
        if (argA.type != VAL_STRING || rest[0].type != VAL_STRING)
        {
            parse_errmsg(is_trsm ? "trsm: A and B must be images" : "trsv: A and b must be images");
        }
        return mk_string("");
    }

    IMAGE *imA  = &data.core.image[image_ID(argA.sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE *imB  = &data.core.image[image_ID(rest[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    uint64_t a0 = 0;
    uint64_t a1 = 0;
    int      is_3d_a = 0;
    cli_calc_flatten2d(imA, &a0, &a1, &is_3d_a);
    if (is_3d_a || a0 != a1 || a0 == 0)
    {
        parse_errmsg(is_trsm ? "trsm: A must be a square 2D matrix"
                             : "trsv: A must be a square 2D matrix");
        return mk_string("");
    }

    uint64_t N       = a0;
    uint64_t NRHS    = 0;
    long     out_nax = 1;
    uint32_t out_sz[3];
    if (resolve_rhs_dims(imA, 0, N, N, imB, &NRHS, &out_nax, out_sz) != 0)
    {
        parse_errmsg(is_trsm ? "trsm: incompatible A and B dimensions"
                             : "trsv: incompatible A and b dimensions");
        return mk_string("");
    }
    if (!is_trsm && NRHS != 1)
    {
        parse_errmsg("trsv: right-hand side b must be a vector (use trsm for matrix RHS)");
        return mk_string("");
    }

    double *A = copy_im_to_dmat(imA, N, N, 0);
    double *X = copy_im_to_dmat(imB, N, NRHS, 0);
    if (!A || !X)
    {
        free(A);
        free(X);
        parse_errmsg("trsv: out of memory");
        return mk_string("");
    }

    int is_upper = (nrest >= 2) ? 0 : detect_uplo(A, N);
    if (nrest >= 2 && parse_uplo_arg(rest[1], &is_upper) != 0)
    {
        free(A);
        free(X);
        parse_errmsg("trsv: uplo must be \"U\" or \"L\"");
        return mk_string("");
    }
    int is_unit = 0;
    if (nrest >= 3 && parse_diag_arg(rest[2], &is_unit) != 0)
    {
        free(A);
        free(X);
        parse_errmsg("trsv: diag must be \"U\" or \"N\"");
        return mk_string("");
    }
    double alpha = 1.0;
    if (nrest >= 4)
    {
        if (rest[3].type == VAL_STRING)
        {
            free(A);
            free(X);
            parse_errmsg("trsm: alpha must be a scalar");
            return mk_string("");
        }
        alpha = to_double(rest[3]);
    }

    if (solve_triangular_inplace(A, N, transA, is_upper, is_unit, alpha, X, NRHS) != 0)
    {
        free(A);
        free(X);
        parse_errmsg(is_trsm ? "trsm: singular triangular matrix (zero diagonal)"
                             : "trsv: singular triangular matrix (zero diagonal)");
        return mk_string("");
    }

    uint8_t out_dt =
        (imA->md[0].datatype == _DATATYPE_DOUBLE || imB->md[0].datatype == _DATATYPE_DOUBLE)
            ? _DATATYPE_DOUBLE
            : _DATATYPE_FLOAT;
    val_t res = write_dmat_to_new_im(X, N * NRHS, out_nax, out_sz, out_dt, fn);
    free(A);
    free(X);
    return res;
}

/**
 * solve_lu_inplace - Solve general square system $A X = B$ via LU with partial pivoting
 * @A:    Column-major `N x N` matrix, overwritten with LU factors
 * @N:    Matrix dimension
 * @X:    Column-major `N x NRHS` right-hand side, overwritten with solution
 * @NRHS: Number of right-hand side columns
 *
 * Return: 0 on success, -1 if @A is singular to working precision.
 */
static int solve_lu_inplace(double *restrict A, uint64_t N, double *restrict X, uint64_t NRHS)
{
    uint64_t *piv = (uint64_t *) malloc((size_t) N * sizeof(uint64_t));
    if (!piv)
    {
        return -1;
    }
    double max_norm = 0.0;
    for (uint64_t i = 0; i < N * N; i++)
    {
        double v = fabs(A[i]);
        if (v > max_norm)
        {
            max_norm = v;
        }
    }
    if (max_norm == 0.0)
    {
        free(piv);
        return -1;
    }
    double tol = 1e-14 * max_norm;

    for (uint64_t k = 0; k < N; k++)
    {
        /* Find pivot row p in column k (rows k .. N-1) */
        uint64_t p                   = k;
        const double *restrict col_k = A + k * N;
        double max_v                 = fabs(col_k[k]);
        for (uint64_t i = k + 1; i < N; i++)
        {
            double v = fabs(col_k[i]);
            if (v > max_v)
            {
                max_v = v;
                p     = i;
            }
        }
        if (max_v <= tol)
        {
            free(piv);
            return -1;
        }
        piv[k] = p;
        if (p != k)
        {
            for (uint64_t j = 0; j < N; j++)
            {
                double tmp   = A[j * N + k];
                A[j * N + k] = A[j * N + p];
                A[j * N + p] = tmp;
            }
        }

        double inv_piv = 1.0 / A[k * N + k];
        for (uint64_t i = k + 1; i < N; i++)
        {
            A[k * N + i] *= inv_piv;
        }

        for (uint64_t j = k + 1; j < N; j++)
        {
            double *restrict col_j       = A + j * N;
            const double *restrict col_l = A + k * N;
            double akj                   = col_j[k];
            if (akj != 0.0)
            {
                for (uint64_t i = k + 1; i < N; i++)
                {
                    col_j[i] -= col_l[i] * akj;
                }
            }
        }
    }

    /* Apply row permutations to RHS X */
    for (uint64_t k = 0; k < N; k++)
    {
        uint64_t p = piv[k];
        if (p != k)
        {
            for (uint64_t r = 0; r < NRHS; r++)
            {
                double tmp   = X[r * N + k];
                X[r * N + k] = X[r * N + p];
                X[r * N + p] = tmp;
            }
        }
    }
    free(piv);

    /* Forward substitution L Y = P B (unit lower triangular) and backward U X = Y */
    if (solve_triangular_inplace(A, N, 0, 0, 1, 1.0, X, NRHS) != 0)
    {
        return -1;
    }
    return solve_triangular_inplace(A, N, 0, 1, 0, 1.0, X, NRHS);
}

/**
 * solve_cholesky_inplace - Solve SPD system $A X = B$ via Cholesky $A = L L^T$
 * @A:    Column-major `N x N` symmetric positive-definite matrix, lower triangle overwritten
 * @N:    Matrix dimension
 * @X:    Column-major `N x NRHS` right-hand side, overwritten with solution
 * @NRHS: Number of right-hand side columns
 *
 * Return: 0 on success, -1 if @A is not symmetric positive-definite.
 */
static int solve_cholesky_inplace(double *restrict A, uint64_t N, double *restrict X, uint64_t NRHS)
{
    /* Symmetrize into lower triangle and check rough symmetry */
    double max_diag = 0.0;
    for (uint64_t j = 0; j < N; j++)
    {
        double d = A[j * N + j];
        if (d > max_diag)
        {
            max_diag = d;
        }
    }
    if (max_diag <= 0.0)
    {
        return -1;
    }
    double sym_tol = 1e-5 * max_diag;
    for (uint64_t j = 0; j < N; j++)
    {
        for (uint64_t i = j + 1; i < N; i++)
        {
            double aij = A[j * N + i];
            double aji = A[i * N + j];
            if (fabs(aij - aji) > sym_tol)
            {
                return -1;
            }
            A[j * N + i] = 0.5 * (aij + aji);
            A[i * N + j] = 0.0;
        }
    }

    /* Cholesky L L^T factorization in lower triangle */
    for (uint64_t j = 0; j < N; j++)
    {
        double *restrict col_j = A + j * N;
        double sum             = col_j[j];
        for (uint64_t k = 0; k < j; k++)
        {
            double ljk = A[k * N + j];
            sum -= ljk * ljk;
        }
        if (sum <= 0.0 || cli_calc_isnan(sum) != 0.0)
        {
            return -1;
        }
        double ljj = sqrt(sum);
        col_j[j]   = ljj;
        double inv = 1.0 / ljj;

        for (uint64_t k = 0; k < j; k++)
        {
            const double *restrict col_k = A + k * N;
            double ljk                   = col_k[j];
            if (ljk != 0.0)
            {
                for (uint64_t i = j + 1; i < N; i++)
                {
                    col_j[i] -= col_k[i] * ljk;
                }
            }
        }
        for (uint64_t i = j + 1; i < N; i++)
        {
            col_j[i] *= inv;
        }
    }

    /* Solve L Y = B then L^T X = Y */
    if (solve_triangular_inplace(A, N, 0, 0, 0, 1.0, X, NRHS) != 0)
    {
        return -1;
    }
    return solve_triangular_inplace(A, N, 1, 0, 0, 1.0, X, NRHS);
}

/**
 * solve_qr_ls - Solve least-squares or minimum-norm system $\min \|A X - B\|_F$ via Householder QR/LQ
 * @A:    Column-major `M x N` matrix, modified in place
 * @M:    Row count of @A
 * @N:    Column count of @A
 * @B:    Column-major `M x NRHS` right-hand side
 * @NRHS: Number of right-hand side columns
 * @X:    Output column-major `N x NRHS` solution buffer
 *
 * Uses Householder QR factorization when $M \ge N$ (least-squares) and Householder
 * QR of $A^T$ ($L Q$ factorization) when $M < N$ (minimum 2-norm solution).
 *
 * Return: 0 on success, -1 if @A is rank-deficient.
 */
static int solve_qr_ls(double *restrict A,
                       uint64_t M,
                       uint64_t N,
                       const double *restrict B,
                       uint64_t NRHS,
                       double *restrict X)
{
    if (M >= N)
    {
        double *W      = (double *) malloc((size_t) (M * NRHS) * sizeof(double));
        double *r_diag = (double *) malloc((size_t) N * sizeof(double));
        double *beta   = (double *) malloc((size_t) N * sizeof(double));
        if (!W || !r_diag || !beta)
        {
            free(W);
            free(r_diag);
            free(beta);
            return -1;
        }
        __builtin_memcpy(W, B, (size_t) (M * NRHS) * sizeof(double));

        double max_r = 0.0;
        for (uint64_t k = 0; k < N; k++)
        {
            double *restrict col_k = A + k * M;
            double norm2           = 0.0;
            for (uint64_t i = k; i < M; i++)
            {
                norm2 += col_k[i] * col_k[i];
            }
            double sigma = sqrt(norm2);
            if (sigma == 0.0)
            {
                free(W);
                free(r_diag);
                free(beta);
                return -1;
            }
            double akk    = col_k[k];
            double alpha  = (akk >= 0.0) ? -sigma : sigma;
            double vk     = akk - alpha;
            double beta_k = -1.0 / (alpha * vk);
            col_k[k]      = vk;
            r_diag[k]     = alpha;
            beta[k]       = beta_k;
            if (fabs(alpha) > max_r)
            {
                max_r = fabs(alpha);
            }

            /* Apply reflector H_k to remaining columns j = k+1 .. N-1 */
            for (uint64_t j = k + 1; j < N; j++)
            {
                double *restrict col_j = A + j * M;
                double dot             = 0.0;
                for (uint64_t i = k; i < M; i++)
                {
                    dot += col_k[i] * col_j[i];
                }
                double s = beta_k * dot;
                for (uint64_t i = k; i < M; i++)
                {
                    col_j[i] -= s * col_k[i];
                }
            }
            /* Apply reflector H_k to RHS columns r = 0 .. NRHS-1 */
            for (uint64_t r = 0; r < NRHS; r++)
            {
                double *restrict col_w = W + r * M;
                double dot             = 0.0;
                for (uint64_t i = k; i < M; i++)
                {
                    dot += col_k[i] * col_w[i];
                }
                double s = beta_k * dot;
                for (uint64_t i = k; i < M; i++)
                {
                    col_w[i] -= s * col_k[i];
                }
            }
        }

        double tol = 1e-13 * max_r;
        for (uint64_t k = 0; k < N; k++)
        {
            if (fabs(r_diag[k]) <= tol)
            {
                free(W);
                free(r_diag);
                free(beta);
                return -1;
            }
        }

        /* Back-substitute R X = W[0:N-1, :] */
        for (uint64_t r = 0; r < NRHS; r++)
        {
            const double *restrict w = W + r * M;
            double *restrict x       = X + r * N;
            for (uint64_t i = 0; i < N; i++)
            {
                x[i] = w[i];
            }
            for (uint64_t jj = N; jj > 0; jj--)
            {
                uint64_t j                   = jj - 1;
                const double *restrict col_j = A + j * M;
                x[j] /= r_diag[j];
                double xj = x[j];
                for (uint64_t i = 0; i < j; i++)
                {
                    x[i] -= col_j[i] * xj;
                }
            }
        }

        free(W);
        free(r_diag);
        free(beta);
        return 0;
    }

    /* Underdetermined M < N: minimum 2-norm solution via Householder QR of A^T (N x M) */
    double *AT     = (double *) malloc((size_t) (N * M) * sizeof(double));
    double *r_diag = (double *) malloc((size_t) M * sizeof(double));
    double *beta   = (double *) malloc((size_t) M * sizeof(double));
    if (!AT || !r_diag || !beta)
    {
        free(AT);
        free(r_diag);
        free(beta);
        return -1;
    }
    for (uint64_t r = 0; r < M; r++)
    {
        for (uint64_t c = 0; c < N; c++)
        {
            AT[r * N + c] = A[c * M + r];
        }
    }

    double max_r = 0.0;
    for (uint64_t k = 0; k < M; k++)
    {
        double *restrict col_k = AT + k * N;
        double norm2           = 0.0;
        for (uint64_t i = k; i < N; i++)
        {
            norm2 += col_k[i] * col_k[i];
        }
        double sigma = sqrt(norm2);
        if (sigma == 0.0)
        {
            free(AT);
            free(r_diag);
            free(beta);
            return -1;
        }
        double akk    = col_k[k];
        double alpha  = (akk >= 0.0) ? -sigma : sigma;
        double vk     = akk - alpha;
        double beta_k = -1.0 / (alpha * vk);
        col_k[k]      = vk;
        r_diag[k]     = alpha;
        beta[k]       = beta_k;
        if (fabs(alpha) > max_r)
        {
            max_r = fabs(alpha);
        }

        for (uint64_t j = k + 1; j < M; j++)
        {
            double *restrict col_j = AT + j * N;
            double dot             = 0.0;
            for (uint64_t i = k; i < N; i++)
            {
                dot += col_k[i] * col_j[i];
            }
            double s = beta_k * dot;
            for (uint64_t i = k; i < N; i++)
            {
                col_j[i] -= s * col_k[i];
            }
        }
    }

    double tol = 1e-13 * max_r;
    for (uint64_t k = 0; k < M; k++)
    {
        if (fabs(r_diag[k]) <= tol)
        {
            free(AT);
            free(r_diag);
            free(beta);
            return -1;
        }
    }

    /* Forward-substitute R^T Y = B into top M rows of X, zero bottom N-M rows, apply Q */
    for (uint64_t r = 0; r < NRHS; r++)
    {
        const double *restrict b = B + r * M;
        double *restrict x       = X + r * N;
        for (uint64_t i = 0; i < M; i++)
        {
            const double *restrict col_i = AT + i * N;
            double sum                   = b[i];
            for (uint64_t j = 0; j < i; j++)
            {
                sum -= col_i[j] * x[j];
            }
            x[i] = sum / r_diag[i];
        }
        for (uint64_t i = M; i < N; i++)
        {
            x[i] = 0.0;
        }
        for (uint64_t kk = M; kk > 0; kk--)
        {
            uint64_t k                   = kk - 1;
            const double *restrict col_k = AT + k * N;
            double beta_k                = beta[k];
            double dot                   = 0.0;
            for (uint64_t i = k; i < N; i++)
            {
                dot += col_k[i] * x[i];
            }
            double s = beta_k * dot;
            for (uint64_t i = k; i < N; i++)
            {
                x[i] -= s * col_k[i];
            }
        }
    }

    free(AT);
    free(r_diag);
    free(beta);
    return 0;
}

/**
 * solve_ridge - Solve Tikhonov-regularized system $\min \|A X - B\|_F^2 + \lambda \|X\|_F^2$
 * @A:    Column-major `M x N` matrix
 * @M:    Row count of @A
 * @N:    Column count of @A
 * @B:    Column-major `M x NRHS` right-hand side
 * @NRHS: Number of right-hand side columns
 * @reg:  Regularization weight $\lambda > 0$
 * @X:    Output column-major `N x NRHS` solution buffer
 *
 * Forms the smaller $\min(M, N) \times \min(M, N)$ Gram system ($A^T A + \lambda I_N$
 * when $M \ge N$, or $A A^T + \lambda I_M$ via the push-through identity when $M < N$)
 * and solves it with Cholesky factorization (or LU fallback if $\lambda < 0$).
 *
 * Return: 0 on success, -1 on failure.
 */
static int solve_ridge(const double *restrict A,
                       uint64_t M,
                       uint64_t N,
                       const double *restrict B,
                       uint64_t NRHS,
                       double   reg,
                       double *restrict X)
{
    if (M >= N)
    {
        /* (A^T A + reg * I_N) X = A^T B */
        double *G = (double *) calloc((size_t) (N * N), sizeof(double));
        if (!G)
        {
            return -1;
        }
        for (uint64_t j = 0; j < N; j++)
        {
            const double *restrict col_j = A + j * M;
            for (uint64_t i = j; i < N; i++)
            {
                const double *restrict col_i = A + i * M;
                double dot                   = 0.0;
                for (uint64_t r = 0; r < M; r++)
                {
                    dot += col_i[r] * col_j[r];
                }
                G[j * N + i] = dot;
                G[i * N + j] = dot;
            }
            G[j * N + j] += reg;
        }
        for (uint64_t r = 0; r < NRHS; r++)
        {
            const double *restrict b = B + r * M;
            double *restrict x       = X + r * N;
            for (uint64_t i = 0; i < N; i++)
            {
                const double *restrict col_i = A + i * M;
                double dot                   = 0.0;
                for (uint64_t k = 0; k < M; k++)
                {
                    dot += col_i[k] * b[k];
                }
                x[i] = dot;
            }
        }
        int rc =
            (reg > 0.0) ? solve_cholesky_inplace(G, N, X, NRHS) : solve_lu_inplace(G, N, X, NRHS);
        free(G);
        return rc;
    }

    /* M < N: X = A^T (A A^T + reg * I_M)^{-1} B */
    double *G = (double *) calloc((size_t) (M * M), sizeof(double));
    double *Y = (double *) malloc((size_t) (M * NRHS) * sizeof(double));
    if (!G || !Y)
    {
        free(G);
        free(Y);
        return -1;
    }
    __builtin_memcpy(Y, B, (size_t) (M * NRHS) * sizeof(double));
    for (uint64_t k = 0; k < N; k++)
    {
        const double *restrict col_k = A + k * M;
        for (uint64_t j = 0; j < M; j++)
        {
            double akj             = col_k[j];
            double *restrict col_g = G + j * M;
            for (uint64_t i = j; i < M; i++)
            {
                col_g[i] += col_k[i] * akj;
            }
        }
    }
    for (uint64_t j = 0; j < M; j++)
    {
        G[j * M + j] += reg;
        for (uint64_t i = j + 1; i < M; i++)
        {
            G[i * M + j] = G[j * M + i];
        }
    }
    int rc = (reg > 0.0) ? solve_cholesky_inplace(G, M, Y, NRHS) : solve_lu_inplace(G, M, Y, NRHS);
    if (rc == 0)
    {
        for (uint64_t r = 0; r < NRHS; r++)
        {
            const double *restrict y = Y + r * M;
            double *restrict x       = X + r * N;
            for (uint64_t i = 0; i < N; i++)
            {
                const double *restrict col_i = A + i * M;
                double dot                   = 0.0;
                for (uint64_t k = 0; k < M; k++)
                {
                    dot += col_i[k] * y[k];
                }
                x[i] = dot;
            }
        }
    }
    free(G);
    free(Y);
    return rc;
}

/**
 * cli_calc_func_gesv - Evaluate `gesv(A, B)` to solve general square system $A X = B$
 *
 * Uses Gaussian elimination with partial pivoting ($P A = L U$). Supports optional
 * `transpose(A)` on the first argument and 1-D or 2-D right-hand sides `B`.
 *
 * Return: `val_t` holding the solution image name, or empty string on error.
 */
val_t cli_calc_func_gesv(void)
{
    val_t argA;
    int   transA = 0;
    if (cli_calc_parse_matmul_arg(&argA, &transA) != 0)
    {
        return mk_string("");
    }
    if (cli_calc_cur_func()()->type != TOK_COMMA)
    {
        parse_errmsg("gesv: expected ','");
        return mk_string("");
    }
    cli_calc_advance_func()();

    val_t argB;
    if (cli_calc_parse_func_argv(1, 1, &argB, "gesv: expected ')'") < 0 ||
        cli_calc_reject_complex(&argA, 1) || cli_calc_reject_complex(&argB, 1))
    {
        return mk_string("");
    }
    if (argA.type != VAL_STRING || argB.type != VAL_STRING || !check_image(argA.sval) ||
        !check_image(argB.sval))
    {
        if (argA.type != VAL_STRING || argB.type != VAL_STRING)
        {
            parse_errmsg("gesv: A and B must be images");
        }
        return mk_string("");
    }

    IMAGE   *imA = &data.core.image[image_ID(argA.sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE   *imB = &data.core.image[image_ID(argB.sval, data.core.image, data.core.NB_MAX_IMAGE)];
    uint64_t a0  = 0;
    uint64_t a1  = 0;
    int      is_3d_a = 0;
    cli_calc_flatten2d(imA, &a0, &a1, &is_3d_a);
    if (is_3d_a || a0 != a1 || a0 == 0)
    {
        parse_errmsg("gesv: A must be a square 2D matrix");
        return mk_string("");
    }

    uint64_t N       = a0;
    uint64_t NRHS    = 0;
    long     out_nax = 1;
    uint32_t out_sz[3];
    if (resolve_rhs_dims(imA, 0, N, N, imB, &NRHS, &out_nax, out_sz) != 0)
    {
        parse_errmsg("gesv: incompatible A and B dimensions");
        return mk_string("");
    }

    double *A = copy_im_to_dmat(imA, N, N, transA);
    double *X = copy_im_to_dmat(imB, N, NRHS, 0);
    if (!A || !X)
    {
        free(A);
        free(X);
        parse_errmsg("gesv: out of memory");
        return mk_string("");
    }
    if (solve_lu_inplace(A, N, X, NRHS) != 0)
    {
        free(A);
        free(X);
        parse_errmsg("gesv: singular matrix");
        return mk_string("");
    }

    uint8_t out_dt =
        (imA->md[0].datatype == _DATATYPE_DOUBLE || imB->md[0].datatype == _DATATYPE_DOUBLE)
            ? _DATATYPE_DOUBLE
            : _DATATYPE_FLOAT;
    val_t res = write_dmat_to_new_im(X, N * NRHS, out_nax, out_sz, out_dt, "gesv");
    free(A);
    free(X);
    return res;
}

/**
 * cli_calc_func_posv - Evaluate `posv(A, B)` to solve SPD system $A X = B$ via Cholesky
 *
 * Factorizes $A = L L^T$ and solves $L L^T X = B$. Reports an error if `A` is not
 * symmetric positive-definite.
 *
 * Return: `val_t` holding the solution image name, or empty string on error.
 */
val_t cli_calc_func_posv(void)
{
    val_t args[2];
    int   nargs = cli_calc_parse_func_argv(2, 2, args, "posv: expected ','");
    if (nargs < 0 || cli_calc_reject_complex(args, nargs))
    {
        return mk_string("");
    }
    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING || !check_image(args[0].sval) ||
        !check_image(args[1].sval))
    {
        if (args[0].type != VAL_STRING || args[1].type != VAL_STRING)
        {
            parse_errmsg("posv: A and B must be images");
        }
        return mk_string("");
    }

    IMAGE *imA  = &data.core.image[image_ID(args[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE *imB  = &data.core.image[image_ID(args[1].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    uint64_t a0 = 0;
    uint64_t a1 = 0;
    int      is_3d_a = 0;
    cli_calc_flatten2d(imA, &a0, &a1, &is_3d_a);
    if (is_3d_a || a0 != a1 || a0 == 0)
    {
        parse_errmsg("posv: A must be a square symmetric positive-definite matrix");
        return mk_string("");
    }

    uint64_t N       = a0;
    uint64_t NRHS    = 0;
    long     out_nax = 1;
    uint32_t out_sz[3];
    if (resolve_rhs_dims(imA, 0, N, N, imB, &NRHS, &out_nax, out_sz) != 0)
    {
        parse_errmsg("posv: incompatible A and B dimensions");
        return mk_string("");
    }

    double *A = copy_im_to_dmat(imA, N, N, 0);
    double *X = copy_im_to_dmat(imB, N, NRHS, 0);
    if (!A || !X)
    {
        free(A);
        free(X);
        parse_errmsg("posv: out of memory");
        return mk_string("");
    }
    if (solve_cholesky_inplace(A, N, X, NRHS) != 0)
    {
        free(A);
        free(X);
        parse_errmsg("posv: matrix is not symmetric positive-definite");
        return mk_string("");
    }

    uint8_t out_dt =
        (imA->md[0].datatype == _DATATYPE_DOUBLE || imB->md[0].datatype == _DATATYPE_DOUBLE)
            ? _DATATYPE_DOUBLE
            : _DATATYPE_FLOAT;
    val_t res = write_dmat_to_new_im(X, N * NRHS, out_nax, out_sz, out_dt, "posv");
    free(A);
    free(X);
    return res;
}

/**
 * cli_calc_func_gels - Evaluate `gels(A, B)` QR/LQ least-squares and minimum-norm solver
 *
 * Solves $\min_X \|op(A) X - B\|_F$ for overdetermined ($M \ge N$) or underdetermined
 * ($M < N$) systems using Householder reflections. Supports `transpose(A)` and 3-D
 * mode cubes `A`.
 *
 * Return: `val_t` holding the solution image name, or empty string on error.
 */
val_t cli_calc_func_gels(void)
{
    val_t argA;
    int   transA = 0;
    if (cli_calc_parse_matmul_arg(&argA, &transA) != 0)
    {
        return mk_string("");
    }
    if (cli_calc_cur_func()()->type != TOK_COMMA)
    {
        parse_errmsg("gels: expected ','");
        return mk_string("");
    }
    cli_calc_advance_func()();

    val_t argB;
    if (cli_calc_parse_func_argv(1, 1, &argB, "gels: expected ')'") < 0 ||
        cli_calc_reject_complex(&argA, 1) || cli_calc_reject_complex(&argB, 1))
    {
        return mk_string("");
    }
    if (argA.type != VAL_STRING || argB.type != VAL_STRING || !check_image(argA.sval) ||
        !check_image(argB.sval))
    {
        if (argA.type != VAL_STRING || argB.type != VAL_STRING)
        {
            parse_errmsg("gels: A and B must be images");
        }
        return mk_string("");
    }

    IMAGE   *imA = &data.core.image[image_ID(argA.sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE   *imB = &data.core.image[image_ID(argB.sval, data.core.image, data.core.NB_MAX_IMAGE)];
    uint64_t a0  = 0;
    uint64_t a1  = 0;
    int      is_3d_a = 0;
    cli_calc_flatten2d(imA, &a0, &a1, &is_3d_a);

    uint64_t M       = transA ? a1 : a0;
    uint64_t N       = transA ? a0 : a1;
    uint64_t NRHS    = 0;
    long     out_nax = 1;
    uint32_t out_sz[3];
    if (resolve_rhs_dims(imA, is_3d_a && !transA, M, N, imB, &NRHS, &out_nax, out_sz) != 0)
    {
        parse_errmsg("gels: incompatible A and B dimensions");
        return mk_string("");
    }

    double *A = copy_im_to_dmat(imA, a0, a1, transA);
    double *B = copy_im_to_dmat(imB, M, NRHS, 0);
    double *X = (double *) malloc((size_t) (N * NRHS) * sizeof(double));
    if (!A || !B || !X)
    {
        free(A);
        free(B);
        free(X);
        parse_errmsg("gels: out of memory");
        return mk_string("");
    }
    if (solve_qr_ls(A, M, N, B, NRHS, X) != 0)
    {
        free(A);
        free(B);
        free(X);
        parse_errmsg("gels: rank-deficient matrix");
        return mk_string("");
    }

    uint8_t out_dt =
        (imA->md[0].datatype == _DATATYPE_DOUBLE || imB->md[0].datatype == _DATATYPE_DOUBLE)
            ? _DATATYPE_DOUBLE
            : _DATATYPE_FLOAT;
    val_t res = write_dmat_to_new_im(X, N * NRHS, out_nax, out_sz, out_dt, "gels");
    free(A);
    free(B);
    free(X);
    return res;
}

/**
 * cli_calc_func_solve - Evaluate `solve(A, B[, reg])` unified linear / ridge solver
 *
 * Dispatches to LU (`M == N`, `reg == 0`), Householder QR/LQ (`M != N`, `reg == 0`),
 * or Cholesky Tikhonov normal equations $(A^T A + \text{reg}\, I) X = A^T B$ when
 * `reg != 0`. Supports `transpose(A)` and 3-D mode cubes `A`.
 *
 * Return: `val_t` holding the solution image name, or empty string on error.
 */
val_t cli_calc_func_solve(void)
{
    val_t argA;
    int   transA = 0;
    if (cli_calc_parse_matmul_arg(&argA, &transA) != 0)
    {
        return mk_string("");
    }
    if (cli_calc_cur_func()()->type != TOK_COMMA)
    {
        parse_errmsg("solve: expected ','");
        return mk_string("");
    }
    cli_calc_advance_func()();

    val_t rest[2];
    int   nrest = cli_calc_parse_func_argv(1, 2, rest, "solve: expected ','");
    if (nrest < 0 || cli_calc_reject_complex(&argA, 1) || cli_calc_reject_complex(rest, nrest))
    {
        return mk_string("");
    }
    if (argA.type != VAL_STRING || rest[0].type != VAL_STRING || !check_image(argA.sval) ||
        !check_image(rest[0].sval))
    {
        if (argA.type != VAL_STRING || rest[0].type != VAL_STRING)
        {
            parse_errmsg("solve: A and B must be images");
        }
        return mk_string("");
    }
    double reg = 0.0;
    if (nrest == 2)
    {
        if (rest[1].type == VAL_STRING)
        {
            parse_errmsg("solve: regularization parameter must be a scalar");
            return mk_string("");
        }
        reg = to_double(rest[1]);
    }

    IMAGE *imA  = &data.core.image[image_ID(argA.sval, data.core.image, data.core.NB_MAX_IMAGE)];
    IMAGE *imB  = &data.core.image[image_ID(rest[0].sval, data.core.image, data.core.NB_MAX_IMAGE)];
    uint64_t a0 = 0;
    uint64_t a1 = 0;
    int      is_3d_a = 0;
    cli_calc_flatten2d(imA, &a0, &a1, &is_3d_a);

    uint64_t M       = transA ? a1 : a0;
    uint64_t N       = transA ? a0 : a1;
    uint64_t NRHS    = 0;
    long     out_nax = 1;
    uint32_t out_sz[3];
    if (resolve_rhs_dims(imA, is_3d_a && !transA, M, N, imB, &NRHS, &out_nax, out_sz) != 0)
    {
        parse_errmsg("solve: incompatible A and B dimensions");
        return mk_string("");
    }

    double *A = copy_im_to_dmat(imA, a0, a1, transA);
    double *B = copy_im_to_dmat(imB, M, NRHS, 0);
    double *X = (double *) malloc((size_t) (N * NRHS) * sizeof(double));
    if (!A || !B || !X)
    {
        free(A);
        free(B);
        free(X);
        parse_errmsg("solve: out of memory");
        return mk_string("");
    }

    int rc = 0;
    if (reg != 0.0)
    {
        rc = solve_ridge(A, M, N, B, NRHS, reg, X);
    }
    else if (M == N)
    {
        __builtin_memcpy(X, B, (size_t) (N * NRHS) * sizeof(double));
        rc = solve_lu_inplace(A, N, X, NRHS);
    }
    else
    {
        rc = solve_qr_ls(A, M, N, B, NRHS, X);
    }
    if (rc != 0)
    {
        free(A);
        free(B);
        free(X);
        parse_errmsg("solve: singular or rank-deficient matrix");
        return mk_string("");
    }

    uint8_t out_dt =
        (imA->md[0].datatype == _DATATYPE_DOUBLE || imB->md[0].datatype == _DATATYPE_DOUBLE)
            ? _DATATYPE_DOUBLE
            : _DATATYPE_FLOAT;
    val_t res = write_dmat_to_new_im(X, N * NRHS, out_nax, out_sz, out_dt, "solve");
    free(A);
    free(B);
    free(X);
    return res;
}

/**
 * cli_calc_func_inv - Evaluate `inv(A)` to compute the inverse $A^{-1}$ of a square matrix
 *
 * Solves $A X = I_N$ via LU factorization with partial pivoting.
 *
 * Return: `val_t` holding the inverse matrix image name, or empty string on error.
 */
val_t cli_calc_func_inv(void)
{
    val_t argA;
    int   transA = 0;
    if (cli_calc_parse_matmul_arg(&argA, &transA) != 0)
    {
        return mk_string("");
    }
    if (cli_calc_cur_func()()->type != TOK_RPAREN)
    {
        parse_errmsg("inv: expected ')'");
        return mk_string("");
    }
    cli_calc_advance_func()();

    if (cli_calc_reject_complex(&argA, 1))
    {
        return mk_string("");
    }
    if (argA.type != VAL_STRING || !check_image(argA.sval))
    {
        if (argA.type != VAL_STRING)
        {
            parse_errmsg("inv: argument must be an image");
        }
        return mk_string("");
    }

    IMAGE   *imA = &data.core.image[image_ID(argA.sval, data.core.image, data.core.NB_MAX_IMAGE)];
    uint64_t a0  = 0;
    uint64_t a1  = 0;
    int      is_3d_a = 0;
    cli_calc_flatten2d(imA, &a0, &a1, &is_3d_a);
    if (is_3d_a || a0 != a1 || a0 == 0)
    {
        parse_errmsg("inv: A must be a square 2D matrix");
        return mk_string("");
    }

    uint64_t N = a0;
    double  *A = copy_im_to_dmat(imA, N, N, transA);
    double  *X = (double *) calloc((size_t) (N * N), sizeof(double));
    if (!A || !X)
    {
        free(A);
        free(X);
        parse_errmsg("inv: out of memory");
        return mk_string("");
    }
    for (uint64_t i = 0; i < N; i++)
    {
        X[i * N + i] = 1.0;
    }
    if (solve_lu_inplace(A, N, X, N) != 0)
    {
        free(A);
        free(X);
        parse_errmsg("inv: singular matrix");
        return mk_string("");
    }

    uint32_t out_sz[3] = { (uint32_t) N, (uint32_t) N, 1 };
    uint8_t out_dt = (imA->md[0].datatype == _DATATYPE_DOUBLE) ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT;
    val_t   res    = write_dmat_to_new_im(X, N * N, 2, out_sz, out_dt, "inv");
    free(A);
    free(X);
    return res;
}

/**
 * compute_pinv_svd - Compute Moore-Penrose pseudo-inverse $A^+$ via one-sided Jacobi SVD
 * @imA:    Input `M x N` matrix image (or 3-D mode cube flattened to `M x N`)
 * @transA: Non-zero if @imA is wrapped in `transpose(...)`
 * @M:      Effective row count of $op(A)$
 * @N:      Effective column count of $op(A)$
 * @rtol:   Relative singular-value cutoff (`>= 0` for hard truncation, `< 0` for Tikhonov)
 * @out:    Output column-major `N x M` buffer receiving $op(A)^+$
 *
 * Orthogonalizes the $Q = \min(M, N)$ columns of the taller orientation ($P \times Q$,
 * $P = \max(M, N)$) via one-sided Jacobi rotations, extracts singular values
 * $\sigma_j = \|W_{:,j}\|_2$, applies hard truncation (`rtol >= 0`) or Tikhonov
 * filtering (`rtol < 0`), and reconstructs $A^+ = V \Sigma^+ U^T$.
 *
 * Return: 0 on success, -1 on allocation failure.
 */
static int compute_pinv_svd(const IMAGE *imA,
                            int          transA,
                            uint64_t     M,
                            uint64_t     N,
                            double       rtol,
                            double      *out)
{
    uint64_t a0   = (imA->md[0].naxis == 3) ? (uint64_t) imA->md[0].size[0] * imA->md[0].size[1]
                                            : imA->md[0].size[0];
    uint64_t a1   = (imA->md[0].naxis == 3)   ? imA->md[0].size[2]
                    : (imA->md[0].naxis >= 2) ? imA->md[0].size[1]
                                              : 1;
    int      tall = (M >= N);
    uint64_t P    = tall ? M : N;
    uint64_t Q    = tall ? N : M;

    /* Copy op(A) if tall (M >= N), or op(A)^T if wide (M < N), into P x Q column-major W */
    double *W     = copy_im_to_dmat(imA, a0, a1, tall ? transA : !transA);
    double *V     = (double *) calloc((size_t) (Q * Q), sizeof(double));
    double *sigma = (double *) malloc((size_t) Q * sizeof(double));
    if (!W || !V || !sigma)
    {
        free(W);
        free(V);
        free(sigma);
        return -1;
    }
    for (uint64_t j = 0; j < Q; j++)
    {
        V[j * Q + j] = 1.0;
    }

    /* One-sided Jacobi SVD sweeps over column pairs (p, q) of W */
    const int max_sweeps = 60;
    for (int sweep = 0; sweep < max_sweeps; sweep++)
    {
        int rotated = 0;
        for (uint64_t p = 0; p < Q; p++)
        {
            for (uint64_t q = p + 1; q < Q; q++)
            {
                double *restrict wp = W + p * P;
                double *restrict wq = W + q * P;
                double app          = 0.0;
                double aqq          = 0.0;
                double apq          = 0.0;
                for (uint64_t i = 0; i < P; i++)
                {
                    double wpi = wp[i];
                    double wqi = wq[i];
                    app += wpi * wpi;
                    aqq += wqi * wqi;
                    apq += wpi * wqi;
                }
                if (fabs(apq) <= 1e-14 * sqrt(app * aqq))
                {
                    continue;
                }
                rotated    = 1;
                double tau = (aqq - app) / (2.0 * apq);
                double t   = copysign(1.0 / (fabs(tau) + sqrt(1.0 + tau * tau)), tau);
                double c   = 1.0 / sqrt(1.0 + t * t);
                double s   = c * t;

                for (uint64_t i = 0; i < P; i++)
                {
                    double wpi = wp[i];
                    double wqi = wq[i];
                    wp[i]      = c * wpi - s * wqi;
                    wq[i]      = s * wpi + c * wqi;
                }
                double *restrict vp = V + p * Q;
                double *restrict vq = V + q * Q;
                for (uint64_t i = 0; i < Q; i++)
                {
                    double vpi = vp[i];
                    double vqi = vq[i];
                    vp[i]      = c * vpi - s * vqi;
                    vq[i]      = s * vpi + c * vqi;
                }
            }
        }
        if (!rotated)
        {
            break;
        }
    }

    double sigma_max = 0.0;
    for (uint64_t j = 0; j < Q; j++)
    {
        double *restrict wj = W + j * P;
        double norm2        = 0.0;
        for (uint64_t i = 0; i < P; i++)
        {
            norm2 += wj[i] * wj[i];
        }
        double sj = sqrt(norm2);
        sigma[j]  = sj;
        if (sj > sigma_max)
        {
            sigma_max = sj;
        }
        if (sj > 0.0)
        {
            double inv_s = 1.0 / sj;
            for (uint64_t i = 0; i < P; i++)
            {
                wj[i] *= inv_s;
            }
        }
    }

    double cutoff = fabs(rtol) * sigma_max;
    double lambda = cutoff * cutoff;
    for (uint64_t j = 0; j < Q; j++)
    {
        double sj = sigma[j];
        double wj = 0.0;
        if (rtol >= 0.0)
        {
            if (sj > cutoff && sj > 0.0)
            {
                wj = 1.0 / sj;
            }
        }
        else if (sj > 0.0)
        {
            wj = sj / (sj * sj + lambda);
        }
        double *restrict vj = V + j * Q;
        for (uint64_t i = 0; i < Q; i++)
        {
            vj[i] *= wj;
        }
    }

    /* Assemble N x M pseudo-inverse in column-major layout (out[c * N + r]) */
    memset(out, 0, (size_t) (N * M) * sizeof(double));
    if (tall)
    {
        /* M >= N (P = M, Q = N): A^+ = V_scaled * U^T (N x M) */
        for (uint64_t j = 0; j < Q; j++)
        {
            const double *restrict uj = W + j * P;
            const double *restrict vj = V + j * Q;
            for (uint64_t c = 0; c < M; c++)
            {
                double ujc             = uj[c];
                double *restrict col_o = out + c * N;
                for (uint64_t r = 0; r < N; r++)
                {
                    col_o[r] += vj[r] * ujc;
                }
            }
        }
    }
    else
    {
        /* M < N (P = N, Q = M): A^+ = U * V_scaled^T (N x M) */
        for (uint64_t j = 0; j < Q; j++)
        {
            const double *restrict uj = W + j * P;
            const double *restrict vj = V + j * Q;
            for (uint64_t c = 0; c < M; c++)
            {
                double vjc             = vj[c];
                double *restrict col_o = out + c * N;
                for (uint64_t r = 0; r < N; r++)
                {
                    col_o[r] += uj[r] * vjc;
                }
            }
        }
    }

    free(W);
    free(V);
    free(sigma);
    return 0;
}

/**
 * cli_calc_func_pinv - Evaluate `pinv(A[, rtol])` Moore-Penrose pseudo-inverse via SVD
 *
 * Computes $A^+ = V \Sigma^+ U^T$ for an `M x N` matrix (or 3-D mode cube flattened
 * to `M x N`). Singular values $\sigma_i \le \text{rtol} \cdot \sigma_{\max}$ are
 * zeroed when `rtol >= 0` (default `1e-6`), or Tikhonov-filtered when `rtol < 0`.
 *
 * Return: `val_t` holding the `N x M` pseudo-inverse image name, or empty string on error.
 */
val_t cli_calc_func_pinv(void)
{
    val_t argA;
    int   transA = 0;
    if (cli_calc_parse_matmul_arg(&argA, &transA) != 0)
    {
        return mk_string("");
    }
    double rtol = 1e-6;
    if (cli_calc_cur_func()()->type == TOK_COMMA)
    {
        cli_calc_advance_func()();
        val_t arg_tol;
        if (cli_calc_parse_func_argv(1, 1, &arg_tol, "pinv: expected ')'") < 0 ||
            cli_calc_reject_complex(&arg_tol, 1))
        {
            return mk_string("");
        }
        if (arg_tol.type == VAL_STRING)
        {
            parse_errmsg("pinv: rtol must be a scalar");
            return mk_string("");
        }
        rtol = to_double(arg_tol);
    }
    else if (cli_calc_cur_func()()->type == TOK_RPAREN)
    {
        cli_calc_advance_func()();
    }
    else
    {
        parse_errmsg("pinv: expected ')'");
        return mk_string("");
    }

    if (cli_calc_reject_complex(&argA, 1))
    {
        return mk_string("");
    }
    if (argA.type != VAL_STRING || !check_image(argA.sval))
    {
        if (argA.type != VAL_STRING)
        {
            parse_errmsg("pinv: first argument must be an image");
        }
        return mk_string("");
    }

    IMAGE   *imA = &data.core.image[image_ID(argA.sval, data.core.image, data.core.NB_MAX_IMAGE)];
    uint64_t a0  = 0;
    uint64_t a1  = 0;
    int      is_3d_a = 0;
    cli_calc_flatten2d(imA, &a0, &a1, &is_3d_a);
    uint64_t M = transA ? a1 : a0;
    uint64_t N = transA ? a0 : a1;
    if (M == 0 || N == 0)
    {
        parse_errmsg("pinv: empty matrix");
        return mk_string("");
    }

    double *out = (double *) malloc((size_t) (N * M) * sizeof(double));
    if (!out || compute_pinv_svd(imA, transA, M, N, rtol, out) != 0)
    {
        free(out);
        parse_errmsg("pinv: out of memory");
        return mk_string("");
    }

    uint32_t out_sz[3] = { (uint32_t) N, (uint32_t) M, 1 };
    uint8_t out_dt = (imA->md[0].datatype == _DATATYPE_DOUBLE) ? _DATATYPE_DOUBLE : _DATATYPE_FLOAT;
    val_t   res    = write_dmat_to_new_im(out, N * M, 2, out_sz, out_dt, "pinv");
    free(out);
    return res;
}

#endif /* CLI_CALC_LAPACK_H */
