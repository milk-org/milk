// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file MVM_CPU.c
 * @brief CPU matrix-vector multiply
 *
 * Computes dmVec = cMat * wfsVec  (M×N matrix
 * times N-vector → M-vector).
 *
 * Uses cblas_sgemv when BLAS is available
 * (MKL or OpenBLAS), otherwise falls back to a
 * restrict + OMP SIMD plain-C implementation.
 */

// MILK_CMAKE_REQUEST_BLAS

#include "milk_blas_lapacke.h"

#ifdef _OPENMP
#    include <omp.h>
#endif

#include <string.h>


/**
 * @brief Matrix-vector multiply (float32)
 *
 * @param cMat   M×N matrix (row-major)
 * @param wfsVec input vector  (length N)
 * @param dmVec  output vector (length M)
 * @param M      number of rows (output size)
 * @param N      number of cols (input size)
 */
void matrixMulCPU(float *restrict cMat, float *restrict wfsVec, float *restrict dmVec, int M, int N)
{
#ifdef HAVE_BLAS
    /* Use BLAS sgemv: y = alpha * A * x + beta * y
     * CblasRowMajor, CblasNoTrans,
     * M rows, N cols, alpha=1, lda=N,
     * incx=1, beta=0, incy=1 */
    cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N, 1.0f, cMat, N, wfsVec, 1, 0.0f, dmVec, 1);
#else
    /* Plain-C fallback with restrict + OMP */
    memset(dmVec, 0, sizeof(float) * M);

#    pragma omp parallel for schedule(static) if (M > 64)
    for (int m = 0; m < M; m++)
    {
        const float *restrict row = &cMat[m * N];
        float acc                 = 0.0f;

#    pragma omp simd reduction(+ : acc)
        for (int n = 0; n < N; n++)
        {
            acc += row[n] * wfsVec[n];
        }
        dmVec[m] = acc;
    }
#endif // #ifdef HAVE_BLAS #else
}
