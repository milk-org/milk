// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file MVM_CPU.c
 * @brief CPU matrix-vector multiply
 */

// MILK_CMAKE_REQUEST_BLAS

#include "COREMOD_arith/image_linalg.h"

/**
 * @brief Matrix-vector multiply (float32)
 *
 * @param cMat   M x N matrix (row-major)
 * @param wfsVec input vector  (length N)
 * @param dmVec  output vector (length M)
 * @param M      number of rows (output size)
 * @param N      number of cols (input size)
 */
void matrixMulCPU(float *restrict cMat, float *restrict wfsVec, float *restrict dmVec, int M, int N)
{
    arith_matmul_kernel_f(cMat, 1, wfsVec, 0, 1.0f, 0.0f, dmVec, (uint64_t) M, (uint64_t) N, 1);
}
