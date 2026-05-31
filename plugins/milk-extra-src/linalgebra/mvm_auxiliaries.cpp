/**
 * @file mvm_auxiliaries.cpp
 * @brief MVM backend implementations
 */

#ifdef HAVE_MKL
#    include <mkl.h>
#elif defined(HAVE_OPENBLAS)
#    include <cblas.h>
#endif

#ifdef HAVE_CUDA
#    include <cuda_runtime.h>
#    include <cublas_v2.h>
#endif

#ifdef _OPENMP
#    include <omp.h>
#endif

#include <cstdio>
#include <cstdlib>

#include "mvm_auxiliaries.hpp"


/* ----------------------------------------------------------------
 * MVMBackend base
 * -------------------------------------------------------------- */
MVMBackend::MVMBackend(const float *matrix, const float *inVec, float *outVec, int M, int N)
    : matrix_(matrix), inVec_(inVec), outVec_(outVec), M_(M), N_(N)
{
}


/* ----------------------------------------------------------------
 * MVMBackendCPU
 * -------------------------------------------------------------- */
void MVMBackendCPU::matrixMulLoopPreload()
{
}
void MVMBackendCPU::matrixMulLoopUnload()
{
}

void MVMBackendCPU::matrixMul()
{
/* y = A x, A col-major M×N, A[m,n] = matrix_[n*M_ + m]
     *
     * Outer loop over rows m; each m writes only to outVec_[m]
     * → omp parallel for is race-free.
     * Inner loop over columns n accumulates into a register acc
     * → omp simd reduction is safe.                           */
#pragma omp parallel for schedule(static) if (M_ > 64)
    for (int m = 0; m < M_; m++)
    {
        float acc = 0.0f;

#pragma omp simd reduction(+ : acc)
        for (int n = 0; n < N_; n++)
        {
            acc += matrix_[n * M_ + m] * inVec_[n];
        }
        outVec_[m] = acc;
    }
}


/* ----------------------------------------------------------------
 * MVMBackendBLAS
 * -------------------------------------------------------------- */
#if defined(HAVE_MKL) || defined(HAVE_OPENBLAS)

void MVMBackendBLAS::matrixMulLoopPreload()
{
}
void MVMBackendBLAS::matrixMulLoopUnload()
{
}

void MVMBackendBLAS::matrixMul()
{
    /* y = A x, A col-major M×N, lda = M */
    cblas_sgemv(CblasColMajor, CblasNoTrans, M_, N_, 1.0f, matrix_, M_, inVec_, 1, 0.0f, outVec_,
                1);
}

#endif /* HAVE_MKL || HAVE_OPENBLAS */


/* ----------------------------------------------------------------
 * MVMBackendCUBLAS
 * -------------------------------------------------------------- */
#ifdef HAVE_CUDA

MVMBackendCUBLAS::MVMBackendCUBLAS(const float *matrix,
                                   const float *inVec,
                                   float       *outVec,
                                   int          M,
                                   int          N)
    : MVMBackend(matrix, inVec, outVec, M, N), handle_(nullptr), d_matrix_(nullptr), d_in_(nullptr),
      d_out_(nullptr)
{
    cublasStatus_t stat = cublasCreate(&handle_);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "MVMBackendCUBLAS: cublasCreate failed (%d)\n", (int) stat);
        exit(EXIT_FAILURE);
    }

    cudaError_t err;

    err = cudaMalloc((void **) &d_matrix_, sizeof(float) * (size_t) (M_ * N_));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "MVMBackendCUBLAS: cudaMalloc d_matrix_ failed\n");
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **) &d_in_, sizeof(float) * (size_t) N_);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "MVMBackendCUBLAS: cudaMalloc d_in_ failed\n");
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **) &d_out_, sizeof(float) * (size_t) M_);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "MVMBackendCUBLAS: cudaMalloc d_out_ failed\n");
        exit(EXIT_FAILURE);
    }

    /* Upload static modes matrix once at construction */
    cudaMemcpy(d_matrix_, matrix_, sizeof(float) * (size_t) (M_ * N_), cudaMemcpyHostToDevice);
}

MVMBackendCUBLAS::~MVMBackendCUBLAS()
{
    if (d_out_)
    {
        cudaFree(d_out_);
        d_out_ = nullptr;
    }
    if (d_in_)
    {
        cudaFree(d_in_);
        d_in_ = nullptr;
    }
    if (d_matrix_)
    {
        cudaFree(d_matrix_);
        d_matrix_ = nullptr;
    }
    if (handle_)
    {
        cublasDestroy(handle_);
        handle_ = nullptr;
    }
}

void MVMBackendCUBLAS::matrixMulLoopPreload()
{
    /* H2D: copy current input vector to GPU */
    cudaMemcpy(d_in_, inVec_, sizeof(float) * (size_t) N_, cudaMemcpyHostToDevice);
}

void MVMBackendCUBLAS::matrixMul()
{
    /* y = A x, A col-major M×N on GPU, lda = M
     *
     * cuBLAS uses Fortran (col-major) convention, same as the
     * host ColMajorMatrix → CUBLAS_OP_N with (M_, N_) works
     * directly, no transposition trick needed.               */
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasSgemv(handle_, CUBLAS_OP_N, M_, N_, &alpha, d_matrix_, M_, d_in_, 1, &beta, d_out_, 1);
}

void MVMBackendCUBLAS::matrixMulLoopUnload()
{
    /* D2H: retrieve result to host output buffer */
    cudaMemcpy(outVec_, d_out_, sizeof(float) * (size_t) M_, cudaMemcpyDeviceToHost);
}

#endif /* HAVE_CUDA */
