/**
 * @file mvm_auxiliaries.hpp
 * @brief MVM backend class hierarchy
 *
 * All backends share col-major matrix convention:
 * A is M×N, lda = M, A[i,j] = matrix[j*M + i].
 *
 * Lifetime:
 *   ctor                   — alloc + upload static matrix
 *   matrixMulLoopPreload() — per-iter H to device transfer
 *   matrixMul()            — pure compute, no memory movement
 *   matrixMulLoopUnload()  — per-iter device to H transfer
 *   dtor                   — free all resources
 *
 * Caller fills inVec[0..N-1] before matrixMulLoopPreload()
 * and reads outVec[0..M-1] after matrixMulLoopUnload().
 */

#pragma once

/* ----------------------------------------------------------------
 * Base class
 * -------------------------------------------------------------- */
class MVMBackend
{
  public:
    /**
     * @param matrix  col-major host matrix, M x N, lda=M (persistent)
     * @param inVec   host input  vector, length N (updated each iter)
     * @param outVec  host output vector, length M
     * @param M       output dimension (NBmodes)
     * @param N       input  dimension (pixels / mask_npix)
     */
    MVMBackend(const float *matrix, const float *inVec, float *outVec, int M, int N);

    virtual ~MVMBackend() = default;

    virtual void matrixMulLoopPreload() = 0;
    virtual void matrixMul()            = 0;
    virtual void matrixMulLoopUnload()  = 0;

  protected:
    const float *matrix_;
    const float *inVec_;
    float       *outVec_;
    int          M_;
    int          N_;
};


/* ----------------------------------------------------------------
 * CPU fallback  (plain C + OMP SIMD)
 * -------------------------------------------------------------- */
class MVMBackendCPU : public MVMBackend
{
  public:
    using MVMBackend::MVMBackend;
    ~MVMBackendCPU() override = default;

    void matrixMulLoopPreload() override;
    void matrixMul() override;
    void matrixMulLoopUnload() override;
};


/* ----------------------------------------------------------------
 * BLAS  (OpenBLAS or Intel MKL)
 * -------------------------------------------------------------- */
#if defined(HAVE_MKL) || defined(HAVE_OPENBLAS)
class MVMBackendBLAS : public MVMBackend
{
  public:
    using MVMBackend::MVMBackend;
    ~MVMBackendBLAS() override = default;

    void matrixMulLoopPreload() override;
    void matrixMul() override;
    void matrixMulLoopUnload() override;
};
#endif /* HAVE_MKL || HAVE_OPENBLAS */


/* ----------------------------------------------------------------
 * cuBLAS
 * -------------------------------------------------------------- */
#ifdef HAVE_CUDA
#    include <cublas_v2.h>

class MVMBackendCUBLAS : public MVMBackend
{
  public:
    MVMBackendCUBLAS(const float *matrix, const float *inVec, float *outVec, int M, int N);
    ~MVMBackendCUBLAS() override;

    /** H2D copy of inVec into d_in_ */
    void matrixMulLoopPreload() override;
    /** cublasSgemv on GPU buffers only */
    void matrixMul() override;
    /** D2H copy of d_out_ into outVec */
    void matrixMulLoopUnload() override;

  private:
    cublasHandle_t handle_;
    float         *d_matrix_; /* GPU col-major M x N */
    float         *d_in_;     /* GPU input  (N)      */
    float         *d_out_;    /* GPU output (M)      */
};
#endif /* HAVE_CUDA */
