/**
 * @file mvm_auxiliaries.hpp
 * @brief MVM backend class hierarchy
 *
 * All backends share col-major matrix convention:
 * A is M×N, lda = M, A[i,j] = matrix[j*M + i].
 *
 * Lifetime:
 *   ctor                   — store I/O buffers and dimension metadata
 *   enable_masking()       — optional; must be called before load_matrix
 *   load_matrix()          — bind matrix; CUDA backend allocates GPU buffers
 *   matrixMul()            — compute MVM, including gather/scatter for CPU/BLAS
 *   dtor                   — free all resources
 *
 * Caller always fills the full inVec (full_size_spatial or nb_modes elements)
 * and reads the full outVec. Masking gather/scatter is handled internally.
 */

#pragma once

#include <cstdint>

/* ----------------------------------------------------------------
 * Base class
 * -------------------------------------------------------------- */
class MVMBackend
{
  public:
    /**
     * @param inVec               host input  vector (updated each iter, caller owns)
     * @param outVec              host output vector (caller owns)
     * @param full_size_spatial   full spatial pixel count (before any masking)
     * @param nb_modes            number of modes
     * @param axmode              0 = extract (spatial→modes), 1 = expand (modes→spatial)
     */
    MVMBackend(const float *inVec,
               float       *outVec,
               uint64_t     full_size_spatial,
               uint64_t     nb_modes,
               uint32_t     axmode);

    virtual ~MVMBackend();

    /**
     * @brief Bind the matrix used for multiplication.
     *
     * Must be called after enable_masking() (if used) and before the
     * first matrixMulLoopPreload().  For CUDA backends this also
     * allocates GPU buffers and uploads the matrix.
     *
     * @param matrix            col-major host matrix (persistent, caller owns lifetime)
     * @param mvm_size_spatial  spatial pixel count used in the MVM (may be mask_npix)
     * @param mvm_size_modes    mode count used in the MVM
     */
    virtual void load_matrix(const float *matrix,
                             uint64_t     mvm_size_spatial,
                             uint64_t     mvm_size_modes);

    virtual void matrixMul() = 0;

    /**
     * @brief Enable pixel masking for this backend.
     *
     * @param mask_idx   array of pixel indices that belong to the mask
     *                   (caller owns lifetime)
     * @param mask_npix  number of entries in mask_idx
     */
    void enable_masking(uint32_t *mask_idx, uint64_t mask_npix);

  protected:
    const float *matrix_;
    const float *inVec_;
    float       *outVec_;
    uint64_t     full_size_spatial_;
    uint64_t     nb_modes_;
    uint32_t     axmode_;

    uint64_t mvm_size_in_;  /* MVM input  size (axmode-dependent) */
    uint64_t mvm_size_out_; /* MVM output size (axmode-dependent) */

    int       masking_;
    uint32_t *mask_idx_;
    uint64_t  mask_npix_;
    float    *masked_array_storage_;
};


/* ----------------------------------------------------------------
 * CPU fallback  (plain C + OMP SIMD)
 * -------------------------------------------------------------- */
class MVMBackendCPU : public MVMBackend
{
  public:
    using MVMBackend::MVMBackend;
    ~MVMBackendCPU() override = default;

    void matrixMul() override;
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

    void matrixMul() override;
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
    MVMBackendCUBLAS(const float *inVec,
                     float       *outVec,
                     uint64_t     full_size_spatial,
                     uint64_t     nb_modes,
                     uint32_t     axmode);
    ~MVMBackendCUBLAS() override;

    /** Allocate GPU buffers and upload matrix (call after enable_masking) */
    void load_matrix(const float *matrix,
                     uint64_t     mvm_size_spatial,
                     uint64_t     mvm_size_modes) override;

    /** cublasSgemv on GPU buffers only */
    void matrixMul() override;

  private:
    cublasHandle_t handle_;
    float         *d_matrix_; /* GPU col-major M x N */
    float         *d_in_;     /* GPU input  (N)      */
    float         *d_out_;    /* GPU output (M)      */
};
#endif /* HAVE_CUDA */
