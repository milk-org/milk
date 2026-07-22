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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "mvm_auxiliaries.hpp"


/* ----------------------------------------------------------------
 * MVMBackend base
 * -------------------------------------------------------------- */
MVMBackend::MVMBackend(const float *inVec,
                       float       *outVec,
                       uint64_t     full_size_spatial,
                       uint64_t     nb_modes,
                       uint32_t     axmode)
    : matrix_(nullptr), inVec_(inVec), outVec_(outVec), full_size_spatial_(full_size_spatial),
      nb_modes_(nb_modes), axmode_(axmode),
      mvm_size_in_(axmode == 0 ? full_size_spatial : nb_modes),
      mvm_size_out_(axmode == 0 ? nb_modes : full_size_spatial), masking_(0), mask_idx_(nullptr),
      mask_npix_(0), masked_array_storage_(nullptr)
{
}

void MVMBackend::load_matrix(const float *matrix,
                             uint64_t     mvm_size_spatial,
                             uint64_t     mvm_size_modes)
{
    matrix_ = matrix;
    if (masking_) // MVM size is masked size
    {
        mvm_size_in_  = (axmode_ == 0) ? mask_npix_ : mvm_size_modes;
        mvm_size_out_ = (axmode_ == 0) ? mvm_size_modes : mask_npix_;
    }
    else // MVM size is full data size
    {
        mvm_size_in_  = (axmode_ == 0) ? mvm_size_spatial : mvm_size_modes;
        mvm_size_out_ = (axmode_ == 0) ? mvm_size_modes : mvm_size_spatial;
    }
}

MVMBackend::~MVMBackend()
{
    if (masked_array_storage_ != nullptr)
    {
        free(masked_array_storage_);
        masked_array_storage_ = nullptr;
    }
}

void MVMBackend::enable_masking(uint32_t *mask_idx, uint64_t mask_npix)
{
    mask_idx_             = mask_idx;
    mask_npix_            = mask_npix;
    masking_              = 1;
    masked_array_storage_ = (float *) malloc(mask_npix * sizeof(float));
}


/* ----------------------------------------------------------------
 * MVMBackendCPU
 * -------------------------------------------------------------- */

void MVMBackendCPU::matrixMul()
{
    if (axmode_ == 0)
    {
/* Extraction: out[j] = Σ_i M[j*mvm_size_in_ + i] * in[in_idx(i)]
         * in_idx(i) = masking_ ? mask_idx_[i] : i */
#pragma omp      parallel for schedule(static) if (mvm_size_out_ > 64)
        for (uint64_t j = 0; j < mvm_size_out_; j++)
        {
            float acc = 0.0f;
            if (masking_)
            {
                for (uint64_t i = 0; i < mvm_size_in_; i++)
                {
                    acc += matrix_[j * mvm_size_in_ + i] * inVec_[mask_idx_[i]];
                }
            }
            else
            {
#pragma omp simd reduction(+ : acc)
                for (uint64_t i = 0; i < mvm_size_in_; i++)
                {
                    acc += matrix_[j * mvm_size_in_ + i] * inVec_[i];
                }
            }
            outVec_[j] = acc;
        }
    }
    else
    {
/* Expansion: out[out_idx(i)] = Σ_j M[j*mvm_size_out_ + i] * in[j]
         * out_idx(i) = masking_ ? mask_idx_[i] : i */
#pragma omp      parallel for schedule(static) if (mvm_size_out_ > 64)
        for (uint64_t i = 0; i < mvm_size_out_; i++)
        {
            float acc = 0.0f;
#pragma omp simd reduction(+ : acc)
            for (uint64_t j = 0; j < mvm_size_in_; j++)
            {
                acc += matrix_[j * mvm_size_out_ + i] * inVec_[j];
            }
            outVec_[masking_ ? mask_idx_[i] : i] = acc;
        }
    }
}


/* ----------------------------------------------------------------
 * MVMBackendBLAS
 * -------------------------------------------------------------- */
#if defined(HAVE_MKL) || defined(HAVE_OPENBLAS)

void MVMBackendBLAS::matrixMul()
{
    if (axmode_ == 0)
    {
        /* Gather masked pixels into compact buffer before BLAS call */
        if (masking_)
        {
            for (uint64_t i = 0; i < mask_npix_; i++)
            {
                masked_array_storage_[i] = inVec_[mask_idx_[i]];
            }
        }
        const float *blas_input = masking_ ? masked_array_storage_ : inVec_;
        /* out = M^T * x, M is mvm_size_in_×mvm_size_out_ col-major */
        cblas_sgemv(CblasColMajor, CblasTrans, (int) mvm_size_in_, (int) mvm_size_out_, 1.0f,
                    matrix_, (int) mvm_size_in_, blas_input, 1, 0.0f, outVec_, 1);
    }
    else
    {
        /* With masking, accumulate into compact buffer then scatter */
        float *blas_output = masking_ ? masked_array_storage_ : outVec_;
        /* out = M * x, M is mvm_size_out_×mvm_size_in_ col-major */
        cblas_sgemv(CblasColMajor, CblasNoTrans, (int) mvm_size_out_, (int) mvm_size_in_, 1.0f,
                    matrix_, (int) mvm_size_out_, inVec_, 1, 0.0f, blas_output, 1);
        if (masking_)
        {
            for (uint64_t i = 0; i < mask_npix_; i++)
            {
                outVec_[mask_idx_[i]] = masked_array_storage_[i];
            }
        }
    }
}

#endif /* HAVE_MKL || HAVE_OPENBLAS */


/* ----------------------------------------------------------------
 * MVMBackendCUBLAS
 * -------------------------------------------------------------- */
#ifdef HAVE_CUDA

MVMBackendCUBLAS::MVMBackendCUBLAS(const float *inVec,
                                   float       *outVec,
                                   uint64_t     full_size_spatial,
                                   uint64_t     nb_modes,
                                   uint32_t     axmode)
    : MVMBackend(inVec, outVec, full_size_spatial, nb_modes, axmode), handle_(nullptr),
      d_matrix_(nullptr), d_in_(nullptr), d_out_(nullptr)
{
    cublasStatus_t stat = cublasCreate(&handle_);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "MVMBackendCUBLAS: cublasCreate failed (%d)\n", (int) stat);
        exit(EXIT_FAILURE);
    }
}

void MVMBackendCUBLAS::load_matrix(const float *matrix,
                                   uint64_t     mvm_size_spatial,
                                   uint64_t     mvm_size_modes)
{
    MVMBackend::load_matrix(matrix, mvm_size_spatial, mvm_size_modes);

    cudaError_t err;

    if (d_matrix_)
    {
        cudaFree(d_matrix_);
        d_matrix_ = nullptr;
    }
    if (d_in_)
    {
        cudaFree(d_in_);
        d_in_ = nullptr;
    }
    if (d_out_)
    {
        cudaFree(d_out_);
        d_out_ = nullptr;
    }

    err = cudaMalloc((void **) &d_matrix_, sizeof(float) * (size_t) (mvm_size_in_ * mvm_size_out_));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "MVMBackendCUBLAS: cudaMalloc d_matrix_ failed\n");
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **) &d_in_, sizeof(float) * (size_t) mvm_size_in_);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "MVMBackendCUBLAS: cudaMalloc d_in_ failed\n");
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **) &d_out_, sizeof(float) * (size_t) mvm_size_out_);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "MVMBackendCUBLAS: cudaMalloc d_out_ failed\n");
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(d_matrix_, matrix_, sizeof(float) * (size_t) (mvm_size_in_ * mvm_size_out_),
               cudaMemcpyHostToDevice);
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

void MVMBackendCUBLAS::matrixMul()
{
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    /* axmode==0 + masking: gather sparse input into compact buffer before H2D */
    const float *src = inVec_;
    if (masking_ && axmode_ == 0)
    {
        for (uint64_t i = 0; i < mask_npix_; i++)
        {
            masked_array_storage_[i] = inVec_[mask_idx_[i]];
        }
        src = masked_array_storage_;
    }
    cudaMemcpy(d_in_, src, sizeof(float) * (size_t) mvm_size_in_, cudaMemcpyHostToDevice);

    if (axmode_ == 0)
    {
        /* out = input @ matrix: A^T*x, A is mvm_size_in_×mvm_size_out_ col-major */
        cublasSgemv(handle_, CUBLAS_OP_T, (int) mvm_size_in_, (int) mvm_size_out_, &alpha,
                    d_matrix_, (int) mvm_size_in_, d_in_, 1, &beta, d_out_, 1);
    }
    else
    {
        /* out = matrix @ input: A*x, A is mvm_size_out_×mvm_size_in_ col-major */
        cublasSgemv(handle_, CUBLAS_OP_N, (int) mvm_size_out_, (int) mvm_size_in_, &alpha,
                    d_matrix_, (int) mvm_size_out_, d_in_, 1, &beta, d_out_, 1);
    }

    /* axmode==1 + masking: D2H into compact buffer then scatter to full output */
    float *dst = outVec_;
    if (masking_ && axmode_ == 1)
    {
        dst = masked_array_storage_;
    }

    cudaMemcpy(dst, d_out_, sizeof(float) * (size_t) mvm_size_out_, cudaMemcpyDeviceToHost);

    if (masking_ && axmode_ == 1)
    {
        for (uint64_t i = 0; i < mask_npix_; i++)
        {
            outVec_[mask_idx_[i]] = masked_array_storage_[i];
        }
    }
}

#endif /* HAVE_CUDA */


/* ----------------------------------------------------------------
 * MVMBackendCUDAGraph
 *
 * Three-node CUDA graph: H2D memcpy → cublasSgemv → D2H memcpy.
 * The graph is captured once in load_matrix() and replayed with a
 * single cudaGraphLaunch() each frame, cutting per-frame API
 * overhead from ~3 blocking calls to one non-blocking dispatch.
 *
 * Pinned host staging buffers (h_in_pinned_, h_out_pinned_) serve
 * as fixed-address intermediaries: the graph nodes always reference
 * the same addresses, while the caller's inVec/outVec may live in
 * pageable memory (/dev/shm).
 * -------------------------------------------------------------- */
#ifdef HAVE_CUDA

MVMBackendCUDAGraph::MVMBackendCUDAGraph(const float *inVec,
                                         float       *outVec,
                                         uint64_t     full_size_spatial,
                                         uint64_t     nb_modes,
                                         uint32_t     axmode)
    : MVMBackend(inVec, outVec, full_size_spatial, nb_modes, axmode), handle_(nullptr),
      stream_(nullptr), d_matrix_(nullptr), d_in_(nullptr), d_out_(nullptr), h_in_pinned_(nullptr),
      h_out_pinned_(nullptr), graph_(nullptr), graph_exec_(nullptr), graph_valid_(false)
{
    cublasStatus_t stat = cublasCreate(&handle_);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "MVMBackendCUDAGraph: cublasCreate failed (%d)\n", (int) stat);
        exit(EXIT_FAILURE);
    }

    cudaError_t err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "MVMBackendCUDAGraph: cudaStreamCreate failed (%d)\n", (int) err);
        exit(EXIT_FAILURE);
    }

    cublasSetStream(handle_, stream_);
}

/* ------------------------------------------------------------------ */

void MVMBackendCUDAGraph::free_buffers_()
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
    if (h_out_pinned_)
    {
        cudaFreeHost(h_out_pinned_);
        h_out_pinned_ = nullptr;
    }
    if (h_in_pinned_)
    {
        cudaFreeHost(h_in_pinned_);
        h_in_pinned_ = nullptr;
    }
}

void MVMBackendCUDAGraph::destroy_graph_()
{
    if (graph_valid_)
    {
        cudaGraphExecDestroy(graph_exec_);
        cudaGraphDestroy(graph_);
        graph_exec_  = nullptr;
        graph_       = nullptr;
        graph_valid_ = false;
    }
}

/* ------------------------------------------------------------------ */

void MVMBackendCUDAGraph::load_matrix(const float *matrix,
                                      uint64_t     mvm_size_spatial,
                                      uint64_t     mvm_size_modes)
{
    destroy_graph_();
    free_buffers_();

    /* Let the base class update mvm_size_in_ / mvm_size_out_ */
    MVMBackend::load_matrix(matrix, mvm_size_spatial, mvm_size_modes);

    cudaError_t  err;
    const size_t sz_in  = sizeof(float) * (size_t) mvm_size_in_;
    const size_t sz_out = sizeof(float) * (size_t) mvm_size_out_;
    const size_t sz_mat = sizeof(float) * (size_t) (mvm_size_in_ * mvm_size_out_);

    /* --- GPU device buffers --- */
    err = cudaMalloc((void **) &d_matrix_, sz_mat);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "MVMBackendCUDAGraph: cudaMalloc d_matrix_ failed\n");
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **) &d_in_, sz_in);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "MVMBackendCUDAGraph: cudaMalloc d_in_ failed\n");
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **) &d_out_, sz_out);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "MVMBackendCUDAGraph: cudaMalloc d_out_ failed\n");
        exit(EXIT_FAILURE);
    }

    /* Upload weight matrix (one-time, not part of the per-frame graph) */
    cudaMemcpy(d_matrix_, matrix_, sz_mat, cudaMemcpyHostToDevice);

    /* --- Pinned host staging buffers --- */
    // TODO wrap the CUDA error checking
    // TODO see if those mallocs can be avoided depending whether the input is a GPU SHM
    err = cudaMallocHost((void **) &h_in_pinned_, sz_in);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "MVMBackendCUDAGraph: cudaMallocHost h_in_pinned_ failed\n");
        exit(EXIT_FAILURE);
    }

    err = cudaMallocHost((void **) &h_out_pinned_, sz_out);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "MVMBackendCUDAGraph: cudaMallocHost h_out_pinned_ failed\n");
        exit(EXIT_FAILURE);
    }

    build_graph_();
}

/* ------------------------------------------------------------------ */

void MVMBackendCUDAGraph::build_graph_()
{
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);

    /* Node 1: H2D — pinned staging → d_in_ */
    cudaMemcpyAsync(d_in_, h_in_pinned_, sizeof(float) * (size_t) mvm_size_in_,
                    cudaMemcpyHostToDevice, stream_);

    /* Node 2: cublasSgemv */
    if (axmode_ == 0)
    {
        /* Extraction: out = M^T * x
         * M is mvm_size_in_ × mvm_size_out_ col-major */
        cublasSgemv(handle_, CUBLAS_OP_T, (int) mvm_size_in_, (int) mvm_size_out_, &alpha,
                    d_matrix_, (int) mvm_size_in_, d_in_, 1, &beta, d_out_, 1);
    }
    else
    {
        /* Expansion: out = M * x
         * M is mvm_size_out_ × mvm_size_in_ col-major */
        cublasSgemv(handle_, CUBLAS_OP_N, (int) mvm_size_out_, (int) mvm_size_in_, &alpha,
                    d_matrix_, (int) mvm_size_out_, d_in_, 1, &beta, d_out_, 1);
    }

    /* Node 3: D2H — d_out_ → pinned staging */
    cudaMemcpyAsync(h_out_pinned_, d_out_, sizeof(float) * (size_t) mvm_size_out_,
                    cudaMemcpyDeviceToHost, stream_);

    cudaStreamEndCapture(stream_, &graph_);
    cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0);
    graph_valid_ = true;
}

/* ------------------------------------------------------------------ */

MVMBackendCUDAGraph::~MVMBackendCUDAGraph()
{
    destroy_graph_();
    free_buffers_();
    if (stream_)
    {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    if (handle_)
    {
        cublasDestroy(handle_);
        handle_ = nullptr;
    }
}

/* ------------------------------------------------------------------ */

void MVMBackendCUDAGraph::matrixMul()
{
    /* --- CPU gather: fill h_in_pinned_ from caller's inVec_ --- */
    if (masking_ && axmode_ == 0)
    {
        /* Sparse gather: pick only masked pixels */
        for (uint64_t i = 0; i < mask_npix_; i++)
        {
            h_in_pinned_[i] = inVec_[mask_idx_[i]];
        }
    }
    else
    {
        memcpy(h_in_pinned_, inVec_, sizeof(float) * (size_t) mvm_size_in_);
    }

    /* --- GPU: single graph dispatch (H2D + sgemv + D2H) --- */
    cudaGraphLaunch(graph_exec_, stream_);
    cudaStreamSynchronize(stream_);

    /* --- CPU scatter: drain h_out_pinned_ into caller's outVec_ --- */
    if (masking_ && axmode_ == 1)
    {
        /* Sparse scatter: write only to masked pixel positions */
        for (uint64_t i = 0; i < mask_npix_; i++)
        {
            outVec_[mask_idx_[i]] = h_out_pinned_[i];
        }
    }
    else
    {
        memcpy(outVec_, h_out_pinned_, sizeof(float) * (size_t) mvm_size_out_);
    }
}

#endif /* HAVE_CUDA (MVMBackendCUDAGraph) */
