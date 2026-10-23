// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file SGEMM.c
 *
 */

// MILK_CMAKE_REQUEST_CUDA
// MILK_CMAKE_REQUEST_BLAS

#include <math.h>
#include <stdlib.h>

#include "milk_runnable.h"
#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_iofits/COREMOD_iofits.h"

#include "libmilkcommon/pixel_dispatch.h"
#include "timeutils.h"

#ifdef HAVE_CUDA
#    include <cublas_v2.h>
#    include <cuda_runtime.h>
#    include <cuda_runtime_api.h>
#    include <cusolverDn.h>
#    include <device_types.h>
#    include <pthread.h>
#endif


// CPU mode: Use MKL if available
// Otherwise use openBLAS
#include "milk_blas_lapacke.h"


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     *inmatA    = NULL;
static char     *inmatB    = NULL;
static uint64_t *transpA   = NULL;
static uint64_t *transpB   = NULL;
static char     *outM      = NULL;
static int32_t  *GPUdevice = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                 \
    X(".matA", &inmatA, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input matrix A") \
    X(".matB", &inmatB, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input matrix B") \
    X(".outM", &outM, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output matrix")


/* ================================================================
 * 5.  PARAMETER BINDINGS
 * ============================================================= */

MILK_PARAMS_DEFINE(SGEMM, FPS_PARAMS);

/**
 * @brief Convert image pixel data to a float array
 *
 * If the image is already FLOAT, returns the array pointer
 * directly (no allocation). Otherwise allocates a float
 * buffer and performs element-wise type conversion.
 *
 * Caller must free the returned pointer if and only if
 * img.md->datatype != _DATATYPE_FLOAT.
 *
 * @param img  Input image
 * @return Pointer to float pixel data
 */
static inline float *to_float_array(IMGID img)
{
    if (img.md->datatype == _DATATYPE_FLOAT)
    {
        return img.im->array.F;
    }

    uint64_t nel = img.md->nelement;
    float   *buf = (float *) malloc(sizeof(float) * nel);

#define _SGEMM_CONV_CASE(DT, ACC, CTYPE)               \
    case DT:                                           \
        for (uint64_t _ii = 0; _ii < nel; _ii++)       \
        {                                              \
            buf[_ii] = (float) img.im->array.ACC[_ii]; \
        }                                              \
        break;

    switch (img.md->datatype)
    {
        FOREACH_REAL_DATATYPE(_SGEMM_CONV_CASE)
    default:
        break;
    }
#undef _SGEMM_CONV_CASE

    return buf;
}


/**
 * @brief Computes the single-precision general matrix multiplication (SGEMM) of two matrices.
 *
 * This function computes the matrix product C = A * B.
 *
 * The function supports both CPU (OpenBLAS/MKL) and GPU (CUDA/cuBLAS) computation.
 * It also handles 2D and 3D input matrices, treating the first two dimensions of a 3D matrix as a single dimension.
 * Type conversion to float is performed if input matrices are not already float.
 *
 * @param imginA The input matrix A. Can be 2D or 3D.
 * @param imginB The input matrix B. Can be 2D or 3D.
 * @param outimg Pointer to the IMGID structure for the output matrix C. The function will create this image.
 * @param TranspA Flag indicating whether to transpose matrix A (1 for transpose, 0 for no transpose).
 * @param TranspB Flag indicating whether to transpose matrix B (1 for transpose, 0 for no transpose).
 * @param GPUdev The GPU device ID to use. If -1 or 99, CPU computation is used.
 * @return errno_t Returns RETURN_SUCCESS on successful computation, or an error code otherwise.
 */
errno_t computeSGEMM(IMGID  imginA,
                     IMGID  imginB,
                     IMGID *outimg,
                     int    TranspA,
                     int    TranspB,
                     int    GPUdev)
{
    DEBUG_TRACE_FSTART();

    printf("SGEMM\n");
    int SGEMMcomputed = 0;

    // Get input matrices A and B sizes (MxN)
    // if 3D cubes, group first 2 dimensions into M

    int inA_Mdim;
    int inA_Mdim0;
    int inA_Mdim1;
    int inA_Mdim1_active = 1; // is axis used ?

    int inA_Ndim;
    int inA_Ndim0;
    int inA_Ndim1;
    int inA_Ndim1_active = 1; // is axis used ?

    printf("inA %s naxis = %d\n", imginA.md->name, imginA.md->naxis);
    fflush(stdout);


    if (imginA.md->naxis == 3)
    {
        printf("inA_Mdim   : %d x %d\n", imginA.md->size[0], imginA.md->size[1]);
        inA_Mdim         = imginA.md->size[0] * imginA.md->size[1];
        inA_Mdim0        = imginA.md->size[0];
        inA_Mdim1        = imginA.md->size[1];
        inA_Mdim1_active = 1;

        printf("inA_Ndim    : %d\n", imginA.md->size[2]);
        inA_Ndim         = imginA.md->size[2];
        inA_Ndim0        = imginA.md->size[2];
        inA_Ndim1        = 1;
        inA_Ndim1_active = 0;
    }
    else
    {
        printf("inA_Mdim   : %d\n", imginA.md->size[0]);
        inA_Mdim         = imginA.md->size[0];
        inA_Mdim0        = imginA.md->size[0];
        inA_Mdim1        = 1;
        inA_Mdim1_active = 0;

        printf("inNdim    : %d\n", imginA.md->size[1]);
        inA_Ndim         = imginA.md->size[1];
        inA_Ndim0        = imginA.md->size[1];
        inA_Ndim1        = 1;
        inA_Ndim1_active = 0;
    }


    int inB_Mdim;
    int inB_Mdim0;
    int inB_Mdim1;
    //int inB_Mdim1_active = 1;

    int inB_Ndim;
    int inB_Ndim0;
    int inB_Ndim1;
    //int inB_Ndim1_active = 1;

    if (imginB.md->naxis == 3)
    {
        printf("inB_Mdim   : %d x %d\n", imginB.md->size[0], imginB.md->size[1]);
        inB_Mdim  = imginB.md->size[0] * imginB.md->size[1];
        inB_Mdim0 = imginB.md->size[0];
        inB_Mdim1 = imginB.md->size[1];
        //inB_Mdim1_active = 1;

        printf("inB_Ndim    : %d\n", imginB.md->size[2]);
        inB_Ndim  = imginB.md->size[2];
        inB_Ndim0 = imginB.md->size[2];
        inB_Ndim1 = 1;
        //inB_Ndim1_active = 0;
    }
    else
    {
        printf("inB_Mdim   : %d\n", imginB.md->size[0]);
        inB_Mdim  = imginB.md->size[0];
        inB_Mdim0 = imginB.md->size[0];
        inB_Mdim1 = 1;
        //inB_Mdim1_active = 0;

        printf("inB_Ndim    : %d\n", imginB.md->size[1]);
        inB_Ndim  = imginB.md->size[1];
        inB_Ndim0 = imginB.md->size[1];
        inB_Ndim1 = 1;
        //inB_Ndim1_active = 0;
    }


    // input to SGEMM function
    int Mdim, Ndim, Kdim;
    int Mdim0, Ndim0; //, Kdim0;
    int Mdim1, Ndim1; //, Kdim1;
    int Mdim1_active = 1;

    //int Ndim1_active = 1;


    // if no transpose
    Mdim         = inA_Mdim;
    Mdim0        = inA_Mdim0;
    Mdim1        = inA_Mdim1;
    Mdim1_active = inA_Mdim1_active;


    Ndim  = inB_Ndim;
    Ndim0 = inB_Ndim0;
    Ndim1 = inB_Ndim1;
    //Ndim1_active = inB_Ndim1_active;

    Kdim = inA_Ndim;

    if (TranspA == 1)
    {
        Mdim         = inA_Ndim;
        Mdim0        = inA_Ndim0;
        Mdim1        = inA_Ndim1;
        Mdim1_active = inA_Ndim1_active;

        Kdim = inA_Mdim;
    }
    if (TranspB == 1)
    {
        Ndim  = inB_Mdim;
        Ndim0 = inB_Mdim0;
        Ndim1 = inB_Mdim1;
        //Ndim1_active = inB_Mdim1_active;
    }

    printf("T %d %d  -> SGEMM  M=%d,(%d %d)  N=%d, (%d %d) K=%d\n", TranspA, TranspB, Mdim, Mdim0,
           Mdim1, Ndim, Ndim0, Ndim1, Kdim);
    printf("INPUT A  M %5d (%5d %5d)   x N %5d (%5d %5d)\n", inA_Mdim, inA_Mdim0, inA_Mdim1,
           inA_Ndim, inA_Ndim0, inA_Ndim1);
    printf("INPUT B  M %5d (%5d %5d)   x N %5d (%5d %5d)\n", inB_Mdim, inB_Mdim0, inB_Mdim1,
           inB_Ndim, inB_Ndim0, inB_Ndim1);


    // Create output
    //
    int outMdim = Mdim;
    int outNdim = Ndim;
    if (Mdim1_active == 0)
    {
        // 2D output
        outimg->mdt->naxis   = 2;
        outimg->mdt->size[0] = outMdim;
        outimg->mdt->size[1] = outNdim;
        outimg->mdt->size[2] = 1;
    }
    else
    {
        // 3D output
        outimg->mdt->naxis   = 3;
        outimg->mdt->size[0] = Mdim0;
        outimg->mdt->size[1] = Mdim1;
        outimg->mdt->size[2] = outNdim;
    }

    printf("OUTPUT  M %d   N %d  (%d %d %d)\n", outMdim, outNdim, outimg->mdt->size[0],
           outimg->mdt->size[1], outimg->mdt->size[2]);

    outimg->mdt->datatype = _DATATYPE_FLOAT;
    if (outimg->ID == -1)
    {
        createimagefromIMGID(outimg);
    }


    // Convert inputs to float (allocates if not already float)
    float *imarrayA = to_float_array(imginA);
    float *imarrayB = to_float_array(imginB);


    if ((GPUdev >= 0) && (GPUdev <= 99))
    {
#ifdef HAVE_CUDA
        //printf("Running SGEMM on GPU device %d\n", GPUdev);
        //fflush(stdout);

        const float  alf   = 1;
        const float  bet   = 0;
        const float *alpha = &alf;
        const float *beta  = &bet;


        float *d_inmatA;

        {
            cudaError_t cudaStat =
                cudaMalloc((void **) &d_inmatA, imginA.md->nelement * sizeof(float));
            if (cudaStat != cudaSuccess)
            {
                printf("device memory allocation failed");
                return EXIT_FAILURE;
            }
        }


        {
            cudaError_t stat = cudaMemcpy(d_inmatA, imarrayA, imginA.md->nelement * sizeof(float),
                                          cudaMemcpyHostToDevice);
            if (stat != cudaSuccess)
            {
                printf("cudaMemcpy failed\n");
                return EXIT_FAILURE;
            }
        }


        float *d_inmatB;

        {
            cudaError_t cudaStat =
                cudaMalloc((void **) &d_inmatB, imginB.md->nelement * sizeof(float));
            if (cudaStat != cudaSuccess)
            {
                printf("device memory allocation failed");
                return EXIT_FAILURE;
            }
        }


        {
            cudaError_t stat = cudaMemcpy(d_inmatB, imarrayB, imginB.md->nelement * sizeof(float),
                                          cudaMemcpyHostToDevice);
            if (stat != cudaSuccess)
            {
                printf("cudaMemcpy failed\n");
                return EXIT_FAILURE;
            }
        }


        float *d_outmat;
        {
            cudaError_t cudaStat =
                cudaMalloc((void **) &d_outmat, outimg->md->nelement * sizeof(float));
            if (cudaStat != cudaSuccess)
            {
                printf("device memory allocation failed");
                return EXIT_FAILURE;
            }
        }


        // Create a handle for CUBLAS
        cublasHandle_t handle;
        {
            cublasStatus_t stat = cublasCreate(&handle);
            if (stat != CUBLAS_STATUS_SUCCESS)
            {
                printf("cublasCreate failed\n");
                return EXIT_FAILURE;
            }
        }


        // Do the actual multiplication

        cublasOperation_t OPA = CUBLAS_OP_N;
        cublasOperation_t OPB = CUBLAS_OP_N;
        if (TranspA == 1)
        {
            OPA = CUBLAS_OP_T;
        }
        if (TranspB == 1)
        {
            OPB = CUBLAS_OP_T;
        }


        {
            cublasStatus_t stat =
                cublasSgemm(handle, OPA, OPB, Mdim, Ndim, Kdim, alpha, d_inmatA, inA_Mdim, d_inmatB,
                            inB_Mdim, beta, d_outmat, outMdim);

            if (stat != CUBLAS_STATUS_SUCCESS)
            {
                printf("cublasSgemm failed\n");
                return EXIT_FAILURE;
            }
        }

        cublasDestroy(handle);


        {
            cudaError_t stat =
                cudaMemcpy(outimg->im->array.F, d_outmat, outimg->md->nelement * sizeof(float),
                           cudaMemcpyDeviceToHost);
            if (stat != cudaSuccess)
            {
                printf("cudaMemcpy failed\n");
                return EXIT_FAILURE;
            }
        }

        cudaFree(d_inmatA);
        cudaFree(d_inmatB);
        cudaFree(d_outmat);

        SGEMMcomputed = 1;

#endif
    }

    if (SGEMMcomputed == 0)
    {
        arith_matmul_kernel_f(imarrayA, TranspA == 1, imarrayB, TranspB == 1, 1.0f, 0.0f,
                              outimg->im->array.F, (uint64_t) Mdim, (uint64_t) Kdim,
                              (uint64_t) Ndim);
        SGEMMcomputed = 1;
    }

    printf("Freeing float arrays\n");
    fflush(stdout);

    if (imginA.md->datatype != _DATATYPE_FLOAT)
    {
        free(imarrayA);
    }

    if (imginB.md->datatype != _DATATYPE_FLOAT)
    {
        free(imarrayB);
    }

    DEBUG_TRACE_FEXIT();
    if (SGEMMcomputed == 0)
    {
        return RETURN_FAILURE;
    }
    return RETURN_SUCCESS;
}


static MILK_HOT errno_t SGEMM_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    // input

    IMGID imginA = imgid_make_from_name(inmatA);
    resolveIMGID(&imginA, ERRMODE_WARN, dcimg, dcnimg);

    IMGID imginB = imgid_make_from_name(inmatB);
    if (imginA.ID == -1)
    {
        return RETURN_FAILURE;
    }
    resolveIMGID(&imginB, ERRMODE_WARN, dcimg, dcnimg);


    // output

    IMGID imgM = imgid_make_from_name(outM);
    if (imginB.ID == -1)
    {
        return RETURN_FAILURE;
    }


    MILK_PROCINFO_INIT(self)


    MILK_PROCINFO_LOOPSTART(self)
    {
        computeSGEMM(imginA, imginB, &imgM, *transpA, *transpB, *GPUdevice);
    }
    MILK_PROCINFO_END(self)

    imgid_free(&imginA);
    imgid_free(&imginB);
    imgid_free(&imgM);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK_RUNNABLE DESCRIPTOR AND FPSEXEC EXPORT
 * ============================================================= */

const MILK_RUNNABLE runnable_linalgebra_SGEMM = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "sgemm",
                 .cmdkey           = "sgemm",
                 .description      = "matrix-matrix multiply",
                 .description_long = "Perform single-precision matrix-matrix multiplication "
                                     "(SGEMM) using BLAS. Computes C = alpha * A * B + beta * C "
                                     "for dense matrices in shared memory." },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(SGEMM),
    .compute = SGEMM_compute,
};
EXPORT_FPSEXEC(milk, "linalg-SGEMM", runnable_linalgebra_SGEMM);
