#include "CommandLineInterface/CLIcore.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef HAVE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#endif

#include <math.h>

// Local variables pointers
static char *inimname;

static CLICMDARGDEF farg[] = {{
        CLIARG_IMG,
        ".in_name",
        "input image",
        "im1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname,
        NULL
    }
};

static CLICMDDATA CLIcmddata = {"PCAdecomp",
                                "Principal Components Analysis decomposition",
                                CLICMD_FIELDS_DEFAULTS
                               };

// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}

void printMatrix(int m, int n, const double *A, int lda, const char *name)
{
    long cnt = 0;

    for(int row = 0; row < m; row++)
    {
        for(int col = 0; col < n; col++)
        {
            if(cnt < 100)
            {
                double Areg = A[row + col * lda];
                printf("%s(%d,%d) = %f\n", name, row, col, Areg);
                cnt++;
            }
        }
    }
}

static imageID image_PCAdecomp(IMGID *img)
{
    DEBUG_TRACE_FSTART();

    // Create image if needed
    //imageID ID = img->ID;

    printf("Image size : %u %u %u\n",
           img->md->size[0],
           img->md->size[1],
           img->md->size[2]);

#ifdef HAVE_CUDA
    cusolverDnHandle_t cusolverH     = NULL;
    cublasHandle_t     cublasH       = NULL;
    cublasStatus_t     cublas_status = CUBLAS_STATUS_SUCCESS;

    const int m = img->md->size[2]; // number of sammples
    const int n = img->md->size[0] *
                  img->md->size[1]; // number of image pixels in each sample

    printf("A size   %d %d\n", m, n);

    const int lda = m;

    double *A = (double *) malloc(sizeof(double) * lda * n);

    for(int ii = 0; ii < n; ii++)  // pixel
    {
        for(int kk = 0; kk < m; kk++)  // sample
        {
            A[ii * m + kk] = 1.0 * img->im->array.UI16[kk * n + ii];
        }
    }

    double  S[n]; // singular value
    double *d_A     = NULL;
    double *d_S     = NULL;
    double *d_U     = NULL;
    double *d_VT    = NULL;
    int    *devInfo = NULL;
    double *d_work  = NULL;
    double *d_rwork = NULL;
    double *d_W     = NULL; // W = S*VT

    int          lwork       = 0;
    int          info_gpu    = 0;
    const double h_one       = 1;
    const double h_minus_one = -1;

    printf("A = \n");
    printMatrix(m, n, A, lda, "A");
    printf("=====\n");

    // step 1: create cusolverDn/cublas handle
    cusolverStatus_t cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    (void) cusolver_status;

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    // step 2: copy A and B to device
    {
        cudaError_t cudaStat =
            cudaMalloc((void **) &d_A, sizeof(double) * lda * n);
        assert(cudaSuccess == cudaStat);
        (void) cudaStat;
    }

    {
        cudaError_t cudaStat = cudaMalloc((void **) &d_S, sizeof(double) * n);
        assert(cudaSuccess == cudaStat);
        (void) cudaStat;
    }

    {
        cudaError_t cudaStat =
            cudaMalloc((void **) &d_U, sizeof(double) * lda * m);
        assert(cudaSuccess == cudaStat);
        (void) cudaStat;
    }

    {
        cudaError_t cudaStat =
            cudaMalloc((void **) &d_VT, sizeof(double) * lda * n);
        assert(cudaSuccess == cudaStat);
        (void) cudaStat;
    }

    {
        cudaError_t cudaStat = cudaMalloc((void **) &devInfo, sizeof(int));
        assert(cudaSuccess == cudaStat);
        (void) cudaStat;
    }

    {
        cudaError_t cudaStat =
            cudaMalloc((void **) &d_W, sizeof(double) * lda * n);
        assert(cudaSuccess == cudaStat);
        (void) cudaStat;
    }

    {
        cudaError_t cudaStat = cudaMemcpy(d_A,
                                          A,
                                          sizeof(double) * lda * n,
                                          cudaMemcpyHostToDevice);
        assert(cudaSuccess == cudaStat);
        (void) cudaStat;
    }

    double dR0    = 0.0;
    cublas_status = cublasDnrm2_v2(cublasH, lda * n, d_A, 1, &dR0);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    (void) cublas_status;

    printf("dR0 = %f\n", dR0);

    // step 3: query working space of SVD
    cusolver_status = cusolverDnDgesvd_bufferSize(cusolverH, m, n, &lwork);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    {
        cudaError_t cudaStat =
            cudaMalloc((void **) &d_work, sizeof(double) * lwork);
        assert(cudaSuccess == cudaStat);
        (void) cudaStat;
    }

    // step 4: compute SVD
    signed char jobu  = 'A'; // all m columns of U
    signed char jobvt = 'A'; // all n columns of VT
    cusolver_status   = cusolverDnDgesvd(cusolverH,
                                         jobu,
                                         jobvt,
                                         m,
                                         n,
                                         d_A,
                                         lda,
                                         d_S,
                                         d_U,
                                         lda, // ldu
                                         d_VT,
                                         lda, // ldvt,
                                         d_work,
                                         lwork,
                                         d_rwork,
                                         devInfo);

    {
        cudaError_t cudaStat = cudaDeviceSynchronize();
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
        assert(cudaSuccess == cudaStat);
        (void) cudaStat;
    }

    double *U = (double *) malloc(sizeof(double) * lda * m);
    {
        cudaError_t cudaStat = cudaMemcpy(U,
                                          d_U,
                                          sizeof(double) * lda * m,
                                          cudaMemcpyDeviceToHost);
        assert(cudaSuccess == cudaStat);
        (void) cudaStat;
    }

    double *VT = (double *) malloc(sizeof(double) * lda * n);
    {
        cudaError_t cudaStat = cudaMemcpy(VT,
                                          d_VT,
                                          sizeof(double) * lda * n,
                                          cudaMemcpyDeviceToHost);
        assert(cudaSuccess == cudaStat);
        (void) cudaStat;
    }

    // Write PCA compents
    uint32_t  lmax = 1000;
    uint32_t *imPCAsize;
    imPCAsize    = (uint32_t *) malloc(sizeof(uint32_t) * 3);
    imPCAsize[0] = img->md->size[0];
    imPCAsize[1] = img->md->size[1];
    imPCAsize[2] = lmax;
    imageID outPCAID;
    create_image_ID("imPCA",
                    3,
                    imPCAsize,
                    _DATATYPE_DOUBLE,
                    0,
                    10,
                    0,
                    &outPCAID);
    for(uint32_t jj = 0; jj < lmax; jj++)
    {
        for(uint32_t ii = 0; ii < (uint32_t) n; ii++)
        {
            data.image[outPCAID].array.D[jj * n + ii] = VT[ii * m + jj];
        }
    }
    free(imPCAsize);

    {
        cudaError_t cudaStat =
            cudaMemcpy(S, d_S, sizeof(double) * n, cudaMemcpyDeviceToHost);
        assert(cudaSuccess == cudaStat);
        (void) cudaStat;
    }

    {
        cudaError_t cudaStat =
            cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
        assert(cudaSuccess == cudaStat);
        (void) cudaStat;
    }

    printf("after gesvd: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);
    printf("=====\n");

    printf("S = \n");
    printMatrix(n, 1, S, lda, "S");
    printf("=====\n");

    printf("U = \n");
    //printMatrix(m, m, U, lda, "U");
    printf("=====\n");

    printf("VT = \n");
    //printMatrix(n, n, VT, lda, "VT");
    printf("=====\n");

    // process S
    double *d_S1 = NULL;
    {
        cudaError_t cudaStat = cudaMalloc((void **) &d_S1, sizeof(double) * n);
        assert(cudaSuccess == cudaStat);
        (void) cudaStat;
    }

    for(uint32_t k = 50; k < (uint32_t) n; k++)
    {
        S[k] = 0.0;
    }

    {
        cudaError_t cudaStat =
            cudaMemcpy(d_S1, S, sizeof(double) * n, cudaMemcpyHostToDevice);
        assert(cudaSuccess == cudaStat);
        (void) cudaStat;
    }

    // step 6: |A - U*S*VT|
    // W = S*VT
    cublas_status = cublasDdgmm(cublasH,
                                CUBLAS_SIDE_LEFT,
                                n,
                                n,
                                d_VT,
                                lda,
                                d_S1,
                                1,
                                d_W,
                                lda);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    // A := -U*W + A
    {
        cudaError_t cudaStat = cudaMemcpy(d_A,
                                          A,
                                          sizeof(double) * lda * n,
                                          cudaMemcpyHostToDevice);
        assert(cudaSuccess == cudaStat);
        (void) cudaStat;
    }

    cublas_status = cublasDgemm_v2(cublasH,
                                   CUBLAS_OP_N,  // U
                                   CUBLAS_OP_N,  // W
                                   m,            // number of rows of A
                                   n,            // number of columns of A
                                   n,            // number of columns of U
                                   &h_minus_one, /* host pointer */
                                   d_U,          // U
                                   lda,
                                   d_W, // W
                                   lda,
                                   &h_one, /* hostpointer */
                                   d_A,
                                   lda);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    double dR_fro = 0.0;
    cublas_status = cublasDnrm2_v2(cublasH, lda * n, d_A, 1, &dR_fro);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    printf("|A - U*S*VT| = %E   %E\n", dR_fro, dR_fro / dR0);

    // copy residual to host
    {
        cudaError_t cudaStat = cudaMemcpy(A,
                                          d_A,
                                          sizeof(double) * lda * n,
                                          cudaMemcpyDeviceToHost);
        assert(cudaSuccess == cudaStat);
        (void) cudaStat;
    }

    IMGID imgAres = makeIMGID_3D("imAres",
                                 img->md->size[0],
                                 img->md->size[1],
                                 img->md->size[2]);
    imcreateIMGID(&imgAres);
    for(int ii = 0; ii < n; ii++)  // pixel
    {
        for(int kk = 0; kk < m; kk++)  // sample
        {
            imgAres.im->array.F[kk * n + ii] = (float) A[ii * m + kk];
        }
    }

    // free resources
    if(d_A)
    {
        cudaFree(d_A);
    }
    if(d_S)
    {
        cudaFree(d_S);
    }
    if(d_U)
    {
        cudaFree(d_U);
    }
    if(d_VT)
    {
        cudaFree(d_VT);
    }
    if(devInfo)
    {
        cudaFree(devInfo);
    }
    if(d_work)
    {
        cudaFree(d_work);
    }
    if(d_rwork)
    {
        cudaFree(d_rwork);
    }
    if(d_W)
    {
        cudaFree(d_W);
    }

    if(cublasH)
    {
        cublasDestroy(cublasH);
    }
    if(cusolverH)
    {
        cusolverDnDestroy(cusolverH);
    }

    cudaDeviceReset();

    free(A);
    free(U);
    free(VT);

#endif

    DEBUG_TRACE_FEXIT();
    return (img->ID);
}

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    DEBUG_TRACEPOINT("PCA of %s", inimname);

    IMGID img = mkIMGID_from_name(inimname);
    resolveIMGID(&img, ERRMODE_ABORT);

    printf("PCA of %s\n", inimname);

    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    image_PCAdecomp(&img);

    //DEBUG_TRACEPOINT("update output ID %ld", img.ID);
    //processinfo_update_output_stream(processinfo, img.ID);
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t
CLIADDCMD_cudacomp__PCAdecomp()
{
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
