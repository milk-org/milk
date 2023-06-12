/** @file GPU_SVD_computeControlMatrix.c
 */

#ifdef HAVE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <device_types.h>
#include <pthread.h>
#endif

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"

#ifdef HAVE_CUDA

extern int cuda_deviceCount;

//
// Computes control matrix
// Conventions:
//   n: number of actuators (= NB_MODES)
//   m: number of sensors  (= # of pixels)
// assumes m = n

errno_t GPU_SVD_computeControlMatrix(int         device,
                                     const char *ID_Rmatrix_name,
                                     const char *ID_Cmatrix_name,
                                     double      SVDeps,
                                     const char *ID_VTmatrix_name)
{
    DEBUG_TRACE_FSTART();

    cusolverDnHandle_t    cudenseH = NULL;
    cublasHandle_t        cublasH  = NULL;
    cublasStatus_t        cublas_status;   // = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t      cusolver_status; // = CUSOLVER_STATUS_SUCCESS;
    struct cudaDeviceProp deviceProp;

    imageID   ID_Rmatrix, ID_Cmatrix, ID_VTmatrix;
    uint8_t   datatype;
    uint32_t *arraysizetmp;
    int       lda, ldu, ldvt;

    float      *d_A      = NULL; // linear memory of GPU
    float      *h_A      = NULL;
    float      *d_S      = NULL; // linear memory of GPU
    float      *d_U      = NULL; // linear memory of GPU
    float      *h_U1     = NULL;
    float      *d_VT     = NULL; // linear memory of GPU
    float      *d_M      = NULL; // linear memory of GPU
    float      *d_U1     = NULL; // linear memory of GPU
    float      *d_Work   = NULL; // linear memory of GPU
    cudaError_t cudaStat = cudaSuccess;
    int        *devInfo  = NULL; // info in gpu (device copy)
    int         Lwork;
    float      *rwork;

    float *Sarray;
    //float *Aarray;
    long  i;
    FILE *fp;
    char  fname[200];

    int info_gpu;

    double          time1sec, time2sec;
    struct timespec tnow;

    float   val;
    float   alpha = 1.0;
    float   beta  = 0.0;
    imageID ID;

    float *h_M;
    long   cnt0;

    cudaGetDeviceCount(&cuda_deviceCount);
    printf("%d devices found\n", cuda_deviceCount);
    fflush(stdout);
    printf("\n");
    for(int k = 0; k < cuda_deviceCount; ++k)
    {
        cudaGetDeviceProperties(&deviceProp, k);
        printf("Device %d [ %20s ]  has compute capability %d.%d.\n",
               k,
               deviceProp.name,
               deviceProp.major,
               deviceProp.minor);
        printf(
            "  Total amount of global memory:                 %.0f MBytes "
            "(%llu bytes)\n",
            (float) deviceProp.totalGlobalMem / 1048576.0f,
            (unsigned long long) deviceProp.totalGlobalMem);
        printf("  (%2d) Multiprocessors\n", deviceProp.multiProcessorCount);
        printf(
            "  GPU Clock rate:                                %.0f MHz (%0.2f "
            "GHz)\n",
            deviceProp.clockRate * 1e-3f,
            deviceProp.clockRate * 1e-6f);
        printf("\n");
    }

    if(device < cuda_deviceCount)
    {
        cudaSetDevice(device);
    }
    else
    {
        printf("Invalid Device : %d / %d\n", device, cuda_deviceCount);
        exit(0);
    }

    cudaDeviceReset();

    printf("step 1a: create cudense handle ...");
    fflush(stdout);
    cusolver_status = cusolverDnCreate(&cudenseH);
    if(cusolver_status != CUSOLVER_STATUS_SUCCESS)
    {
        printf("CUSOLVER initialization failed\n");
        return EXIT_FAILURE;
    }
    printf(" done\n");
    fflush(stdout);

    printf("step 1b: create cublas handle ...");
    fflush(stdout);
    cublas_status = cublasCreate(&cublasH);
    if(cublas_status != CUBLAS_STATUS_SUCCESS)
    {
        printf("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    printf(" done\n");
    fflush(stdout);

    clock_gettime(CLOCK_MILK, &tnow);
    time1sec = 1.0 * ((long) tnow.tv_sec) + 1.0e-9 * tnow.tv_nsec;

    list_image_ID();

    ID_Rmatrix = image_ID(ID_Rmatrix_name);

    datatype = data.image[ID_Rmatrix].md[0].datatype;
    if(datatype != _DATATYPE_FLOAT)
    {
        printf("wrong type\n");
        exit(EXIT_FAILURE);
    }

    uint32_t m;
    uint32_t n;

    if(data.image[ID_Rmatrix].md[0].naxis == 3)
    {
        m = data.image[ID_Rmatrix].md[0].size[0] *
            data.image[ID_Rmatrix].md[0].size[1];
        n = data.image[ID_Rmatrix].md[0].size[2];
        printf("3D image -> %d %d\n", m, n);
        fflush(stdout);
    }
    else
    {
        m = data.image[ID_Rmatrix].md[0].size[0];
        n = data.image[ID_Rmatrix].md[0].size[1];
        printf("2D image -> %d %d\n", m, n);
        fflush(stdout);
    }

    if(m != n)
    {
        printf("ERROR: m must be equal to n\n");
        exit(EXIT_FAILURE);
    }

    cudaStat = cudaMalloc((void **) &d_A, sizeof(float) * n * m);
    if(cudaStat != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n",
               cudaStat,
               __LINE__);
        exit(EXIT_FAILURE);
    }

    h_A = (float *) malloc(sizeof(float) * m * n);

    cudaStat = cudaMemcpy(d_A,
                          data.image[ID_Rmatrix].array.F,
                          sizeof(float) * m * n,
                          cudaMemcpyHostToDevice);
    if(cudaStat != cudaSuccess)
    {
        printf("cudaMemcpy d_A returned error code %d, line(%d)\n",
               cudaStat,
               __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaStat = cudaMalloc((void **) &d_S, sizeof(float) * n);
    if(cudaStat != cudaSuccess)
    {
        printf("cudaMalloc d_S returned error code %d, line(%d)\n",
               cudaStat,
               __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaStat = cudaMalloc((void **) &d_U, sizeof(float) * m * m);
    if(cudaStat != cudaSuccess)
    {
        printf("cudaMalloc d_U returned error code %d, line(%d)\n",
               cudaStat,
               __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaStat = cudaMalloc((void **) &d_VT, sizeof(float) * n * n);
    if(cudaStat != cudaSuccess)
    {
        printf("cudaMalloc d_VT returned error code %d, line(%d)\n",
               cudaStat,
               __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaStat = cudaMalloc((void **) &devInfo, sizeof(int));
    if(cudaStat != cudaSuccess)
    {
        printf("cudaMalloc devInfo returned error code %d, line(%d)\n",
               cudaStat,
               __LINE__);
        exit(EXIT_FAILURE);
    }

    lda             = m;
    ldu             = m;
    ldvt            = n;
    cusolver_status = cusolverDnSgesvd_bufferSize(cudenseH, m, n, &Lwork);
    if(cusolver_status != CUSOLVER_STATUS_SUCCESS)
    {
        printf("CUSOLVER DnSgesvd_bufferSize failed\n");
        return EXIT_FAILURE;
    }

    cudaStat = cudaMalloc((void **) &d_Work, sizeof(float) * Lwork);
    if(cudaStat != cudaSuccess)
    {
        printf("cudaMalloc d_Work returned error code %d, line(%d)\n",
               cudaStat,
               __LINE__);
        exit(EXIT_FAILURE);
    }

    rwork = (float *) malloc(5 * sizeof(float) * n);

    printf("START GPU COMPUTATION (%d x %d)  buffer size = %d ...",
           m,
           n,
           Lwork);
    fflush(stdout);
    cusolverDnSgesvd(cudenseH,
                     'A',
                     'A',
                     m,
                     n,
                     d_A,
                     lda,
                     d_S,
                     d_U,
                     ldu,
                     d_VT,
                     ldvt,
                     d_Work,
                     Lwork,
                     NULL,
                     devInfo);
    printf(" SYNC ");
    fflush(stdout);
    cudaStat = cudaDeviceSynchronize();
    printf(" DONE\n");
    fflush(stdout);

    cudaStat =
        cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    printf("after gesvd: info_gpu = %d\n", info_gpu);

    FUNC_CHECK_RETURN(create_2Dimage_ID(ID_VTmatrix_name, n, n, &ID_VTmatrix));

    cudaStat = cudaMemcpy(data.image[ID_VTmatrix].array.F,
                          d_VT,
                          sizeof(float) * n * n,
                          cudaMemcpyDeviceToHost);
    if(cudaStat != cudaSuccess)
    {
        printf("cudaMemcpy returned error code %d, line(%d)\n",
               cudaStat,
               __LINE__);
        exit(EXIT_FAILURE);
    }

    save_fits(ID_VTmatrix_name, "matVT0.fits");

    Sarray = (float *) malloc(sizeof(float) * n);
    //    Aarray = (float*) malloc(sizeof(float)*m*n);
    cudaStat =
        cudaMemcpy(Sarray, d_S, sizeof(float) * n, cudaMemcpyDeviceToHost);
    if(cudaStat != cudaSuccess)
    {
        printf("cudaMemcpy returned error code %d, line(%d)\n",
               cudaStat,
               __LINE__);
        exit(EXIT_FAILURE);
    }

    sprintf(fname, "eigenv.dat.gsl");
    if((fp = fopen(fname, "w")) == NULL)
    {
        printf("ERROR: cannot create file \"%s\"\n", fname);
        exit(0);
    }
    for(i = 0; i < n; i++)
    {
        fprintf(fp, "%5ld %20g %20g\n", i, Sarray[i], Sarray[i] / Sarray[0]);
    }
    fclose(fp);

    FUNC_CHECK_RETURN(create_2Dimage_ID("matU", m, m, &ID));

    cudaMemcpy(data.image[ID].array.F,
               d_U,
               sizeof(float) * m * m,
               cudaMemcpyDeviceToHost);
    save_fits("matU", "matU.fits");

    h_U1     = (float *) malloc(sizeof(float) * m * n);
    cudaStat = cudaMalloc((void **) &d_U1, sizeof(float) * m * n);
    if(cudaStat != cudaSuccess)
    {
        printf("cudaMalloc d_U1 returned error code %d, line(%d)\n",
               cudaStat,
               __LINE__);
        exit(EXIT_FAILURE);
    }
    for(uint32_t ii = 0; ii < m; ii++)
        for(uint32_t jj = 0; jj < n; jj++)
        {
            h_U1[jj * m + ii] = data.image[ID].array.F[jj * m + ii];
        }
    cudaMemcpy(d_U1, h_U1, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    free(h_U1);

    FUNC_CHECK_RETURN(create_2Dimage_ID("matU1", m, n, &ID));

    cudaMemcpy(data.image[ID].array.F,
               d_U1,
               sizeof(float) * m * n,
               cudaMemcpyDeviceToHost);
    save_fits("matU1", "matU1.fits");

    printf("SVDeps = %f\n", SVDeps);
    cnt0 = 0;
    // multiply lines of VT by 1/eigenval
    for(uint32_t ii = 0; ii < n; ii++)
    {
        if(Sarray[ii] > Sarray[0] * SVDeps)
        {
            val = 1.0 / (Sarray[ii]);
            cnt0++;
        }
        else
        {
            val = 0.0;
        }

        for(uint32_t jj = 0; jj < n; jj++)
        {
            data.image[ID_VTmatrix].array.F[jj * n + ii] *= val;
        }
    }
    printf("%ld eigenvalues kept\n", cnt0);

    // copy VT back to GPU
    cudaStat = cudaMemcpy(d_VT,
                          data.image[ID_VTmatrix].array.F,
                          sizeof(float) * n * n,
                          cudaMemcpyHostToDevice);
    if(cudaStat != cudaSuccess)
    {
        printf("cudaMemcpy returned error code %d, line(%d)\n",
               cudaStat,
               __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaStat = cudaMalloc((void **) &d_M, sizeof(float) * n * m);
    if(cudaStat != cudaSuccess)
    {
        printf("cudaMalloc d_M returned error code %d, line(%d)\n",
               cudaStat,
               __LINE__);
        exit(EXIT_FAILURE);
    }

    save_fits(ID_VTmatrix_name, "matVT.fits");

    cublasStatus_t cublasStat = cublasSgemm(cublasH,
                                            CUBLAS_OP_T,
                                            CUBLAS_OP_T,
                                            n,
                                            m,
                                            n,
                                            &alpha,
                                            d_VT,
                                            n,
                                            d_U,
                                            m,
                                            &beta,
                                            d_M,
                                            n);
    if(cublasStat != CUBLAS_STATUS_SUCCESS)
    {
        printf("cublasSgemm returned error code %d, line(%d)\n",
               cudaStat,
               __LINE__);
        exit(EXIT_FAILURE);
    }

    arraysizetmp = (uint32_t *) malloc(sizeof(uint32_t) * 3);

    if(data.image[ID_Rmatrix].md[0].naxis == 3)
    {
        arraysizetmp[0] = data.image[ID_Rmatrix].md[0].size[0];
        arraysizetmp[1] = data.image[ID_Rmatrix].md[0].size[1];
        arraysizetmp[2] = n;
    }
    else
    {
        arraysizetmp[0] = m;
        arraysizetmp[1] = n;
    }

    FUNC_CHECK_RETURN(create_image_ID(ID_Cmatrix_name,
                                      data.image[ID_Rmatrix].md[0].naxis,
                                      arraysizetmp,
                                      _DATATYPE_FLOAT,
                                      0,
                                      0,
                                      0,
                                      &ID_Cmatrix));

    //   cudaStat = cudaMemcpy(data.image[ID_Cmatrix].array.F, d_M, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

    h_M = (float *) malloc(sizeof(float) * m * n);
    cudaStat =
        cudaMemcpy(h_M, d_M, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    for(uint32_t ii = 0; ii < m; ii++)
        for(uint32_t jj = 0; jj < n; jj++)
        {
            data.image[ID_Cmatrix].array.F[jj * m + ii] = h_M[ii * n + jj];
        }

    //cudaStat = cudaMemcpy(data.image[ID_Cmatrix].array.F, d_VT, sizeof(float)*n*n, cudaMemcpyDeviceToHost);
    if(cudaStat != cudaSuccess)
    {
        printf("cudaMemcpy returned error code %d, line(%d)\n",
               cudaStat,
               __LINE__);
        free(arraysizetmp);
        exit(EXIT_FAILURE);
    }

    cudaFree(d_A);
    cudaFree(d_S);
    cudaFree(d_U);
    cudaFree(d_VT);
    cudaFree(d_Work);
    cudaFree(devInfo);
    cudaFree(d_M);
    cudaFree(d_U1);

    clock_gettime(CLOCK_MILK, &tnow);
    time2sec = 1.0 * ((long) tnow.tv_sec) + 1.0e-9 * tnow.tv_nsec;

    printf("time = %8.3f s\n", 1.0 * (time2sec - time1sec));

    if(cublasH)
    {
        cublasDestroy(cublasH);
    }
    if(cudenseH)
    {
        cusolverDnDestroy(cudenseH);
    }

    cudaDeviceReset();

    free(arraysizetmp);
    free(Sarray);
    free(rwork);
    free(h_A);
    free(h_M);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

#endif
